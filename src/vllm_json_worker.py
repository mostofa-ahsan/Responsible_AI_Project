"""vLLM worker for the local evaluation (runs under .venv-vllm). Standard library + vLLM only.

Input JSONL rows {"id", "messages"}; results are APPENDED to --output after every chunk, so a killed run
keeps its work and a rerun only sends the missing ids (the caller filters).
  --mode json   greedy decoding with guided JSON (--schema) -> {"id", "text"}
  --mode yesno  one greedy token with logprobs -> {"id", "p_yes"} = P(Yes) / (P(Yes) + P(No))  (MiniCheck)
--deadline (unix seconds): no new chunk starts after it.
"""

import argparse
import json
import math
import os
import sys
import time


def block_api_clients():
    def _blocked(self, *a, **k):
        raise AssertionError("API client construction is forbidden (LLM_OFFLINE=1)")
    for mod, names in (("anthropic", ("Anthropic", "AsyncAnthropic")), ("openai", ("OpenAI", "AsyncOpenAI"))):
        try:
            m = __import__(mod)
        except Exception:
            continue
        for n in names:
            if getattr(m, n, None) is not None:
                getattr(m, n).__init__ = _blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["json", "yesno"], default="json")
    ap.add_argument("--schema")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=1500)
    ap.add_argument("--deadline", type=float)
    ap.add_argument("--mistral3", action="store_true", help="Mistral3 multimodal checkpoint: text-only")
    ap.add_argument("--mistral-format", action="store_true", help="tokenizer/config/load format = mistral")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--tokenizer", help="separate tokenizer directory")
    args = ap.parse_args()
    assert os.environ.get("LLM_OFFLINE") == "1", "LLM_OFFLINE=1 required"
    block_api_clients()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    kw = dict(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, trust_remote_code=args.trust_remote_code, seed=0)
    if args.tokenizer:
        kw["tokenizer"] = args.tokenizer
    if args.mistral3:
        kw["limit_mm_per_prompt"] = {"image": 0}
    if args.mistral_format:
        kw.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[worker] model loaded in {time.time() - t0:.0f}s; {len(rows)} prompts", flush=True)
    if args.mode == "json":
        schema = json.load(open(args.schema))
        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                structured_outputs=StructuredOutputsParams(json=schema))
    else:
        params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    done = 0
    with open(args.output, "a", encoding="utf-8") as f:
        for i in range(0, len(rows), args.chunk):
            if args.deadline and time.time() > args.deadline:
                print(f"[worker] deadline reached; {done}/{len(rows)} done", flush=True)
                break
            pause = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", ".gpu_pause")
            while os.path.exists(pause):      # thermal pause from scripts/gpu_watchdog.sh
                print(f"[worker] {time.strftime('%H:%M:%S')} GPU pause flag set: sleeping 5 min", flush=True)
                time.sleep(300)
            part = rows[i:i + args.chunk]
            t1 = time.time()
            outs = llm.chat([r["messages"] for r in part], params, use_tqdm=True,
                            chat_template_kwargs={"enable_thinking": False})
            for r, o in zip(part, outs):
                if args.mode == "json":
                    rec = {"id": r["id"], "text": o.outputs[0].text, "finish": o.outputs[0].finish_reason}
                else:
                    py = pn = 0.0
                    lp = (o.outputs[0].logprobs or [{}])[0]
                    for tok_id, l in lp.items():
                        s = (l.decoded_token or "").replace("\u2581", "").replace("\u0120", "").strip().lower()
                        if s == "yes":
                            py += math.exp(l.logprob)
                        elif s == "no":
                            pn += math.exp(l.logprob)
                    rec = {"id": r["id"], "p_yes": py / (py + pn) if py + pn > 0 else None,
                           "top": (o.outputs[0].text or "")[:10]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done += len(part)
            print(f"[worker] {time.strftime('%H:%M:%S')} {done}/{len(rows)} "
                  f"({len(part) / max(time.time() - t1, 1e-6):.1f}/s)", flush=True)
    print(f"[worker] finished {done}/{len(rows)} in {time.time() - t0:.0f}s", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
