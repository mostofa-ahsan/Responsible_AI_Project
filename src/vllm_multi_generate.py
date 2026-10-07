"""vLLM generation worker for several prompt variants / LoRA adapters of ONE base model (runs under .venv-vllm).

Input JSONL rows {"id", "system", "question", "adapter": path or null}. Results are APPENDED per chunk to --output as
{"id", "prediction", "n_tokens"}; ids already in --output are skipped by the caller. Greedy decoding, chat template with
enable_thinking=False (same settings as src/vllm_generate.py). With --lora, the engine is started with LoRA enabled and
each row uses its adapter; without it, adapters must be null. --stats writes {"seconds", "output_tokens", "prompts"} for
throughput. Standard library + vLLM only.
"""

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--quantization")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()
    assert os.environ.get("LLM_OFFLINE") == "1", "LLM_OFFLINE=1 required"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    kw = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    if args.lora:
        kw.update(enable_lora=True, max_lora_rank=args.max_lora_rank, max_loras=1)
    if args.quantization:
        kw["quantization"] = args.quantization
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[multi] model loaded in {time.time() - t0:.0f}s; {len(rows)} prompts", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    lora_ids = {}
    rows.sort(key=lambda r: r.get("adapter") or "")          # one adapter at a time
    gen_s, out_tok = 0.0, 0
    pause = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", ".gpu_pause")
    with open(args.output, "a", encoding="utf-8") as f:
        for i in range(0, len(rows), args.chunk):
            while os.path.exists(pause):
                print(f"[multi] {time.strftime('%H:%M:%S')} GPU pause flag set: sleeping 5 min", flush=True)
                time.sleep(300)
            part = rows[i:i + args.chunk]
            groups = {}
            for r in part:
                groups.setdefault(r.get("adapter"), []).append(r)
            for ad, rs in groups.items():
                lr = None
                if ad:
                    lora_ids.setdefault(ad, len(lora_ids) + 1)
                    lr = LoRARequest(f"a{lora_ids[ad]}", lora_ids[ad], ad)
                t1 = time.time()
                if all(r.get("prompt_token_ids") for r in rs):     # pre-tokenized with the training chat function
                    from vllm.inputs import TokensPrompt
                    outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in rs], params,
                                        lora_request=lr, use_tqdm=True)
                else:
                    outs = llm.chat([[{"role": "system", "content": r["system"]}, {"role": "user", "content": r["question"]}]
                                     for r in rs], params, lora_request=lr, use_tqdm=True,
                                    chat_template_kwargs={"enable_thinking": False})
                gen_s += time.time() - t1
                for r, o in zip(rs, outs):
                    n = len(o.outputs[0].token_ids)
                    out_tok += n
                    f.write(json.dumps({"id": r["id"], "prediction": o.outputs[0].text.strip(), "n_tokens": n},
                                       ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"[multi] {time.strftime('%H:%M:%S')} {min(i + args.chunk, len(rows))}/{len(rows)}", flush=True)
    if args.stats:
        json.dump({"seconds": round(gen_s, 1), "output_tokens": out_tok, "prompts": len(rows),
                   "load_seconds": round(time.time() - t0 - gen_s, 1)}, open(args.stats, "w"))
    print(f"[multi] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
