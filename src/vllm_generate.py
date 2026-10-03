"""vLLM generation worker for eval_closedbook.py (runs under .venv-vllm, which has its own torch).

Reads JSONL rows {"qa_id", "messages"} and writes JSONL rows {"qa_id", "prediction"}. Greedy
decoding; the chat template is applied with enable_thinking=False. With --adapter, the base model
is loaded with LoRA enabled and every request uses that adapter.

Standalone on purpose: imports only the standard library and vLLM.

Usage (normally called by eval_closedbook.py):
    .venv-vllm/bin/python src/vllm_generate.py --model Qwen/Qwen3-8B [--adapter DIR] --input in.jsonl --output out.jsonl
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--check", action="store_true", help="only verify vLLM imports and the API used here")
    args = ap.parse_args()

    import inspect
    import os

    # FlashInfer's sampler JIT-compiles kernels with nvcc at engine start-up; this machine has the
    # NVIDIA driver but no CUDA toolkit, so use vLLM's built-in sampler (attention is unaffected).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    if args.check:
        params = inspect.signature(LLM.chat).parameters
        missing = [p for p in ("lora_request", "chat_template_kwargs") if p not in params]
        print(json.dumps({"vllm_ok": not missing, "missing": missing}))
        return

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    llm = LLM(model=args.model, dtype="bfloat16", enable_lora=bool(args.adapter),
              max_lora_rank=args.max_lora_rank, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    lora = LoRARequest("finetuned", 1, args.adapter) if args.adapter else None
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outs = llm.chat([r["messages"] for r in rows], params, lora_request=lora,
                    chat_template_kwargs={"enable_thinking": False})
    with open(args.output, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"qa_id": r["qa_id"], "prediction": o.outputs[0].text.strip()},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
