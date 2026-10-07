"""Diagnostic helper (runs under .venv-vllm): the exact prompt token ids vLLM builds for chat requests, plus greedy
answers with the same settings as production (src/vllm_generate.py / src/vllm_multi_generate.py: llm.chat, greedy,
max_tokens 512, chat_template_kwargs enable_thinking=False, LoRA adapter). Standard library + vLLM only.

Input JSONL {"id", "messages"}; output JSONL {"id", "prompt_token_ids", "output_token_ids", "text", "finish_reason"}.
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--quantization", help="e.g. bitsandbytes (in-flight NF4) to match the QLoRA training base")
    args = ap.parse_args()
    assert os.environ.get("LLM_OFFLINE") == "1"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    kw = {"quantization": args.quantization} if args.quantization else {}
    llm = LLM(model=args.model, dtype="bfloat16", enable_lora=bool(args.adapter), max_lora_rank=16,
              max_model_len=4096, gpu_memory_utilization=0.85, **kw)
    lora = LoRARequest("a", 1, args.adapter) if args.adapter else None
    outs = llm.chat([r["messages"] for r in rows], SamplingParams(temperature=0.0, max_tokens=512), lora_request=lora,
                    chat_template_kwargs={"enable_thinking": False})
    with open(args.output, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "prompt_token_ids": list(o.prompt_token_ids),
                                "output_token_ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text,
                                "finish_reason": o.outputs[0].finish_reason}) + "\n")


if __name__ == "__main__":
    main()
