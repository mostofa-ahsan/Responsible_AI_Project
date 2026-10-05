"""Merge a LoRA adapter into its bf16 base IN MEMORY and export a 4-bit AWQ (W4A16, asymmetric, group 128) model with
llm-compressor (compressed-tensors format, served by vLLM). Nothing bf16 is written to disk.

Runs under .venv-vllm with the overlay models_quant/overlay on PYTHONPATH (llm-compressor 0.14 needs newer
compressed-tensors / older transformers than vLLM 0.30 pins; the overlay shadows them for this process only).
Calibration: 128 training conversations (system + question + trained answer, chat template), max 512 tokens.
If AWQ fails for an architecture (llm-compressor has no Gemma 4 mapping; Gemma 3's is tried), falls back to
round-to-nearest W4A16 (QuantizationModifier only) and records that in <out>/quant_info.json.

    PYTHONPATH=models_quant/overlay .venv-vllm/bin/python src/trained_quant.py --model M --adapter DIR --out DIR
"""

import argparse
import json
import os
import random
import time


def block_api_clients():
    def _blocked(self, *a, **k):
        raise AssertionError("API client construction is forbidden (LLM_OFFLINE=1)")
    for mod, names in (("anthropic", ("Anthropic",)), ("openai", ("OpenAI", "AsyncOpenAI"))):
        try:
            m = __import__(mod)
        except Exception:
            continue
        for n in names:
            if getattr(m, n, None) is not None:
                getattr(m, n).__init__ = _blocked


def generate_hf(args):
    """Fallback generation with transformers + compressed-tensors (if vLLM 0.30 cannot load the export).
    Input/output JSONL as src/vllm_multi_generate.py; appends; greedy; batch 16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.out, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(args.out, device_map={"": 0}, dtype=torch.bfloat16).eval()
    load_gb = torch.cuda.max_memory_allocated() / 1e9
    gen_s, ntok = 0.0, 0
    rows.sort(key=lambda r: len(r["question"]))
    with open(args.output, "a", encoding="utf-8") as f:
        for b in range(0, len(rows), 16):
            part = rows[b:b + 16]
            texts = [tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["question"]}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in part]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            t1 = time.time()
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            gen_s += time.time() - t1
            for r, o in zip(part, out):
                new = o[enc["input_ids"].shape[1]:]
                n = int((new != tok.pad_token_id).sum())
                ntok += n
                f.write(json.dumps({"id": r["id"], "prediction": tok.decode(new, skip_special_tokens=True).strip(),
                                    "n_tokens": n}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[quant-gen] {b + len(part)}/{len(rows)}", flush=True)
    json.dump({"seconds": round(gen_s, 1), "output_tokens": ntok, "prompts": len(rows), "weights_vram_gb": round(load_gb, 2),
               "backend": "transformers + compressed-tensors (vLLM fallback)"}, open(args.stats, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="data/splits/train.jsonl")
    ap.add_argument("--system")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--generate", action="store_true", help="fallback: generate with the exported model (transformers)")
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--stats")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    assert os.environ.get("LLM_OFFLINE") == "1"
    block_api_clients()
    if args.generate:
        return generate_hf(args)
    import uuid

    import datasets.fingerprint as _fp
    import torch
    from datasets import Dataset
    # datasets hashes transforms with dill to fingerprint its cache; dill 0.4.1 (vLLM venv) cannot pickle newer pyarrow
    # types. No caching is needed here, so use random fingerprints instead of hashing.
    _fp.update_fingerprint = lambda *a, **k: uuid.uuid4().hex
    _fp.generate_fingerprint = lambda *a, **k: uuid.uuid4().hex
    _fp.Hasher.hash = classmethod(lambda cls, value: uuid.uuid4().hex)
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)        # CPU, bf16
    if args.adapter != "none":
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()        # merged in memory
    arch = type(model).__name__
    print(f"[quant] {args.model} + {args.adapter} merged ({arch}) in {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    texts = [tok.apply_chat_template([{"role": "system", "content": args.system},
                                      {"role": "user", "content": r["question"]},
                                      {"role": "assistant", "content": r["answer"]}], tokenize=False,
                                     enable_thinking=False) for r in rows[:args.n_calib]]
    enc = tok(texts, max_length=512, truncation=True, add_special_tokens=False)
    ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    ignore = ["lm_head", "re:.*(vision|audio|multi_modal_projector|embed_vision|embed_audio).*"]
    quant = QuantizationModifier(ignore=ignore, scheme="W4A16_ASYM", targets=["Linear"])
    method = "AWQ W4A16 asym (group 128)"
    try:
        kw = {}
        if "Gemma4" in arch:
            # llm-compressor has no Gemma 4 mapping, and the Gemma 3 regexes also match the audio/vision towers.
            # Gemma-style mappings anchored to the language model's decoder layers:
            from llmcompressor.modifiers.transform.awq.mappings import AWQMapping
            L = "re:.*language_model.layers.\\d+."
            kw["mappings"] = [AWQMapping(L + "self_attn.v_proj$", [L + "self_attn.o_proj$"]),
                              AWQMapping(L + "pre_feedforward_layernorm$", [L + "mlp.gate_proj$", L + "mlp.up_proj$"]),
                              AWQMapping(L + "mlp.up_proj$", [L + "mlp.down_proj$"])]
            method += ", Gemma-style smoothing mappings on the language-model layers"
        oneshot(model=model, dataset=ds, recipe=[AWQModifier(duo_scaling="both", **kw), quant],
                max_seq_length=512, num_calibration_samples=len(texts))
    except Exception as e:
        print(f"[quant] AWQ failed ({type(e).__name__}: {str(e)[:300]}); falling back to RTN W4A16", flush=True)
        method = f"RTN W4A16 asym (AWQ failed: {type(e).__name__})"
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        if args.adapter != "none":
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        oneshot(model=model, recipe=[QuantizationModifier(ignore=ignore, scheme="W4A16_ASYM", targets=["Linear"])])
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)
    for extra in ("chat_template.jinja", "preprocessor_config.json", "processor_config.json", "generation_config.json"):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(args.model, extra)
            if not os.path.exists(os.path.join(args.out, extra)):
                import shutil
                shutil.copy(p, os.path.join(args.out, extra))
        except Exception:
            pass
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(args.out) for f in fs)
    info = {"method": method, "architecture": arch, "calibration": f"{len(texts)} training conversations, <= 512 tokens",
            "minutes": round((time.time() - t0) / 60, 1), "size_gb": round(size / 1e9, 2)}
    json.dump(info, open(os.path.join(args.out, "quant_info.json"), "w"), indent=2)
    print(f"[quant] saved {args.out}: {info}", flush=True)


if __name__ == "__main__":
    main()
