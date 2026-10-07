"""Diagnostic (no training, no result files changed): is there a train/inference prompt-format mismatch?

For each model (Qwen3-8B, Gemma 4 E4B, Llama 3.1 8B as control), on items of test_trained_exact (= training items):
 1. TRAINING sequence exactly as src/train_qlora.py builds it (examples() + tokenize(): chat template, system prompt,
    generation prompt, enable_thinking=False, answer, end-of-turn tokens; labels from the prompt boundary) vs the exact
    prompt_token_ids vLLM used (src/diag_vllm_prompts.py, production settings). Token-by-token diff.
 2. QA-only ep3 adapter in HF transformers: mean answer-token loss on 20 items with (a) the training prompt and (b) the
    vLLM prompt, both on the NF4 base the adapter was trained on, and (b') the vLLM prompt on the bf16 base (what vLLM
    serves); greedy answers for 5 items in each setting, ROUGE-L vs the training answer, next to the vLLM answers.
Writes results/diagnostics/format_check.md (+ format_check.json). Local only (LLM_OFFLINE=1).
"""

import difflib
import json
import os
import random
import subprocess
import time

from eval_closedbook import rouge_l
from local_common import VLLM_PY, read_jsonl, write_jsonl
from train_qlora import examples, tokenize
from utils import load_config, repo_path

OUT = repo_path("results/diagnostics")
MODELS = [("qwen3-8b", "Qwen/Qwen3-8B"), ("gemma-4-e4b-it", "google/gemma-4-E4B-it"),
          ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct")]
N_SHOW, N_LOSS, N_GEN = 10, 20, 5


def vllm_prompts(model, adapter, msgs, tag):
    inp, outp = OUT / f"{tag}.in.jsonl", OUT / f"{tag}.vllm.jsonl"
    if not outp.exists():
        write_jsonl(inp, [{"id": i, "messages": m} for i, m in enumerate(msgs)])
        env = dict(os.environ, LLM_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0",
                   PYTHONPATH=str(repo_path("src/vllm_shims")))
        with open(repo_path("logs/diag_format_worker.log"), "a") as lf:
            subprocess.run([str(VLLM_PY), str(repo_path("src/diag_vllm_prompts.py")), "--model", model, "--adapter",
                            str(adapter), "--input", str(inp), "--output", str(outp)], env=env, stdout=lf, stderr=lf,
                           check=True)
        inp.unlink(missing_ok=True)
    return sorted(read_jsonl(outp), key=lambda r: r["id"])


def token_diff(tok, a, b):
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            out.append({"op": op, "train_pos": [i1, i2], "vllm_pos": [j1, j2],
                        "train_tokens": tok.convert_ids_to_tokens(a[i1:i2]), "vllm_tokens": tok.convert_ids_to_tokens(b[j1:j2])})
    return out


def hf_eval(model_id, adapter, seqs, gens, four_bit):
    """seqs: [(input_ids, label_start)] -> mean answer-token losses; gens: [prompt_ids] -> greedy texts."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    m = PeftModel.from_pretrained(m, str(adapter)).eval()
    losses = []
    with torch.no_grad():
        for ids, start in seqs:
            x = torch.tensor([ids], device="cuda")
            lab = x.clone()
            lab[:, :start] = -100
            losses.append(float(m(input_ids=x, labels=lab).loss))
        texts = []
        for p in gens:
            x = torch.tensor([p], device="cuda")
            o = m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=512, do_sample=False,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
            texts.append(tok.decode(o[0, x.shape[1]:], skip_special_tokens=True).strip())
    del m
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return losses, texts


def main():
    assert os.environ.get("LLM_OFFLINE") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    tcfg = dict(cfg["train"])
    from transformers import AutoTokenizer
    test = read_jsonl(repo_path("data/splits/test_trained_exact.jsonl"))
    train = {r["qa_id"]: r for r in read_jsonl(repo_path("data/splits/train.jsonl"))}
    rows = [train[r["qa_id"]] for r in random.Random(0).sample(test, N_LOSS)]
    res = json.loads((OUT / "format_check.json").read_text()) if (OUT / "format_check.json").exists() else {}
    # vLLM first for every model: once this process has loaded an HF model, it keeps GPU memory that vLLM needs
    for fam, model in MODELS:
        if fam not in res:
            exs = examples(rows, tcfg, "closedbook", False)
            vllm_prompts(model, repo_path("models_epochs") / fam / "epoch3" / "adapter", [e["messages"][:-1] for e in exs], fam)
    for fam, model in MODELS:
        if fam in res:
            continue
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        exs = examples(rows, tcfg, "closedbook", False)                     # original question, as in training
        st = {"prefix_mismatch": 0}
        tr = [tokenize(tok, e, tcfg["max_seq_length"], st) for e in exs]
        p_len = [next(i for i, l in enumerate(t["labels"]) if l != -100) for t in tr]
        adapter = repo_path("models_epochs") / fam / "epoch3" / "adapter"
        vl = vllm_prompts(model, adapter, [e["messages"][:-1] for e in exs], fam)
        diffs = [token_diff(tok, t["input_ids"][:p], v["prompt_token_ids"]) for t, p, v in zip(tr, p_len, vl)]
        prod = {r["qa_id"]: r["prediction"] for r in
                read_jsonl(repo_path("results/trained_eval/answers") / f"{fam}__ep3" / "test_trained_exact_predictions.jsonl")}
        # loss: same answer tokens (training labels) after (a) training prompt, (b) vLLM prompt
        seq_a = [(t["input_ids"], p) for t, p in zip(tr, p_len)]
        seq_b = [(v["prompt_token_ids"] + t["input_ids"][p:], len(v["prompt_token_ids"])) for t, p, v in zip(tr, p_len, vl)]
        gen_a = [t["input_ids"][:p] for t, p in zip(tr[:N_GEN], p_len[:N_GEN])]
        gen_b = [v["prompt_token_ids"] for v in vl[:N_GEN]]
        la, ga = hf_eval(model, adapter, seq_a, gen_a, True)
        lb, gb = hf_eval(model, adapter, seq_b, gen_b, True)
        lc, gc_ = hf_eval(model, adapter, seq_b, gen_b, False)
        ref = [r["answer"] for r in rows]
        rl = lambda texts: [round(rouge_l(t, r), 3) for t, r in zip(texts, ref)]
        res[fam] = {
            "model": model, "adapter": str(adapter), "prefix_mismatch_in_training_tokenizer": st["prefix_mismatch"],
            "items_identical_prompt": sum(1 for d in diffs if not d), "items": len(diffs),
            "diffs_first_10": diffs[:N_SHOW],
            "train_prompt_text_example": tok.decode(tr[0]["input_ids"][:p_len[0]]),
            "vllm_prompt_text_example": tok.decode(vl[0]["prompt_token_ids"]),
            "train_supervised_tail_example": tok.convert_ids_to_tokens(tr[0]["input_ids"][-4:]),
            "vllm_output_tail_example": tok.convert_ids_to_tokens(vl[0]["output_token_ids"][-4:]),
            "vllm_finish_reasons": sorted({v["finish_reason"] for v in vl}),
            "loss_nf4_train_format": la, "loss_nf4_vllm_format": lb, "loss_bf16_vllm_format": lc,
            "rougeL_hf_nf4_train_format": rl(ga), "rougeL_hf_nf4_vllm_format": rl(gb), "rougeL_hf_bf16_vllm_format": rl(gc_),
            "rougeL_vllm_fresh_5": rl([v["text"].strip() for v in vl[:N_GEN]]),
            "rougeL_vllm_fresh_20": rl([v["text"].strip() for v in vl]),
            "rougeL_production_20": rl([prod.get(r["qa_id"], "") for r in rows]),
            "examples": [{"qa_id": rows[i]["qa_id"], "reference": ref[i], "hf_nf4_train": ga[i], "hf_nf4_vllm": gb[i],
                          "hf_bf16_vllm": gc_[i], "vllm": vl[i]["text"].strip()} for i in range(N_GEN)],
            "minutes": round((time.time() - t0) / 60, 1)}
        (OUT / "format_check.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"{fam}: identical prompts {res[fam]['items_identical_prompt']}/{len(diffs)}; loss a/b/c "
              f"{sum(la) / len(la):.4f} / {sum(lb) / len(lb):.4f} / {sum(lc) / len(lc):.4f}", flush=True)
    write_md(res)


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def write_md(res):
    L = ["# Train/inference prompt-format check (QA-only ep3 adapters)", "",
         f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by `src/diag_format.py` (no training, no result files changed). "
         f"{N_LOSS} items of test_trained_exact (= training items, original question), seed 0._", "",
         "## 1. Training token sequence vs the prompt vLLM received", "",
         "| Model | items with identical prompt tokens | training-tokenizer prefix mismatches | vLLM finish reasons |",
         "|---|---|---|---|"]
    for fam, r in res.items():
        L.append(f"| {fam} | {r['items_identical_prompt']}/{r['items']} | {r['prefix_mismatch_in_training_tokenizer']} | "
                 f"{', '.join(r['vllm_finish_reasons'])} |")
    for fam, r in res.items():
        L += ["", f"### {fam}", "", "Training prompt (decoded, first item):", "", "```", r["train_prompt_text_example"], "```",
              "vLLM prompt (decoded, same item):", "", "```", r["vllm_prompt_text_example"], "```",
              f"Last supervised training tokens: `{r['train_supervised_tail_example']}`; last vLLM output tokens: "
              f"`{r['vllm_output_tail_example']}`.", ""]
        nd = [d for d in r["diffs_first_10"] if d]
        if not nd:
            L.append("No token differences in the first 10 items.")
        else:
            L.append("Token differences (first 10 items):")
            for i, d in enumerate(r["diffs_first_10"]):
                for x in d:
                    L.append(f"- item {i}: {x['op']} training {x['train_tokens']} (pos {x['train_pos']}) vs vLLM "
                             f"{x['vllm_tokens']} (pos {x['vllm_pos']})")
    L += ["", "## 2–3. Effect on the QA-only ep3 adapter (HF transformers)", "",
          "| Model | answer loss: NF4, training prompt | NF4, vLLM prompt | bf16, vLLM prompt | ROUGE-L (5 gens): HF NF4 train / "
          "HF NF4 vLLM / HF bf16 vLLM / vLLM | ROUGE-L vLLM fresh (20) | production answers (20) |",
          "|---|---|---|---|---|---|---|"]
    for fam, r in res.items():
        L.append(f"| {fam} | {mean(r['loss_nf4_train_format']):.4f} | {mean(r['loss_nf4_vllm_format']):.4f} | "
                 f"{mean(r['loss_bf16_vllm_format']):.4f} | {mean(r['rougeL_hf_nf4_train_format']):.3f} / "
                 f"{mean(r['rougeL_hf_nf4_vllm_format']):.3f} / {mean(r['rougeL_hf_bf16_vllm_format']):.3f} / "
                 f"{mean(r['rougeL_vllm_fresh_5']):.3f} | {mean(r['rougeL_vllm_fresh_20']):.3f} | "
                 f"{mean(r['rougeL_production_20']):.3f} |")
    L += ["", "Generated examples are in `results/diagnostics/format_check.json`."]
    (OUT / "format_check.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
