"""Trained-question recall evaluation: base vs fine-tuned on EXACT training questions, plus the paraphrased (seen
facts) and unseen (held-out documents) tests, and deployment variants of each model's best epoch. Local only.

Stages (scripts/run_trained_eval.sh; markers in results/trained_eval/.done/):
  testset     data/splits/test_trained_exact.jsonl: 500 train items, all 279 test_seen_facts sources + 221 more,
              stratified by q_type x dimension, at most 8 per document for the added items, seed 42
  gen_vllm    vLLM answers: base and concise base (plain engine); 1-epoch run, ep1-3 (LoRA engine)
  gen_nf4     best epoch's adapter on an NF4 4-bit base (transformers + bitsandbytes, the training config)
  awq / gguf  merged best-epoch model as 4-bit AWQ (vLLM) / GGUF Q4_K_M (see src/trained_quant.py)
  keyfacts, grade, checks, stats, pack (see the later functions)

Systems: <family>__<variant>, variant in base, concise, ft1run, ep1, ep2, ep3, nf4_epN, awq_epN, gguf_epN.
Answers: existing run_2026-10-03 / run_epochs files where they exist, else results/trained_eval/answers/<system>/.
"""

import argparse
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from local_common import (FAMILIES, FAMILY_LABEL, OUT as LOCAL_OUT, VLLM_PY, items, log, pause_wait, pred_path,
                          read_jsonl, write_jsonl)
from utils import load_config, repo_path

TE = repo_path("results/trained_eval")
DONE = TE / ".done"
ANS = TE / "answers"
TEST_FILE = repo_path("data/splits/test_trained_exact.jsonl")
SPLITS = ["test_trained_exact", "test_seen_facts", "test_heldout_docs"]
MODEL_ID = {"qwen3-8b": "Qwen/Qwen3-8B", "gemma-4-e4b-it": "google/gemma-4-E4B-it",
            "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct"}
CONCISE = " Answer in 40 words or fewer."           # appended to the training system prompt (length control)
for d in (TE, DONE, ANS):
    d.mkdir(parents=True, exist_ok=True)


def cfg():
    return load_config()


def system_prompt(variant):
    s = cfg()["train"]["system_prompt"]
    return s + CONCISE if variant == "concise" else s


def best_epochs():
    """Best fine-tuned epoch per family: highest mean local-judge accuracy over the 3 earlier test splits (the rule
    used for the paper pack, src/local_pack.best_epochs)."""
    p = TE / "best_epochs.json"
    if p.exists():
        return json.loads(p.read_text())
    import pandas as pd
    mt = pd.read_csv(LOCAL_OUT / "metrics_by_system.csv")
    out = {}
    for fam in FAMILIES:
        sc = {v: mt[(mt.family == fam) & (mt.variant == v)].judge_lenient.mean() for v in ("ep1", "ep2", "ep3")}
        out[fam] = max(sc, key=sc.get)
    p.write_text(json.dumps(out, indent=2))
    return out


def adapter_path(fam, var):
    if var == "ft1run":
        return repo_path("models") / fam / "adapter"
    return repo_path("models_epochs") / fam / f"epoch{var[-1]}" / "adapter"


def variants(fam):
    be = best_epochs()[fam]
    return ["base", "concise", "ft1run", "ep1", "ep2", "ep3", f"nf4_{be}", f"awq_{be}", f"gguf_{be}"]


def is_new(var, split):
    """True if this system x split needs a newly generated answer file (no earlier run produced it)."""
    if split == "test_trained_exact":
        return True
    return var not in ("base", "ft1run", "ep1", "ep2", "ep3")


def answer_path(fam, var, split):
    if not is_new(var, split):
        return pred_path(fam, var, split)
    return ANS / f"{fam}__{var}" / f"{split}_predictions.jsonl"


def preds(fam, var, split):
    return {r["qa_id"]: r["prediction"] for r in read_jsonl(answer_path(fam, var, split))}


_items = None


def all_items():
    """{split: {qa_id: row}}: the new exact-question set plus the eval-subset ids of seen facts / held-out docs."""
    global _items
    if _items is None:
        its = items()
        _items = {"test_trained_exact": {r["qa_id"]: r for r in read_jsonl(TEST_FILE)},
                  "test_seen_facts": its["test_seen_facts"], "test_heldout_docs": its["test_heldout_docs"]}
    return _items


# ------------------------------------------------------------------ test set

def cmd_testset():
    if TEST_FILE.exists():
        log().info(f"[T0] {TEST_FILE.name} exists ({len(read_jsonl(TEST_FILE))} items)")
        return
    train = read_jsonl(repo_path("data/splits/train.jsonl"))
    by_id = {r["qa_id"]: r for r in train}
    seen = read_jsonl(repo_path("data/splits/test_seen_facts.jsonl"))
    must = [by_id[r["source_qa_id"]] for r in seen]
    chosen = {r["qa_id"] for r in must}
    per_doc = Counter(r["doc_id"] for r in must)
    stratum = lambda r: (r["q_type"], r["dimension"])
    share = Counter(stratum(r) for r in train)
    target = {k: 500 * v / len(train) for k, v in share.items()}
    have = Counter(stratum(r) for r in must)
    rng = random.Random(42)
    pool = [r for r in train if r["qa_id"] not in chosen]
    rng.shuffle(pool)
    while len(chosen) < 500:
        # stratum with the largest remaining deficit that still has an eligible item (doc cap 8 for added items)
        added = False
        for k in sorted(target, key=lambda k: have[k] - target[k]):
            cand = next((r for r in pool if stratum(r) == k and r["qa_id"] not in chosen and per_doc[r["doc_id"]] < 8), None)
            if cand:
                chosen.add(cand["qa_id"])
                per_doc[cand["doc_id"]] += 1
                have[k] += 1
                added = True
                break
        if not added:          # every stratum exhausted under the cap: relax the cap
            cand = next(r for r in pool if r["qa_id"] not in chosen)
            chosen.add(cand["qa_id"])
            per_doc[cand["doc_id"]] += 1
    rows = []
    seen_src = {r["source_qa_id"]: r["qa_id"] for r in seen}
    for r in train:
        if r["qa_id"] in chosen:
            rows.append({"qa_id": r["qa_id"], "question": r["question"], "answer": r["answer"],
                         "q_type": r["q_type"], "dimension": r["dimension"], "difficulty": r["difficulty"],
                         "doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "evidence": r["evidence"],
                         "citation": r["citation"], "paraphrase_test_id": seen_src.get(r["qa_id"]),
                         "split": "test_trained_exact"})
    write_jsonl(TEST_FILE, rows)
    qt = Counter(r["q_type"] for r in rows)
    log().info(f"[T0] test_trained_exact: {len(rows)} items ({sum(1 for r in rows if r['paraphrase_test_id'])} paired "
               f"with seen facts), {len({r['doc_id'] for r in rows})} documents, max per doc "
               f"{max(Counter(r['doc_id'] for r in rows).values())}; q_type {dict(qt)}")


# ------------------------------------------------------------------ generation

def run_multi(model, rows, out, lora, stats_path, label, quantization=None, gpu_util=0.85):
    """rows: [{"id", "system", "question", "adapter"}]; resumable (ids already in out are skipped)."""
    have = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["id"] not in have]
    if not todo:
        return 0
    inp = Path(str(out) + ".in.jsonl")
    write_jsonl(inp, todo)
    cmd = [str(VLLM_PY), str(repo_path("src/vllm_multi_generate.py")), "--model", model, "--input", str(inp),
           "--output", str(out), "--stats", str(stats_path), "--max-new-tokens", str(cfg()["eval"]["max_new_tokens"]),
           "--gpu-memory-utilization", str(gpu_util)]
    if lora:
        cmd.append("--lora")
    if quantization:
        cmd += ["--quantization", quantization]
    log().info(f"[{label}] vLLM: {len(todo)} prompts ({model}{', LoRA' if lora else ''})")
    import os
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0", LLM_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=str(repo_path("src/vllm_shims")))
    wl = repo_path(f"logs/trained_eval_worker.log")
    with wl.open("a") as lf:
        lf.write(f"\n===== {time.strftime('%F %T')} {label}: {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=lf).returncode
    inp.unlink(missing_ok=True)
    left = len(todo) - len({r["id"] for r in read_jsonl(out)} & {r["id"] for r in todo})
    log().info(f"[{label}] worker exit {rc}; {len(todo) - left}/{len(todo)} new answers")
    return left


def split_out(raw_path, fam, var_of):
    """Distribute raw worker output {"id": "<var>|<split>|<qa_id>"} into per-system prediction files."""
    rows = defaultdict(list)
    for r in read_jsonl(raw_path):
        var, split, q = r["id"].split("|", 2)
        rows[(var, split)].append({"qa_id": q, "prediction": r["prediction"], "n_tokens": r.get("n_tokens")})
    for (var, split), rs in rows.items():
        p = ANS / f"{fam}__{var}" / f"{split}_predictions.jsonl"
        uniq = {r["qa_id"]: r for r in rs}
        write_jsonl(p, list(uniq.values()))


def cmd_gen_vllm():
    its = all_items()
    for fam, model in MODEL_ID.items():
        pause_wait("T1")
        raw_dir = TE / "raw"
        raw_dir.mkdir(exist_ok=True)
        plain, lora = [], []
        for var in ("base", "concise", "ft1run", "ep1", "ep2", "ep3"):
            for split in SPLITS:
                if not is_new(var, split):
                    continue
                for q, r in its[split].items():
                    row = {"id": f"{var}|{split}|{q}", "system": system_prompt(var), "question": r["question"],
                           "adapter": None if var in ("base", "concise") else str(adapter_path(fam, var))}
                    (plain if row["adapter"] is None else lora).append(row)
        run_multi(model, plain, raw_dir / f"{fam}__plain.jsonl", False, raw_dir / f"{fam}__plain.stats.json",
                  f"T1 {fam} plain")
        run_multi(model, lora, raw_dir / f"{fam}__lora.jsonl", True, raw_dir / f"{fam}__lora.stats.json",
                  f"T1 {fam} LoRA")
        split_out(raw_dir / f"{fam}__plain.jsonl", fam, None)
        split_out(raw_dir / f"{fam}__lora.jsonl", fam, None)
    missing = [(fam, var, s) for fam in MODEL_ID for var in ("base", "concise", "ft1run", "ep1", "ep2", "ep3")
               for s in SPLITS if len(preds(fam, var, s)) < len(its[s])]
    if missing:
        raise SystemExit(f"T1 incomplete: {missing[:6]}")


def cmd_gen_nf4():
    """Best epoch's adapter on the NF4 4-bit base (training quantization config), HF generate, greedy."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    its = all_items()
    be = best_epochs()
    bs = 16
    for fam, model in MODEL_ID.items():
        var = f"nf4_{be[fam]}"
        todo = [(s, q, r) for s in SPLITS for q, r in its[s].items() if q not in preds(fam, var, s)]
        if not todo:
            continue
        pause_wait("T2")
        log().info(f"[T2] {fam}: NF4 base + {be[fam]} adapter, {len(todo)} answers")
        tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        m = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
        m = PeftModel.from_pretrained(m, str(adapter_path(fam, be[fam]))).eval()
        load_mem = torch.cuda.max_memory_allocated() / 1e9
        todo.sort(key=lambda x: len(x[2]["question"]))
        sp = system_prompt("base")
        gen_s, out_tok = 0.0, 0
        files = {s: (ANS / f"{fam}__{var}" / f"{s}_predictions.jsonl") for s in SPLITS}
        for f in files.values():
            f.parent.mkdir(parents=True, exist_ok=True)
        for b in range(0, len(todo), bs):
            if b % (bs * 10) == 0:
                pause_wait("T2")
            part = todo[b:b + bs]
            texts = [tok.apply_chat_template([{"role": "system", "content": sp}, {"role": "user", "content": r["question"]}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for _, _, r in part]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            t1 = time.time()
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=cfg()["eval"]["max_new_tokens"], do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            gen_s += time.time() - t1
            for (s, q, _), o in zip(part, out):
                new = o[enc["input_ids"].shape[1]:]
                n = int((new != tok.pad_token_id).sum())
                out_tok += n
                with files[s].open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"qa_id": q, "prediction": tok.decode(new, skip_special_tokens=True).strip(),
                                        "n_tokens": n}, ensure_ascii=False) + "\n")
            if (b // bs) % 10 == 0:
                log().info(f"[T2] {fam}: {b + len(part)}/{len(todo)} ({out_tok / max(gen_s, 1e-6):.0f} tok/s)")
        st = TE / "deploy_stats.json"
        d = json.loads(st.read_text()) if st.exists() else {}
        d[f"{fam}__{var}"] = {"backend": "transformers + bitsandbytes NF4 (double quant) + PEFT LoRA, batch 16",
                              "seconds": round(gen_s, 1), "output_tokens": out_tok,
                              "tokens_per_s": round(out_tok / max(gen_s, 1e-6), 1),
                              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                              "weights_vram_gb": round(load_mem, 2)}
        st.write_text(json.dumps(d, indent=2))
        del m
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    a = ap.parse_args()
    fn = globals().get(f"cmd_{a.cmd}")
    if fn is None:
        raise SystemExit(f"unknown command {a.cmd}")
    fn()


if __name__ == "__main__":
    main()
