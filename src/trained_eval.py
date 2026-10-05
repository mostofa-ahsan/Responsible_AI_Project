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


# ------------------------------------------------------------------ disk / model management

def free_gb():
    import shutil
    return shutil.disk_usage(str(TE)).free / 1e9


def fallback(stage, what, why):
    p = TE / "fallbacks.json"
    d = json.loads(p.read_text()) if p.exists() else []
    d.append({"time": time.strftime("%F %T"), "stage": stage, "fallback": what, "why": why})
    p.write_text(json.dumps(d, indent=2))
    log().info(f"[{stage}] FALLBACK: {what} ({why})")


def delete_hf_model(repo, tag):
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    revs = [r.commit_hash for repo_ in info.repos if repo_.repo_id == repo for r in repo_.revisions]
    if revs:
        st = info.delete_revisions(*revs)
        st.execute()
        log().info(f"[{tag}] deleted {repo} from the HF cache ({st.expected_freed_size_str}); free {free_gb():.1f} GB")


def cmd_purge_minicheck():
    import local_checks as L
    delete_hf_model(L.MC7B, "T3")


def loading_gib(label):
    """'Model loading took X GiB' of the most recent worker run with this label (vLLM weights in VRAM)."""
    import re
    txt = repo_path("logs/trained_eval_worker.log").read_text(errors="ignore")
    part = txt[txt.rfind(f"===== "):] if label is None else txt[txt.rfind(f" {label}: "):]
    m = re.findall(r"Model loading took ([0-9.]+) GiB", part)
    return float(m[0]) if m else None


def record_deploy(key, d):
    st = TE / "deploy_stats.json"
    all_ = json.loads(st.read_text()) if st.exists() else {}
    all_[key] = {**all_.get(key, {}), **d}
    st.write_text(json.dumps(all_, indent=2))


_VLLM_OK = None


def vllm_reads_export():
    """vLLM 0.30 ships compressed-tensors 0.17; llm-compressor 0.14 writes with 0.19. Probe once with a tiny export
    (Qwen3-0.6B, models_quant/test_qwen06_awq); if vLLM cannot load it, generate with transformers instead."""
    global _VLLM_OK
    if _VLLM_OK is not None:
        return _VLLM_OK
    probe = repo_path("models_quant/test_qwen06_awq")
    if not (probe / "quant_info.json").exists():
        _VLLM_OK = True
        return True
    out = TE / "raw" / "probe.jsonl"
    out.unlink(missing_ok=True)
    left = run_multi(str(probe), [{"id": "probe|x|1", "system": "You are helpful.", "question": "Say hello.", "adapter": None}],
                     out, False, TE / "raw" / "probe.stats.json", "T4 probe", gpu_util=0.5)
    _VLLM_OK = left == 0
    log().info(f"[T4] vLLM {'CAN' if _VLLM_OK else 'CANNOT'} load llm-compressor 0.14 exports")
    if not _VLLM_OK:
        fallback("T4", "4-bit models served by transformers + compressed-tensors instead of vLLM",
                 "vLLM 0.30 (compressed-tensors 0.17) could not load the llm-compressor 0.14 export")
    return _VLLM_OK


def hf_generate_quant(qdir, rows, raw, stats, label):
    import os
    have = {r["id"] for r in read_jsonl(raw)}
    todo = [r for r in rows if r["id"] not in have]
    if todo:
        inp = Path(str(raw) + ".in.jsonl")
        write_jsonl(inp, todo)
        cmd = [str(VLLM_PY), str(repo_path("src/trained_quant.py")), "--generate", "--out", str(qdir), "--input", str(inp),
               "--output", str(raw), "--stats", str(stats), "--max-new-tokens", str(cfg()["eval"]["max_new_tokens"])]
        env = dict(os.environ, LLM_OFFLINE="1", PYTHONPATH=str(repo_path("models_quant/overlay")))
        log().info(f"[{label}] transformers generation: {len(todo)} prompts")
        with repo_path("logs/trained_eval_worker.log").open("a") as lf:
            lf.write(f"\n===== {time.strftime('%F %T')} {label}: {' '.join(cmd)}\n")
            lf.flush()
            subprocess.run(cmd, env=env, stdout=lf, stderr=lf)
        inp.unlink(missing_ok=True)
    got = {r["id"] for r in read_jsonl(raw)}
    return sum(1 for r in rows if r["id"] not in got)


def awq_one(fam, model, its, be):
    """Quantize the best epoch to 4-bit (in-memory merge), generate on the 3 splits, record stats, remove the export.
    Returns True when every answer exists."""
    import os
    import shutil
    var = f"awq_{be[fam]}"
    if all(len(preds(fam, var, s)) >= len(its[s]) for s in SPLITS):
        return True
    qdir = repo_path("models_quant") / f"{fam}_{var}"
    if not (qdir / "quant_info.json").exists():
        if free_gb() < 10 + 11:
            fallback("T4", f"{fam}: AWQ skipped", f"only {free_gb():.1f} GB free (needs >= 21)")
            return False
        pause_wait("T4")
        cmd = [str(VLLM_PY), str(repo_path("src/trained_quant.py")), "--model", model, "--adapter",
               str(adapter_path(fam, be[fam])), "--out", str(qdir), "--system", system_prompt("base")]
        env = dict(os.environ, LLM_OFFLINE="1", PYTHONPATH=str(repo_path("models_quant/overlay")),
                   TOKENIZERS_PARALLELISM="false")
        log().info(f"[T4] {fam}: merging {be[fam]} adapter in memory and quantizing to 4-bit (llm-compressor)")
        with repo_path("logs/trained_eval_worker.log").open("a") as lf:
            lf.write(f"\n===== {time.strftime('%F %T')} T4 quant {fam}: {' '.join(cmd)}\n")
            lf.flush()
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=lf).returncode
        if rc != 0 or not (qdir / "quant_info.json").exists():
            fallback("T4", f"{fam}: no 4-bit export", f"quantization exit {rc} (logs/trained_eval_worker.log)")
            return False
    info = json.loads((qdir / "quant_info.json").read_text())
    log().info(f"[T4] {fam}: {info}")
    rows = [{"id": f"{var}|{s}|{q}", "system": system_prompt("base"), "question": r["question"], "adapter": None}
            for s in SPLITS for q, r in its[s].items()]
    raw = TE / "raw" / f"{fam}__{var}.jsonl"
    raw.parent.mkdir(exist_ok=True)
    label = f"T4 {fam} {var}"
    stp = TE / "raw" / f"{fam}__{var}.stats.json"
    if vllm_reads_export():
        left = run_multi(str(qdir), rows, raw, False, stp, label)
    else:
        left = hf_generate_quant(qdir, rows, raw, stp, label)
    split_out(raw, fam, None)
    stats = json.loads(stp.read_text()) if stp.exists() else {}
    record_deploy(f"{fam}__{var}", {"backend": (f"vLLM, compressed-tensors {info['method']}" if _VLLM_OK else
                                                stats.get("backend", "transformers") + f", {info['method']}"),
                                    "size_gb": info["size_gb"], "quant_minutes": info["minutes"],
                                    "weights_vram_gb": loading_gib(label) if _VLLM_OK else stats.get("weights_vram_gb"),
                                    "seconds": stats.get("seconds"), "output_tokens": stats.get("output_tokens"),
                                    "tokens_per_s": round(stats["output_tokens"] / stats["seconds"], 1)
                                    if stats.get("seconds") else None})
    if left:
        fallback("T4", f"{fam}: 4-bit model generation incomplete", f"{left} answers missing")
        return False
    shutil.rmtree(qdir, ignore_errors=True)       # disk reserve for the judge; regenerate with src/trained_quant.py
    log().info(f"[T4] {fam}: 4-bit model evaluated and removed (free {free_gb():.1f} GB)")
    return True


def cmd_awq():
    its = all_items()
    be = best_epochs()
    ok = sum(awq_one(fam, model, its, be) for fam, model in MODEL_ID.items())
    if ok == 0:
        raise SystemExit("T4: no 4-bit variant evaluated")


def redo_rtn_with_awq():
    """One-time fix (before the judge is downloaded): a model whose 4-bit export fell back to RTN because the Gemma 3
    regexes also matched the audio tower is re-quantized with language-model-anchored AWQ mappings."""
    import shutil
    st = TE / "deploy_stats.json"
    if not st.exists():
        return
    d = json.loads(st.read_text())
    its, be = all_items(), best_epochs()
    for fam, model in MODEL_ID.items():
        key = f"{fam}__awq_{be[fam]}"
        if "RTN" not in str(d.get(key, {}).get("backend", "")) or (TE / f".redo_{fam}").exists():
            continue
        (TE / f".redo_{fam}").write_text(time.strftime("%F %T"))
        src = ANS / key
        bak = ANS / (key + "_rtn")
        if src.exists():
            shutil.rmtree(bak, ignore_errors=True)
            src.rename(bak)
        (TE / "raw" / f"{key}.jsonl").unlink(missing_ok=True)
        shutil.rmtree(repo_path("models_quant") / f"{fam}_awq_{be[fam]}", ignore_errors=True)
        fallback("T6", f"{fam}: 4-bit export redone with AWQ (RTN answers kept in answers/{key}_rtn, not graded)",
                 "the first AWQ attempt failed because the Gemma 3 mapping regexes also matched the audio tower")
        awq_one(fam, model, its, be)


def cmd_gguf():
    import shutil
    reasons = []
    if not (shutil.which("llama-quantize") or shutil.which("llama-cli")):
        reasons.append("llama.cpp is not installed; the only local GGUF route is Ollama, whose model store belongs to "
                       "another project on this machine (not touched)")
    if free_gb() < 10 + 16 + 6:
        reasons.append(f"a GGUF export needs the merged bf16 model on disk (~16 GB) plus the Q4_K_M file; "
                       f"{free_gb():.1f} GB free")
    if reasons:
        fallback("T5", "GGUF Q4_K_M variant skipped", "; ".join(reasons))
        (TE / "gguf_skipped.json").write_text(json.dumps({"reasons": reasons}, indent=2))
        return


# ------------------------------------------------------------------ judge (downloaded once), key facts, grading

def cmd_judge_keyfacts():
    from local_common import JUDGES, KEYFACTS, CACHE, run_worker, parse_json, deadline_from_env
    import local_facts as F
    redo_rtn_with_awq()
    if free_gb() < 10 + 16:
        raise SystemExit(f"only {free_gb():.1f} GB free; the judge needs ~15 GB plus the 10 GB reserve")
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download(JUDGES["A"]["model"])
    log().info(f"[T6] judge downloaded in {(time.time() - t0) / 60:.1f} min; free {free_gb():.1f} GB")
    its = all_items()["test_trained_exact"]
    schema = {"type": "object", "properties": {"facts": {"type": "array", "minItems": 1, "maxItems": 5,
                                                         "items": {"type": "string", "maxLength": 300}}},
              "required": ["facts"], "additionalProperties": False}
    cp = CACHE / f"keyfacts__{JUDGES['A']['name']}.jsonl"
    rows = [{"id": f"test_trained_exact|{q}", "messages": [
        {"role": "system", "content": F.KF_SYSTEM},
        {"role": "user", "content": F.KF_PROMPT.format(question=r["question"], reference=r["answer"],
                                                     evidence=(r.get("evidence") or "")[:2000] or "(none)")}]}
        for q, r in its.items()]
    run_worker(JUDGES["A"]["model"], rows, cp, schema=schema, max_tokens=500, deadline=deadline_from_env(40),
               extra=["--mistral3"], label="T6 key facts")
    got = {}
    for r in read_jsonl(cp):
        if r["id"].startswith("test_trained_exact|"):
            g = parse_json(r.get("text", ""))
            if g and isinstance(g.get("facts"), list):
                got[r["id"].split("|", 1)[1]] = [f.strip() for f in g["facts"] if isinstance(f, str) and f.strip()][:5]
    out = [{"qa_id": q, "split": "test_trained_exact", "question": r["question"], "reference": r["answer"],
            "facts": got[q], "decomposer": JUDGES["A"]["name"]} for q, r in its.items() if got.get(q)]
    write_jsonl(KEYFACTS / "test_trained_exact.jsonl", out)
    log().info(f"[T6] key facts: {len(out)}/{len(its)} exact-question references decomposed")
    if len(out) < 0.95 * len(its):
        raise SystemExit("T6: key facts incomplete")


def seed43_systems():
    out = []
    for fam in MODEL_ID:
        d = repo_path("results/seed_replication") / fam / "seed43_epoch1"
        for s in ("test_indomain", "test_heldout_docs", "test_seen_facts"):
            p = {r["qa_id"]: r["prediction"] for r in read_jsonl(d / f"{s}_predictions.jsonl")}
            if p:
                out.append((fam, "seed43_ep1", s, p))
    return out


def grade_pairs_list():
    its = all_items()
    base_its = items()
    pairs = []
    for fam in MODEL_ID:
        for var in variants(fam):
            for s in SPLITS:
                if not is_new(var, s):
                    continue
                p = preds(fam, var, s)
                for q, r in its[s].items():
                    if q in p:
                        pairs.append({"split": s, "qa_id": q, "system": f"{fam}__{var}", "family": fam, "variant": var,
                                      "question": r["question"], "reference": r["answer"], "answer": p[q]})
    for fam, var, s, p in seed43_systems():
        for q, r in base_its[s].items():
            if q in p:
                pairs.append({"split": s, "qa_id": q, "system": f"{fam}__{var}", "family": fam, "variant": var,
                              "question": r["question"], "reference": r["answer"], "answer": p[q]})
    return pairs


def cmd_grade():
    import local_judge as J
    from local_common import deadline_from_env
    ch = json.loads((LOCAL_OUT / "judge_choice.json").read_text())
    pairs = grade_pairs_list()
    log().info(f"[T7] grading {len(pairs)} answers with {ch['judge_name']} prompt {ch['prompt']} (unchanged)")
    g = J.grade_pairs(ch["judge"], ch["prompt"], pairs, deadline_from_env(120), "T7")
    out = []
    for p in pairs:
        r = g.get(J.key(p["split"], p["qa_id"], p["system"]))
        out.append({k: p[k] for k in ("split", "qa_id", "system", "family", "variant")} |
                   {"local_grade": r["grade"] if r else None,
                    "hallucinated_specific": bool(r.get("hallucinated_specific")) if r else None,
                    "local_reasoning": r.get("reasoning") if r else None})
    (TE / "per_item").mkdir(exist_ok=True)
    write_jsonl(TE / "per_item" / "judge_grades.jsonl", out)
    n = sum(1 for o in out if o["local_grade"])
    log().info(f"[T7] graded {n}/{len(out)}")
    if n < 0.95 * len(out):
        raise SystemExit("T7 incomplete")


# ------------------------------------------------------------------ checkers (judge deleted, MiniCheck back)

def check_pairs():
    from local_common import KEYFACTS
    kf = {s: {r["qa_id"]: r["facts"] for r in read_jsonl(KEYFACTS / f"{s}.jsonl")} for s in SPLITS}
    out = []
    for fam in MODEL_ID:
        for var in variants(fam):
            for s in SPLITS:
                if not is_new(var, s):
                    continue
                sid = f"{fam}__{var}"
                for q, a in preds(fam, var, s).items():
                    for k, f in enumerate(kf[s].get(q, [])):
                        out.append((f"{s}|{q}|{sid}|{k}", s, q, sid, k, a, f))
    return out


def cmd_checks():
    import local_checks as L
    from local_common import JUDGES, deadline_from_env
    deadline = deadline_from_env(60)
    delete_hf_model(JUDGES["A"]["model"], "T8")
    if not L.minicheck_local():
        if free_gb() < 10 + 16:
            raise SystemExit(f"only {free_gb():.1f} GB free for MiniCheck-7B")
        from huggingface_hub import snapshot_download
        snapshot_download(L.MC7B, allow_patterns=["*.json", "*.py", "*.model", "*.safetensors"])
    pairs = check_pairs()
    pi = TE / "per_item"
    pi.mkdir(exist_ok=True)
    (TE / "cache").mkdir(exist_ok=True)
    log().info(f"[T8] MiniCheck-7B + NLI on {len(pairs)} (answer, fact) pairs")
    n = L.score_minicheck7b(pairs, TE / "cache" / "minicheck7b_raw.jsonl", pi / "support_minicheck7b.jsonl",
                            time.time() + (deadline - time.time()) * 0.7, "T8")
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    L.score_nli_facts(L.NLIScorer(), pairs, pi / "nli_facts.jsonl", deadline, "T8 NLI")
    if n < 0.9 * len(pairs):
        raise SystemExit(f"T8: MiniCheck covered {n}/{len(pairs)}")


# ------------------------------------------------------------------ statistics

VAR_LABEL = {"base": "base", "concise": "concise base (<= 40 words)", "ft1run": "1-epoch (run 10-03)", "ep1": "ep1",
             "ep2": "ep2", "ep3": "ep3", "seed43_ep1": "ep1, seed 43"}


def var_label(v):
    if v.startswith(("nf4_", "awq_", "gguf_")):
        kind, ep = v.split("_")
        return {"nf4": "NF4 base + adapter", "awq": "merged 4-bit (AWQ/W4A16, vLLM)", "gguf": "GGUF Q4_K_M"}[kind] + f" ({ep})"
    return VAR_LABEL.get(v, v)


def per_item_table():
    """One row per (system, split, qa_id) for every table system, joining the new and the earlier per-item files."""
    import numpy as np
    import pandas as pd
    from eval_closedbook import rouge_l, token_f1
    from local_common import KEYFACTS, PER_ITEM
    its = all_items()
    kf = {(s, r["qa_id"]): len(r["facts"]) for s in SPLITS for r in read_jsonl(KEYFACTS / f"{s}.jsonl")}

    def keyed(path, field, by="k"):
        d = {}
        for r in read_jsonl(path):
            d.setdefault((r["split"], r["qa_id"], r["system"]), {})[r[by]] = r[field]
        return d
    judge = {(r["split"], r["qa_id"], r["system"]): r for r in read_jsonl(TE / "per_item" / "judge_grades.jsonl")}
    judge.update({(r["split"], r["qa_id"], r["system"]): r for r in read_jsonl(PER_ITEM / "judge_grades.jsonl")
                  if (r["split"], r["qa_id"], r["system"]) not in judge})
    sup = keyed(PER_ITEM / "support_minicheck7b.jsonl", "p")
    sup.update(keyed(TE / "per_item" / "support_minicheck7b.jsonl", "p"))
    con = keyed(PER_ITEM / "nli_facts.jsonl", "contradicted")
    con.update(keyed(TE / "per_item" / "nli_facts.jsonl", "contradicted"))
    SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    rows = []
    for fam in MODEL_ID:
        for var in variants(fam):
            for s in SPLITS:
                p = preds(fam, var, s)
                sid = f"{fam}__{var}"
                for q, it in its[s].items():
                    if q not in p:
                        continue
                    k = (s, q, sid)
                    g = judge.get(k, {})
                    lg = g.get("local_grade")
                    n = kf.get((s, q))
                    sp, cn = sup.get(k, {}), con.get(k, {})
                    rl = rouge_l(p[q], it["answer"])
                    rows.append({"system": sid, "family": fam, "variant": var, "split": s, "qa_id": q,
                                 "q_type": it.get("q_type"), "dimension": it.get("dimension"),
                                 "paraphrase_test_id": it.get("paraphrase_test_id"),
                                 "answer_words": len(p[q].split()), "token_f1": token_f1(p[q], it["answer"]),
                                 "rouge_l": rl, "exact_repro": float(rl >= 0.8),
                                 "local_grade": lg, "judge_strict": float(lg == "correct") if lg else np.nan,
                                 "judge_lenient": SC[lg] if lg else np.nan,
                                 "hallucination": float(bool(g.get("hallucinated_specific"))) if lg else np.nan,
                                 "keyfact_recall": sum(v >= 0.5 for v in sp.values()) / n if n and len(sp) >= n else np.nan,
                                 "contradiction_rate": sum(cn.values()) / n if n and len(cn) >= n else np.nan})
    return pd.DataFrame(rows)


METRICS = ["judge_lenient", "judge_strict", "hallucination", "keyfact_recall", "contradiction_rate", "token_f1",
           "rouge_l", "exact_repro", "answer_words"]


def cmd_stats():
    import numpy as np
    import pandas as pd
    from local_stats import boot_ci, holm, mcnemar, paired_boot
    df = per_item_table()
    df.to_csv(TE / "master_items.csv", index=False)
    rows = []
    for (sid, s), g in df.groupby(["system", "split"], sort=False):
        r = {"system": sid, "family": g.family.iloc[0], "variant": g.variant.iloc[0], "split": s, "n": len(g)}
        for c in METRICS:
            m, lo, hi, n = boot_ci(g[c].tolist())
            r.update({c: m, c + "_lo": lo, c + "_hi": hi, c + "_n": n})
        rows.append(r)
    mt = pd.DataFrame(rows)
    sig = []
    for fam in MODEL_ID:
        for var in variants(fam)[1:]:
            for s in SPLITS:
                a = df[(df.system == f"{fam}__base") & (df.split == s)].set_index("qa_id")
                b = df[(df.system == f"{fam}__{var}") & (df.split == s)].set_index("qa_id")
                common = a.index.intersection(b.index)
                if not len(common):
                    continue
                a, b = a.loc[common], b.loc[common]
                for c in METRICS:
                    pb = paired_boot((b[c] - a[c]).to_numpy(dtype=float))
                    sig.append({"family": fam, "variant": var, "split": s, "metric": c, "test": "paired_bootstrap",
                                "diff": pb["diff"], "diff_lo": pb["lo"], "diff_hi": pb["hi"], "p": pb["p"],
                                "effect": pb["dz"], "n": pb["n"]})
                ok = a.local_grade.notna() & b.local_grade.notna()
                if ok.sum():
                    mc = mcnemar((a.judge_strict[ok] == 1).to_numpy(), (b.judge_strict[ok] == 1).to_numpy())
                    sig.append({"family": fam, "variant": var, "split": s, "metric": "judge_strict", "test": "mcnemar",
                                "diff": float(b.judge_strict[ok].mean() - a.judge_strict[ok].mean()), "p": mc["p"],
                                "effect": mc["cohen_g"], "n": int(ok.sum())})
    sig = pd.DataFrame(sig)
    sig["p_holm"] = np.nan
    for _, g in sig.groupby(["metric", "test"]):
        sig.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    mt.to_csv(TE / "metrics_by_system.csv", index=False)
    sig.to_csv(TE / "significance.csv", index=False)
    # robustness gap: exact training question vs its paraphrase (279 paired items)
    gap = []
    ex = df[df.split == "test_trained_exact"].dropna(subset=["paraphrase_test_id"])
    pa = df[df.split == "test_seen_facts"]
    for sid, g in ex.groupby("system"):
        p2 = pa[pa.system == sid].set_index("qa_id")
        g = g[g.paraphrase_test_id.isin(p2.index)]
        if g.empty:
            continue
        for c in ("judge_lenient", "keyfact_recall", "token_f1", "exact_repro"):
            d = g[c].to_numpy(dtype=float) - p2.loc[g.paraphrase_test_id, c].to_numpy(dtype=float)
            pb = paired_boot(d)
            gap.append({"system": sid, "family": g.family.iloc[0], "variant": g.variant.iloc[0], "metric": c,
                        "exact": float(np.nanmean(g[c])), "paraphrase": float(np.nanmean(p2.loc[g.paraphrase_test_id, c])),
                        "drop_exact_minus_paraphrase": pb["diff"], "lo": pb["lo"], "hi": pb["hi"], "p": pb["p"], "n": pb["n"]})
    pd.DataFrame(gap).to_csv(TE / "robustness_gap.csv", index=False)
    deployment_table(mt)
    seed43_judge()
    log().info(f"[T9] stats: {len(mt)} system x split rows, {len(sig)} tests, {len(gap)} robustness rows")


def base_param_count(model):
    """Total parameters of the base checkpoint (sum of tensor sizes in its safetensors files)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = Path(snapshot_download(model, allow_patterns=["*.safetensors"], local_files_only=True))
    n = 0
    for f in d.glob("*.safetensors"):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                c = 1
                for x in h.get_slice(k).get_shape():
                    c *= x
                n += c
    return n


def base_size_gb(model):
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(model, allow_patterns=["*.safetensors"], local_files_only=True))
    return sum(f.resolve().stat().st_size for f in d.glob("*.safetensors")) / 1e9


def deployment_table(mt):
    import pandas as pd
    st = json.loads((TE / "deploy_stats.json").read_text()) if (TE / "deploy_stats.json").exists() else {}
    be = best_epochs()
    rows = []
    for fam, model in MODEL_ID.items():
        ep = be[fam]
        ad = adapter_path(fam, ep) / "adapter_model.safetensors"
        bsz = base_size_gb(model)
        lst = TE / "raw" / f"{fam}__lora.stats.json"
        lora = json.loads(lst.read_text()) if lst.exists() else {}
        ref = {}
        for s in SPLITS:
            r = mt[(mt.system == f"{fam}__{ep}") & (mt.split == s)]
            ref[s] = r.iloc[0] if len(r) else None
        for var, label, size, d in (
                (ep, "bf16 base + LoRA (vLLM)", bsz + ad.stat().st_size / 1e9,
                 {"weights_vram_gb": loading_gib(f"T1 {fam} LoRA"),
                  "tokens_per_s": round(lora["output_tokens"] / lora["seconds"], 1) if lora.get("seconds") else None}),
                (f"nf4_{ep}", "NF4 base + LoRA (transformers)", bsz + ad.stat().st_size / 1e9, st.get(f"{fam}__nf4_{ep}", {})),
                (f"awq_{ep}", "merged 4-bit (vLLM)", st.get(f"{fam}__awq_{ep}", {}).get("size_gb"),
                 st.get(f"{fam}__awq_{ep}", {}))):
            r = {"family": fam, "epoch": ep, "variant": var, "deployment": label, "disk_gb": round(size, 2) if size else None,
                 "weights_vram_gb": d.get("weights_vram_gb"), "tokens_per_s": d.get("tokens_per_s"),
                 "backend": d.get("backend", "vLLM 0.30, bf16, LoRA r16")}
            accs = []
            for s in SPLITS:
                m = mt[(mt.system == f"{fam}__{var}") & (mt.split == s)]
                acc = float(m.judge_lenient.iloc[0]) if len(m) else None
                kfr = float(m.keyfact_recall.iloc[0]) if len(m) else None
                r[f"judge_{s}"] = acc
                r[f"kf_{s}"] = kfr
                if acc is not None and ref[s] is not None and ref[s].judge_lenient:
                    r[f"retained_{s}"] = acc / float(ref[s].judge_lenient)
                    accs.append(r[f"retained_{s}"])
            r["retained_mean"] = sum(accs) / len(accs) if accs else None
            rows.append(r)
    pd.DataFrame(rows).to_csv(TE / "deployment.csv", index=False)


def seed43_judge():
    """Append judge accuracy (seed 43 vs seed 42, epoch 1) to the seed-replication summary."""
    import pandas as pd
    from local_stats import paired_boot
    jg = pd.DataFrame(read_jsonl(TE / "per_item" / "judge_grades.jsonl"))
    if jg.empty:
        return
    old = pd.DataFrame(read_jsonl(LOCAL_OUT / "per_item" / "judge_grades.jsonl"))
    SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    out = []
    for fam in MODEL_ID:
        for s in ("test_indomain", "test_heldout_docs", "test_seen_facts"):
            a = old[(old.system == f"{fam}__ep1") & (old.split == s)].set_index("qa_id").local_grade.map(SC)
            b = jg[(jg.system == f"{fam}__seed43_ep1") & (jg.split == s)].set_index("qa_id").local_grade.map(SC)
            c = a.index.intersection(b.index)
            if not len(c):
                continue
            pb = paired_boot((b.loc[c] - a.loc[c]).to_numpy(dtype=float))
            out.append({"family": fam, "split": s, "metric": "judge_lenient", "seed42": float(a.loc[c].mean()),
                        "seed43": float(b.loc[c].mean()), "diff_43_minus_42": pb["diff"], "abs_diff": abs(pb["diff"]),
                        "diff_lo": pb["lo"], "diff_hi": pb["hi"], "p": pb["p"], "n": pb["n"]})
    if not out:
        return
    d = pd.DataFrame(out)
    d.to_csv(repo_path("results/seed_replication/judge_summary.csv"), index=False)
    sp = repo_path("results/seed_replication/SUMMARY.md")
    txt = sp.read_text().split("\n## Judge accuracy")[0].rstrip() + "\n"
    L = ["", "## Judge accuracy (graded later by the same calibrated local judge, prompt v1)", "",
         "| Model | Split | Seed 42 | Seed 43 | Δ (43 − 42) [95% CI] |", "|---|---|---|---|---|"]
    for r in d.itertuples():
        L.append(f"| {FAMILY_LABEL[r.family]} | {r.split} | {r.seed42:.3f} | {r.seed43:.3f} | "
                 f"{r.diff_43_minus_42:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}] |")
    excl = int(((d.diff_lo > 1e-9) | (d.diff_hi < -1e-9)).sum())
    L.append(f"\n{excl} of {len(d)} judge-accuracy differences have a 95% CI excluding 0; largest |Δ| "
             f"{d.abs_diff.max():.3f}.")
    sp.write_text(txt + "\n".join(L) + "\n")


# ------------------------------------------------------------------ paper pack additions + STATUS

def f3(x, d=3):
    import math
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def lora_facts():
    """Adapter size and trainable fraction per model, computed from the files (for the method description)."""
    from safetensors import safe_open
    out = {}
    for fam, model in MODEL_ID.items():
        f = adapter_path(fam, "ep1") / "adapter_model.safetensors"
        with safe_open(str(f), "pt") as h:
            n = 0
            for k in h.keys():
                c = 1
                for x in h.get_slice(k).get_shape():
                    c *= x
                n += c
            dt = h.get_slice(next(iter(h.keys()))).get_dtype()
        tot = base_param_count(model)
        out[fam] = {"lora_params": n, "base_params": tot, "fraction_pct": 100 * n / tot, "adapter_mb": f.stat().st_size / 1e6,
                    "dtype": dt}
    return out


def cmd_pack():
    import numpy as np
    import pandas as pd
    import local_pack as P
    plt = P.plt_setup()
    mt = pd.read_csv(TE / "metrics_by_system.csv") if (TE / "metrics_by_system.csv").exists() else None
    sig = pd.read_csv(TE / "significance.csv") if (TE / "significance.csv").exists() else None
    dep = pd.read_csv(TE / "deployment.csv") if (TE / "deployment.csv").exists() else None
    gap = pd.read_csv(TE / "robustness_gap.csv") if (TE / "robustness_gap.csv").exists() else None
    be = best_epochs()
    ss = set() if sig is None else {(r.family, r.variant, r.split, r.metric) for r in sig.itertuples()
                                    if r.test == "paired_bootstrap" and r.p_holm < 0.05}

    def row(fam, var, s):
        if mt is None:
            return None
        r = mt[(mt.system == f"{fam}__{var}") & (mt.split == s)]
        return r.iloc[0] if len(r) else None
    cols = [("judge_lenient", "Judge acc", True), ("keyfact_recall", "KF recall", True),
            ("contradiction_rate", "Contra.", False), ("token_f1", "F1", True), ("exact_repro", "Exact repro.", True),
            ("rouge_l", "ROUGE-L", True), ("answer_words", "Words", None)]
    order = ["base", "concise", "ft1run", "ep1", "ep2", "ep3"]
    for s, tname in (("test_trained_exact", "T7_trained_recall"), ("test_seen_facts", "T7b_seen_facts_all_variants"),
                     ("test_heldout_docs", "T7c_heldout_all_variants")):
        rows, vals = [], []
        for fam in MODEL_ID:
            for var in order + [f"nf4_{be[fam]}", f"awq_{be[fam]}"]:
                r = row(fam, var, s)
                if r is None:
                    continue
                cells, vs = [], []
                for c, _, _ in cols:
                    v = r[c]
                    vs.append(v)
                    cells.append(f3(v, 1 if c == "answer_words" else 3) + ("†" if (fam, var, s, c) in ss else ""))
                rows.append([FAMILY_LABEL[fam], var_label(var)] + cells)
                vals.append(vs)
        bold = set()
        if vals:
            arr = np.array(vals, dtype=float)
            for j, (_, _, hib) in enumerate(cols):
                if hib is None or np.all(np.isnan(arr[:, j])):
                    continue
                best = np.nanmax(arr[:, j]) if hib else np.nanmin(arr[:, j])
                for i in np.where(np.isclose(arr[:, j], best))[0]:
                    bold.add((int(i), j + 2))
        title = {"test_trained_exact": "Table 7. Trained-question recall: the exact training questions (n = 500)",
                 "test_seen_facts": "Table 7b. Paraphrased training questions (seen facts, n = 279), all variants",
                 "test_heldout_docs": "Table 7c. Held-out documents (n = 500), all variants"}[s]
        P.write_table(tname, title, ["Model", "Variant"] + [c[1] for c in cols], rows, bold,
                      note="Judge acc = (correct + 0.5 partial) / N (local judge, prompt v1). KF recall = share of reference "
                           "key facts supported by the answer (MiniCheck-7B). Exact repro. = share of answers with ROUGE-L "
                           "≥ 0.8 against the trained answer. † = differs from the same model's base (paired bootstrap, "
                           "Holm-adjusted p < 0.05). Best per column in bold.")
    if dep is not None:
        rows = []
        for r in dep.itertuples():
            accs = [getattr(r, f"judge_{s}") for s in SPLITS]
            rows.append([FAMILY_LABEL[r.family], r.deployment, f3(r.disk_gb, 1), f3(r.weights_vram_gb, 1),
                         f3(r.tokens_per_s, 0)] + [f3(a) for a in accs] +
                        [f"{100 * r.retained_mean:.1f}%" if r.retained_mean == r.retained_mean and r.retained_mean else "–"])
        P.write_table("T8_deployment", "Table 8. Deployment variants of each model's best epoch",
                      ["Model", "Deployment", "Disk (GB)", "Weights VRAM (GB)", "Tokens/s"] +
                      [f"Judge acc {s.replace('test_', '')}" for s in SPLITS] + ["Accuracy retained vs bf16"], rows,
                      note="Best epoch per model by mean local-judge accuracy over the earlier test splits ("
                           + ", ".join(f"{FAMILY_LABEL[f]} {e}" for f, e in be.items()) + "). Disk: base checkpoint + adapter "
                           "(NF4 quantizes the bf16 checkpoint at load time) or the exported 4-bit model. Weights VRAM: vLLM "
                           "'model loading' memory, or torch peak allocation after loading for NF4. Tokens/s: batched greedy "
                           "generation throughput on this workload (vLLM batches the whole split; transformers batch 16), so "
                           "it compares serving stacks rather than kernels. Retained = mean over the 3 splits of variant / bf16 "
                           "judge accuracy.")
    # F9: exact vs paraphrase vs unseen
    if mt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
        tests = [("test_trained_exact", "exact\ntrained Q"), ("test_seen_facts", "paraphrased\ntrained Q"),
                 ("test_heldout_docs", "unseen\ndocuments")]
        styles = {"base": ("#999999", "o", "--"), "concise": ("#56B4E9", "s", "--"), "ep1": ("#E69F00", "^", "-"),
                  "ep2": ("#D55E00", "v", "-"), "ep3": ("#882255", "D", "-")}
        for ax, fam in zip(axes, MODEL_ID):
            for var, (c, m, ls) in styles.items():
                rs = [row(fam, var, s) for s, _ in tests]
                P.errbar(ax, range(3), rs, "judge_lenient", c, var_label(var), marker=m, ls=ls)
            ax.set_xticks(range(3), [t[1] for t in tests])
            ax.set_title(FAMILY_LABEL[fam])
        axes[0].set_ylabel("judge accuracy (correct + 0.5 partial)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle("Figure 9. The three-test picture: exact training questions vs paraphrases vs unseen documents", y=1.03,
                     fontsize=9)
        P.savefig(fig, "F9_exact_paraphrase_unseen")
    # F10: accuracy vs model size
    if dep is not None and len(dep):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        mk = {"bf16 base + LoRA (vLLM)": "o", "NF4 base + LoRA (transformers)": "s", "merged 4-bit (vLLM)": "^"}
        for r in dep.itertuples():
            accs = [getattr(r, f"judge_{s}") for s in SPLITS]
            accs = [a for a in accs if a == a and a is not None]
            size = r.weights_vram_gb if r.weights_vram_gb == r.weights_vram_gb and r.weights_vram_gb else r.disk_gb
            if accs and size == size and size:
                ax.scatter(size, np.mean(accs), color=P.FAMILY_COLOR[r.family], marker=mk.get(r.deployment, "x"), s=50,
                           edgecolor="black", lw=0.5)
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], color=P.FAMILY_COLOR[f], marker="o", ls="none", label=FAMILY_LABEL[f]) for f in MODEL_ID] + \
             [Line2D([], [], color="grey", marker=m, ls="none", label=k) for k, m in mk.items()]
        ax.legend(handles=hs, fontsize=7, loc="best")
        ax.set_xlabel("model weights in GPU memory (GB)")
        ax.set_ylabel("judge accuracy, mean of 3 tests")
        fig.suptitle("Figure 10. Accuracy vs model size: bf16 vs 4-bit deployments of the best epoch", y=1.02, fontsize=9)
        P.savefig(fig, "F10_accuracy_vs_size")
    # RESULTS.md section (local_pack.results_md appends results/trained_eval/RESULTS_section.md)
    lf = lora_facts()
    fr = [v["fraction_pct"] for v in lf.values()]
    mb = [v["adapter_mb"] for v in lf.values()]
    L = ["## Method: QLoRA", "",
         f"All fine-tuned models are QLoRA adapters: the base model is frozen and quantized to 4-bit NF4 (double "
         f"quantization, bf16 compute) during training, and only a bf16 LoRA adapter of rank 16 (alpha 32, dropout 0.05) on "
         f"the q, k, v, o, gate, up and down projections is trained: {min(fr):.2f}–{max(fr):.2f}% of the parameters "
         f"({', '.join(f'{FAMILY_LABEL[f]} {v['lora_params'] / 1e6:.1f}M of {v['base_params'] / 1e9:.2f}B' for f, v in lf.items())}), "
         f"stored in {min(mb):.0f}–{max(mb):.0f} MB per adapter (bf16 safetensors; the run 10-03 adapters hold the same "
         "parameters in fp32). At inference the adapter is applied to the bf16 base (vLLM) unless stated otherwise.", "",
         "## Trained-question recall", "",
         "The exact-question test asks 500 training questions verbatim (stratified by question type × dimension; all 279 "
         "sources of the paraphrased seen-facts test are included, so those items form exact/paraphrase pairs). It measures "
         "recall of trained content and must be read together with the paraphrase test (robustness to rewording) and the "
         "unseen-document test (generalization): Table 7, Table 7b, Table 7c and Figure 9."]
    if mt is not None:
        for fam in MODEL_ID:
            parts = []
            for var in ("base", "concise", "ep1", "ep2", "ep3"):
                r = row(fam, var, "test_trained_exact")
                if r is not None:
                    parts.append(f"{var_label(var)} {f3(r['judge_lenient'])} (KF recall {f3(r['keyfact_recall'])}, "
                                 f"exact repro. {f3(r['exact_repro'])})")
            L.append(f"- {FAMILY_LABEL[fam]}, exact questions, judge accuracy: " + "; ".join(parts) + ".")
    if gap is not None and len(gap):
        g = gap[gap.metric == "judge_lenient"]
        L.append("- Robustness gap (judge accuracy on the exact question minus its paraphrase, 279 pairs): " + "; ".join(
            f"{FAMILY_LABEL[r.family]} {var_label(r.variant)} {r.drop_exact_minus_paraphrase:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
            for r in g.itertuples() if r.variant in ("base", "ep1", "ep2", "ep3")) + " (`results/trained_eval/robustness_gap.csv`).")
    if dep is not None and len(dep):
        L.append("- Deployment (Table 8, Figure 10): " + "; ".join(
            f"{FAMILY_LABEL[r.family]} {r.deployment} retains {100 * r.retained_mean:.1f}% of bf16 judge accuracy"
            for r in dep.itertuples() if r.retained_mean == r.retained_mean and r.retained_mean and "bf16" not in r.deployment) + ".")
    fb = TE / "fallbacks.json"
    if fb.exists():
        for f in json.loads(fb.read_text()):
            L.append(f"- Note ({f['stage']}): {f['fallback']} — {f['why']}.")
    (TE / "RESULTS_section.md").write_text("\n".join(L) + "\n")
    P.build()


def cmd_status():
    import local_pack as P
    st = TE / "stage_status.tsv"
    rows = [l.split("\t") for l in st.read_text().splitlines()] if st.exists() else []
    last = {}
    for r in rows:
        if r and r[0].startswith("T"):
            last[r[0]] = r
    order = ["T0_testset", "T1_gen_vllm", "T2_gen_nf4", "T3_purge_minicheck", "T4_awq", "T5_gguf", "T6_judge_keyfacts",
             "T7_grade", "T8_checks", "T9_stats", "T10_pack"]
    L = ["# Trained-question recall evaluation: STATUS", "", f"_Written {time.strftime('%Y-%m-%d %H:%M')}._", "",
         "| Stage | Result | Wall time (min) | Note |", "|---|---|---|---|"]
    for n in order:
        r = last.get(n)
        L.append(f"| {n} | {r[1]} | {r[2]} | {r[3] if len(r) > 3 else ''} |" if r else
                 (f"| {n} | done (earlier run) | – | |" if (DONE / n).exists() else f"| {n} | not run | – | |"))
    L += ["", "## Decisions and fallbacks", ""]
    pl = [r for r in rows if r and r[0] == "power_limit_at_start"]
    if pl:
        w = float(pl[-1][2])
        L.append(f"- GPU power limit at start ({pl[-1][1]}): {pl[-1][2]} W" + (" — **WARNING: above 270 W**" if w > 270 else
                                                                            " (260 W cap active)"))
    L.append(f"- Best epoch per model (mean local-judge accuracy, earlier splits): {json.dumps(best_epochs())}")
    L.append(f"- Concise base: training system prompt + '{CONCISE.strip()}'.")
    if (TE / "fallbacks.json").exists():
        for f in json.loads((TE / "fallbacks.json").read_text()):
            L.append(f"- {f['stage']}: {f['fallback']} ({f['why']})")
    if (TE / "deploy_stats.json").exists():
        for k, v in json.loads((TE / "deploy_stats.json").read_text()).items():
            L.append(f"- {k}: {v.get('backend')}; tokens/s {v.get('tokens_per_s')}; weights VRAM {v.get('weights_vram_gb')} GB")
    L += P.gpu_lines()[1:2] + P.gpu_lines()[2:3]
    L += ["", "Outputs: `results/trained_eval/` (SUMMARY via metrics_by_system.csv, significance.csv, robustness_gap.csv, "
          "deployment.csv), paper pack Table 7/7b/7c/8 and Figures 9/10, `results/seed_replication/SUMMARY.md` (seed-43 "
          "judge accuracy).", ""]
    (TE / "STATUS.md").write_text("\n".join(L))
    print("\n".join(L))


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
