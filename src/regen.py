"""Serve every adapter on the base it was trained on (fix from results/diagnostics/format_check.md). Local only.

The QLoRA adapters were trained on the NF4-quantized base; earlier evaluations served them on the bf16 base. Those
results stay untouched and are reported as the "__bf16serve" ablation. Here, per model (one at a time):
  dequant  build the NF4-DEQUANTIZED bf16 base (training BitsAndBytesConfig -> bitsandbytes dequantize_4bit -> bf16),
           verify the QA-only ep3 adapter's answer loss on 20 training items matches NF4 within 2%, save it
           (models_dequant/<model>/, ~16 GB, deleted after generation)
  gen      vLLM + LoRA on that base; prompts PRE-TOKENIZED with the training chat function (train_qlora.chat /
           tokenize: no vLLM template re-rendering); greedy, max_new_tokens 512; eval-subset ids of the 4 test types.
           Systems: QA-only ep1-3 (P1/P2), mixed-CPT epochs (P3), the dequantized base without adapter (P4), the
           1-epoch run and seed 43 (deferred), and for Gemma the base / concise base re-rendered with the training
           chat function on the ORIGINAL bf16 base (deferred).
  grade / checks --tier P12|rest   local judge (prompt v1, unchanged) / MiniCheck-7B + DeBERTa NLI + Flan-T5
  stats    metrics with CIs, significance vs base and vs the bf16-served counterpart, serving-precision table,
           robustness gap, length control, paper-pack tables (T7 regen, serving precision)
Internal system ids: <family>__<variant>__mb (matching base) and <family>__<variant>__rr (re-rendered, bf16 base).
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from local_common import FAMILY_LABEL, KEYFACTS, VLLM_PY, log, pause_wait, read_jsonl, write_jsonl
from utils import load_config, repo_path

RG = repo_path("results/regen")
ANS = RG / "answers"
PI = RG / "per_item"
DQ = repo_path("models_dequant")
ORDER = ["qwen3-8b", "gemma-4-e4b-it", "llama-3.1-8b-instruct"]
MODEL_ID = {"qwen3-8b": "Qwen/Qwen3-8B", "gemma-4-e4b-it": "google/gemma-4-E4B-it",
            "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct"}
TESTS = ["test_trained_exact", "test_seen_facts", "test_indomain", "test_heldout_docs"]
CONCISE = " Answer in at most 40 words."
for d in (RG, ANS, PI):
    d.mkdir(parents=True, exist_ok=True)


def cfg():
    return load_config()


def cpt_epochs(fam):
    p = repo_path("models_cpt") / fam / "mixC" / "epochs.json"
    return [e["epoch"] for e in json.loads(p.read_text())["epochs"]] if p.exists() else []


def adapter_path(fam, var):
    if var.startswith("cpt_ep"):
        return repo_path("models_cpt") / fam / "mixC" / f"epoch{var[6:]}" / "adapter"
    if var.startswith("ep"):
        return repo_path("models_epochs") / fam / f"epoch{var[2:]}" / "adapter"
    if var == "ft1run":
        return repo_path("models") / fam / "adapter"
    if var == "seed43_ep1":
        return repo_path("models_seeds") / fam / "seed43" / "epoch1" / "adapter"
    return None                                            # nf4base: the dequantized base itself


def tier_systems(tier, fam=None):
    """[(family, variant, suffix)] for a priority tier; suffix 'mb' = matching (dequantized NF4) base, 'rr' = re-render."""
    fams = [fam] if fam else ORDER
    out = []
    for f in fams:
        if tier in ("P1", "P12") and f in ("qwen3-8b", "gemma-4-e4b-it"):
            out += [(f, v, "mb") for v in ("ep1", "ep2", "ep3")]
        if tier in ("P2", "P12") and f == "llama-3.1-8b-instruct":
            out += [(f, v, "mb") for v in ("ep1", "ep2", "ep3")]
        if tier in ("P3", "rest"):
            out += [(f, f"cpt_ep{e}", "mb") for e in cpt_epochs(f)]
        if tier in ("P4", "rest"):
            out.append((f, "nf4base", "mb"))
        if tier in ("D", "rest"):
            out += [(f, "ft1run", "mb"), (f, "seed43_ep1", "mb")]
            if f == "gemma-4-e4b-it":
                out += [(f, "base", "rr"), (f, "concise40", "rr")]
    return [s for s in out if s[1] == "nf4base" or s[2] == "rr" or adapter_path(s[0], s[1]).exists()]


def sid(fam, var, suf):
    return f"{fam}__{var}__{suf}"


def answer_path(fam, var, suf, split):
    return ANS / sid(fam, var, suf) / f"{split}_predictions.jsonl"


def preds(fam, var, suf, split):
    return {r["qa_id"]: r["prediction"] for r in read_jsonl(answer_path(fam, var, suf, split))}


_items = None


def all_items():
    global _items
    if _items is None:
        import cpt
        _items = cpt.all_items()
    return _items


def prompt_ids(tok, system, question):
    """Prompt tokens exactly as training built them (train_qlora.chat with enable_thinking=False, no extra specials)."""
    from train_qlora import chat
    text = chat(tok, [{"role": "system", "content": system}, {"role": "user", "content": question}], True)
    return tok(text, add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------------ dequantized base

def verify_items(tok):
    from train_qlora import examples, tokenize
    test = read_jsonl(repo_path("data/splits/test_trained_exact.jsonl"))
    train = {r["qa_id"]: r for r in read_jsonl(repo_path("data/splits/train.jsonl"))}
    rows = [train[r["qa_id"]] for r in random.Random(0).sample(test, 20)]
    st = {"prefix_mismatch": 0}
    tr = [tokenize(tok, e, 1024, st) for e in examples(rows, dict(cfg()["train"]), "closedbook", False)]
    return [(t["input_ids"], next(i for i, l in enumerate(t["labels"]) if l != -100)) for t in tr]


def answer_losses(m, seqs):
    import torch
    out = []
    with torch.no_grad():
        for ids, start in seqs:
            x = torch.tensor([ids], device="cuda")
            lab = x.clone()
            lab[:, :start] = -100
            out.append(float(m(input_ids=x, labels=lab).loss))
    return out


def cmd_dequant(a):
    import bitsandbytes as bnbm
    import bitsandbytes.functional as F
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    fam = a.model
    model = MODEL_ID[fam]
    out = DQ / fam
    vp = RG / f"dequant_verify_{fam}.json"
    if (out / "DONE").exists() and vp.exists() and json.loads(vp.read_text())["passed"]:
        log().info(f"[R1 {fam}] dequantized base already built and verified")
        return
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model)
    seqs = verify_items(tok)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0})
    ad = adapter_path(fam, "ep3")
    pm = PeftModel.from_pretrained(m, str(ad)).eval()
    l_nf4 = answer_losses(pm, seqs)
    m = pm.unload()
    n = 0
    for _, mod in list(m.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, bnbm.nn.Linear4bit):
                w = F.dequantize_4bit(child.weight.data, child.weight.quant_state).to(torch.bfloat16)
                lin = torch.nn.Linear(child.in_features, child.out_features, bias=child.bias is not None,
                                      dtype=torch.bfloat16, device=w.device)
                lin.weight.data = w
                if child.bias is not None:
                    lin.bias.data = child.bias.data.to(torch.bfloat16)
                setattr(mod, cname, lin)
                n += 1
    for attr in ("hf_quantizer",):
        if hasattr(m, attr):
            setattr(m, attr, None)
    m.is_quantized = False
    if hasattr(m.config, "quantization_config"):
        del m.config.quantization_config
    pm = PeftModel.from_pretrained(m, str(ad)).eval()
    l_dq = answer_losses(pm, seqs)
    m = pm.unload()
    a_nf4, a_dq = sum(l_nf4) / len(l_nf4), sum(l_dq) / len(l_dq)
    rel = abs(a_dq - a_nf4) / a_nf4
    res = {"model": model, "adapter_for_check": str(ad), "items": len(seqs), "dequantized_linear_layers": n,
           "answer_loss_nf4": round(a_nf4, 5), "answer_loss_dequantized_bf16": round(a_dq, 5),
           "relative_difference": round(rel, 5), "tolerance": 0.02, "passed": rel <= 0.02}
    vp.write_text(json.dumps(res, indent=2))
    log().info(f"[R1 {fam}] dequant check: NF4 {a_nf4:.4f} vs dequantized {a_dq:.4f} (rel {rel:.3%}) -> "
               f"{'PASS' if res['passed'] else 'FAIL'}")
    if not res["passed"]:
        raise SystemExit(2)
    m.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(str(out))
    snap = Path(snapshot_download(model, allow_patterns=["*.json", "*.jinja", "*.model", "*.txt"], local_files_only=True))
    for f in snap.iterdir():
        if f.is_file() and f.suffix in (".json", ".jinja", ".model", ".txt") and not (out / f.name).exists() \
                and "safetensors.index" not in f.name:
            shutil.copy(f.resolve(), out / f.name)
    cfgj = json.loads((out / "config.json").read_text())
    cfgj.pop("quantization_config", None)
    (out / "config.json").write_text(json.dumps(cfgj, indent=2))
    (out / "DONE").write_text(time.strftime("%F %T"))
    size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    log().info(f"[R1 {fam}] dequantized bf16 base saved: {size:.1f} GB in {(time.time() - t0) / 60:.1f} min")


def cmd_purge_dequant(a):
    shutil.rmtree(DQ / a.model, ignore_errors=True)
    log().info(f"[R1 {a.model}] dequantized base deleted")


# ------------------------------------------------------------------ generation

def run_worker(model_dir, rows, raw, label, max_rank):
    have = {r["id"] for r in read_jsonl(raw)}
    todo = [r for r in rows if r["id"] not in have]
    if not todo:
        return 0
    inp = Path(str(raw) + ".in.jsonl")
    write_jsonl(inp, todo)
    cmd = [str(VLLM_PY), str(repo_path("src/vllm_multi_generate.py")), "--model", str(model_dir), "--input", str(inp),
           "--output", str(raw), "--stats", str(raw) + ".stats.json", "--max-new-tokens", str(cfg()["eval"]["max_new_tokens"]),
           "--gpu-memory-utilization", "0.85"]
    if any(r.get("adapter") for r in todo):
        cmd += ["--lora", "--max-lora-rank", str(max_rank)]
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0", LLM_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=str(repo_path("src/vllm_shims")))
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    log().info(f"[{label}] vLLM: {len(todo)} prompts on {model_dir}")
    with repo_path("logs/regen_worker.log").open("a") as lf:
        lf.write(f"\n===== {time.strftime('%F %T')} {label}: {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=lf).returncode
    inp.unlink(missing_ok=True)
    left = len({r["id"] for r in todo} - {r["id"] for r in read_jsonl(raw)})
    log().info(f"[{label}] worker exit {rc}; {len(todo) - left}/{len(todo)} new answers")
    return left


def write_answers(raw):
    rows = defaultdict(dict)
    for r in read_jsonl(raw):
        s, split, q = r["id"].split("|", 2)
        rows[(s, split)][q] = {"qa_id": q, "prediction": r["prediction"], "n_tokens": r.get("n_tokens")}
    for (s, split), d in rows.items():
        write_jsonl(ANS / s / f"{split}_predictions.jsonl", list(d.values()))


def cmd_gen(a):
    from transformers import AutoTokenizer
    fam = a.model
    its = all_items()
    tok = AutoTokenizer.from_pretrained(MODEL_ID[fam])
    system = cfg()["train"]["system_prompt"]
    cache = {}

    def pids(sysmsg, q):
        k = (sysmsg, q)
        if k not in cache:
            cache[k] = prompt_ids(tok, sysmsg, q)
        return cache[k]
    raw = RG / "raw"
    raw.mkdir(exist_ok=True)
    # 1. adapters + no-adapter control on the dequantized NF4 base
    systems = [s for t in ("P12", "P3", "P4", "D") for s in tier_systems(t, fam) if s[2] == "mb"]
    rows, ranks = [], [16]
    for f, var, suf in systems:
        ad = adapter_path(f, var)
        if ad is not None:
            ranks.append(json.loads((ad / "adapter_config.json").read_text())["r"])
        for s in TESTS:
            for q, r in its[s].items():
                rows.append({"id": f"{sid(f, var, suf)}|{s}|{q}", "system": system, "question": r["question"],
                             "prompt_token_ids": pids(system, r["question"]), "adapter": str(ad) if ad else None})
    pause_wait("R1 gen")
    left = run_worker(DQ / fam, rows, raw / f"{fam}__mb.jsonl", f"R1 {fam} matching base", max(ranks))
    write_answers(raw / f"{fam}__mb.jsonl")
    # 2. Gemma: base / concise base re-rendered with the training chat function, original bf16 base
    if fam == "gemma-4-e4b-it":
        rr = []
        for var, sysmsg in (("base", system), ("concise40", system + CONCISE)):
            for s in TESTS:
                for q, r in its[s].items():
                    rr.append({"id": f"{sid(fam, var, 'rr')}|{s}|{q}", "system": sysmsg, "question": r["question"],
                               "prompt_token_ids": pids(sysmsg, r["question"]), "adapter": None})
        run_worker(MODEL_ID[fam], rr, raw / f"{fam}__rr.jsonl", f"R1 {fam} re-render", 16)
        write_answers(raw / f"{fam}__rr.jsonl")
    if left:
        raise SystemExit(f"{fam}: {left} answers missing on the matching base")


# ------------------------------------------------------------------ grading and checks

def pairs_for(systems):
    its = all_items()
    out = []
    for f, var, suf in systems:
        for s in TESTS:
            p = preds(f, var, suf, s)
            for q, r in its[s].items():
                if q in p:
                    out.append({"split": s, "qa_id": q, "system": sid(f, var, suf), "family": f, "variant": var,
                                "suffix": suf, "question": r["question"], "reference": r["answer"], "answer": p[q]})
    return out


def judge_local():
    from huggingface_hub import snapshot_download
    from local_common import JUDGES
    try:
        p = Path(snapshot_download(JUDGES["A"]["model"], allow_patterns=["*.safetensors"], local_files_only=True))
        return len(list(p.glob("*.safetensors"))) >= 4
    except Exception:
        return False


def download(repo, patterns=None, tag="R"):
    from huggingface_hub import snapshot_download
    for i in range(3):
        try:
            snapshot_download(repo, allow_patterns=patterns)
            return True
        except Exception as e:
            log().info(f"[{tag}] download of {repo} failed (try {i + 1}/3): {e}")
            time.sleep(30 * (i + 1))
    return False


def cmd_grade(a):
    import local_checks as L
    import local_judge as J
    from local_common import JUDGES, OUT, deadline_from_env
    from trained_eval import delete_hf_model, free_gb
    deadline = deadline_from_env(120)
    if L.minicheck_local():
        delete_hf_model(L.MC7B, "R grade")
    if not judge_local():
        if free_gb() < 10 + 16:
            raise SystemExit(f"only {free_gb():.1f} GB free for the judge download")
        if not download(JUDGES["A"]["model"], tag="R grade"):
            raise SystemExit("judge download failed 3 times")
    ch = json.loads((OUT / "judge_choice.json").read_text())
    tiers = ["P1", "P2"] if a.tier == "P12" else ["P3", "P4", "D"]
    out = {(r["split"], r["qa_id"], r["system"]): r for r in read_jsonl(PI / "judge_grades.jsonl")}
    for t in tiers:                                         # priority order
        pairs = [p for p in pairs_for(tier_systems(t)) if (p["split"], p["qa_id"], p["system"]) not in out]
        if not pairs:
            continue
        log().info(f"[R grade {t}] grading {len(pairs)} answers ({ch['judge_name']}, prompt {ch['prompt']}, unchanged)")
        g = J.grade_pairs(ch["judge"], ch["prompt"], pairs, deadline, f"R grade {t}")
        for p in pairs:
            r = g.get(J.key(p["split"], p["qa_id"], p["system"]))
            if r:
                out[(p["split"], p["qa_id"], p["system"])] = {k: p[k] for k in ("split", "qa_id", "system", "family", "variant", "suffix")} | {
                    "local_grade": r["grade"], "hallucinated_specific": bool(r.get("hallucinated_specific"))}
        write_jsonl(PI / "judge_grades.jsonl", list(out.values()))
    need = pairs_for([s for t in tiers for s in tier_systems(t)])
    got = sum(1 for p in need if (p["split"], p["qa_id"], p["system"]) in out)
    log().info(f"[R grade {a.tier}] graded {got}/{len(need)}")
    if got < 0.95 * len(need):
        raise SystemExit("grading incomplete")


def cmd_checks(a):
    import gc

    import local_checks as L
    import torch
    from local_common import JUDGES, deadline_from_env
    from trained_eval import delete_hf_model, free_gb
    deadline = deadline_from_env(60)
    if (RG / ".done" / f"R_grade_{a.tier}").exists() and judge_local():
        delete_hf_model(JUDGES["A"]["model"], "R checks")      # only after grading succeeded
    if not L.minicheck_local():
        if free_gb() < 10 + 16 or not download(L.MC7B, ["*.json", "*.py", "*.model", "*.safetensors"], "R checks"):
            raise SystemExit("MiniCheck-7B not available (disk or download)")
    tiers = ["P1", "P2"] if a.tier == "P12" else ["P3", "P4", "D"]
    kf = {s: {r["qa_id"]: r["facts"] for r in read_jsonl(KEYFACTS / f"{s}.jsonl")} for s in TESTS}
    pairs = []
    for f, var, suf in [s for t in tiers for s in tier_systems(t)]:
        for s in TESTS:
            for q, ans in preds(f, var, suf, s).items():
                for k, fact in enumerate(kf[s].get(q, [])):
                    pairs.append((f"{s}|{q}|{sid(f, var, suf)}|{k}", s, q, sid(f, var, suf), k, ans, fact))
    log().info(f"[R checks {a.tier}] {len(pairs)} (answer, fact) pairs")
    (RG / "cache").mkdir(exist_ok=True)
    have = {(r["split"], r["qa_id"], r["system"], r["k"]) for r in read_jsonl(PI / "support_minicheck7b.jsonl")}
    todo = [p for p in pairs if (p[1], p[2], p[3], p[4]) not in have]
    if todo:
        pause_wait("R checks")
        L.score_minicheck7b(todo, RG / "cache" / "minicheck7b_raw.jsonl", RG / "cache" / "mc_tmp.jsonl", deadline, "R checks")
        with (PI / "support_minicheck7b.jsonl").open("a", encoding="utf-8") as fh:
            for r in read_jsonl(RG / "cache" / "mc_tmp.jsonl"):
                fh.write(json.dumps(r) + "\n")
    gc.collect()
    torch.cuda.empty_cache()
    L.score_nli_facts(L.NLIScorer(), pairs, PI / "nli_facts.jsonl", deadline, "R NLI")
    gc.collect()
    torch.cuda.empty_cache()
    L.score_flant5(pairs, PI / "support_flant5.jsonl", deadline, "R Flan-T5")
    have = {(r["split"], r["qa_id"], r["system"], r["k"]) for r in read_jsonl(PI / "support_minicheck7b.jsonl")}
    miss = sum(1 for p in pairs if (p[1], p[2], p[3], p[4]) not in have)
    if miss > 0.05 * len(pairs):
        raise SystemExit(f"MiniCheck covered only {len(pairs) - miss}/{len(pairs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--model")
    ap.add_argument("--tier")
    a = ap.parse_args()
    if a.cmd in ("stats", "status"):
        import regen_report
        return getattr(regen_report, f"cmd_{a.cmd}")(a)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
