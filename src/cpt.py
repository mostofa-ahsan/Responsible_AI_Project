"""Arm C (mixed continued pretraining) evaluation stages, driven by scripts/run_cpt.sh. Local only (LLM_OFFLINE=1).

gen_concise              concise base ("Answer in at most 40 words.") on the 4 test types
gen_epoch --model --epoch  vLLM greedy answers with models_cpt/<m>/mixC/epoch<N>/adapter on the 4 test types
checks [--model]         MiniCheck-7B support (if its weights are on disk), DeBERTa NLI contradiction and Flan-T5
                         support against the cached key facts, for every new answer (or one model's)
interim                  results/cpt/INTERIM.md: F1, ROUGE-L, length, exact reproduction, key-fact recall, contradiction
judge                    delete MiniCheck, download the calibrated judge (3 retries), key facts for test_trained_exact
                         if missing, grade every new answer (concise base + arm C)
checks_final             delete the judge, re-download MiniCheck-7B only if any new pair lacks it, finish all checks
report / status          results/cpt/SUMMARY.md, paper-pack T7 (per test type), F9, F10, robustness gap, RESULTS.md
                         section, STATUS.md

Systems: <family>__<variant>; variants base, concise40 (new), ep1-3 (QA-only arm, models_epochs), cpt_ep1-3 (arm C).
"""

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from local_common import FAMILIES, FAMILY_LABEL, KEYFACTS, OUT as LOCAL_OUT, PER_ITEM as LOCAL_PI, items, log, \
    pause_wait, pred_path, read_jsonl, write_jsonl
from utils import load_config, repo_path

CP = repo_path("results/cpt")
ANS = CP / "answers"
PI = CP / "per_item"
TE = repo_path("results/trained_eval")
ORDER = ["llama-3.1-8b-instruct", "qwen3-8b", "gemma-4-e4b-it"]
MODEL_ID = {"llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct", "qwen3-8b": "Qwen/Qwen3-8B",
            "gemma-4-e4b-it": "google/gemma-4-E4B-it"}
TESTS = ["test_trained_exact", "test_seen_facts", "test_indomain", "test_heldout_docs"]
TEST_LABEL = {"test_trained_exact": "exact training questions", "test_seen_facts": "paraphrased training questions",
              "test_indomain": "in-domain (unseen chunks)", "test_heldout_docs": "held-out documents"}
CONCISE = " Answer in at most 40 words."
for d in (CP, ANS, PI):
    d.mkdir(parents=True, exist_ok=True)


def cfg():
    return load_config()


_items = None


def all_items():
    global _items
    if _items is None:
        its = items()
        _items = {"test_trained_exact": {r["qa_id"]: r for r in read_jsonl(repo_path("data/splits/test_trained_exact.jsonl"))},
                  **{s: its[s] for s in ("test_seen_facts", "test_indomain", "test_heldout_docs")}}
    return _items


def answer_path(fam, var, split):
    """New answers live in results/cpt/answers; earlier systems are read where they were produced."""
    if var == "concise40" or var.startswith("cpt_"):
        return ANS / f"{fam}__{var}" / f"{split}_predictions.jsonl"
    if split == "test_trained_exact":
        return TE / "answers" / f"{fam}__{var}" / f"{split}_predictions.jsonl"
    return pred_path(fam, var, split)


def preds(fam, var, split):
    return {r["qa_id"]: r["prediction"] for r in read_jsonl(answer_path(fam, var, split))}


def cpt_epochs(fam):
    p = repo_path("models_cpt") / fam / "mixC" / "epochs.json"
    return json.loads(p.read_text()) if p.exists() else {"epochs": []}


def new_systems(fams=None):
    out = []
    for fam in (fams or ORDER):
        out.append((fam, "concise40"))
        out += [(fam, f"cpt_ep{e['epoch']}") for e in cpt_epochs(fam)["epochs"]]
    return out


def run_vllm(model, rows, raw, label, lora_rank=16, lora=False):
    from trained_eval import run_multi
    return run_multi(model, rows, raw, lora, Path(str(raw) + ".stats.json"), label, max_lora_rank=lora_rank)


def write_answers(raw, fam):
    rows = defaultdict(dict)
    for r in read_jsonl(raw):
        var, split, q = r["id"].split("|", 2)
        rows[(var, split)][q] = {"qa_id": q, "prediction": r["prediction"], "n_tokens": r.get("n_tokens")}
    for (var, split), d in rows.items():
        write_jsonl(ANS / f"{fam}__{var}" / f"{split}_predictions.jsonl", list(d.values()))


def system_prompt(var):
    s = cfg()["train"]["system_prompt"]
    return s + CONCISE if var == "concise40" else s


def cmd_gen_concise(a):
    its = all_items()
    for fam in ORDER:
        pause_wait("C1")
        rows = [{"id": f"concise40|{s}|{q}", "system": system_prompt("concise40"), "question": r["question"],
                 "adapter": None} for s in TESTS for q, r in its[s].items()]
        raw = CP / "raw" / f"{fam}__concise40.jsonl"
        raw.parent.mkdir(exist_ok=True)
        run_vllm(MODEL_ID[fam], rows, raw, f"C1 {fam} concise40")
        write_answers(raw, fam)
    miss = [(f, s) for f in ORDER for s in TESTS if len(preds(f, "concise40", s)) < len(its[s])]
    if miss:
        raise SystemExit(f"concise answers incomplete: {miss}")


def cmd_gen_epoch(a):
    fam = a.model
    ep = a.epoch
    ad = repo_path("models_cpt") / fam / "mixC" / f"epoch{ep}" / "adapter"
    its = all_items()
    var = f"cpt_ep{ep}"
    rank = json.loads((ad / "adapter_config.json").read_text())["r"]
    rows = [{"id": f"{var}|{s}|{q}", "system": system_prompt(var), "question": r["question"], "adapter": str(ad)}
            for s in TESTS for q, r in its[s].items()]
    raw = CP / "raw" / f"{fam}__{var}.jsonl"
    raw.parent.mkdir(exist_ok=True)
    pause_wait("C3 gen")
    left = run_vllm(MODEL_ID[fam], rows, raw, f"C3 {fam} {var}", lora_rank=rank, lora=True)
    write_answers(raw, fam)
    if left:
        raise SystemExit(f"{fam} {var}: {left} answers missing")


# ------------------------------------------------------------------ checks

def fact_pairs(systems):
    kf = {s: {r["qa_id"]: r["facts"] for r in read_jsonl(KEYFACTS / f"{s}.jsonl")} for s in TESTS}
    out = []
    for fam, var in systems:
        sid = f"{fam}__{var}"
        for s in TESTS:
            for q, a in preds(fam, var, s).items():
                for k, f in enumerate(kf[s].get(q, [])):
                    out.append((f"{s}|{q}|{sid}|{k}", s, q, sid, k, a, f))
    return out


def run_checks(systems, minicheck=True, flan=True, deadline=None):
    import local_checks as L
    pairs = fact_pairs(systems)
    deadline = deadline or time.time() + 3 * 3600
    done = {}
    if minicheck and L.minicheck_local():
        pause_wait("checks")
        have = {(r["split"], r["qa_id"], r["system"], r["k"]) for r in read_jsonl(PI / "support_minicheck7b.jsonl")}
        todo = [p for p in pairs if (p[1], p[2], p[3], p[4]) not in have]
        if todo:
            # score_minicheck7b rewrites its output from the raw cache, which holds every pair scored so far
            L.score_minicheck7b(todo, CP / "cache" / "minicheck7b_raw.jsonl", CP / "cache" / "mc_tmp.jsonl", deadline,
                                "C checks")
            new = read_jsonl(CP / "cache" / "mc_tmp.jsonl")
            with (PI / "support_minicheck7b.jsonl").open("a", encoding="utf-8") as f:
                for r in new:
                    f.write(json.dumps(r) + "\n")
        done["minicheck"] = True
    else:
        done["minicheck"] = False
    import gc

    import torch
    if flan:
        pause_wait("checks")
        L.score_flant5(pairs, PI / "support_flant5.jsonl", deadline, "C Flan-T5")
        gc.collect()
        torch.cuda.empty_cache()
    pause_wait("checks")
    L.score_nli_facts(L.NLIScorer(), pairs, PI / "nli_facts.jsonl", deadline, "C NLI")
    gc.collect()
    torch.cuda.empty_cache()
    return done


def cmd_checks(a):
    (CP / "cache").mkdir(exist_ok=True)
    fams = [a.model] if a.model else None
    systems = new_systems(fams) if not a.concise_only else [(f, "concise40") for f in ORDER]
    log().info(f"[C checks] {len(systems)} systems")
    d = run_checks(systems)
    if not d["minicheck"]:
        log().info("[C checks] MiniCheck-7B weights not on disk: key-fact recall pending (Stage 4)")


# ------------------------------------------------------------------ per-item table and interim report

SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def per_item(fams=None, variants=None):
    import numpy as np
    import pandas as pd
    from eval_closedbook import rouge_l, token_f1
    its = all_items()
    kf = {(s, r["qa_id"]): len(r["facts"]) for s in TESTS for r in read_jsonl(KEYFACTS / f"{s}.jsonl")}

    def keyed(paths, field, by="k"):
        d = {}
        for p in paths:
            for r in read_jsonl(p):
                d.setdefault((r["split"], r["qa_id"], r["system"]), {})[r[by]] = r[field]
        return d
    sup = keyed([LOCAL_PI / "support_minicheck7b.jsonl", TE / "per_item" / "support_minicheck7b.jsonl",
                 PI / "support_minicheck7b.jsonl"], "p")
    con = keyed([LOCAL_PI / "nli_facts.jsonl", TE / "per_item" / "nli_facts.jsonl", PI / "nli_facts.jsonl"], "contradicted")
    judge = {}
    for p in (LOCAL_PI / "judge_grades.jsonl", TE / "per_item" / "judge_grades.jsonl", PI / "judge_grades.jsonl"):
        for r in read_jsonl(p):
            if r.get("local_grade"):
                judge[(r["split"], r["qa_id"], r["system"])] = r
    rows = []
    for fam in (fams or ORDER):
        vs = variants or (["base", "concise40", "ep1", "ep2", "ep3"] +
                          [f"cpt_ep{e['epoch']}" for e in cpt_epochs(fam)["epochs"]])
        for var in vs:
            sid = f"{fam}__{var}"
            for s in TESTS:
                p = preds(fam, var, s)
                for q, it in its[s].items():
                    if q not in p:
                        continue
                    k = (s, q, sid)
                    n = kf.get((s, q))
                    sp, cn = sup.get(k, {}), con.get(k, {})
                    lg = judge.get(k, {}).get("local_grade")
                    rl = rouge_l(p[q], it["answer"])
                    rows.append({"system": sid, "family": fam, "variant": var, "split": s, "qa_id": q,
                                 "paraphrase_test_id": it.get("paraphrase_test_id"),
                                 "answer_words": len(p[q].split()), "token_f1": token_f1(p[q], it["answer"]),
                                 "rouge_l": rl, "exact_repro": float(rl >= 0.8),
                                 "judge_lenient": SC[lg] if lg else np.nan,
                                 "judge_strict": float(lg == "correct") if lg else np.nan,
                                 "keyfact_recall": sum(v >= 0.5 for v in sp.values()) / n if n and len(sp) >= n else np.nan,
                                 "contradiction_rate": sum(cn.values()) / n if n and len(cn) >= n else np.nan})
    return pd.DataFrame(rows)


def f3(x, d=3):
    import math
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def cmd_interim(a):
    import pandas as pd
    fams = [f for f in ORDER if cpt_epochs(f)["epochs"]]
    L = ["# Arm C (mixed continued pretraining): interim results", "",
         f"_Updated {time.strftime('%Y-%m-%d %H:%M')} by `src/cpt.py interim`. Judge grades come in Stage 4; '–' = not yet "
         "measured. Key-fact recall: MiniCheck-7B on the cached key facts. Exact repro. = ROUGE-L ≥ 0.8 vs the trained "
         "answer._", ""]
    for fam in fams:
        st = cpt_epochs(fam)
        L += [f"## {FAMILY_LABEL[fam]} (LoRA r={st.get('rank')}, micro-batch {st.get('micro_batch')})", "",
              "| Epoch | minutes | train loss (raw / QA) | val QA loss | held-out ppl | peak VRAM (GB) |", "|---|---|---|---|---|---|",
              f"| 0 (base) | – | – | {f3(st.get('epoch0', {}).get('val_qa_loss'))} | {f3(st.get('epoch0', {}).get('heldout_ppl'))} | – |"]
        for e in st["epochs"]:
            L.append(f"| {e['epoch']} | {e['minutes']} | {f3(e.get('raw_loss_mean'))} / {f3(e.get('qa_loss_mean'))} | "
                     f"{f3(e.get('val_qa_loss'))} | {f3(e.get('heldout_ppl'))} | {e.get('peak_vram_gb')} |")
        df = per_item([fam])
        L += ["", "| Test | Variant | F1 | ROUGE-L | Words | Exact repro. | KF recall | Contra. |", "|---|---|---|---|---|---|---|---|"]
        for s in TESTS:
            for var in ["base", "concise40", "ep1", "ep2", "ep3"] + [f"cpt_ep{e['epoch']}" for e in st["epochs"]]:
                g = df[(df.variant == var) & (df.split == s)]
                if g.empty:
                    continue
                L.append(f"| {s.replace('test_', '')} | {var} | {f3(g.token_f1.mean())} | {f3(g.rouge_l.mean())} | "
                         f"{f3(g.answer_words.mean(), 1)} | {f3(g.exact_repro.mean())} | {f3(g.keyfact_recall.mean())} | "
                         f"{f3(g.contradiction_rate.mean())} |")
        L.append("")
    (CP / "INTERIM.md").write_text("\n".join(L) + "\n")
    log().info(f"[C interim] INTERIM.md updated ({', '.join(fams)})")


# ------------------------------------------------------------------ Stage 4: judge (one download), then checks

def download(repo, tag, patterns=None):
    from huggingface_hub import snapshot_download
    for i in range(3):
        try:
            snapshot_download(repo, allow_patterns=patterns)
            return True
        except Exception as e:
            log().info(f"[{tag}] download of {repo} failed (try {i + 1}/3): {e}")
            time.sleep(60 * (i + 1))
    return False


def fallback(stage, what, why):
    p = CP / "fallbacks.json"
    d = json.loads(p.read_text()) if p.exists() else []
    d.append({"time": time.strftime("%F %T"), "stage": stage, "fallback": what, "why": why})
    p.write_text(json.dumps(d, indent=2))
    log().info(f"[{stage}] FALLBACK: {what} ({why})")


def judge_local():
    from huggingface_hub import snapshot_download
    from local_common import JUDGES
    try:
        p = Path(snapshot_download(JUDGES["A"]["model"], allow_patterns=["*.safetensors"], local_files_only=True))
        return len(list(p.glob("*.safetensors"))) >= 4
    except Exception:
        return False


def cmd_judge(a):
    import local_checks as L
    import local_judge as J
    from local_common import JUDGES, deadline_from_env
    from trained_eval import delete_hf_model, free_gb
    deadline = deadline_from_env(180)
    if L.minicheck_local():
        delete_hf_model(L.MC7B, "C4")
    if not judge_local() and free_gb() < 10 + 16:          # headroom only matters if it still has to be downloaded
        raise SystemExit(f"only {free_gb():.1f} GB free for the judge")
    if not download(JUDGES["A"]["model"], "C4"):
        fallback("C4", "judge grades pending", "judge download failed 3 times")
        raise SystemExit(2)
    if not (KEYFACTS / "test_trained_exact.jsonl").exists():
        import trained_eval as T
        T.cmd_judge_keyfacts()
    ch = json.loads((LOCAL_OUT / "judge_choice.json").read_text())
    its = all_items()
    pairs = []
    for fam, var in new_systems():
        for s in TESTS:
            p = preds(fam, var, s)
            for q, r in its[s].items():
                if q in p:
                    pairs.append({"split": s, "qa_id": q, "system": f"{fam}__{var}", "family": fam, "variant": var,
                                  "question": r["question"], "reference": r["answer"], "answer": p[q]})
    log().info(f"[C4] grading {len(pairs)} answers with {ch['judge_name']} prompt {ch['prompt']} (unchanged)")
    g = J.grade_pairs(ch["judge"], ch["prompt"], pairs, deadline, "C4")
    out = []
    for p in pairs:
        r = g.get(J.key(p["split"], p["qa_id"], p["system"]))
        out.append({k: p[k] for k in ("split", "qa_id", "system", "family", "variant")} |
                   {"local_grade": r["grade"] if r else None,
                    "hallucinated_specific": bool(r.get("hallucinated_specific")) if r else None})
    write_jsonl(PI / "judge_grades.jsonl", out)
    n = sum(1 for o in out if o["local_grade"])
    log().info(f"[C4] graded {n}/{len(out)}")
    if n < 0.95 * len(out):
        raise SystemExit("C4: grading incomplete")


def cmd_checks_final(a):
    import local_checks as L
    from local_common import JUDGES
    from trained_eval import delete_hf_model, free_gb
    if (CP / ".done" / "C4_judge").exists():
        delete_hf_model(JUDGES["A"]["model"], "C5")   # only once grading succeeded (never lose an ungraded judge)
    else:
        fallback("C5", "judge kept on disk", "C4 did not finish; deleting the judge now would force another download")
    systems = new_systems()
    pairs = fact_pairs(systems)
    have = {(r["split"], r["qa_id"], r["system"], r["k"]) for r in read_jsonl(PI / "support_minicheck7b.jsonl")}
    missing = [p for p in pairs if (p[1], p[2], p[3], p[4]) not in have]
    log().info(f"[C5] {len(missing)} of {len(pairs)} new (answer, fact) pairs lack MiniCheck-7B")
    if missing and not L.minicheck_local():
        if free_gb() < 10 + 16 or not download(L.MC7B, "C5", ["*.json", "*.py", "*.model", "*.safetensors"]):
            fallback("C5", "MiniCheck-7B key-fact recall pending for some answers", "download failed or disk short")
            run_checks(systems, minicheck=False)
            return
    run_checks(systems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--model")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--concise-only", action="store_true")
    a = ap.parse_args()
    if a.cmd in ("report", "status"):
        import cpt_report
        return getattr(cpt_report, f"cmd_{a.cmd}")(a)
    fn = globals().get(f"cmd_{a.cmd}")
    if fn is None:
        raise SystemExit(f"unknown command {a.cmd}")
    fn(a)


if __name__ == "__main__":
    main()
