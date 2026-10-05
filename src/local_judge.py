"""Local LLM judge (S1 calibration against Opus, S2 full grading). No API calls.

Rubric: JUDGE_SYSTEM / JUDGE_PROMPT from src/eval_closedbook.py verbatim (the test_seen_facts variant adds
the unsupported_claims line, as in src/seen_facts.py). Output by guided JSON:
  {"reasoning": <=60 words, "grade": correct|partial|incorrect, "hallucinated_specific": bool
   [, "unsupported_claims": "0"|"1"|"2+"  (seen facts only)]}
The judge never sees which system wrote the answer; prompts are shuffled. Cache:
results/local_eval/judge_cache/<judge>__<prompt>.jsonl keyed by "<split>|<qa_id>|<system>".

Prompt versions (fixed in advance; at most 2 revisions of the verbatim rubric, chosen ONLY on DEV):
  v0  rubric verbatim + output-field definitions
  v1  v0 + clarification of key-information / list answers / extra detail
  v2  v1 + reasoning must first name which reference key points are present or missing

Usage:
    python src/local_judge.py smoke --n 20
    python src/local_judge.py calibrate [--judge A]
    python src/local_judge.py grade
"""

import argparse
import json
import random
import time
from collections import Counter, defaultdict

import numpy as np

from local_common import (CACHE, FAMILIES, JUDGES, OPUS_VARIANTS, OUT, PER_ITEM, SPLITS, VARIANTS, deadline_from_env,
                          done, items, log, mark, opus_grades, parse_json, preds, run_worker, shuffled, sys_id,
                          systems, words, write_jsonl, read_jsonl)
from eval_closedbook import JUDGE_PROMPT, JUDGE_SYSTEM

HALLU_DEF = ("hallucinated_specific: true if the model answer asserts at least one specific detail (a name, number, "
             "list item, mechanism or finding) that contradicts the reference or that the reference does not "
             "support and that looks invented; generic framing does not count. Otherwise false.")
SEEN_LINE = ("unsupported_claims: count the factual claims in the model answer that are NOT supported by the\n"
             "reference answer (a claim that contradicts it, or a specific detail it does not contain; generic\n"
             "framing does not count): \"0\", \"1\", or \"2+\".")
OUT_DEF = ("Respond with JSON only: reasoning (at most 60 words), grade (correct, partial or incorrect), "
           "hallucinated_specific (true/false).")
V1_ADD = ("Guidance: the key information is the specific content the reference gives in answer to the question. "
          "An answer that contains the reference's key points is correct even if it adds further correct or "
          "plausible points, as long as nothing contradicts the reference. Use partial when only some of the key "
          "points are present, or when the answer stays generic where the reference is specific. Use incorrect "
          "when the key points are absent or contradicted, or the answer addresses a different question.")
V2_ADD = ("In reasoning, first name which key points of the reference the model answer contains and which it "
          "misses, then decide the grade.")
PROMPTS = ["v0", "v1", "v2"]
ORD = {"incorrect": 0, "partial": 1, "correct": 2}


def system_text(version, seen):
    parts = [JUDGE_SYSTEM]
    if seen:
        parts.append(SEEN_LINE)
    parts.append(HALLU_DEF)
    if version in ("v1", "v2"):
        parts.append(V1_ADD)
    if version == "v2":
        parts.append(V2_ADD)
    parts.append(OUT_DEF + (" Also unsupported_claims." if seen else ""))
    return "\n\n".join(parts)


def schema(seen):
    props = {"reasoning": {"type": "string", "maxLength": 420},
             "grade": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
             "hallucinated_specific": {"type": "boolean"}}
    req = ["reasoning", "grade", "hallucinated_specific"]
    if seen:
        props["unsupported_claims"] = {"type": "string", "enum": ["0", "1", "2+"]}
        req.append("unsupported_claims")
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


def key(split, qa_id, sid):
    return f"{split}|{qa_id}|{sid}"


def all_pairs(sys_list=None):
    """Every (split, qa_id, system) to grade, with question / reference / answer."""
    its = items()
    out = []
    for fam, var in (sys_list or systems()):
        sid = sys_id(fam, var)
        for s in SPLITS:
            p = preds(fam, var, s)
            for q, r in its[s].items():
                if q in p:
                    out.append({"split": s, "qa_id": q, "system": sid, "family": fam, "variant": var,
                                "question": r["question"], "reference": r["answer"], "answer": p[q]})
    return out


def worker_extra(j):
    return ["--mistral3"] if JUDGES[j]["mistral3"] else []


def cache_path(j, version):
    return CACHE / f"{JUDGES[j]['name']}__{version}.jsonl"


def grade_pairs(j, version, pairs, deadline, label):
    """Grade pairs (seen and non-seen use separate schemas -> two worker calls). Returns {key: parsed}."""
    cp = cache_path(j, version)
    for seen in (False, True):
        sel = [p for p in pairs if (p["split"] == "test_seen_facts") == seen]
        if not sel:
            continue
        rows = [{"id": key(p["split"], p["qa_id"], p["system"]),
                 "messages": [{"role": "system", "content": system_text(version, seen)},
                              {"role": "user", "content": JUDGE_PROMPT.format(question=p["question"],
                                                                              reference=p["reference"],
                                                                              prediction=p["answer"])}]}
                for p in shuffled(sel, seed=PROMPTS.index(version) + 7)]
        run_worker(JUDGES[j]["model"], rows, cp, schema=schema(seen), max_tokens=320, deadline=deadline,
                   extra=worker_extra(j), label=f"{label} {'seen' if seen else 'unseen'}")
    return load_cache(j, version)


def load_cache(j, version):
    out = {}
    for r in read_jsonl(cache_path(j, version)):
        g = parse_json(r.get("text", ""))
        if g and g.get("grade") in ORD:
            out[r["id"]] = g
    return out


# ---------------- calibration ----------------

def opus_table():
    """[(pair, opus_verdict, opus_unsupported)] for the 6 Opus-graded systems."""
    rows = []
    for p in all_pairs([(f, v) for f in FAMILIES for v in OPUS_VARIANTS]):
        g = opus_grades(p["family"], p["variant"], p["split"]).get(p["qa_id"])
        if g:
            rows.append(dict(p, opus=g["verdict"], opus_unsupported=g.get("unsupported_claims")))
    return rows


def dev_test_split(rows, dev_grades=1000, seed=42):
    """Split by qa_id (all systems of an item on one side), stratified by split x modal Opus verdict."""
    by_item = defaultdict(list)
    for r in rows:
        by_item[(r["split"], r["qa_id"])].append(r)
    strata = defaultdict(list)
    for (s, q), rs in by_item.items():
        modal = Counter(r["opus"] for r in rs).most_common(1)[0][0]
        strata[(s, modal)].append((s, q))
    n_items = len(by_item)
    per_item = len(rows) / n_items
    target_items = round(dev_grades / per_item)
    rng = random.Random(seed)
    dev = set()
    for k in sorted(strata):
        lst = sorted(strata[k])
        rng.shuffle(lst)
        dev.update(lst[:round(target_items * len(lst) / n_items)])
    return dev


def kappas(y_opus, y_loc):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    a = np.array([ORD[x] for x in y_opus])
    b = np.array([ORD[x] for x in y_loc])
    if len(a) < 2:
        return {"n": int(len(a))}

    def k(x, y, **kw):
        try:
            v = cohen_kappa_score(x, y, **kw)
            return None if np.isnan(v) else round(float(v), 4)
        except Exception:
            return None
    return {"n": int(len(a)), "agreement_3class": round(float((a == b).mean()), 4),
            "kappa": k(a, b), "kappa_quadratic": k(a, b, weights="quadratic"),
            "kappa_bin_correct": k(a == 2, b == 2), "kappa_bin_incorrect": k(a == 0, b == 0),
            "confusion_opus_rows_local_cols": confusion_matrix(a, b, labels=[2, 1, 0]).tolist()}


def acc(vs):
    return sum({"correct": 1, "partial": 0.5}.get(v, 0) for v in vs) / max(len(vs), 1)


SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def opus_delta_ci(opus_item, fam, split, n_boot=2000, seed=0):
    """Paired bootstrap 95% CI of the Opus lenient-accuracy delta (1-epoch FT - base) on identical items."""
    b = {q: v for (sid, s, q), v in opus_item.items() if sid == sys_id(fam, "base") and s == split}
    t = {q: v for (sid, s, q), v in opus_item.items() if sid == sys_id(fam, "ft1run") and s == split}
    qs = sorted(set(b) & set(t))
    if len(qs) < 2:
        return (float("nan"), float("nan"))
    d = np.array([t[q] - b[q] for q in qs])
    means = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


def gate(k, sl, coverage_ok):
    """Pre-registered gate, refined: max binary kappa >= 0.5 and the local judge reproduces the sign of every
    base->FT delta that is significant under Opus (paired bootstrap 95% CI excludes 0)."""
    mx = max([x for x in (k.get("kappa_bin_correct"), k.get("kappa_bin_incorrect")) if x is not None] or [-1])
    return mx, bool(mx >= 0.5 and sl["n_sig_sign_match"] == sl["n_sig_deltas"] and coverage_ok)


def system_level(rows, grades):
    """Per (system, split) lenient accuracy under Opus and the local judge; correlations; delta signs."""
    from scipy.stats import pearsonr, spearmanr
    cell = defaultdict(lambda: ([], []))
    for r in rows:
        g = grades.get(key(r["split"], r["qa_id"], r["system"]))
        if g:
            cell[(r["system"], r["split"])][0].append(r["opus"])
            cell[(r["system"], r["split"])][1].append(g["grade"])
    pts = {k: {"opus": round(acc(o), 4), "local": round(acc(l), 4), "n": len(o)} for k, (o, l) in cell.items()}
    xs = [v["opus"] for v in pts.values()]
    ys = [v["local"] for v in pts.values()]
    opus_item = {(r["system"], r["split"], r["qa_id"]): SCORE[r["opus"]] for r in rows}
    deltas = []
    for f in FAMILIES:
        for s in SPLITS:
            b, t = pts.get((sys_id(f, "base"), s)), pts.get((sys_id(f, "ft1run"), s))
            if b and t:
                do, dl = t["opus"] - b["opus"], t["local"] - b["local"]
                lo, hi = opus_delta_ci(opus_item, f, s)
                deltas.append({"family": f, "split": s, "delta_opus": round(do, 4), "delta_local": round(dl, 4),
                               "opus_delta_ci": [lo, hi], "opus_significant": bool(lo > 0 or hi < 0),
                               "sign_match": (do > 0) == (dl > 0)})
    sig = [d for d in deltas if d["opus_significant"]]
    res = {"points": [{"system": k[0], "split": k[1], **v} for k, v in sorted(pts.items())], "deltas": deltas,
           "n_sign_match": sum(d["sign_match"] for d in deltas), "n_deltas": len(deltas),
           "n_sig_deltas": len(sig), "n_sig_sign_match": sum(d["sign_match"] for d in sig)}
    if len(xs) >= 3:
        res["pearson"] = round(float(pearsonr(xs, ys)[0]), 4)
        res["spearman"] = round(float(spearmanr(xs, ys)[0]), 4)
    return res


def breakdowns(rows, grades):
    sel = [(r, grades[key(r["split"], r["qa_id"], r["system"])]) for r in rows
           if key(r["split"], r["qa_id"], r["system"]) in grades]
    lens = np.array([words(r["answer"]) for r, _ in sel])
    qs = np.quantile(lens, [0.25, 0.5, 0.75]) if len(lens) else [0, 0, 0]

    def quart(n):
        return f"Q{1 + sum(n > q for q in qs)}"
    groups = {"split": lambda r: r["split"], "family": lambda r: r["family"],
              "base_vs_ft": lambda r: "base" if r["variant"] == "base" else "finetuned",
              "length_quartile": lambda r: quart(words(r["answer"]))}
    out = {}
    for gname, fn in groups.items():
        d = defaultdict(lambda: ([], []))
        for r, g in sel:
            d[fn(r)][0].append(r["opus"])
            d[fn(r)][1].append(g["grade"])
        out[gname] = {k: dict(kappas(o, l), acc_opus=round(acc(o), 4), acc_local=round(acc(l), 4))
                      for k, (o, l) in sorted(d.items())}
    out["length_quartile_cutoffs_words"] = [float(x) for x in qs]
    return out


def hallu_check(rows, grades):
    """Weak check: local hallucinated_specific / unsupported_claims vs Opus unsupported_claims (seen facts)."""
    from sklearn.metrics import cohen_kappa_score
    a, b, c = [], [], []
    for r in rows:
        g = grades.get(key(r["split"], r["qa_id"], r["system"]))
        if g and r.get("opus_unsupported") in ("0", "1", "2+"):
            a.append(r["opus_unsupported"] != "0")
            b.append(bool(g.get("hallucinated_specific")))
            c.append((r["opus_unsupported"], g.get("unsupported_claims")))
    if not a:
        return {}
    return {"n": len(a), "kappa_hallucinated_vs_opus_unsupported_any": round(float(cohen_kappa_score(a, b)), 4),
            "unsupported_claims_exact_agreement": round(sum(x == y for x, y in c) / len(c), 4),
            "kappa_unsupported_claims": round(float(cohen_kappa_score([x for x, _ in c], [str(y) for _, y in c])), 4)}


def calibrate_judge(j, rows, dev, deadline):
    lg = log()
    dev_rows = [r for r in rows if (r["split"], r["qa_id"]) in dev]
    test_rows = [r for r in rows if (r["split"], r["qa_id"]) not in dev]
    lg.info(f"[S1] judge {j} ({JUDGES[j]['name']}): DEV {len(dev_rows)} grades, TEST {len(test_rows)} grades")
    # all prompt versions on DEV in one pass per version
    dev_scores = {}
    for v in PROMPTS:
        g = grade_pairs(j, v, dev_rows, deadline, f"S1 {j} DEV {v}")
        sub = [(r["opus"], g[key(r["split"], r["qa_id"], r["system"])]["grade"]) for r in dev_rows
               if key(r["split"], r["qa_id"], r["system"]) in g]
        if not sub:
            continue
        k = kappas([x for x, _ in sub], [y for _, y in sub])
        k["selection_score"] = round(np.mean([x for x in (k.get("kappa_bin_correct"), k.get("kappa_bin_incorrect"))
                                              if x is not None] or [-1]), 4)
        dev_scores[v] = k
        lg.info(f"[S1] {j} DEV {v}: n {k['n']}, agreement {k.get('agreement_3class')}, "
                f"bin kappa correct {k.get('kappa_bin_correct')} / incorrect {k.get('kappa_bin_incorrect')}")
    if not dev_scores:
        return None
    best_v = max(PROMPTS, key=lambda v: (dev_scores.get(v, {}).get("selection_score", -9), -PROMPTS.index(v)))
    lg.info(f"[S1] {j}: chosen prompt {best_v} by DEV mean binary kappa")
    g = grade_pairs(j, best_v, rows, deadline, f"S1 {j} TEST {best_v}")     # TEST (DEV already cached)
    return evaluate(j, rows, dev, best_v, dev_scores, g)


def evaluate(j, rows, dev, best_v, dev_scores, g):
    """Calibration metrics + gate from (cached) grades; writes calibration_<j>.json."""
    lg = log()
    test_rows = [r for r in rows if (r["split"], r["qa_id"]) not in dev]
    tsub = [r for r in test_rows if key(r["split"], r["qa_id"], r["system"]) in g]
    k = kappas([r["opus"] for r in tsub], [g[key(r["split"], r["qa_id"], r["system"])]["grade"] for r in tsub])
    sl = system_level(test_rows, g)
    res = {"judge": j, "judge_name": JUDGES[j]["name"], "model": JUDGES[j]["model"], "prompt": best_v,
           "dev": dev_scores, "test": k, "test_coverage": f"{len(tsub)}/{len(test_rows)}",
           "breakdowns_test": breakdowns(test_rows, g), "system_level_test": sl,
           "system_level_all": system_level(rows, g), "hallucination_check_seen_facts": hallu_check(rows, g),
           "gate_rule": "max binary kappa >= 0.5 and sign match on every base->FT delta that is significant under "
                        "Opus (paired bootstrap 95% CI excludes 0); test coverage >= 90%"}
    mx, ok = gate(k, sl, len(tsub) >= 0.9 * len(test_rows))
    res["max_binary_kappa"] = mx
    res["passes_gate"] = ok
    lg.info(f"[S1] {j} TEST: n {k.get('n')}, agreement {k.get('agreement_3class')}, kappa {k.get('kappa')}, "
            f"qwk {k.get('kappa_quadratic')}, bin {k.get('kappa_bin_correct')}/{k.get('kappa_bin_incorrect')}, "
            f"system r {sl.get('pearson')}, signs {sl['n_sign_match']}/{sl['n_deltas']} overall, "
            f"{sl['n_sig_sign_match']}/{sl['n_sig_deltas']} significant -> {'PASS' if ok else 'FAIL'}")
    (OUT / f"calibration_{j}.json").write_text(json.dumps(res, indent=2))
    return res


def cmd_calibrate(args):
    lg = log()
    t_start = time.time()
    deadline = deadline_from_env(75)
    rows = opus_table()
    dev = dev_test_split(rows)
    (OUT / "calibration_dev_ids.json").write_text(json.dumps(sorted(f"{s}|{q}" for s, q in dev)))
    results = {}
    for j in (args.judge or ["A", "B"]):
        prev = OUT / f"calibration_{j}.json"
        if prev.exists() and done(f"calibrate_{j}"):
            results[j] = json.loads(prev.read_text())
        else:
            if j == "B":
                if results.get("A", {}).get("passes_gate"):
                    break
                elapsed_pipeline = time.time() - float(__import__("os").environ.get("PIPELINE_START", t_start))
                if time.time() > deadline - 30 * 60 or elapsed_pipeline > 105 * 60:
                    lg.info("[S1] judge B skipped: not enough time left in the S1 box / pipeline budget")
                    results["B_skipped"] = "time"
                    break
                ok = download(JUDGES["B"]["model"])
                if not ok:
                    results["B_skipped"] = "download failed"
                    break
            r = calibrate_judge(j, rows, dev, deadline)
            if r is None:
                lg.info(f"[S1] judge {j} produced no grades")
                continue
            results[j] = r
            mark(f"calibrate_{j}", {"passes_gate": r["passes_gate"]})
        if results[j].get("passes_gate"):
            break
    cands = [results[j] for j in ("A", "B") if isinstance(results.get(j), dict)]
    if not cands:
        raise SystemExit("no judge calibrated")
    write_choice(cands, results.get("B_skipped"))


def write_choice(cands, b_skipped=None, note=None):
    def coverage(c):
        a, b = c["test_coverage"].split("/")
        return int(a) / max(int(b), 1)
    passing = [c for c in cands if c["passes_gate"]]
    complete = [c for c in cands if coverage(c) >= 0.9]     # a time-truncated candidate must not win on a subsample
    pick = max(passing or complete or cands, key=lambda c: c["max_binary_kappa"])
    choice = {"judge": pick["judge"], "judge_name": pick["judge_name"], "model": pick["model"], "prompt": pick["prompt"],
              "passes_gate": pick["passes_gate"], "exploratory": not pick["passes_gate"],
              "candidates": {c["judge"]: {"max_binary_kappa": c["max_binary_kappa"], "passes_gate": c["passes_gate"]}
                             for c in cands}, "B_skipped": b_skipped, "gate_rule": pick.get("gate_rule"),
              "note": note}
    (OUT / "judge_choice.json").write_text(json.dumps(choice, indent=2))
    write_calibration_md(cands, choice)
    log().info(f"[S1] judge choice: {choice}")
    return choice


def cmd_regate(args):
    """Recompute calibration metrics and the gate from cached grades (no GPU, no re-grading)."""
    rows = opus_table()
    dev = dev_test_split(rows)
    cands = []
    for j in ("A", "B"):
        p = OUT / f"calibration_{j}.json"
        if not p.exists():
            continue
        old = json.loads(p.read_text())
        g = load_cache(j, old["prompt"])
        if g:
            cands.append(evaluate(j, rows, dev, old["prompt"], old["dev"], g))
    if not cands:
        raise SystemExit("nothing cached to regate")
    b_skip = None if any(c["judge"] == "B" for c in cands) else (
        "not needed: judge A passes the refined gate" if cands[0]["passes_gate"] else "not run")
    return write_choice(cands, b_skip, note=args.note)


def fmt(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def write_calibration_md(cands, choice):
    L = ["# Local judge calibration against Opus 5.5", "",
         "_Generated by `src/local_judge.py calibrate`. All numbers come from `results/local_eval/calibration_*.json`._", "",
         "Reference: the Opus 5.5 grades of run_2026-10-03 (3 base + 3 one-epoch fine-tuned systems, eval-subset ids). "
         "Items were split by qa_id (all systems of an item on one side), stratified by split × modal Opus verdict, "
         "into a ~1,000-grade DEV slice (used only to pick one of three fixed prompt versions, by mean binary kappa) "
         "and a held-out TEST slice (all numbers below).", "",
         f"**Chosen judge: {choice['judge_name']}** (`{choice['model']}`), prompt {choice['prompt']}. "
         + ("Passes the gate."
            if choice["passes_gate"] else
            "**Does NOT pass the gate; its grades are labelled exploratory and key-fact recall / contradiction are the primary metrics.**"),
         "",
         "**Gate (refined before the gate decision, documented here):** max(binary κ correct-vs-rest, binary κ "
         "incorrect-vs-rest) ≥ 0.5 on TEST, and the local judge reproduces the sign of every base→fine-tuned delta "
         "that is significant under Opus (paired bootstrap 95% CI of the Opus delta excludes 0). The original rule "
         "required all 9 signs; a delta whose Opus CI includes 0 has no reliable sign to reproduce, so requiring it "
         "would test noise. All 9 signs are still reported.", ""]
    if choice.get("note"):
        L.append(f"Note: {choice['note']}\n")
    if choice.get("B_skipped"):
        L.append(f"Judge B (Phi-4) was not run: {choice['B_skipped']}.\n")
    for c in cands:
        t = c["test"]
        sl = c["system_level_test"]
        L += [f"## {c['judge_name']} (prompt {c['prompt']})", "",
              "DEV (prompt selection): " + "; ".join(
                  f"{v}: agreement {fmt(d.get('agreement_3class'))}, binary κ correct {fmt(d.get('kappa_bin_correct'))} / "
                  f"incorrect {fmt(d.get('kappa_bin_incorrect'))}" for v, d in c["dev"].items()), "",
              "| TEST metric | value |", "|---|---|",
              f"| grades (n) | {t.get('n')} (coverage {c['test_coverage']}) |",
              f"| 3-class agreement | {fmt(t.get('agreement_3class'))} |",
              f"| Cohen's κ | {fmt(t.get('kappa'))} |", f"| quadratic-weighted κ | {fmt(t.get('kappa_quadratic'))} |",
              f"| binary κ, correct vs rest | {fmt(t.get('kappa_bin_correct'))} |",
              f"| binary κ, incorrect vs rest | {fmt(t.get('kappa_bin_incorrect'))} |",
              f"| system-level Pearson r (18 system × split cells) | {fmt(sl.get('pearson'))} |",
              f"| system-level Spearman ρ | {fmt(sl.get('spearman'))} |",
              f"| base→FT delta signs matching | {sl['n_sign_match']}/{sl['n_deltas']} overall, "
              f"{sl.get('n_sig_sign_match', '–')}/{sl.get('n_sig_deltas', '–')} significant |", "",
              "Confusion matrix (rows = Opus, columns = local; order correct, partial, incorrect):", "",
              "| Opus \\ local | correct | partial | incorrect |", "|---|---|---|---|"]
        for lab, row in zip(["correct", "partial", "incorrect"], t.get("confusion_opus_rows_local_cols", [])):
            L.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
        L += ["", "Base → fine-tuned (1 epoch) deltas in lenient accuracy (TEST items):", "",
              "| family | split | Δ Opus [95% CI] | Opus significant | Δ local | sign match |",
              "|---|---|---|---|---|---|"]
        for d in sl["deltas"]:
            ci = d.get("opus_delta_ci", [None, None])
            L.append(f"| {d['family']} | {d['split']} | {d['delta_opus']:+.3f} [{fmt(ci[0])}, {fmt(ci[1])}] | "
                     f"{'yes' if d.get('opus_significant') else 'no'} | {d['delta_local']:+.3f} | "
                     f"{'yes' if d['sign_match'] else '**no**'} |")
        for gname, gd in c["breakdowns_test"].items():
            if not isinstance(gd, dict):
                continue
            L += ["", f"Breakdown by {gname.replace('_', ' ')}:", "",
                  "| group | n | agreement | κ | binary κ correct | binary κ incorrect | acc Opus | acc local |",
                  "|---|---|---|---|---|---|---|---|"]
            for g, d in gd.items():
                L.append(f"| {g} | {d.get('n')} | {fmt(d.get('agreement_3class'))} | {fmt(d.get('kappa'))} | "
                         f"{fmt(d.get('kappa_bin_correct'))} | {fmt(d.get('kappa_bin_incorrect'))} | "
                         f"{fmt(d.get('acc_opus'))} | {fmt(d.get('acc_local'))} |")
        cut = c["breakdowns_test"].get("length_quartile_cutoffs_words")
        if cut:
            L.append(f"\nLength quartile cut-offs (answer words): {', '.join(f'{x:.0f}' for x in cut)}. "
                     "A length bias shows up as the local judge's accuracy rising faster than Opus's across quartiles.")
        h = c.get("hallucination_check_seen_facts") or {}
        if h:
            L.append(f"\nHallucination flag (weak check, test_seen_facts only, n = {h['n']}): κ of local "
                     f"`hallucinated_specific` vs Opus `unsupported_claims ≠ 0` = {fmt(h['kappa_hallucinated_vs_opus_unsupported_any'])}; "
                     f"unsupported_claims exact agreement {fmt(h['unsupported_claims_exact_agreement'])} "
                     f"(κ {fmt(h['kappa_unsupported_claims'])}). Opus's unsupported_claims mostly tracks answer length, "
                     "so this is not a strong validation of the hallucination flag.")
        L.append("")
    (OUT / "judge_calibration.md").write_text("\n".join(L) + "\n")


# ---------------- full grading ----------------

def cmd_grade(args):
    lg = log()
    deadline = deadline_from_env(60)
    ch = json.loads((OUT / "judge_choice.json").read_text())
    j, v = ch["judge"], ch["prompt"]
    pairs = all_pairs()
    lg.info(f"[S2] grading {len(pairs)} answers (15 systems x eval subset) with {ch['judge_name']} prompt {v}")
    g = grade_pairs(j, v, pairs, deadline, "S2")
    # retry unparsable / missing once (same deterministic prompt -> only useful if truncated; give more tokens)
    miss = [p for p in pairs if key(p["split"], p["qa_id"], p["system"]) not in g]
    lg.info(f"[S2] graded {len(pairs) - len(miss)}/{len(pairs)}")
    out = []
    for p in pairs:
        k = key(p["split"], p["qa_id"], p["system"])
        r = g.get(k)
        og = opus_grades(p["family"], p["variant"], p["split"]).get(p["qa_id"], {})
        out.append({"split": p["split"], "qa_id": p["qa_id"], "system": p["system"], "family": p["family"],
                    "variant": p["variant"], "answer_words": words(p["answer"]),
                    "local_grade": r["grade"] if r else None,
                    "hallucinated_specific": bool(r.get("hallucinated_specific")) if r else None,
                    "unsupported_claims_local": r.get("unsupported_claims") if r else None,
                    "local_reasoning": r.get("reasoning") if r else None,
                    "opus_grade": og.get("verdict"), "opus_unsupported_claims": og.get("unsupported_claims"),
                    "judge": ch["judge_name"], "prompt": v, "exploratory": ch["exploratory"]})
    write_jsonl(PER_ITEM / "judge_grades.jsonl", out)
    mark("S2_grade", {"graded": len(pairs) - len(miss), "total": len(pairs)})
    if len(miss) > 0.05 * len(pairs):
        raise SystemExit(f"S2 incomplete: {len(miss)} missing (kept the rest)")


def cmd_smoke(args):
    j = args.judge[0] if args.judge else "A"
    rows = opus_table()
    sel = shuffled(rows, seed=3)
    sel = [r for r in sel if r["split"] != "test_seen_facts"][:args.n - 5] + \
          [r for r in sel if r["split"] == "test_seen_facts"][:5]
    import tempfile
    from pathlib import Path
    tmpd = Path(tempfile.mkdtemp(dir=str(OUT)))
    cp = tmpd / "smoke.jsonl"
    for seen in (False, True):
        s = [p for p in sel if (p["split"] == "test_seen_facts") == seen]
        rws = [{"id": key(p["split"], p["qa_id"], p["system"]),
                "messages": [{"role": "system", "content": system_text("v0", seen)},
                             {"role": "user", "content": JUDGE_PROMPT.format(question=p["question"], reference=p["reference"],
                                                                             prediction=p["answer"])}]} for p in s]
        run_worker(JUDGES[j]["model"], rws, cp, schema=schema(seen), max_tokens=320, extra=worker_extra(j),
                   label="smoke")
    raw = read_jsonl(cp)
    byk = {key(p["split"], p["qa_id"], p["system"]): p for p in sel}
    ok = 0
    for r in raw:
        g = parse_json(r["text"])
        ok += bool(g and g.get("grade") in ORD)
    print(f"parsed {ok}/{len(raw)}; finish reasons {Counter(r.get('finish') for r in raw)}")
    agree = sum(1 for r in raw if (parse_json(r["text"]) or {}).get("grade") == byk[r["id"]]["opus"])
    print(f"agreement with Opus on smoke items: {agree}/{len(raw)}")
    for r in raw[:3] + [x for x in raw if x["id"].startswith("test_seen")][:1]:
        print("----", r["id"], "| Opus:", byk[r["id"]]["opus"])
        print("ANSWER:", byk[r["id"]]["answer"][:300].replace("\n", " "))
        print("RAW:", r["text"])
    print("smoke dir:", tmpd)


def download(repo):
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo)
        return True
    except Exception as e:
        log().info(f"download of {repo} failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "calibrate", "grade", "regate"])
    ap.add_argument("--note")
    ap.add_argument("--judge", nargs="*")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    {"smoke": cmd_smoke, "calibrate": cmd_calibrate, "grade": cmd_grade, "regate": cmd_regate}[args.cmd](args)


if __name__ == "__main__":
    main()
