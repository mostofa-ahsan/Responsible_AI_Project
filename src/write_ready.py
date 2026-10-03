"""Write logs/READY_FOR_TRAINING.md from the run's artifacts (local files only, no API calls).

Sources: logs/after_filter_steps.tsv (step status), data/qa_pairs/<run>.jsonl (pair counts),
data/splits/split_manifest.json, logs/llm_usage.jsonl (spend), logs/smoke_test_summary.json,
eval/base_closedbook/*_summary.json, logs/UNATTENDED_DECISIONS.md.

Usage:
    python src/write_ready.py [--run full_v1]
"""

import argparse
import json
from collections import Counter
from datetime import datetime

from qa_common import read_jsonl, run_paths
from utils import load_config, repo_path


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing artifacts are reported as such
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="full_v1")
    args = ap.parse_args()
    cfg = load_config()
    logs = repo_path(cfg["paths"]["logs"])
    L = []

    L.append(f"# Ready for training: {args.run}")
    L.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M} by src/write_ready.py._\n")

    L.append("## Step status")
    steps = logs / "after_filter_steps.tsv"
    if steps.exists():
        L.append("| Time | Step | Status | Detail |\n|---|---|---|---|")
        for line in steps.read_text(encoding="utf-8").splitlines():
            t, name, status, detail = (line.split("\t") + ["", "", "", ""])[:4]
            L.append(f"| {t} | {name} | {status} | {detail} |")
    else:
        L.append("No step log found.")

    L.append("\n## Dataset")
    final = run_paths(cfg, args.run)["final"]
    rows = read_jsonl(final) if final.exists() else []
    passed = [r for r in rows if r["passed_filters"]]
    if rows:
        L.append(f"- Pairs generated: {len(rows):,}; **passed: {len(passed):,}** ({len(passed) / len(rows):.1%}); "
                 f"repaired share of passed: {sum(bool(r['repair']) for r in passed) / max(len(passed), 1):.1%}")
        L.append(f"- Training questions incl. paraphrases: {sum(1 + len(r['paraphrases']) for r in passed):,}")
        L.append(f"- q_type: {dict(Counter(r['q_type'] for r in passed))}")
        L.append(f"- difficulty: {dict(Counter(r['difficulty'] for r in passed))}")
    man = load(repo_path(cfg["paths"]["splits"]) / "split_manifest.json")
    if man:
        L.append(f"- Splits: {man['counts']}; held-out doc groups: {len(man['heldout_doc_groups'])}; "
                 f"checks: {', '.join(k + (' OK' if v else ' FAILED') for k, v in man.get('checks', {}).items())}")
    L.append(f"- Final report: `logs/{args.run}_final_report.txt`; split report: `logs/split_{args.run}_report.txt`")

    L.append("\n## Spend vs the $200 cap (full_v1 + evaluation)")
    usage = [json.loads(line) for line in (logs / "llm_usage.jsonl").open(encoding="utf-8")] \
        if (logs / "llm_usage.jsonl").exists() else []
    run_cost = sum(u["cost"] for u in usage if u["stage"].endswith(f":{args.run}") and not u["cached"])
    eval_cost = sum(u["cost"] for u in usage if u["stage"].startswith("eval_closedbook_judge") and not u["cached"])
    other = sum(u["cost"] for u in usage if not u["cached"]) - run_cost - eval_cost
    b = cfg.get("budget", {})
    cap = (b.get("run_max_usd") or {}).get(args.run)
    L.append(f"- {args.run}: **${run_cost:.2f}**" + (f" (cap ${cap})" if cap else ""))
    L.append(f"- Evaluation grading so far: **${eval_cost:.2f}** (cap ${b.get('eval_max_usd')}, "
             f"${b.get('eval_max_usd', 0) - eval_cost:.2f} left for the fine-tuned model)")
    L.append(f"- **Total against the $200 cap: ${run_cost + eval_cost:.2f}** "
             f"(${200 - run_cost - eval_cost:.2f} left)")
    L.append(f"- Outside the cap (pilots, tests, comparisons): ${other:.2f}")

    L.append("\n## Smoke test (30 steps, GPU)")
    sm = load(logs / "smoke_test_summary.json")
    if sm:
        L.append(f"- Loss: {sm['loss_trend']}")
        L.append(f"- Peak VRAM: {sm['peak_vram_gb']} GB; {sm['sec_per_step']} s/step; ~{sm['tokens_per_sec']} tokens/s")
        L.append(f"- Full training: {sm['full_steps_for_epochs']} steps ({sm['epochs']} epochs, "
                 f"{sm['train_examples']:,} examples) **~ {sm['projected_full_hours']} h**")
    else:
        L.append("- Not available (see step status).")

    L.append("\n## Base-model closed-book baseline (Qwen3-8B, thinking off)")
    any_eval = False
    for split in ("test_indomain", "test_heldout_docs"):
        s = load(repo_path(cfg["eval"]["output_dir"]) / "base_closedbook" / f"{split}_summary.json")
        if not s:
            L.append(f"- {split}: not available")
            continue
        any_eval = True
        o = s["overall"]

        def f(m):
            x = o[m]
            return f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"
        L.append(f"- **{split}** ({s['n_graded']}/{s['n_questions']} graded): judge accuracy {f('judge_score')}, "
                 f"token F1 {f('token_f1')}, ROUGE-L {f('rouge_l')}")
        L.append("  - by q_type: " + ", ".join(f"{t} {v['judge_score']['mean']:.2f}" for t, v in s["by_q_type"].items()))
    if any_eval:
        L.append("- Full tables with CIs by q_type and dimension: `logs/eval_closedbook_base_closedbook_<split>_report.txt`")

    L.append("\n## Decisions made while unattended")
    dec = logs / "UNATTENDED_DECISIONS.md"
    if dec.exists():
        body = [ln for ln in dec.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
        L += body if len(body) > 2 else ["None beyond the defaults."]

    L.append("\n## Commands for tonight")
    L.append("```bash\ncd ~/Responsible_AI_Project")
    L.append("# 1. launch full training (2 epochs; best checkpoint by val loss -> models/qwen3-8b-qlora/closedbook/final)")
    L.append("tmux new-session -d -s train '.venv/bin/python src/train_qlora.py --format closedbook; exec bash'")
    L.append("# 2. monitor (detach: Ctrl-b then d)")
    L.append("tmux attach -t train            # or: tail -f logs/train.log | grep -E 'step|eval|SMOKE|ETA'")
    L.append("# 3. after training: evaluate the fine-tuned model (vLLM + LoRA, Opus judge via batch)")
    for split in ("test_indomain", "test_heldout_docs"):
        L.append(f".venv/bin/python src/eval_closedbook.py --arm ft_closedbook --split {split} "
                 f"--adapter models/qwen3-8b-qlora/closedbook/final")
    L.append("```")
    out = logs / "READY_FOR_TRAINING.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
