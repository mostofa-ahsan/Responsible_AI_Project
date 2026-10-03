"""Full QA run over the Message Batches API (50% price), stage by stage.

For each stage (extract -> generate -> filter) the stage script is run in collect mode
(LLM_COLLECT): every uncached LLM request is recorded instead of sent. The recorded requests go
out as ONE batch; succeeded results are written into the normal disk cache and llm_usage.jsonl
(at batch prices), and the stage is run again. Dependent calls (the filter's judge -> repair and
paraphrase regeneration -> re-judge -> ...) therefore become one batch per round. When a collect
pass records nothing, the stage is run normally (all cache hits) to write its outputs.

Fallbacks: a batch not ended after llm.batch_timeout_min is cancelled; its unfinished requests,
and errored / expired / refused / schema-invalid results, are retried once through the standard
API (parallel workers, refusal fallbacks enabled). Billing or spend-limit errors stop the run
cleanly (exit code 3) with state saved; re-running resumes from the cache.

Usage:
    python src/full_run.py --run full_v1            # run (use tmux; logs to logs/full_run.log)
    python src/full_run.py --run full_v1 --status   # current stage, batch, items, spend, ETA
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

from llm import BatchRunner, BillingError, estimate_cost
from qa_common import read_jsonl, run_paths
from utils import get_logger, load_config, repo_path

STAGES = [("extract", "src/qa_extract.py"), ("generate", "src/qa_generate.py"), ("filter", "src/qa_filter.py")]
EXPECTED_ROUNDS = {"extract": 1, "generate": 1, "filter": 5}   # for the ETA
MAX_ROUNDS = 8


class Status:
    def __init__(self, cfg, run):
        self.path = repo_path(cfg["paths"]["logs"]) / f"full_run_{run}_status.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data.update(run=run, pid=os.getpid())
        self.data.setdefault("started", time.time())
        self.data.setdefault("round_minutes", [])

    def update(self, **kw):
        self.data.update(kw, updated=time.time())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def run_stage_script(script, run, collect_file, log_path, extra=(), offline=False):
    env = dict(os.environ)
    if collect_file:
        env["LLM_COLLECT"] = str(collect_file)
    else:
        env.pop("LLM_COLLECT", None)
    if offline:
        env["LLM_OFFLINE"] = "1"            # budget cap: uncached calls raise instead of calling the API
    else:
        env.pop("LLM_OFFLINE", None)
    cmd = [sys.executable, script, "--run", run, *extra]
    with open(log_path, "a", encoding="utf-8") as out:
        out.write(f"\n===== {datetime.now():%H:%M:%S} {' '.join(cmd)} collect={bool(collect_file)}\n")
        out.flush()
        res = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=repo_path("."))
    if res.returncode == 3:
        raise BillingError(f"{script} stopped on a billing error (see {log_path})")
    if res.returncode != 0:
        raise RuntimeError(f"{script} exited with {res.returncode} (see {log_path})")


def items_done(cfg, run):
    p = run_paths(cfg, run)
    sel = json.loads(p["chunks"].read_text()) if p["chunks"].exists() else {"chunk_ids": []}
    units = {u["chunk_id"] for u in read_jsonl(p["units"])}
    sp = p["chunks"].with_name(f"{run}_chunk_status.json")
    usable = sum(s["generate"] for s in json.loads(sp.read_text()).values()) if sp.exists() else None
    gen = {g["chunk_id"] for g in read_jsonl(p["generated"])}
    final = read_jsonl(p["final"]) if p["final"].exists() else []
    return {"chunks_selected": len(sel["chunk_ids"]), "chunks_extracted": len(units),
            "chunks_usable": usable, "chunks_generated": len(gen),
            "pairs_generated": sum(1 for _ in read_jsonl(p["generated"])),
            "pairs_final": len(final), "pairs_passed": sum(r["passed_filters"] for r in final)}


def spend(cfg, run):
    tot, batch = 0.0, 0.0
    path = repo_path(cfg["llm"]["usage_log"])
    if path.exists():
        for line in path.open(encoding="utf-8"):
            u = json.loads(line)
            if u["stage"].endswith(f":{run}") and not u["cached"]:
                tot += u["cost"]
                batch += u["cost"] if u.get("batch") else 0.0
    return round(tot, 2), round(batch, 2)


def cap_for(cfg, run):
    return (cfg.get("budget", {}).get("run_max_usd") or {}).get(run)


def log_decision(cfg, step, decision, why):
    path = repo_path(cfg["paths"]["logs"]) / "UNATTENDED_DECISIONS.md"
    if path.exists():
        with path.open("a", encoding="utf-8") as f:
            f.write(f"| {datetime.now():%H:%M} | {step} | {decision} | {why} |\n")


def eta(data):
    mins = data.get("round_minutes") or []
    per_round = sum(mins) / len(mins) if mins else 30.0
    names = [s for s, _ in STAGES]
    cur = data.get("stage")
    if data.get("state") == "done" or cur not in names:
        return None, per_round
    i = names.index(cur)
    done_rounds = max(0, data.get("round", 1) - 1)
    remaining = max(EXPECTED_ROUNDS[cur] - done_rounds, 1) + sum(EXPECTED_ROUNDS[s] for s in names[i + 1:])
    if data.get("batch_status") in ("in_progress", "canceling") and data.get("batch_elapsed_min"):
        remaining -= min(data["batch_elapsed_min"] / per_round, 0.9)
    return datetime.now() + timedelta(minutes=remaining * per_round), per_round


def print_status(cfg, run):
    st = Status.__new__(Status)
    st.path = repo_path(cfg["paths"]["logs"]) / f"full_run_{run}_status.json"
    if not st.path.exists():
        print(f"no status for run {run} yet")
        return
    d = json.loads(st.path.read_text())
    alive = False
    try:
        os.kill(d.get("pid", 0), 0)
        alive = True
    except OSError:
        pass
    total, batch = spend(cfg, run)
    when, per_round = eta(d)
    print(f"run {run}: state={d.get('state')} (process {'running' if alive else 'not running'}), "
          f"started {datetime.fromtimestamp(d['started']):%H:%M}, updated "
          f"{datetime.fromtimestamp(d.get('updated', d['started'])):%H:%M:%S}")
    print(f"stage: {d.get('stage')} round {d.get('round')}  ({d.get('stage_note', '')})")
    if d.get("batch_id"):
        print(f"batch: {d['batch_id']} {d.get('batch_status')} {d.get('request_counts')} "
              f"{d.get('batch_elapsed_min', 0)} min, {d.get('batch_requests')} requests")
    print(f"items: {json.dumps(items_done(cfg, run))}")
    cap = cap_for(cfg, run)
    print(f"spend so far: ${total:.2f} (batch ${batch:.2f}, standard ${total - batch:.2f})"
          + (f" | cap ${cap:.0f}: {total / cap:.0%} used, ${cap - total:.2f} left" if cap else ""))
    if d.get("budget_stop"):
        print(f"budget: {d['budget_stop']}")
    print(f"ETA: {when:%H:%M} (rough; ~{per_round:.0f} min per batch round)" if when else "ETA: -")
    if d.get("error"):
        print(f"error: {d['error']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="full_v1")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stages", default="extract,generate,filter")
    args = ap.parse_args()
    cfg = load_config()
    if args.status:
        print_status(cfg, args.run)
        return
    log = get_logger("full_run", cfg)
    lcfg = cfg["llm"]
    st = Status(cfg, args.run)
    st.update(state="running", error=None)
    runner = BatchRunner(cfg, log, status_cb=lambda **kw: st.update(**kw))
    work_dir = repo_path(lcfg["batch_dir"])
    stage_log = repo_path(cfg["paths"]["logs"]) / f"full_run_{args.run}_stages.log"
    wanted = args.stages.split(",")
    log.info(f"=== full run {args.run}: stages {wanted} ===")
    try:
        for stage, script in STAGES:
            if stage not in wanted:
                continue
            extra = ["--skip-dedup"] if stage == "filter" else []
            gave_up = set()
            offline = False
            prefix = f"{args.run}:{stage}"
            for bid in runner.pending(prefix):        # resume batches submitted before a crash
                log.info(f"resuming uncollected batch {bid}")
                st.update(stage=stage, stage_note=f"resuming batch {bid}", batch_id=bid)
                runner.poll(bid, interval_s=lcfg["batch_poll_s"], timeout_min=lcfg["batch_timeout_min"])
                ok, failed, billing = runner.collect(bid)
                log.info(f"resumed batch {bid}: {ok} succeeded, {len(failed)} to retry")
                if billing:
                    raise BillingError(f"billing error in batch {bid} results")
                if failed:
                    runner.retry_standard(failed, workers=lcfg["standard_workers"])
            for rnd in range(1, MAX_ROUNDS + 1):
                st.update(stage=stage, round=rnd, stage_note="collecting requests", batch_id=None,
                          batch_status=None, request_counts=None, batch_elapsed_min=None)
                collect = work_dir / f"{args.run}_{stage}_r{rnd}_requests.jsonl"
                collect.unlink(missing_ok=True)
                run_stage_script(script, args.run, collect, stage_log, extra)
                recs = {}
                for line in (collect.open(encoding="utf-8") if collect.exists() else []):
                    r = json.loads(line)
                    recs.setdefault(r["cache_key"], r)
                todo = [r for k, r in recs.items() if k not in gave_up]
                log.info(f"{stage} round {rnd}: {len(recs)} uncached requests ({len(recs) - len(todo)} given up)")
                if not todo:
                    break
                cap = cap_for(cfg, args.run)
                if cap:
                    spent, _ = spend(cfg, args.run)
                    est = estimate_cost(cfg, todo, batch=True)
                    log.info(f"{stage} round {rnd}: spend ${spent:.2f} + estimate ${est:.2f} vs cap ${cap:.0f}")
                    if spent + est > cap:
                        msg = (f"skipped {stage} round {rnd} ({len(todo)} requests, est ${est:.2f}): "
                               f"spend ${spent:.2f} + est > cap ${cap:.0f}; finalizing offline")
                        log.warning(msg)
                        st.update(budget_stop=msg)
                        log_decision(cfg, f"full_run {args.run}", msg,
                                     "budget guard: repairs/paraphrase fixes that would exceed the cap are skipped")
                        offline = True
                        break
                t0 = time.time()
                st.update(stage_note=f"batch of {len(todo)} requests", batch_requests=len(todo))
                bid = runner.submit_batch(todo, stage_prefix=prefix)
                st.update(batch_id=bid, batch_status="in_progress")
                _, timed_out = runner.poll(bid, interval_s=lcfg["batch_poll_s"], timeout_min=lcfg["batch_timeout_min"])
                ok, failed, billing = runner.collect(bid)
                log.info(f"{stage} round {rnd}: batch {bid} -> {ok} succeeded, {len(failed)} to retry"
                         + (" (timed out, cancelled)" if timed_out else ""))
                if billing:
                    raise BillingError(f"billing error in batch {bid} results")
                if failed:
                    st.update(stage_note=f"standard-API retry of {len(failed)} requests")
                    why = {}
                    for f in failed:
                        why[f["_why"].split(":")[0]] = why.get(f["_why"].split(":")[0], 0) + 1
                    log.info(f"retrying via standard API: {why}")
                    ok2, still = runner.retry_standard(failed, workers=lcfg["standard_workers"])
                    log.info(f"standard retry: {ok2} ok, {len(still)} still failing")
                    for s in still:
                        gave_up.add(s["cache_key"])
                        log.warning(f"giving up on {s['stage']} {s['cache_key'][:12]}: {s['_why'][:200]}")
                st.data["round_minutes"].append(round((time.time() - t0) / 60, 1))
                st.update()
            else:
                log.warning(f"{stage}: stopped after {MAX_ROUNDS} rounds")
            st.update(stage_note="writing outputs", batch_id=None)
            run_stage_script(script, args.run, None, stage_log, offline=offline)   # writes outputs
            log.info(f"{stage} done: {items_done(cfg, args.run)}")
        total, batch = spend(cfg, args.run)
        st.update(state="done", stage="done", stage_note=f"finished, spend ${total:.2f}")
        log.info(f"=== full run {args.run} finished: {items_done(cfg, args.run)}; spend ${total:.2f} "
                 f"(batch ${batch:.2f}) ===")
    except BillingError as e:
        st.update(state="stopped_billing", error=str(e)[:500])
        log.error(f"STOPPED on billing/spend limit: {e}. State saved; re-run the same command to resume.")
        sys.exit(3)
    except Exception as e:
        st.update(state="error", error=f"{type(e).__name__}: {e}"[:500])
        log.exception("full run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
