"""S5 local checkers (and the S8 gate). No API calls; one GPU model at a time.

purge_judge   delete the local judge weights from the HF cache (all judge-model work is cached by then)
minicheck7b   download Bespoke-MiniCheck-7B and score support: premise = model answer, claim = reference key
              fact; P(Yes) via vLLM; supported if p >= 0.5. On failure, Flan-T5-Large becomes primary.
flant5        MiniCheck-Flan-T5-Large support scores (secondary checker; answers > 400 words are split into
              400-word windows, max over windows, as in MiniCheck)
nli           DeBERTa-v3-large NLI (MoritzLaurer ...-ling-wanli):
              (a) contradiction of reference facts: premise = answer (sentence windows if > 450 tokens; max
                  entailment / max contradiction over windows), hypothesis = fact
              (b) Phase 2c: premise = reference + evidence (windows), hypothesis = each answer claim
rag_gate      S8: checks the time / disk conditions for the optional RAG baseline and records the decision

Outputs: results/local_eval/per_item/{support_minicheck7b,support_flant5,nli_facts,nli_claims}.jsonl
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from local_common import (JUDGES, KEYFACTS, OUT, PER_ITEM, SPLITS, deadline_from_env, items, log, pause_wait, preds,
                          read_jsonl, run_worker, sys_id, systems, write_jsonl)

DEV = __import__("os").environ.get("LOCALEVAL_DEVICE", "cuda")
MC7B = "bespokelabs/Bespoke-MiniCheck-7B"
FLAN = "lytang/MiniCheck-Flan-T5-Large"
NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# prompt of the MiniCheck package for Bespoke-MiniCheck-7B
MC_SYSTEM = ("Determine whether the provided claim is consistent with the corresponding document. Consistency in "
             "this context implies that all information presented in the claim is substantiated by the document. "
             "If not, it should be considered inconsistent. Please assess the claim's consistency with the document "
             "by responding with either \"Yes\" or \"No\".")
MC_USER = "Document: {doc}\nClaim: {claim}"


def fallback(stage, what, why):
    p = OUT / "fallbacks.json"
    d = json.loads(p.read_text()) if p.exists() else []
    d.append({"time": time.strftime("%F %T"), "stage": stage, "fallback": what, "why": why})
    p.write_text(json.dumps(d, indent=2))
    log().info(f"[{stage}] FALLBACK: {what} ({why})")


def free_gb():
    return shutil.disk_usage(str(OUT)).free / 1e9


def fact_pairs():
    """[(id, split, qa_id, system, k, answer, fact)] for every system x eval-subset item with key facts."""
    its = items()
    kf = {s: {r["qa_id"]: r["facts"] for r in read_jsonl(KEYFACTS / f"{s}.jsonl")} for s in SPLITS}
    out = []
    for fam, var in systems():
        sid = sys_id(fam, var)
        for s in SPLITS:
            p = preds(fam, var, s)
            for q in its[s]:
                if q in p and q in kf[s]:
                    for k, f in enumerate(kf[s][q]):
                        out.append((f"{s}|{q}|{sid}|{k}", s, q, sid, k, p[q], f))
    return out


def cmd_purge_judge(args):
    from huggingface_hub import scan_cache_dir
    lg = log()
    before = free_gb()
    info = scan_cache_dir()
    targets = {JUDGES[j]["model"] for j in JUDGES}
    revs = [rev.commit_hash for repo in info.repos if repo.repo_id in targets for rev in repo.revisions]
    if revs:
        strat = info.delete_revisions(*revs)
        lg.info(f"[S5a] deleting judge weights {sorted(targets)}: frees {strat.expected_freed_size_str}")
        strat.execute()
    lg.info(f"[S5a] free disk {before:.1f} -> {free_gb():.1f} GB")


def minicheck_tokenizer():
    """The checkpoint's remote-code InternLM2 tokenizer is broken under transformers 5.18 (character-level,
    spaces dropped). Its tokenizer.json loaded as a plain fast tokenizer is correct, so build a tokenizer
    directory without the remote code and hand it to vLLM."""
    from huggingface_hub import snapshot_download
    from utils import repo_path
    src = Path(snapshot_download(MC7B, allow_patterns=["*.json", "*.py", "*.model"]))
    out = repo_path("models/minicheck7b_tokenizer")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "tokenizer.json", out / "tokenizer.json")
    tc = json.loads((src / "tokenizer_config.json").read_text())
    tc.pop("auto_map", None)
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=1))
    if (src / "special_tokens_map.json").exists():
        shutil.copy(src / "special_tokens_map.json", out / "special_tokens_map.json")
    return out


def minicheck_local():
    """True if the MiniCheck-7B weights are already in the HF cache (no download, no disk headroom needed)."""
    from huggingface_hub import snapshot_download
    try:
        p = Path(snapshot_download(MC7B, allow_patterns=["*.safetensors", "*.json"], local_files_only=True))
        return len(list(p.glob("*.safetensors"))) >= 4
    except Exception:
        return False


class Appender:
    """Append JSONL rows and fsync at most every `every` seconds, so a crash loses about a minute of work."""
    def __init__(self, path, every=60):
        self.path, self.every, self.buf, self.t = Path(path), every, [], time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def add(self, row):
        self.buf.append(row)
        if time.time() - self.t > self.every:
            self.flush()

    def flush(self):
        if self.buf:
            with self.path.open("a", encoding="utf-8") as f:
                for r in self.buf:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                __import__("os").fsync(f.fileno())
            self.n += len(self.buf)
            self.buf = []
        self.t = time.time()


def done_keys(path, fields):
    return {tuple(r[f] for f in fields) for r in read_jsonl(path)}


def score_minicheck7b(pairs, raw_cache, out_path, deadline, label):
    """pairs: [(id, split, qa_id, system, k, answer, fact)]. Raw P(Yes) is appended per worker chunk to raw_cache
    (resumable); out_path is rebuilt from it. Returns the number of scored pairs."""
    tok_dir = minicheck_tokenizer()
    pairs = sorted(pairs, key=lambda x: (x[3], x[1], x[2], x[4]))      # same answer adjacent -> prefix-cache reuse
    rows = [{"id": pid, "messages": [{"role": "system", "content": MC_SYSTEM},
                                     {"role": "user", "content": MC_USER.format(doc=ans, claim=f)}]}
            for pid, s, q, sid, k, ans, f in pairs]
    run_worker(MC7B, rows, raw_cache, mode="yesno", deadline=deadline, label=label,
               extra=["--trust-remote-code", "--max-num-seqs", "256", "--tokenizer", str(tok_dir)], gpu_util=0.88)
    raw = {r["id"]: r.get("p_yes") for r in read_jsonl(raw_cache)}
    out = [{"split": s, "qa_id": q, "system": sid, "k": k, "p": raw[pid]} for pid, s, q, sid, k, _, _ in pairs
           if raw.get(pid) is not None]
    write_jsonl(out_path, out)
    log().info(f"[{label}] MiniCheck-7B scored {len(out)}/{len(pairs)} pairs "
               f"({sum(1 for v in raw.values() if v is None)} without Yes/No logprobs)")
    return len(out)


def cmd_minicheck7b(args):
    lg = log()
    deadline = deadline_from_env(50)
    if not minicheck_local():
        if free_gb() < 10 + 16:
            fallback("S5b", "Flan-T5-Large is the primary support checker",
                     f"only {free_gb():.1f} GB free for the 15.5 GB download")
            raise SystemExit(1)
        from huggingface_hub import snapshot_download
        t0 = time.time()
        try:
            snapshot_download(MC7B, allow_patterns=["*.json", "*.py", "*.model", "*.safetensors"])
        except Exception as e:
            fallback("S5b", "Flan-T5-Large is the primary support checker", f"download failed: {e}")
            raise SystemExit(1)
        lg.info(f"[S5b] MiniCheck-7B downloaded in {(time.time() - t0) / 60:.1f} min; free {free_gb():.1f} GB")
    pairs = fact_pairs()
    lg.info(f"[S5b] MiniCheck-7B: {len(pairs)} (answer, fact) pairs")
    n = score_minicheck7b(pairs, OUT / "judge_cache" / "minicheck7b_raw.jsonl", PER_ITEM / "support_minicheck7b.jsonl",
                          deadline, "S5b")
    if n < 0.9 * len(pairs):
        fallback("S5b", "Flan-T5-Large is the primary support checker", f"MiniCheck-7B covered only {n}/{len(pairs)} pairs")
        raise SystemExit(1)
    fp = OUT / "fallbacks.json"
    if fp.exists() and any(f["stage"] == "S5b" and not f["fallback"].startswith("reverted")
                           for f in json.loads(fp.read_text())):
        fallback("S5b", "reverted: MiniCheck-7B is the primary support checker again",
                 "rerun after the post-crash relaunch with the vLLM InternLM2 fix (src/vllm_shims/sitecustomize.py)")


def score_flant5(pairs, out_path, deadline, tag, model=None):
    """MiniCheck-Flan-T5-Large P(supported); resumable and appended incrementally (only complete pairs written)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    lg = log()
    have = done_keys(out_path, ("split", "qa_id", "system", "k"))
    todo = [x for x in pairs if (x[1], x[2], x[3], x[4]) not in have]
    lg.info(f"[{tag}] Flan-T5: {len(todo)} of {len(pairs)} pairs left to score")
    if not todo:
        return True
    tok = AutoTokenizer.from_pretrained(FLAN)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN, dtype=torch.bfloat16).to(DEV).eval()
    units, expected = [], {}
    for i, (pid, s, q, sid, k, ans, f) in enumerate(todo):
        w = ans.split()
        # MiniCheck: documents longer than ~400 words are chunked; the claim's score is the max over chunks
        chunks = [" ".join(w[j:j + 400]) for j in range(0, max(len(w), 1), 350)] if len(w) > 400 else [ans]
        expected[i] = len(chunks)
        units += [(i, f"premise: {c} hypothesis: {f}") for c in chunks]
    units = block_sort(units, key=lambda u: len(u[1]))
    best, cnt = {}, {}
    app = Appender(out_path)
    bs, t0 = 32, time.time()
    with torch.no_grad():
        for n_b, b in enumerate(range(0, len(units), bs)):
            if time.time() > deadline:
                lg.info(f"[{tag}] deadline: {b}/{len(units)} units scored")
                app.flush()
                return False
            if n_b % 50 == 0:
                pause_wait(tag)
            part = units[b:b + bs]
            enc = tok([u[1] for u in part], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEV)
            dec = torch.zeros((len(part), 1), dtype=torch.long, device=DEV)
            logits = model(**enc, decoder_input_ids=dec).logits[:, 0, :]
            probs = torch.softmax(logits[:, [3, 209]].float(), dim=-1)[:, 1].tolist()    # MiniCheck: id 209 = "1"
            for (i, _), p in zip(part, probs):
                best[i] = max(best.get(i, 0.0), p)
                cnt[i] = cnt.get(i, 0) + 1
                if cnt[i] == expected[i]:
                    _, s, q, sid, k, _, _ = todo[i]
                    app.add({"split": s, "qa_id": q, "system": sid, "k": k, "p": round(best[i], 5)})
            if n_b % 300 == 0:
                lg.info(f"[{tag}] Flan-T5 {b}/{len(units)} units ({time.time() - t0:.0f}s)")
    app.flush()
    lg.info(f"[{tag}] Flan-T5 done: {app.n} new pairs in {(time.time() - t0) / 60:.1f} min")
    return True


def cmd_flant5(args):
    score_flant5(fact_pairs(), PER_ITEM / "support_flant5.jsonl", deadline_from_env(15), "S5c")


def block_sort(units, key, block=4096, seed=0):
    """Shuffle, then sort by length only within blocks: padding stays efficient, and a time-boxed stop leaves a
    random subset unscored instead of the longest answers (which are mostly base-model answers)."""
    import random
    units = list(units)
    random.Random(seed).shuffle(units)
    return [u for b in range(0, len(units), block) for u in sorted(units[b:b + block], key=key)]


SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def windows(tok, text, max_tokens=450):
    """Sentence windows of <= max_tokens (deberta tokens), one-sentence overlap; one window if short."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return [text]
    sents = [s for s in SENT.split(text) if s.strip()]
    lens = [len(tok(s, add_special_tokens=False)["input_ids"]) for s in sents]
    out, cur, cl = [], [], 0
    for s, l in zip(sents, lens):
        if cur and cl + l > max_tokens:
            out.append(" ".join(cur))
            cur, cl = [cur[-1]], lens[sents.index(cur[-1])] if cur else 0
        cur.append(s)
        cl += l
    if cur:
        out.append(" ".join(cur))
    return out


class NLIScorer:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(NLI)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI, dtype=torch.bfloat16).to(DEV).eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ie, self.ic = lab["entailment"], lab["contradiction"]

    def score(self, units, expected, emit, deadline, tag, bs=64):
        """units: [(key, premise, hypothesis)]. emit(key, max_entail, max_contra) is called once every window of a
        key has been scored. Returns False if the deadline stopped it."""
        torch = self.torch
        units = block_sort(units, key=lambda u: len(u[1]) + len(u[2]))
        res = {}
        t0 = time.time()
        with torch.no_grad():
            for n_b, b in enumerate(range(0, len(units), bs)):
                if time.time() > deadline:
                    log().info(f"[{tag}] deadline: {b}/{len(units)} NLI units scored")
                    return False
                if n_b % 50 == 0:
                    pause_wait(tag)
                part = units[b:b + bs]
                enc = self.tok([u[1] for u in part], [u[2] for u in part], return_tensors="pt", padding=True,
                               truncation="only_first", max_length=512).to(DEV)
                pr = torch.softmax(self.model(**enc).logits.float(), dim=-1).tolist()
                for (k, _, _), p in zip(part, pr):
                    e, c, n = res.get(k, (0.0, 0.0, 0))
                    res[k] = (max(e, p[self.ie]), max(c, p[self.ic]), n + 1)
                    if res[k][2] == expected[k]:
                        emit(k, res[k][0], res[k][1])
                if n_b % 300 == 0:
                    log().info(f"[{tag}] {b}/{len(units)} units ({time.time() - t0:.0f}s)")
        return True


def score_nli_facts(nli, pairs, out_path, deadline, tag):
    """Contradiction of reference facts: premise = answer (sentence windows), hypothesis = fact. Resumable."""
    have = done_keys(out_path, ("split", "qa_id", "system", "k"))
    todo = [x for x in pairs if (x[1], x[2], x[3], x[4]) not in have]
    win_cache, units, expected, meta = {}, [], {}, {}
    for pid, s, q, sid, k, ans, f in todo:
        if ans not in win_cache:
            win_cache[ans] = windows(nli.tok, ans)
        units += [(pid, w, f) for w in win_cache[ans]]
        expected[pid] = len(win_cache[ans])
        meta[pid] = {"split": s, "qa_id": q, "system": sid, "k": k}
    log().info(f"[{tag}] NLI facts: {len(todo)} of {len(pairs)} pairs left, {len(units)} window units")
    app = Appender(out_path)

    def emit(key, e, c):
        app.add(dict(meta[key], p_entail=round(e, 5), p_contra=round(c, 5), contradicted=bool(c >= 0.5 and c > e)))
    complete = nli.score(units, expected, emit, deadline, tag)
    app.flush()
    return complete


def cmd_nli(args):
    lg = log()
    deadline = deadline_from_env(20)
    nli = NLIScorer()
    # (a) reference facts vs answer; keep part of the box for (b)
    score_nli_facts(nli, fact_pairs(), PER_ITEM / "nli_facts.jsonl", time.time() + (deadline - time.time()) * 0.6, "S5d")
    # (b) Phase 2c: answer claims vs reference + evidence (an answer counts in S6 only when all its claims are scored)
    outp = PER_ITEM / "nli_claims.jsonl"
    have = done_keys(outp, ("split", "qa_id", "system", "i"))
    its = items()
    prem_cache, units, expected, meta = {}, [], {}, {}
    for r in read_jsonl(PER_ITEM / "claims.jsonl"):
        it = its[r["split"]].get(r["qa_id"])
        if not it:
            continue
        pk = (r["split"], r["qa_id"])
        for i, c in enumerate(r["claims"]):
            if (r["split"], r["qa_id"], r["system"], i) in have:
                continue
            if pk not in prem_cache:
                ref, ev = it["answer"], it.get("evidence") or ""
                ref_len = len(nli.tok(ref, add_special_tokens=False)["input_ids"])
                prem_cache[pk] = [f"{ref} {w}" for w in windows(nli.tok, ev, max(120, 440 - ref_len))] if ev else [ref]
            key = f"{r['split']}|{r['qa_id']}|{r['system']}|{i}"
            meta[key] = {"split": r["split"], "qa_id": r["qa_id"], "system": r["system"], "i": i}
            units += [(key, p, c) for p in prem_cache[pk]]
            expected[key] = len(prem_cache[pk])
    lg.info(f"[S5d] NLI (b, Phase 2c): {len(meta)} claims left ({len(have)} done), {len(units)} window units")
    app = Appender(outp)

    def emit(key, e, c):
        app.add(dict(meta[key], p_entail=round(e, 5), p_contra=round(c, 5), contradicted=bool(c >= 0.5 and c > e),
                     supported=bool(e >= 0.5 and e >= c)))
    complete = nli.score(units, expected, emit, deadline, "S5d claims")
    app.flush()
    if not complete:
        fallback("S5d", "Phase 2c metrics on the completed subset", "time box hit; S6 counts only fully scored answers")


def cmd_rag_gate(args):
    import os
    start = float(os.environ.get("PIPELINE_START", time.time()))
    left = 6 * 60 - (time.time() - start) / 60
    free = free_gb()
    reasons = []
    if left < 75:
        reasons.append(f"only {left:.0f} min of the 6 h budget left (< 75)")
    if free < 15:
        reasons.append(f"only {free:.1f} GB free (< 15)")
    reasons.append("the judge weights were deleted after S4 by design, so RAG answers could not be graded with the "
                   "same calibrated judge in this run; re-downloading it (15 GB) plus the retrieval models does not "
                   "fit the disk reserve next to MiniCheck-7B")
    d = {"run": False, "minutes_left": round(left), "free_gb": round(free, 1), "reasons": reasons,
         "future_work": "closed-book vs RAG baseline (BM25 + dense + Qwen3-Reranker, recall@5 92%) graded by the "
                        "same local judge and checkers"}
    (OUT / "S8_rag.json").write_text(json.dumps(d, indent=2))
    log().info(f"[S8] optional RAG baseline skipped: {'; '.join(reasons)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["purge_judge", "minicheck7b", "flant5", "nli", "rag_gate"])
    args = ap.parse_args()
    {"purge_judge": cmd_purge_judge, "minicheck7b": cmd_minicheck7b, "flant5": cmd_flant5, "nli": cmd_nli,
     "rag_gate": cmd_rag_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
