"""test_seen_facts: can the models answer questions about facts that ARE in the training data?

build  300 TRAIN pairs, stratified by q_type x dimension (seed 42). For each, the generator writes ONE
       new paraphrase (different wording from the question and its two training paraphrases); the judge
       checks it asks the same thing; failures (and banned-phrase hits) are dropped. Writes
       data/splits/test_seen_facts.jsonl with the original reference answer. Calls go through the
       Message Batches path (cached).
grade  grades every model/arm's test_seen_facts predictions with the same judge rubric plus an
       unsupported_claims count (0 / 1 / 2+). This field is only added for this split: adding it to the
       existing splits would change the schema and invalidate their cached grades.

Usage:
    python src/seen_facts.py build
    python src/seen_facts.py grade
"""

import argparse
import json
import random
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from eval_closedbook import JUDGE_PROMPT, JUDGE_SYSTEM
from llm import LLM, batched_map
from qa_common import banned_regex, read_jsonl
from qa_generate import system_prompt
from utils import get_logger, load_config, repo_path

N_SEEN = 300
SEED = 42
RUN = "run_2026-10-03"
MODELS = ["Qwen/Qwen3-8B", "google/gemma-4-E4B-it", "meta-llama/Llama-3.1-8B-Instruct"]
ARMS = ["base", "finetuned"]

PARA_PROMPT = """Question: {question}

Existing phrasings (do NOT reuse their wording):
1. {p1}
2. {p2}

Write ONE new paraphrase of the question: same scope, same entities, same expected answer, standalone,
with wording clearly different from the question and both existing phrasings."""

CHECK_SYSTEM = """You check paraphrases for a QA dataset. A paraphrase is equivalent only if it asks for \
exactly the same information as the question (same scope, entities and expected answer) and is \
standalone (no "the text", "this study", "the authors", etc.)."""

CHECK_PROMPT = """Question: {question}

Candidate paraphrase: {para}"""

SEEN_JUDGE_SYSTEM = JUDGE_SYSTEM + """

unsupported_claims: count the factual claims in the model answer that are NOT supported by the
reference answer (a claim that contradicts it, or a specific detail it does not contain; generic
framing does not count): "0", "1", or "2+"."""


class NewParaphrase(BaseModel):
    paraphrase: str


class Check(BaseModel):
    equivalent: bool
    reason: str = Field(description="One short sentence")


class SeenGrade(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    rationale: str = Field(description="One sentence")
    unsupported_claims: Literal["0", "1", "2+"]


def stratified(rows, n, rng):
    strata = defaultdict(list)
    for r in rows:
        strata[(r["q_type"], r["dimension"])].append(r)
    raw = {k: n * len(v) / len(rows) for k, v in strata.items()}
    quota = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - quota[k], reverse=True)[:n - sum(quota.values())]:
        quota[k] += 1
    out = []
    for k in sorted(strata):
        out += rng.sample(strata[k], min(quota[k], len(strata[k])))
    return out


def build(cfg, log):
    out_path = repo_path(cfg["paths"]["splits"]) / "test_seen_facts.jsonl"
    rows = read_jsonl(repo_path(cfg["paths"]["splits"]) / "train.jsonl")
    sample = stratified(rows, N_SEEN, random.Random(SEED))
    stage = "seen_facts_build"
    llm = LLM(cfg, stage)
    banned = banned_regex(cfg)

    def para(r):
        p = (r["paraphrases"] + ["", ""])[:2]
        return llm.complete(PARA_PROMPT.format(question=r["question"], p1=p[0], p2=p[1]), system_prompt(cfg),
                            role="generator", json_schema=NewParaphrase).parsed.paraphrase.strip()
    paras = batched_map(cfg, log, stage, para, sample, workers=cfg["llm"]["standard_workers"])

    def check(item):
        r, p = item
        return llm.complete(CHECK_PROMPT.format(question=r["question"], para=p), CHECK_SYSTEM, role="judge",
                            json_schema=Check).parsed
    cands = [(r, paras[i]) for i, r in enumerate(sample) if isinstance(paras[i], str) and paras[i]]
    checks = batched_map(cfg, log, stage, check, cands, workers=cfg["llm"]["standard_workers"])
    kept, dropped = [], defaultdict(int)
    for i, (r, p) in enumerate(cands):
        c = checks[i]
        if isinstance(c, Exception):
            dropped["error"] += 1
        elif not c.equivalent:
            dropped["not_equivalent"] += 1
        elif banned.search(p):
            dropped["banned_phrase"] += 1
        elif p.strip().lower() in {r["question"].lower(), *[x.lower() for x in r["paraphrases"]]}:
            dropped["same_wording"] += 1
        else:
            kept.append({"qa_id": f"seen:{r['qa_id']}", "source_qa_id": r["qa_id"], "question": p,
                         "train_question": r["question"], "answer": r["answer"], "citation": r["citation"],
                         "doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "q_type": r["q_type"],
                         "dimension": r["dimension"], "difficulty": r["difficulty"], "split": "test_seen_facts"})
    with out_path.open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    log.info(f"test_seen_facts: {len(kept)} kept of {len(sample)} sampled; dropped {dict(dropped)} -> {out_path}")


def grade(cfg, log):
    rd = repo_path("results") / RUN
    refs = {r["qa_id"]: r for r in read_jsonl(repo_path(cfg["paths"]["splits"]) / "test_seen_facts.jsonl")}
    stage = f"eval_grade_seen:{RUN}"
    judge = LLM(cfg, stage)
    items = []
    for m in MODELS:
        ms = m.split("/")[-1].lower()
        for arm in ARMS:
            p = rd / ms / arm / "test_seen_facts_predictions.jsonl"
            if p.exists():
                for x in read_jsonl(p):
                    if x["qa_id"] in refs:
                        items.append((ms, arm, x["qa_id"], x["prediction"]))

    def g(item):
        _, _, q, pred = item
        r = refs[q]
        return judge.complete(JUDGE_PROMPT.format(question=r["question"], reference=r["answer"], prediction=pred),
                              SEEN_JUDGE_SYSTEM, role="judge", json_schema=SeenGrade).parsed
    res = batched_map(cfg, log, stage, g, items, workers=cfg["llm"]["standard_workers"])
    by = defaultdict(list)
    for i, (ms, arm, q, _) in enumerate(items):
        if not isinstance(res[i], Exception):
            by[(ms, arm)].append({"qa_id": q, "verdict": res[i].verdict, "rationale": res[i].rationale,
                                  "unsupported_claims": res[i].unsupported_claims})
    for (ms, arm), rows in by.items():
        with (rd / ms / arm / "test_seen_facts_grades.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"graded {sum(len(v) for v in by.values())} test_seen_facts answers for {len(by)} model/arm runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "grade"])
    args = ap.parse_args()
    cfg = load_config()
    log = get_logger("seen_facts", cfg)
    build(cfg, log) if args.cmd == "build" else grade(cfg, log)


if __name__ == "__main__":
    main()
