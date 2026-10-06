"""Build the mixed continued-pretraining corpus (raw text + QA) for arm C, per model tokenizer. CPU only, no API calls.

RAW  every chunk of the 80 TRAINING documents (doc_ids of data/splits/train.jsonl), in document order, with the
     chunk-overlap tokens removed (no duplicated text), documents concatenated with EOS between them and packed into
     1024-token sequences (BOS first if the tokenizer has one; no chat template); loss on all tokens. This includes the
     chunks behind test_indomain questions (by design). Held-out documents are NEVER included.
QA   train-split QA pairs in the previous training format (closed-book chat, answer-only loss). Per epoch: the original
     question + ONE paraphrase, rotating the paraphrase each epoch. Never a test item, a test_seen_facts paraphrase or a
     val item.
Each epoch = RAW + QA shuffled together (seed 42 + epoch), padded to a multiple of the effective batch (16) by
duplicating a few QA examples of that epoch (recorded), so epoch boundaries fall on optimizer steps.
Also: held-out-document perplexity set (200 packed sequences of the 14 held-out documents, never trained on) and the
val QA set (data/splits/val.jsonl, as before).

Leakage audit (fails the build): no held-out doc in RAW or QA; no test/val qa_id in QA; no test_seen_facts question
text among QA questions. Reports the word 13-gram overlap between RAW and each test split's questions.

Output: data/cpt/<model>/{epoch1,epoch2,epoch3,heldout_ppl,val_qa}.npz, data/cpt/stats.json, data/cpt/leakage_audit.json
    python src/build_cpt_mix.py [--models qwen3-8b ...] [--smoke]
"""

import argparse
import json
import random
import re
from collections import defaultdict

import numpy as np

from train_qlora import examples, tokenize
from utils import load_config, repo_path

MODEL_ID = {"llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct", "qwen3-8b": "Qwen/Qwen3-8B",
            "gemma-4-e4b-it": "google/gemma-4-E4B-it"}
SEQ = 1024
EFF_BATCH = 16
EPOCHS = 3
OUT = repo_path("data/cpt")


def rd(p):
    return [json.loads(l) for l in open(repo_path(p), encoding="utf-8") if l.strip()]


def raw_doc_texts(doc_ids):
    """{doc_id: text} from the chunks, in chunk order, with each chunk's leading overlap tokens removed (the overlap
    was measured with the Qwen3 tokenizer, so it is removed with that tokenizer)."""
    from transformers import AutoTokenizer
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    by_doc = defaultdict(list)
    for c in rd("data/chunks/chunks.jsonl"):
        if c["doc_id"] in doc_ids:
            by_doc[c["doc_id"]].append(c)
    out = {}
    for d, cs in by_doc.items():
        parts = []
        for c in sorted(cs, key=lambda c: int(c["chunk_id"].rsplit(":", 1)[1])):
            t = c["text"]
            ov = int(c.get("overlap_tokens") or 0)
            if ov:
                ids = qtok(t, add_special_tokens=False)["input_ids"]
                t = qtok.decode(ids[ov:]).lstrip()
            parts.append(t)
        out[d] = "\n\n".join(parts)
    return out


def pack(tok, texts):
    """Concatenate documents (EOS between them) and cut into SEQ-token sequences (BOS first when available)."""
    stream = []
    for d in sorted(texts):
        stream += tok(texts[d], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    body = SEQ - len(bos)
    return [bos + stream[i:i + body] for i in range(0, len(stream), body)], len(stream)


def save_npz(path, seqs):
    """seqs: list of (ids, is_raw, prompt_len). Stored as a flat int32 array + offsets."""
    ids = np.concatenate([np.asarray(s[0], dtype=np.int32) for s in seqs]) if seqs else np.zeros(0, np.int32)
    off = np.cumsum([0] + [len(s[0]) for s in seqs]).astype(np.int64)
    np.savez(path, ids=ids, offsets=off, is_raw=np.asarray([s[1] for s in seqs], np.int8),
             prompt_len=np.asarray([s[2] for s in seqs], np.int32))


def ngrams(text, n=13):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODEL_ID))
    ap.add_argument("--smoke", action="store_true", help="tiny corpus (3 docs, 64 QA items) into data/cpt_smoke/")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        OUT = repo_path("data/cpt_smoke")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    tcfg = dict(cfg["train"])
    train = rd("data/splits/train.jsonl")
    val = rd("data/splits/val.jsonl")
    tests = {s: rd(f"data/splits/{s}.jsonl") for s in ("test_indomain", "test_heldout_docs", "test_seen_facts")}
    train_docs = {r["doc_id"] for r in train}
    heldout_docs = {r["doc_id"] for r in tests["test_heldout_docs"]}
    if args.smoke:
        train_docs = set(sorted(train_docs)[:3])
        train = [r for r in train if r["doc_id"] in train_docs][:64]
        val = val[:16]
        heldout_docs = set(sorted(heldout_docs)[:1])

    # ---- leakage audit (fails the build)
    test_ids = {r["qa_id"] for s in tests.values() for r in s} | {r["qa_id"] for r in val}
    seen_q = {r["question"].strip().lower() for r in tests["test_seen_facts"]}
    qa_questions = set()
    for r in train:
        qa_questions.add(r["question"].strip().lower())
        qa_questions.update(p.strip().lower() for p in r.get("paraphrases") or [])
    audit = {"train_docs": len(train_docs), "heldout_docs": len(heldout_docs),
             "heldout_docs_in_raw": sorted(train_docs & heldout_docs),
             "heldout_docs_in_qa": sorted({r["doc_id"] for r in train} & heldout_docs),
             "test_or_val_qa_ids_in_qa": sorted({r["qa_id"] for r in train} & test_ids),
             "seen_facts_questions_in_qa": len(seen_q & qa_questions),
             "note": "test_trained_exact items are TRAIN items by design (trained-question recall test)."}
    failed = [k for k in ("heldout_docs_in_raw", "heldout_docs_in_qa", "test_or_val_qa_ids_in_qa") if audit[k]] + \
             (["seen_facts_questions_in_qa"] if audit["seen_facts_questions_in_qa"] else [])

    texts = raw_doc_texts(train_docs)
    raw_grams = set()
    for t in texts.values():
        raw_grams |= ngrams(t)
    ov = {}
    for s, rows in list(tests.items()) + [("test_trained_exact", rd("data/splits/test_trained_exact.jsonl"))]:
        hit, share = 0, []
        for r in rows:
            g = ngrams(r["question"])
            if g:
                k = len(g & raw_grams)
                hit += k > 0
                share.append(k / len(g))
        ov[s] = {"questions": len(rows), "with_any_13gram_in_raw": hit, "share": round(hit / max(len(rows), 1), 4),
                 "mean_13gram_share": round(float(np.mean(share)) if share else 0.0, 4)}
    audit["question_13gram_overlap_with_raw"] = ov
    audit["passed"] = not failed
    audit["failed_checks"] = failed
    (OUT / "leakage_audit.json").write_text(json.dumps(audit, indent=2))
    if failed:
        raise SystemExit(f"LEAKAGE AUDIT FAILED: {failed}")
    ho_texts = raw_doc_texts(heldout_docs)

    from transformers import AutoTokenizer
    stats = json.loads((OUT / "stats.json").read_text()) if (OUT / "stats.json").exists() else {}
    for fam in args.models:
        tok = AutoTokenizer.from_pretrained(MODEL_ID[fam])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        d = OUT / fam
        d.mkdir(parents=True, exist_ok=True)
        raw, raw_tokens = pack(tok, texts)
        ho, _ = pack(tok, ho_texts)
        ho = random.Random(0).sample(ho, min(200, len(ho)))
        save_npz(d / "heldout_ppl.npz", [(s, 1, 0) for s in ho])
        tst = {"prefix_mismatch": 0}
        vq = [tokenize(tok, e, SEQ, tst) for e in examples(val, tcfg, "closedbook", False)]
        save_npz(d / "val_qa.npz", [(x["input_ids"], 0, sum(1 for l in x["labels"] if l == -100)) for x in vq])
        st = {"raw_sequences": len(raw), "raw_tokens": raw_tokens, "heldout_ppl_sequences": len(ho), "epochs": {}}
        for ep in range(1, EPOCHS + 1):
            qa_rows = []
            for r in train:
                paras = r.get("paraphrases") or []
                rr = dict(r, paraphrases=[paras[(ep - 1) % len(paras)]] if paras else [])
                qa_rows.append(rr)
            qa = [tokenize(tok, e, SEQ, tst) for e in examples(qa_rows, tcfg, "closedbook", True)]
            qa_seqs = [(x["input_ids"], 0, sum(1 for l in x["labels"] if l == -100)) for x in qa]
            seqs = [(s, 1, 0) for s in raw] + qa_seqs
            rng = random.Random(42 + ep)
            rng.shuffle(seqs)
            pad = (-len(seqs)) % EFF_BATCH
            dup = rng.sample(qa_seqs, pad) if pad else []
            seqs += dup
            save_npz(d / f"epoch{ep}.npz", seqs)
            qa_tok = sum(len(s[0]) for s in qa_seqs)
            qa_loss_tok = sum(len(s[0]) - s[2] for s in qa_seqs)
            st["epochs"][ep] = {"sequences": len(seqs), "raw_sequences": len(raw), "qa_examples": len(qa_seqs),
                                "qa_duplicates_for_batch_alignment": pad, "raw_tokens_in_sequences": sum(len(s) for s in raw),
                                "qa_tokens": qa_tok, "qa_loss_tokens": qa_loss_tok,
                                "steps": len(seqs) // EFF_BATCH}
        e1 = st["epochs"][1]
        st["raw_to_qa_token_ratio"] = f"{e1['raw_tokens_in_sequences'] / (e1['raw_tokens_in_sequences'] + e1['qa_tokens']):.2f}" \
                                      f" / {e1['qa_tokens'] / (e1['raw_tokens_in_sequences'] + e1['qa_tokens']):.2f}"
        st["raw_to_qa_loss_token_ratio"] = f"{e1['raw_tokens_in_sequences'] / (e1['raw_tokens_in_sequences'] + e1['qa_loss_tokens']):.2f}" \
                                           f" / {e1['qa_loss_tokens'] / (e1['raw_tokens_in_sequences'] + e1['qa_loss_tokens']):.2f}"
        st["prefix_mismatches"] = tst["prefix_mismatch"]
        stats[fam] = st
        (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
        print(f"{fam}: {json.dumps({k: v for k, v in st.items() if k != 'epochs'})}; epoch 1 {st['epochs'][1]}", flush=True)
    print(json.dumps(audit["question_13gram_overlap_with_raw"]), flush=True)


if __name__ == "__main__":
    main()
