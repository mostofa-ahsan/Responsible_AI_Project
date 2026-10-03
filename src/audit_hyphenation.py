"""Audit line-break hyphen joins made by parse.fix_text.

Replays the join decision on the cached Docling output and lists joined words
that never occur unhyphenated anywhere in the corpus (the likeliest bad joins),
plus pairs kept hyphenated. Writes logs/hyphenation_audit_report.txt.

Usage:
    python src/audit_hyphenation.py [--top 30]
"""

import argparse
from collections import Counter

from docling_core.types.doc import DoclingDocument

from parse import HYPHEN_BREAK_RX, doc_vocab, fix_text, raw_items
from utils import Report, load_config, repo_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config()

    docs = {}
    for f in sorted(repo_path(cfg["paths"]["docling_cache"]).glob("*.json")):
        docs[f.stem] = [it["text"] for it in raw_items(DoclingDocument.load_from_json(f))]

    corpus = Counter()
    for texts in docs.values():
        corpus.update(doc_vocab(texts))

    joined, kept, joined_docs = Counter(), Counter(), {}
    for doc_id, texts in docs.items():
        vocab = doc_vocab(texts)
        for t in texts:
            for m in HYPHEN_BREAK_RX.finditer(t.replace("­", "")):
                left, right = m.group(1), m.group(2)
                word = (left + right).lower()
                out = fix_text(m.group(0), vocab)
                if " – " in out or "-" in out:
                    kept[out.lower()] += 1
                else:
                    joined[word] += 1
                    joined_docs.setdefault(word, doc_id)

    unseen = Counter({w: c for w, c in joined.items() if not corpus[w]})
    rep = Report("hyphenation_audit", cfg)
    rep(f"=== Hyphenation audit over {len(docs)} docs ===")
    rep(f"joins: {sum(joined.values()):,} ({len(joined):,} distinct words); "
        f"kept hyphenated: {sum(kept.values()):,} ({len(kept):,} distinct)")
    rep(f"joined words never seen unhyphenated in corpus: {len(unseen):,} distinct, "
        f"{sum(unseen.values()):,} occurrences")
    rep(f"\n-- Top {args.top} joined words not found elsewhere in the corpus --")
    for w, c in unseen.most_common(args.top):
        rep(f"  {c:4d}  {w:28s} (e.g. {joined_docs[w]})")
    rep(f"\n-- Top {args.top} pairs kept hyphenated or written as a dash --")
    for w, c in kept.most_common(args.top):
        rep(f"  {c:4d}  {w}")
    rep.save()


if __name__ == "__main__":
    main()
