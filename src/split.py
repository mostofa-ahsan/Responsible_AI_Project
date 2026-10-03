"""Train / val / test splits without leakage.

Input: passed pairs of data/qa_pairs/<run>.jsonl. Output: data/splits/{train,val,test_indomain,
test_heldout_docs}.jsonl and data/splits/split_manifest.json.

1. test_heldout_docs: whole documents held out (split.min/max_heldout_docs, about
   split.heldout_pair_share of the pairs). Documents are chosen by doc_group, so multi-file
   books (AI Horizons) stay together; books and articles in proportion to their pair counts,
   greedily favouring documents that add readiness dimensions the held-out set lacks.
2. The remaining pairs are split by CHUNK (all questions from one chunk, with their paraphrases,
   stay in one split, so no evidence is shared across splits) into val, test_indomain and train.

Usage:
    python src/split.py [--run full_v1]
"""

import argparse
import json
import random
from collections import Counter, defaultdict

from qa_common import read_jsonl, run_paths
from utils import Report, get_logger, load_config, repo_path


def doc_groups(cfg):
    """doc_id -> doc_group from the parsed docs (books split into chapter files share one)."""
    out = {}
    for f in repo_path(cfg["paths"]["parsed"]).glob("*.json"):
        if f.name == "metadata.csv":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["doc_id"]] = (d.get("doc_group") or d["doc_id"], d["folder"])
    return out


def choose_heldout(pairs, groups, scfg, rng):
    by_group = defaultdict(list)
    for p in pairs:
        by_group[groups[p["doc_id"]][0]].append(p)
    folder = {g: groups[ps[0]["doc_id"]][1] for g, ps in by_group.items()}
    total = len(pairs)
    target = scfg["heldout_pair_share"] * total
    share = Counter()
    for g, ps in by_group.items():
        share[folder[g]] += len(ps)
    want = {f: target * n / total for f, n in share.items()}          # pairs per folder
    all_dims = {p["dimension"] for p in pairs}
    chosen, have, dims = [], Counter(), set()
    cands = list(by_group)
    rng.shuffle(cands)
    # Avoid holding out a document that alone exceeds half the target.
    cands = [g for g in cands if len(by_group[g]) <= max(target / 2, 1)]
    while len(chosen) < scfg["max_heldout_docs"] and cands:
        def gain(g):
            f = folder[g]
            room = want[f] - have[f]
            if room <= 0:
                return -1
            new_dims = len({p["dimension"] for p in by_group[g]} - dims)
            fit = 1 - abs(room - len(by_group[g])) / max(room, 1)
            return new_dims * 2 + fit
        best = max(cands, key=gain)
        if gain(best) < 0 and len(chosen) >= scfg["min_heldout_docs"]:
            break
        chosen.append(best)
        have[folder[best]] += len(by_group[best])
        dims |= {p["dimension"] for p in by_group[best]}
        cands.remove(best)
        if sum(have.values()) >= target and len(chosen) >= scfg["min_heldout_docs"]:
            break
    return set(chosen), all_dims - dims


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="full_v1")
    args = ap.parse_args()
    cfg = load_config()
    scfg = cfg["split"]
    log = get_logger("split", cfg)
    rng = random.Random(scfg["seed"])
    pairs = [p for p in read_jsonl(run_paths(cfg, args.run)["final"]) if p["passed_filters"]]
    if not pairs:
        raise SystemExit(f"no passed pairs in {args.run}")
    groups = doc_groups(cfg)
    held_groups, missing_dims = choose_heldout(pairs, groups, scfg, rng)

    splits = defaultdict(list)
    rest_by_chunk = defaultdict(list)
    for p in pairs:
        g = groups[p["doc_id"]][0]
        row = {k: p[k] for k in ("qa_id", "group_id", "question", "paraphrases", "answer", "citation",
                                 "evidence", "doc_id", "chunk_id", "section_path", "q_type", "dimension",
                                 "difficulty")}
        row["doc_group"] = g
        if g in held_groups:
            splits["test_heldout_docs"].append(row)
        else:
            rest_by_chunk[p["chunk_id"]].append(row)
    chunk_ids = sorted(rest_by_chunk)
    rng.shuffle(chunk_ids)
    n_val = round(len(chunk_ids) * scfg["val_share"])
    n_test = round(len(chunk_ids) * scfg["test_indomain_share"])
    for i, cid in enumerate(chunk_ids):
        name = "val" if i < n_val else "test_indomain" if i < n_val + n_test else "train"
        splits[name].extend(rest_by_chunk[cid])

    out_dir = repo_path(cfg["paths"]["splits"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test_indomain", "test_heldout_docs"):
        rows = sorted(splits[name], key=lambda r: r["qa_id"])
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({**r, "split": name}, ensure_ascii=False) + "\n")
    # leakage checks
    chunk_split = defaultdict(set)
    doc_split = defaultdict(set)
    for name, rows in splits.items():
        for r in rows:
            chunk_split[r["chunk_id"]].add(name)
            doc_split[r["doc_group"]].add(name)
    assert all(len(s) == 1 for s in chunk_split.values()), "a chunk appears in two splits"
    assert all(s == {"test_heldout_docs"} for g, s in doc_split.items() if g in held_groups)
    manifest = {"run": args.run, "heldout_doc_groups": sorted(held_groups),
                "counts": {k: len(v) for k, v in splits.items()}, "seed": scfg["seed"]}
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    rep = Report(f"split_{args.run}", cfg)
    rep(f"=== Splits from {args.run}: {len(pairs):,} passed pairs ===")
    for name in ("train", "val", "test_indomain", "test_heldout_docs"):
        rows = splits[name]
        rep(f"{name:18s} pairs={len(rows):6,} ({len(rows) / len(pairs):.0%})  chunks={len({r['chunk_id'] for r in rows}):5,}  "
            f"docs={len({r['doc_id'] for r in rows}):3d}  paraphrase examples={sum(len(r['paraphrases']) for r in rows):6,}")
    rep(f"\nheld-out documents ({len(held_groups)}): "
        + ", ".join(f"{g} [{groups[next(p['doc_id'] for p in pairs if groups[p['doc_id']][0] == g)][1]}]"
                    for g in sorted(held_groups)))
    rep(f"dimensions missing from the held-out set: {sorted(missing_dims) or 'none'}")
    for key in ("q_type", "dimension", "difficulty"):
        rep(f"\n-- {key} share by split --")
        vals = sorted({r[key] for r in pairs})
        rep(f"{'':22s}" + "".join(f"{v[:12]:>13s}" for v in vals))
        for name in ("train", "val", "test_indomain", "test_heldout_docs"):
            c = Counter(r[key] for r in splits[name])
            n = max(len(splits[name]), 1)
            rep(f"{name:22s}" + "".join(f"{c[v] / n:>13.0%}" for v in vals))
    rep("\nleakage checks passed: no chunk in two splits; held-out documents appear only in test_heldout_docs")
    rep.save()
    log.info(f"wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
