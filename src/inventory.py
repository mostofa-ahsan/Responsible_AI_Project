"""Phase 1: inventory the raw PDF corpus before parsing.

Scans data/raw_pdfs recursively (read-only) and writes data/inventory.csv with
doc_id, size, page count, text-layer status, title/author/year, and likely
duplicates (identical hash, or most of a doc's text contained in another doc).

Resumable: each PDF's scan result is cached under paths.inventory_cache keyed by
its sha256, so re-runs skip PyMuPDF work for unchanged files.

Usage:
    python src/inventory.py [--limit N] [--no-cache]
"""

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

from utils import Report, get_logger, load_config, repo_path, slugify

pymupdf.TOOLS.mupdf_display_errors(False)

SCAN_VERSION = 3  # bump when scan logic changes, to invalidate cached scans

FIELDS = [
    # spec columns
    "doc_id", "path", "folder", "size_mb", "pages", "text_layer", "title",
    "author", "year", "sha256", "duplicate_of",
    # extras
    "doc_group", "duplicate_reason", "similar_title", "title_source",
    "year_source", "text_page_ratio", "avg_chars_per_page", "status", "error",
]
# Fields that depend only on file content, so they can be cached by sha256.
SCAN_FIELDS = ["pages", "text_layer", "title", "author", "year", "title_source",
               "year_source", "text_page_ratio", "avg_chars_per_page", "status", "error"]

BAD_META_TITLE = re.compile(r"^(untitled|microsoft word|document\d*|title|unknown)\b", re.I)
BAD_META_AUTHOR = re.compile(r"^(user|admin|administrator|owner|unknown|author|pc|hp|dell|lenovo)$"
                             r"|\.(indd|indb|doc|docx|tex)$|^\W*$", re.I)
# Largest-font text that is a running header or journal name, not the document title.
GENERIC_FONT_TITLE = re.compile(r"^(sciencedirect|.*\bjournal\b.*|chapter\s*\d+|contents)$", re.I)
COPYRIGHT_YEAR = re.compile(r"(?:©|\(c\)|copyright)\s*(?:by\s+)?(?:[A-Za-z&.,]+\s+){0,4}?((?:19|20)\d{2})\b", re.I)
ARTICLE_YEAR = re.compile(r"(?:available online|published online|accepted|received)\D{0,25}((?:19|20)\d{2})\b", re.I)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_stem(stem):
    s = re.sub(r"^JA_", "", stem)
    s = re.sub(r"(_\d{2}){6}$", "", s)          # download timestamp suffix
    s = re.sub(r"_?97[89][-\d]{10,}", "", s)     # ISBN
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def unspace(text):
    """'C H A P T E R 1' -> 'CHAPTER 1', 'C ontents' -> 'Contents'."""
    text = re.sub(r"\b(?:\w ){2,}\w\b", lambda m: m.group(0).replace(" ", ""), text)
    return re.sub(r"^([A-Za-z]) ([a-z]+)$", r"\1\2", text)  # whole heading only; "2 Fairness" stays


def normalize_title(title):
    t = unspace(clean_stem(title)).lower()
    t = re.sub(r"\bchapter\s+\d+\b", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def font_title(doc):
    """Largest-font text on the first 3 pages, joining consecutive lines of that size."""
    best_size, best_text = 0.0, ""
    for page in doc.pages(0, min(3, doc.page_count)):
        lines = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                text = " ".join(s["text"] for s in line["spans"]).strip()
                size = max((s["size"] for s in line["spans"]), default=0)
                if text:
                    lines.append((size, text))
        sizes = [sz for sz, t in lines if len(re.sub(r"\W", "", t)) >= 2]
        if not sizes or max(sizes) <= best_size:
            continue
        top = max(sizes)
        run, parts = False, []
        for sz, t in lines:
            if sz >= 0.9 * top:
                parts.append(t)
                run = True
            elif run:
                break
        best_size, best_text = top, " ".join(parts)
    return unspace(re.sub(r"\s+", " ", best_text).strip())[:200]


def detect_title(doc, path):
    meta = ((doc.metadata or {}).get("title") or "").strip()
    if (len(meta) > 4 and not BAD_META_TITLE.match(meta)
            and normalize_title(meta) != normalize_title(path.stem)
            and not meta.lower().endswith((".pdf", ".doc", ".docx", ".indd", ".indb"))):
        return meta, "metadata"
    ft = font_title(doc)
    if ft and not GENERIC_FONT_TITLE.match(ft):
        return ft, "font"
    return clean_stem(path.stem), "filename"


def detect_author(doc, path):
    meta = re.sub(r"\s+", " ", (doc.metadata or {}).get("author") or "").strip()
    if meta and not BAD_META_AUTHOR.search(meta):
        return meta
    m = re.match(r"^(.+?)_(?:19|20)\d{2}_", path.stem)   # "Name, Name_2021_Title"
    return m.group(1).strip(" ,") if m else ""


def clean_author(author):
    a = author.replace("\ufeff", "").strip()
    return re.sub(r"^(edited|ed\.)\s+by\s+", "", a, flags=re.I)


def fill_from_group(rows):
    """Multi-file documents (split chapters) share year/author across their files."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["doc_group"]].append(r)
    for members in groups.values():
        for field, src in (("year", "year_source"), ("author", None)):
            donor = next((m for m in members if m[field]), None)
            for m in members:
                if donor and not m[field]:
                    m[field] = donor[field]
                    if src:
                        m[src] = "doc_group"


def detect_year(doc, path, page_texts):
    m = re.search(r"_((?:19|20)\d{2})_", path.stem)
    if m:
        return m.group(1), "filename"
    head = "\n".join(page_texts[:8])
    for rx in (COPYRIGHT_YEAR, ARTICLE_YEAR):
        years = [int(y) for y in rx.findall(head) if 1950 <= int(y) <= 2030]
        if years:
            return str(max(years)), "text"
    m = re.match(r"D:((?:19|20)\d{2})", (doc.metadata or {}).get("creationDate") or "")
    if m:
        return m.group(1), "creation_date"
    return "", ""


def shingle_hashes(text, n_words, sample_mod):
    """Hashed word n-grams, consistently sampled (keep hash % mod == 0)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for i in range(max(len(words) - n_words + 1, 0)):
        h = int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n_words]).encode(),
                                           digest_size=8).digest(), "big")
        if h % sample_mod == 0:
            out.add(h)
    return out


def scan_pdf(path, icfg):
    """Content-only scan. Returns (fields dict, shingle set)."""
    res = dict.fromkeys(SCAN_FIELDS, "")
    res["status"] = "ok"
    shingles = set()
    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise ValueError("encrypted (password required)")
            if doc.page_count == 0:
                raise ValueError("zero pages")
            page_texts = [page.get_text() for page in doc]
            counts = [len(re.sub(r"\s", "", t)) for t in page_texts]
            ratio = sum(c >= icfg["min_text_chars"] for c in counts) / doc.page_count
            res.update(
                pages=doc.page_count,
                text_page_ratio=round(ratio, 3),
                avg_chars_per_page=round(sum(counts) / doc.page_count),
                text_layer=("yes" if ratio >= icfg["text_yes_ratio"]
                            else "no" if ratio <= icfg["text_no_ratio"] else "partial"),
                author=detect_author(doc, path),
            )
            res["title"], res["title_source"] = detect_title(doc, path)
            res["year"], res["year_source"] = detect_year(doc, path, page_texts)
            shingles = shingle_hashes("\n".join(page_texts), icfg["shingle_words"],
                                      icfg["shingle_sample_mod"])
    except Exception as e:  # corrupt / unreadable file
        res["status"] = "corrupt"
        res["error"] = f"{type(e).__name__}: {e}"
    return res, shingles


def assign_doc_ids(rows):
    """Slug of the cleaned filename; bulk-download files get a folder prefix.
    Collisions get a short hash suffix so ids stay unique and stable."""
    for r in rows:
        rel = Path(r["path"])
        base = clean_stem(rel.stem)
        if "bulk-download" in rel.parts:
            base = f"bulk-download {base}"
        r["doc_id"] = slugify(base)
    counts = Counter(r["doc_id"] for r in rows)
    for r in rows:
        if counts[r["doc_id"]] > 1:
            r["doc_id"] = f"{r['doc_id']}-{r['sha256'][:6]}"


def containment(a, b):
    """Fraction of the smaller shingle set found in the larger one."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return len(small & large) / len(small)


def find_duplicates(rows, shingles, icfg):
    """Likely duplicate = identical hash or >= containment_threshold shared text.

    Similar titles are reported separately with their content overlap; on their
    own they are not evidence (e.g. two different books both titled
    "Artificial Intelligence ...").
    """
    dups = defaultdict(dict)    # doc_id -> {other_doc_id: reason}
    similar = defaultdict(list)
    ok = [r for r in rows if r["status"] == "ok"]

    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r["doc_id"])
    for ids in by_hash.values():
        for a in ids:
            for b in ids:
                if a != b:
                    dups[a][b] = "hash"

    norm = {r["doc_id"]: normalize_title(r["title"]) for r in ok}
    for i, ra in enumerate(ok):
        for rb in ok[i + 1:]:
            a, b = ra["doc_id"], rb["doc_id"]
            if b in dups[a]:
                continue
            frac = containment(shingles.get(a), shingles.get(b))
            if frac >= icfg["containment_threshold"]:
                dups[a][b] = dups[b][a] = f"content:{frac:.0%}"
            ta, tb = norm[a], norm[b]
            if (len(ta) >= 12 and len(tb) >= 12 and not re.match(r"chapter \d+$", ta)
                    and difflib.SequenceMatcher(None, ta, tb).ratio() >= icfg["title_similarity"]):
                similar[a].append(f"{b} (overlap {frac:.0%})")
                similar[b].append(f"{a} (overlap {frac:.0%})")

    for r in rows:
        d = dups.get(r["doc_id"], {})
        r["duplicate_of"] = ";".join(d)
        r["duplicate_reason"] = ";".join(d.values())
        r["similar_title"] = ";".join(similar.get(r["doc_id"], []))


def report(rows, n_cached, rep):
    rep(f"=== Phase 1 inventory: {len(rows)} PDFs ({n_cached} from cache) ===")
    for folder in ("book", "article"):
        fr = [r for r in rows if r["folder"] == folder]
        rep(f"{folder:8s} files={len(fr):3d}  pages={sum(r['pages'] or 0 for r in fr):,}"
            f"  size={sum(r['size_mb'] for r in fr):,.1f} MB")

    rep("\n-- Text layer (no/partial = OCR candidates) --")
    for status in ("yes", "partial", "no"):
        fr = [r for r in rows if r["text_layer"] == status]
        rep(f"{status:8s} {len(fr)}")
        if status != "yes":
            for r in fr:
                rep(f"    {r['doc_id']}  {r['path']}  (pages={r['pages']}, text ratio={r['text_page_ratio']})")

    bad = [r for r in rows if r["status"] != "ok"]
    rep(f"\n-- Corrupt / unreadable: {len(bad)} --")
    for r in bad:
        rep(f"    {r['path']}: {r['error']}")

    dup = [r for r in rows if r["duplicate_of"]]
    rep(f"\n-- Likely duplicates: {len(dup)} --")
    for r in dup:
        rep(f"  {r['doc_id']} ~ {r['duplicate_of']}  [{r['duplicate_reason']}]")

    sim = [r for r in rows if r["similar_title"]]
    rep(f"\n-- Similar titles (not duplicates unless overlap is high): {len(sim)} --")
    for r in sim:
        rep(f"  {r['doc_id']} ~ {r['similar_title']}")

    groups = defaultdict(list)
    for r in rows:
        groups[r["doc_group"]].append(r["doc_id"])
    multi = {g: ids for g, ids in groups.items() if len(ids) > 1}
    rep(f"\n-- Multi-file documents (keep in one split): {len(multi)} --")
    for g, ids in multi.items():
        rep(f"  {g}: {len(ids)} files")

    rep("\n-- Metadata coverage --")
    rep(f"  title sources: {dict(Counter(r['title_source'] for r in rows))}")
    rep(f"  year sources:  {dict(Counter(r['year_source'] or 'none' for r in rows))}"
        "  (creation_date is often the download date, not publication)")
    rep(f"  missing author: {sum(not r['author'] for r in rows)}, "
        f"missing year: {sum(not r['year'] for r in rows)}")

    # Pages for QA cost: keep the larger file of each duplicate pair.
    pages = {r["doc_id"]: r["pages"] or 0 for r in rows}
    unique_pages = sum(
        r["pages"] for r in rows
        if r["status"] == "ok" and not any(
            (pages[o], o) > (r["pages"], r["doc_id"]) for o in r["duplicate_of"].split(";") if o))
    rep(f"\nPages after removing duplicates (QA cost base): {unique_pages:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, help="only scan the first N PDFs (quick test)")
    ap.add_argument("--no-cache", action="store_true", help="rescan every PDF")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    icfg = cfg["inventory"]
    log = get_logger("inventory", cfg)
    root = repo_path(cfg["paths"]["raw_pdfs"])
    out = repo_path(cfg["paths"]["inventory"])
    cache_dir = repo_path(cfg["paths"]["inventory_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Cached results are only valid for the settings that produced them.
    params = hashlib.sha256(json.dumps([icfg, SCAN_VERSION], sort_keys=True).encode()).hexdigest()[:12]

    pdfs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if args.limit:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        sys.exit(f"No PDFs found under {root}")
    log.info(f"Scanning {len(pdfs)} PDFs under {root}")

    rows, shingles, n_cached = [], {}, 0
    for i, path in enumerate(pdfs, 1):
        rel = path.relative_to(root)
        digest = sha256(path)
        cache_file = cache_dir / f"{digest}.json"
        cached = None
        if not args.no_cache and cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("params") != params:
                cached = None
        if cached:
            res, sh = cached["fields"], set(cached["shingles"])
            n_cached += 1
            log.debug(f"[{i}/{len(pdfs)}] cached {rel}")
        else:
            log.info(f"[{i}/{len(pdfs)}] {rel}")
            res, sh = scan_pdf(path, icfg)
            cache_file.write_text(json.dumps({"params": params, "fields": res,
                                              "shingles": sorted(sh)}), encoding="utf-8")
        if res["status"] != "ok":
            log.warning(f"corrupt: {rel}: {res['error']}")
        row = dict.fromkeys(FIELDS, "")
        row.update(res)
        row.update(
            path=str(rel),
            folder="book" if rel.parts[0] == "Books_AI" else "article",
            size_mb=round(path.stat().st_size / 1e6, 2),
            sha256=digest,
            # Chapters split into separate files belong to one document for splits.
            doc_group=str(rel.parent) if "bulk-download" in rel.parts else str(rel),
        )
        row["author"] = clean_author(row["author"])
        rows.append(row)
        shingles[str(rel)] = sh

    fill_from_group(rows)
    assign_doc_ids(rows)
    shingles = {r["doc_id"]: shingles[r["path"]] for r in rows}
    find_duplicates(rows, shingles, icfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    log.info(f"Wrote {len(rows)} rows to {out}")

    rep = Report("inventory", cfg)
    report(rows, n_cached, rep)
    rep.save()
    log.debug("\n".join(rep.lines))


if __name__ == "__main__":
    main()
