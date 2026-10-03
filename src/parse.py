"""Phase 2: parse PDFs with Docling into cleaned, ordered blocks.

Reads data/inventory.csv, converts each PDF with Docling (GPU), and writes one
JSON per document to data/parsed/<doc_id>.json:

    {doc_id, path, folder, doc_group, n_pages, extracted: {...}, blocks: [
        {block_id, type, text, section_path, page, page_end, skip, skip_reason}]}

type is heading | paragraph | table | caption | list. Non-content material
(references, index, TOC, copyright, acknowledgements, declarations, footnotes,
article front matter) is kept but tagged skip=true with a skip_reason. Running
headers/footers and page numbers are dropped.

Also writes data/parsed/metadata.csv (inventory metadata with gaps filled from
the first pages) and data/parsed/metadata_review.csv (rows needing a human look).

Resumable: docs with an existing data/parsed/<doc_id>.json are skipped. Raw
Docling output is cached in paths.docling_cache, so --reclean re-runs the
cleaning rules without re-parsing.

Usage:
    python src/parse.py [--limit N] [--doc DOC_ID ...] [--reclean] [--force]
"""

import argparse
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from inventory import ARTICLE_YEAR, COPYRIGHT_YEAR, clean_stem, unspace
from utils import Report, get_logger, load_config, repo_path

PARSER_VERSION = 13

# --- section-level skip rules (matched against normalized heading text) -----
SKIP_SECTIONS = [
    ("references", r"(references?( list)?|bibliography|works cited|literature cited|"
                   r"further reading|suggested readings?|recommended readings?|sources)"),
    ("index", r"((subject|author|name) )?index( \d+)?"),
    ("toc", r"((table of |brief )?contents|list of (figures|tables|abbreviations|illustrations))"),
    ("front_matter", r"(copyright( page)?|dedication|also by.*|other books.*|article info|"
                     r"about the (authors?|editors?|contributors?)|(notes on )?contributors|"
                     r"editors? and contributors)"),
    ("back_matter", r"(acknowledge?ments?|declaration of competing interests?|"
                    r"(declaration of )?conflicts? of interests?|competing interests?|"
                    r"funding( sources| information| statement)?|credit authorship contribution statement|"
                    r"authors? contributions?|data availability( statement)?|"
                    r"ethic(s|al) (statement|approval).*|declaration of (generative )?ai.*|"
                    r"(appendix [a-z0-9]+ )?supplementary (data|materials?)|"
                    r"(statements and )?declarations|consent.*|informed consent.*)"),
]
SKIP_SECTIONS = [(reason, re.compile(rf"^{rx}$")) for reason, rx in SKIP_SECTIONS]

LABEL_TYPE = {
    "title": "heading", "section_header": "heading",
    "text": "paragraph", "paragraph": "paragraph", "formula": "paragraph", "code": "paragraph",
    "footnote": "paragraph", "reference": "paragraph", "document_index": "paragraph",
    "list_item": "list", "caption": "caption", "table": "table",
}
LABEL_SKIP = {"footnote": "footnote", "reference": "references", "document_index": "toc"}

CHAPTER_RX = re.compile(r"^(chapter|part)\s+(\d+|[ivxlc]+)\b\.?", re.I)
NUMBERED_RX = re.compile(r"^((?:\d+\.)*\d+)\.?\s+\S")
PAGE_NUM_RX = re.compile(r"^\s*(page\s*)?(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.I)
HEADER_WITH_PAGE_RX = re.compile(r"^\d{1,4}\s*[·•|]\s*\S|\S\s*[·•|]\s*\d{1,4}$")
COPYRIGHT_RX = re.compile(r"all rights reserved|library of congress|\bisbn\b|"
                          r"©.{0,80}(publish|springer|elsevier|routledge|press|wiley|taylor)", re.I)
ARTICLE_META_RX = re.compile(r"^(received|accepted|revised|available online|e-?mail|corresponding author)|"
                             r"https?://doi\.org|creativecommons|open access article|peer review under|"
                             r"journal homepage|contents lists available", re.I)
META_MAX_CHARS = 250  # longer blocks are body text even if they contain a DOI/e-mail
TOC_LINE_RX = re.compile(r"(\.\s?){4,}\s*\d{1,4}\s*$")
INDEX_LINE_RX = re.compile(r"^[^.!?]{2,80},\s*\d{1,4}(\s*[-–]\s*\d{1,4})?(\s*,\s*\d{1,4}(\s*[-–]\s*\d{1,4})?)*$")
AFFILIATION_RX = re.compile(r"universit|department|school of|institute|college|faculty|"
                            r"centre|center|@|\bP\.?O\.? box\b", re.I)
JOURNAL_RX = re.compile(r"journal|computers and education|computers & education|"
                        r"heliyon|sciencedirect|elsevier|^vol\.|proceedings", re.I)
GENERIC_HEADING_RX = re.compile(r"^(abstract|article info|keywords|introduction|highlights|"
                                r"research article|original article|review article|contents|preface)$", re.I)
NAMES_RX = re.compile(r"^(?:(?:[A-Z][a-zA-Z'’.-]+|[A-Z]\.|van|von|de|del|da|di|la|le|al|el)"
                      r"(?:\s+|\s*[,·&]\s*|\s+and\s+)?){2,12}$")
FIRST_PUBLISHED = re.compile(r"first published\D{0,20}((?:19|20)\d{2})", re.I)
JOURNAL_CITATION_YEAR = re.compile(r"\(((?:19|20)\d{2})\)\s*\d{3,}")


def norm_heading(text):
    t = unspace(text).lower()
    t = re.sub(r"^((\d+\.)*\d+\.?|[ivxlc]+\.|appendix [a-z0-9]+\.?)\s+", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


FUNCTION_WORDS = {"the", "a", "an", "of", "to", "for", "that", "with", "in", "on", "at", "by",
                  "is", "are", "was", "it", "this", "which", "from", "as", "be"}
COMPOUND_PREFIXES = {"self", "non", "co", "post", "pre", "multi", "cross", "anti", "semi", "human",
                     "privacy", "data", "student", "teacher", "user", "ai", "well", "long", "short",
                     "high", "low", "real", "open", "machine", "lip", "socio", "meta", "mid", "attention"}
HYPHEN_BREAK_RX = re.compile(r"\b([A-Za-z]{2,})- (?!(?:and|or|to|nor|versus|vs)\b)([a-z]{2,})\b")


def doc_vocab(texts):
    """Lowercase words in a document, ignoring fragments next to '- ' breaks."""
    vocab = Counter()
    for t in texts:
        t = re.sub(r"\w+- \w+", " ", t.replace("\u00ad", ""))
        vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", t))
    return vocab


def fix_text(text, vocab=None):
    t = re.sub(r"\u00ad\s*", "", text)          # soft hyphens
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)  # stray control chars
    t = re.sub(r"\s+", " ", t).strip()

    def join(m):
        left, right = m.group(1), m.group(2)
        joined = (left + right).lower()
        if left.lower() in FUNCTION_WORDS or right.lower() in FUNCTION_WORDS:
            return f"{left} – {right}"           # a dash read as a hyphen: "implicit- that"
        if vocab is not None and vocab[joined]:
            return left + right                  # joined form attested in the document
        if left.lower() in COMPOUND_PREFIXES:
            return f"{left}-{right}"             # "self- identity" -> "self-identity"
        if vocab is None or not vocab[right.lower()]:
            return left + right                  # "infor- mation" -> "information"
        return f"{left}-{right}"                 # both real words: "constraints-disparities"
    return HYPHEN_BREAK_RX.sub(join, t)


# --- Docling ----------------------------------------------------------------
_converters = {}


def get_converter(pcfg, ocr):
    if ocr not in _converters:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        opts = PdfPipelineOptions(
            do_ocr=ocr,
            do_table_structure=pcfg["do_table_structure"],
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice(pcfg["device"])),
        )
        _converters[ocr] = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converters[ocr]


def load_or_convert(row, pdf_path, cache_file, pcfg, force, log):
    """Return (DoclingDocument, seconds spent converting or None if cached)."""
    from docling_core.types.doc import DoclingDocument
    if cache_file.exists() and not force:
        return DoclingDocument.load_from_json(cache_file), None
    ocr = row["text_layer"] in pcfg["ocr_for_text_layer"]
    t0 = time.time()
    res = get_converter(pcfg, ocr).convert(pdf_path, raises_on_error=False)
    secs = time.time() - t0
    status = res.status.value if hasattr(res.status, "value") else str(res.status)
    if status not in ("success", "partial_success"):
        errs = "; ".join(str(e.error_message) for e in (res.errors or []))[:500]
        raise RuntimeError(f"docling status={status} {errs}")
    if status == "partial_success":
        log.warning(f"{row['doc_id']}: partial_success ({len(res.errors or [])} errors)")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    res.document.save_as_json(tmp)
    tmp.replace(cache_file)
    return res.document, secs


# --- cleaning ---------------------------------------------------------------
def raw_items(doc):
    out = []
    for item, _ in doc.iterate_items():
        label = item.label.value
        if label not in LABEL_TYPE:
            continue  # pictures, checkboxes, etc.
        if label == "table":
            text = item.export_to_markdown(doc=doc)
        else:
            text = getattr(item, "text", "") or ""
        if not text.strip():
            continue
        page = item.prov[0].page_no if item.prov else None
        out.append({"label": label, "text": text, "page": page})
    return out


def drop_running_headers(items, n_pages, min_pages):
    """Remove page numbers and short lines repeated across pages. Returns (kept, header texts)."""
    def key(t):
        return re.sub(r"[\d·•|]+", "", t).strip().lower()

    def header_like(it):
        # short body-label line that doesn't read as a sentence
        t = it["text"].strip()
        return it["label"] == "text" and len(t) < 100 and not re.search(r"[.!?]$", t)

    pages_by_key = defaultdict(set)
    for it in items:
        if header_like(it):
            pages_by_key[key(it["text"])].add(it["page"])
    threshold = max(min_pages, int(0.05 * n_pages))
    repeated = {k for k, pages in pages_by_key.items() if k and len(pages) >= threshold}

    kept, headers = [], Counter()
    for it in items:
        t = it["text"].strip()
        if it["label"] != "table" and (
                PAGE_NUM_RX.match(t)
                or (header_like(it) and key(t) in repeated)
                or (it["label"] in ("section_header", "text") and len(t) < 100
                    and HEADER_WITH_PAGE_RX.search(t))):
            if not PAGE_NUM_RX.match(t):
                headers[re.sub(r"\s*[·•|]?\s*\d{1,4}\s*[·•|]?\s*", " ", t).strip()] += 1
            continue
        kept.append(it)
    return kept, headers


INDEX_RUN_RX = re.compile(r"[A-Za-z][\w\s,'()/-]{2,60},?\s\d{1,3}(?:\s*[-–]\s*\d{1,3})?(?:,\s*\d{1,3})+")


def number_ratio(text):
    words = text.split()
    return sum(bool(re.fullmatch(r"\d{1,4}[,;.]?(?:[-–]\d{1,4}[,;.]?)?", w)) for w in words) / max(len(words), 1)


def is_meta_line(text):
    return len(text) < META_MAX_CHARS and bool(ARTICLE_META_RX.search(text))


def is_interruption(b):
    """Blocks that can sit between the two halves of a split paragraph."""
    return (b["_label"] in ("footnote", "caption", "table") or is_meta_line(b["text"])
            or (len(b["text"]) < META_MAX_CHARS and COPYRIGHT_RX.search(b["text"]))
            or PAGE_NUM_RX.match(b["text"]))


def continuation_target(blocks, max_back=4):
    """Last body paragraph a lowercase-starting text could continue, looking back over
    footnotes, captions, tables and article meta lines (never across a heading)."""
    for b in reversed(blocks[-max_back:]):
        if b["type"] == "heading":
            return None
        if b["_label"] == "text" and not is_interruption(b):
            return b
        if not is_interruption(b):
            return None
    return None


def is_skip_heading(text):
    nh = norm_heading(text)
    return any(rx.match(nh) for _, rx in SKIP_SECTIONS)


def numbering(text):
    m = NUMBERED_RX.match(text)
    return [int(x) for x in m.group(1).split(".")] if m else None


def find_chapter_starts(blocks):
    """Indices (into blocks) of book chapter-title headings. Three numbering styles:

    - "Chapter N" headings are always chapter starts.
    - Depth-1 numbers that never restart ("1 Intro", "1.1", "2 Fairness", "2.1"):
      each depth-1 heading is a chapter.
    - Depth-1 numbers that restart ("... 8 Conclusion", "1 Introduction"): an
      edited volume; a chapter starts at each restart.
    - Otherwise ("X.1", "X.2" under unnumbered chapter titles): a chapter starts at each new X.
    For the last two, the chapter title is the nearest unnumbered, non-generic
    heading shortly before the restart (stepping over "Abstract", "Keywords").
    """
    heads = [(i, numbering(b["text"])) for i, b in enumerate(blocks) if b["type"] == "heading"]
    d1 = [n[0] for _, n in heads if n and len(n) == 1]
    restarts = any(b < a for a, b in zip(d1, d1[1:]))
    # "X Title" style only if most "X.1" headings follow a depth-1 "X" heading.
    x1, parented, last_d1 = 0, 0, None
    for _, n in heads:
        if n and len(n) == 1:
            last_d1 = n[0]
        elif n and len(n) == 2 and n[1] == 1:
            x1 += 1
            parented += last_d1 == n[0]
    xtitle_style = bool(d1) and not restarts and x1 > 0 and parented / x1 >= 0.5
    starts, prev_d1, prev_lead2 = set(), None, None

    def walk_back(i):
        for steps, j in enumerate(range(i - 1, -1, -1), 1):
            if steps > 15:
                break
            if j in starts:
                return j                  # already the chapter's title
            bj = blocks[j]
            if bj["type"] != "heading":
                continue
            if numbering(bj["text"]) or is_skip_heading(bj["text"]):
                break
            if not GENERIC_HEADING_RX.match(norm_heading(bj["text"])):
                return j
        return i

    for i, num in heads:
        if CHAPTER_RX.match(blocks[i]["text"]):
            starts.add(i)
            continue
        if not num:
            continue
        if len(num) == 1:
            if xtitle_style:
                starts.add(i)
            elif restarts and (prev_d1 is None or num[0] < prev_d1):
                starts.add(walk_back(i))
            prev_d1 = num[0]
        elif (not xtitle_style and not restarts and len(num) == 2 and num[1] == 1
              and num[0] != prev_lead2):
            starts.add(walk_back(i))
        if len(num) >= 2:
            prev_lead2 = num[0]
    return starts


def heading_levels(blocks, is_chapter_file, is_book=False):
    """Docling gives every heading level 1; infer depth from numbering, chapters and style."""
    heads = [b for b in blocks if b["type"] == "heading"]
    has_numbered = any(numbering(b["text"]) for b in heads)
    has_caps = any(len(re.sub(r"[^A-Za-z]", "", b["text"])) >= 4 and b["text"].isupper()
                   and not re.match(r"^(\w )+\w$", b["text"]) for b in heads)
    chapters = find_chapter_starts(blocks) if is_book and has_numbered else set()
    if is_chapter_file and heads:
        chapters.add(blocks.index(heads[0]))
    has_chapter = bool(chapters) or any(CHAPTER_RX.match(b["text"]) for b in heads)
    base = 1 if has_chapter else 0
    offset, last_num_level = base, None
    for i, b in enumerate(blocks):
        if b["type"] != "heading":
            continue
        t = b["text"]
        num = numbering(t)
        if i in chapters or CHAPTER_RX.match(t):
            b["level"] = 1
            # sections under a numbered "X Title" chapter are X.1, X.2 -> depth 2 = level 2
            offset = 0 if (num and len(num) == 1) else base
            last_num_level = None
        elif num:
            b["level"] = max(offset + len(num), base + 1)
            last_num_level = b["level"]
        elif is_skip_heading(t) and has_numbered:
            b["level"] = base + 1   # References, Funding, ... sit at the top of their chapter/article
        elif last_num_level is not None:
            b["level"] = last_num_level + 1   # unnumbered sub-heading inside a numbered section
        elif t.isupper() and len(re.sub(r"[^A-Za-z]", "", t)) >= 4:
            b["level"] = base + 1
        elif has_numbered:
            b["level"] = base + 1   # e.g. "ABSTRACT" before "1. Introduction"
        else:
            b["level"] = base + 1 + has_caps


def clean(doc, row, pcfg, n_pages):
    return clean_items(raw_items(doc), row, pcfg, n_pages)


def clean_items(items, row, pcfg, n_pages):
    """Clean a list of {label, text, page} items (Docling-independent, for tests)."""
    items, headers = drop_running_headers(items, n_pages, pcfg["header_min_pages"])

    # Merge "Chapter 3" + following heading into one heading.
    merged = []
    for it in items:
        if (merged and merged[-1]["label"] in ("section_header", "title")
                and it["label"] in ("section_header", "title")
                and re.fullmatch(CHAPTER_RX.pattern, merged[-1]["text"].strip(), re.I)):
            merged[-1]["text"] = f"{merged[-1]['text'].strip()}: {it['text'].strip()}"
            continue
        merged.append(dict(it))

    vocab = doc_vocab(it["text"] for it in merged)
    blocks = []
    for it in merged:
        typ = LABEL_TYPE[it["label"]]
        text = it["text"].strip() if typ == "table" else fix_text(it["text"], vocab)
        if typ == "heading":
            text = unspace(text)
            letters = len(re.sub(r"[^A-Za-z]", "", text))
            if re.match(r"^(?i:table|figure|fig\.)\s*(\d+|[A-Z]\d*)(\.\d+)*\b", text):
                typ = "caption"     # "Table 4 (continued)" mislabelled as a heading
            elif (letters == 0 or text.startswith("[") or re.fullmatch(r"\(.*\d{4}[a-z]?\)\.?", text) or re.search(r"https?://|www\.|doi\.org", text)
                    or (letters < 2 and not re.fullmatch(r"[A-Z]", text) and not NUMBERED_RX.match(text))):
                typ = "paragraph"   # citation fragments / symbols mislabelled as headings
        elif row["folder"] == "article" and (it["page"] or 99) <= 2 and len(text) >= META_MAX_CHARS:
            # Docling sometimes glues the DOI footer onto a body paragraph.
            text = re.sub(r"\s*https?://doi\.org/\S+\s*", " ", text).strip()
        prev = continuation_target(blocks) if it["label"] == "text" else None
        # Re-join paragraphs split across columns/pages.
        if (prev and not re.search(r"[.!?:;\"”)\]]$", prev["text"]) and re.match(r"[a-z]", text)):
            sep = "" if prev["text"].endswith("-") else " "
            prev["text"] = (prev["text"][:-1] if sep == "" else prev["text"]) + sep + text
            prev["page_end"] = it["page"]
            continue
        blocks.append({"type": typ, "text": text, "page": it["page"], "page_end": it["page"],
                       "_label": it["label"]})

    heading_levels(blocks, row["in_bulk_download"] == "yes" and "chapter" in row["path"].lower(),
                   is_book=row["folder"] == "book")

    is_article = row["folder"] == "article"
    has_abstract = is_article and any(
        b["type"] == "heading" and norm_heading(b["text"]) in ("abstract", "a b s t r a c t")
        for b in blocks)
    seen_abstract = False
    # Books: the copyright page and everything before it (half-title, title pages) is front matter.
    copyright_pages = sorted({b["page"] for b in blocks if b["page"] and b["type"] != "heading"
                              and b["page"] <= max(15, n_pages * 0.05) and COPYRIGHT_RX.search(b["text"])})
    front_until = copyright_pages[0] if (copyright_pages and not is_article) else 0
    stack, skip_level, skip_reason = [], None, None
    late_page = n_pages * 0.9
    for i, b in enumerate(blocks):
        reason = None
        if b["type"] == "heading":
            nh = norm_heading(b["text"])
            lvl = b["level"]
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, b["text"]))
            if nh in ("abstract", "a b s t r a c t"):
                seen_abstract = True
            match = next((r for r, rx in SKIP_SECTIONS if rx.match(nh)), None)
            if match:
                skip_level, skip_reason = lvl, match
            elif skip_level is not None and len(nh) > 1 and (
                    skip_reason in ("references", "back_matter", "front_matter")
                    or lvl <= skip_level or lvl == 1 or numbering(b["text"])):
                # References/back/front matter end at the next non-skip heading (appendices,
                # next chapter); TOC and index regions, whose entries can look like headings,
                # end at a heading of the same or higher level, a numbered heading or a
                # chapter. Single-letter index headings ("A", "B") never end a region.
                skip_level = skip_reason = None
        b["section_path"] = [t for _, t in stack]

        if (skip_level is not None and skip_reason in ("toc", "index") and b["type"] != "heading"
                and len(b["text"].split()) >= 40 and re.search(r"[.!?]\s", b["text"])
                and number_ratio(b["text"]) < 0.1):
            skip_level = skip_reason = None   # prose: TOC/index entries are never this long
        if skip_level is not None:
            reason = skip_reason
        elif b["_label"] in LABEL_SKIP:
            reason = LABEL_SKIP[b["_label"]]
        elif b["type"] != "heading":
            t = b["text"]
            if b["page"] in copyright_pages:
                reason = "copyright"
            elif b["page"] and b["page"] < front_until:
                reason = "front_matter"
            elif is_article and b["page"] and b["page"] <= 2 and is_meta_line(t):
                reason = "front_matter"
            elif is_article and has_abstract and not seen_abstract and b["page"] == 1:
                reason = "front_matter"
            elif b["page"] and b["page"] <= 30 and TOC_LINE_RX.search(t):
                reason = "toc"
            elif b["page"] and b["page"] >= late_page and n_pages > 50 and (
                    INDEX_LINE_RX.match(t)
                    or (len(INDEX_RUN_RX.findall(t)) >= 3 and number_ratio(t) >= 0.2)):
                reason = "index"   # merged index columns: "privacy 7, 12, 45-47 pre-processing 129"
        elif is_article and has_abstract and not seen_abstract and b["page"] == 1:
            reason = "front_matter"   # journal name / title / "ARTICLE INFO" headings
        elif b["page"] and b["page"] < front_until:
            reason = "front_matter"   # half-title / title page headings
        b["skip"] = reason is not None
        b["skip_reason"] = reason or ""

    out = []
    for n, b in enumerate(blocks):
        out.append({"block_id": f"b{n:05d}", "type": b["type"], "text": b["text"],
                    "section_path": b["section_path"], "page": b["page"],
                    "page_end": b["page_end"], "skip": b["skip"], "skip_reason": b["skip_reason"]})
    return out, headers


# --- metadata from first pages ---------------------------------------------
def clean_article_authors(t):
    """'Tomás Matos a,1 , Walter Santos b,1' -> 'Tomás Matos; Walter Santos'."""
    t = re.sub(r"[*†‡§¶|✉]", " ", t)
    # affiliation markers after a surname: " a", " b,1", " a, ," -> separator
    t = re.sub(r"(?<=[a-zà-ÿ.])\s+[a-h](?:\s*,\s*\d+)*(?=\s*(?:,|$|\band\b|[A-Z]))", ",", t)
    t = re.sub(r"(?<=[a-zà-ÿ])\d+(?:,\d+)*", "", t)
    names = []
    for seg in re.split(r",|;|\band\b|&", t):
        seg = re.sub(r"\s+\d+$", "", re.sub(r"\s+", " ", seg)).strip(" .")
        if len(re.findall(r"\b[A-ZÀ-Þ]", seg)) >= 2 and len(seg.split()) <= 5:
            names.append(seg)
    return "; ".join(names)


NON_NAME_WORDS = {
    "the", "of", "for", "in", "on", "using", "with", "to", "a", "an", "series", "technology",
    "technologies", "science", "sciences", "education", "educational", "communication", "learning",
    "intelligence", "artificial", "ai", "ethical", "responsible", "implement", "creating", "signals",
    "systems", "data", "studies", "postdigital", "management", "handbook", "springer", "press",
    "university", "lecture", "notes", "advances", "computing", "trustworthy", "distributed",
    "ledger", "python", "foreword", "preface", "editor", "editors", "contents", "volume",
}


def clean_book_authors(t):
    """Keep only personal-name segments; drop locations/affiliations. '' if not names."""
    t = re.sub(r"\s+[A-Z][a-z]+,\s*(?:UK|USA|[A-Z]{2})\b.*$", "", t)   # "London, UK"
    parts = []
    for seg in re.split(r"\s*[·;]\s*|\s+and\s+|\s*&\s*|,\s*", t):
        if not seg:
            continue
        if AFFILIATION_RX.search(seg) or re.search(r"\b(UK|USA|[A-Z]{2})$", seg):
            break
        words = re.findall(r"[A-Za-z'’.-]+", seg)
        if any(w.lower().strip(".") in NON_NAME_WORDS for w in words):
            return ""
        parts.append(seg.strip())
    return " · ".join(parts)


def extract_metadata(blocks, raw, row, pcfg):
    max_page = pcfg["article_meta_pages"] if row["folder"] == "article" else pcfg["book_meta_pages"]
    first = [b for b in raw if b["page"] and b["page"] <= max_page]
    text = "\n".join(b["text"] for b in first)

    title = ""
    titles = [b["text"] for b in first if b["label"] == "title"]
    if titles:
        title = titles[0]
    elif row["folder"] == "article":
        cands = [b["text"] for b in first if b["label"] == "section_header" and b["page"] == 1
                 and len(b["text"]) >= 20 and not JOURNAL_RX.search(b["text"])
                 and not GENERIC_HEADING_RX.match(norm_heading(b["text"]))]
        title = max(cands, key=len) if cands else ""
    title = fix_text(unspace(title)) if title else ""

    authors = ""
    if row["folder"] == "article" and title:
        idx = next((i for i, b in enumerate(first) if fix_text(unspace(b["text"])) == title), None)
        for b in first[idx + 1: idx + 4] if idx is not None else []:
            t = b["text"].strip()
            if b["label"] == "text" and len(t) < 400 and not AFFILIATION_RX.search(t) \
                    and len(re.findall(r"\b[A-Z][a-z]+", t)) >= 2 and not t.endswith("."):
                authors = clean_article_authors(fix_text(unspace(t)).replace(title, ""))
                break
    else:
        m = re.search(r"(?:^|\n)\s*(?:edited by|editors?)\s*:?\s*\n?\s*([A-Z][^\n]{3,150})", text, re.I)
        if m and NAMES_RX.match(m.group(1).strip()):
            authors = clean_book_authors(m.group(1).strip())
        else:
            # Title-page line made only of personal names, e.g. "Sray Agarwal · Shashin Mishra".
            for b in first:
                t = b["text"].strip()
                if (b["page"] <= 3 and b["label"] == "text" and NAMES_RX.match(t)
                        and fix_text(unspace(t)) != title and len(t.split()) >= 2):
                    authors = clean_book_authors(t)
                    if authors:
                        break

    year, year_kind = "", ""
    cblocks = " ".join(b["text"] for b in first if COPYRIGHT_RX.search(b["text"]) or "©" in b["text"])
    cyears = [int(y) for y in re.findall(r"(?<![\d-])((?:19|20)\d{2})(?![\d-])", cblocks)
              if 1950 <= int(y) <= 2035]
    for rx, kind in ((FIRST_PUBLISHED, "first_published"),
                     (re.compile(r"(?!x)x"), "copyright_block"),   # placeholder, handled below
                     (COPYRIGHT_YEAR, "copyright"),
                     (JOURNAL_CITATION_YEAR, "journal_citation"), (ARTICLE_YEAR, "article_dates")):
        years = cyears if kind == "copyright_block" else \
            [int(y) for y in rx.findall(text) if 1950 <= int(y) <= 2035]
        if years:
            year, year_kind = str(min(years) if kind == "first_published" else max(years)), kind
            break
    return {"title": title, "authors": authors, "year": year, "year_kind": year_kind}


def build_metadata(inv_rows, parsed, pcfg):
    """Combine inventory metadata with first-page extraction; flag uncertain rows."""
    group_titles = {}
    by_group = defaultdict(list)
    for r in inv_rows:
        by_group[r["doc_group"]].append(r["doc_id"])
    for g, ids in by_group.items():
        if len(ids) < 2:
            continue
        counts = Counter()
        for d in ids:
            for h in (parsed.get(d, {}).get("running_headers") or {}):
                counts[h] += 1
        common = [h for h, c in counts.most_common() if c >= min(2, len(ids)) and len(h) > 3]
        group_titles[g] = common[0] if common else ""

    rows, review = [], []
    for r in inv_rows:
        p = parsed.get(r["doc_id"])
        if not p:
            continue
        ex = p["extracted"]
        flags = []
        title, title_src = r["title"], r["title_source"]
        if r["doc_group"] in group_titles:
            book = group_titles[r["doc_group"]] or "UNKNOWN BOOK"
            title, title_src = f"{book}: {clean_stem(Path(r['path']).stem)}", "doc_group"
            if not group_titles[r["doc_group"]]:
                flags.append("group_title_unknown")
        elif title_src == "filename" and ex["title"]:
            title, title_src = ex["title"], "text"
            flags.append("title_from_text_heuristic")
        if title_src == "filename":
            flags.append("title_from_filename")

        authors, author_src = r["author"], "metadata" if r["author"] else ""
        if not authors and ex["authors"]:
            authors, author_src = ex["authors"], "text"
            flags.append("authors_from_text_heuristic")
        if not authors:
            flags.append("authors_missing")

        year, year_src = r["year"], r["year_source"]
        if year_src == "creation_date" and ex["year"]:
            year, year_src = ex["year"], "text"
        elif year_src == "text" and ex["year"] and abs(int(ex["year"]) - int(year)) > 1:
            flags.append(f"year_conflict({year} vs {ex['year']})")
        if year_src == "creation_date":
            flags.append("year_from_creation_date")
        if year and int(year) > pcfg["max_year"]:
            flags.append("future_year")

        out = {"doc_id": r["doc_id"], "folder": r["folder"], "title": title, "authors": authors,
               "year": year, "title_source": title_src, "author_source": author_src,
               "year_source": year_src, "flags": ";".join(flags)}
        rows.append(out)
        if flags:
            review.append({**out, "path": r["path"], "human_title": "", "human_authors": "",
                           "human_year": ""})
    return rows, review


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, help="only process the first N documents")
    ap.add_argument("--doc", nargs="+", help="only these doc_ids")
    ap.add_argument("--reclean", action="store_true",
                    help="redo cleaning for all docs from cached Docling output")
    ap.add_argument("--force", action="store_true", help="re-run Docling even if cached")
    ap.add_argument("--seed", type=int, default=0, help="seed for report samples")
    args = ap.parse_args()

    cfg = load_config()
    pcfg = cfg["parse"]
    log = get_logger("parse", cfg)
    root = repo_path(cfg["paths"]["raw_pdfs"])
    out_dir = repo_path(cfg["paths"]["parsed"])
    cache_dir = repo_path(cfg["paths"]["docling_cache"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with repo_path(cfg["paths"]["inventory"]).open(encoding="utf-8") as f:
        inv = list(csv.DictReader(f))
    for r in inv:
        r["in_bulk_download"] = "yes" if "bulk-download" in r["path"] else "no"
    pages_of = {r["doc_id"]: int(r["pages"] or 0) for r in inv}

    todo = []
    for r in inv:
        if r["status"] != "ok":
            log.warning(f"skip corrupt: {r['doc_id']}")
            continue
        # Of a duplicate set, parse only the largest file.
        others = [o for o in r["duplicate_of"].split(";") if o]
        if any((pages_of.get(o, 0), o) > (pages_of[r["doc_id"]], r["doc_id"]) for o in others):
            log.info(f"skip duplicate: {r['doc_id']} (kept {r['duplicate_of']})")
            continue
        todo.append(r)
    if args.doc:
        todo = [r for r in todo if r["doc_id"] in set(args.doc)]
    if args.limit:
        todo = todo[:args.limit]

    stats = Counter()
    failures, convert_times = [], []
    for i, r in enumerate(todo, 1):
        out_file = out_dir / f"{r['doc_id']}.json"
        if out_file.exists() and not (args.reclean or args.force):
            stats["resumed"] += 1
            continue
        try:
            doc, secs = load_or_convert(r, root / r["path"], cache_dir / f"{r['doc_id']}.json",
                                        pcfg, args.force, log)
            n_pages = len(doc.pages) or int(r["pages"])
            if secs is None:
                stats["from_docling_cache"] += 1
            else:
                stats["converted"] += 1
                convert_times.append((n_pages, secs))
                log.info(f"[{i}/{len(todo)}] {r['doc_id']}: {n_pages} pages in {secs:.1f}s")
            raw = raw_items(doc)
            blocks, headers = clean(doc, r, pcfg, n_pages)
            payload = {
                "doc_id": r["doc_id"], "path": r["path"], "folder": r["folder"],
                "doc_group": r["doc_group"], "n_pages": n_pages, "parser_version": PARSER_VERSION,
                "parse_seconds": round(secs, 1) if secs else None,
                "extracted": extract_metadata(blocks, raw, r, pcfg),
                "running_headers": dict(headers.most_common(10)),
                "blocks": blocks,
            }
            tmp = out_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(out_file)
        except Exception as e:
            log.exception(f"FAILED {r['doc_id']}: {e}")
            failures.append((r["doc_id"], f"{type(e).__name__}: {e}"))

    # Load every parsed doc (new + resumed) for metadata and the report.
    parsed = {}
    for r in inv:
        f = out_dir / f"{r['doc_id']}.json"
        if f.exists():
            parsed[r["doc_id"]] = json.loads(f.read_text(encoding="utf-8"))
    meta_rows, review = build_metadata(inv, parsed, pcfg)
    write_csv(out_dir / "metadata.csv", meta_rows)
    write_csv(out_dir / "metadata_review.csv", review)

    report(parsed, inv, todo, stats, failures, convert_times, review, cfg, args.seed)


def report(parsed, inv, todo, stats, failures, convert_times, review, cfg, seed):
    rep = Report("parse", cfg)
    rep(f"=== Phase 2 parse: {len(parsed)}/{len(inv)} docs parsed ===")
    rep(f"this run: {len(todo)} selected, {stats['converted']} converted, "
        f"{stats['from_docling_cache']} re-cleaned from Docling cache, {stats['resumed']} already done")
    if convert_times:
        pages = sum(p for p, _ in convert_times)
        secs = sum(s for _, s in convert_times)
        rep(f"Docling time: {secs:.0f}s for {pages:,} pages ({secs / max(pages, 1):.2f} s/page)")

    rep(f"\n-- Failures: {len(failures)} --")
    for d, e in failures:
        rep(f"  {d}: {e}")

    per_doc, skip_totals, type_totals = [], Counter(), Counter()
    for d, p in parsed.items():
        bl = p["blocks"]
        kept = [b for b in bl if not b["skip"]]
        skip_totals.update(b["skip_reason"] for b in bl if b["skip"])
        type_totals.update(b["type"] for b in kept)
        per_doc.append({
            "doc_id": d, "folder": p["folder"], "pages": p["n_pages"], "blocks": len(bl),
            "kept": len(kept), "skipped": len(bl) - len(kept),
            "headings": sum(b["type"] == "heading" for b in kept),
            "tables": sum(b["type"] == "table" for b in kept),
            "kept_words": sum(len(b["text"].split()) for b in kept),
        })
    write_csv(repo_path(cfg["paths"]["logs"]) / "parse_blocks.csv", per_doc)

    total_blocks = sum(x["blocks"] for x in per_doc)
    rep(f"\n-- Blocks: {total_blocks:,} total, {sum(x['kept'] for x in per_doc):,} kept, "
        f"{sum(x['kept_words'] for x in per_doc):,} kept words --")
    rep(f"kept by type: {dict(type_totals)}")
    rep(f"skipped by reason: {dict(skip_totals)}")

    rep("\n-- Per-doc block counts (full table: logs/parse_blocks.csv) --")
    rep(f"{'doc_id':46s} {'pages':>5} {'blocks':>6} {'kept':>5} {'skip':>5} {'head':>4} {'tbl':>3} {'words':>7}")
    for x in sorted(per_doc, key=lambda x: (x["folder"], x["doc_id"])):
        rep(f"{x['doc_id'][:46]:46s} {x['pages']:>5} {x['blocks']:>6} {x['kept']:>5} "
            f"{x['skipped']:>5} {x['headings']:>4} {x['tables']:>3} {x['kept_words']:>7}")
    odd = [x for x in per_doc if x["pages"] and x["kept_words"] / x["pages"] < 100]
    if odd:
        rep(f"\nLow text yield (<100 kept words/page), check these: "
            f"{', '.join(x['doc_id'] for x in odd)}")

    rep(f"\n-- Metadata: {len(review)} docs flagged in data/parsed/metadata_review.csv --")
    flag_counts = Counter(f.split("(")[0] for r in review for f in r["flags"].split(";") if f)
    rep(f"flags: {dict(flag_counts)}")

    rng = random.Random(seed)
    rep("\n-- 3 random samples (6 consecutive kept blocks each) --")
    for d in rng.sample(sorted(parsed), min(3, len(parsed))):
        kept = [b for b in parsed[d]["blocks"] if not b["skip"]]
        if not kept:
            continue
        start = rng.randrange(max(len(kept) - 6, 1))
        rep(f"\n## {d}")
        for b in kept[start:start + 6]:
            text = b["text"] if len(b["text"]) <= 300 else b["text"][:300] + " …"
            rep(f"  [{b['type']} p{b['page']}] {' > '.join(b['section_path'])}")
            rep(f"     {text}")
    rep.save()


if __name__ == "__main__":
    main()
