"""Regression tests for the Phase 2 cleaning rules (src/parse.py)."""

from parse import clean_items, doc_vocab, extract_metadata, fix_text

PCFG = {"header_min_pages": 3, "article_meta_pages": 2, "book_meta_pages": 8}
BOOK = {"folder": "book", "path": "Books_AI/x.pdf", "in_bulk_download": "no"}


def item(label, text, page):
    return {"label": label, "text": text, "page": page}


def by_text(blocks, text):
    return [b for b in blocks if b["text"] == text]


def test_em_dash_preserved():
    assert fix_text("Material constraints—disparities between students") == \
        "Material constraints—disparities between students"


def test_line_break_hyphen_joined_but_real_compound_kept():
    texts = ["information is shared", "Material constraints- disparities between students",
             "constraints matter", "disparities persist", "infor- mation flows"]
    vocab = doc_vocab(texts)
    # "mation" is not a word in the document -> line-break hyphenation, join it
    assert fix_text("infor- mation flows", vocab) == "information flows"
    # both halves are real words and the joined form is not -> keep as a hyphenated pair
    assert fix_text("Material constraints- disparities between", vocab) == \
        "Material constraints-disparities between"
    # coordinated compounds are never joined
    assert fix_text("short- and long-term", vocab) == "short- and long-term"


def test_compound_prefix_and_dash_cases():
    vocab = doc_vocab(["identity matters", "implicit rules", "that is"])
    assert fix_text("self- identity", vocab) == "self-identity"
    assert fix_text("is implicit- that is", vocab) == "is implicit – that is"


def test_soft_hyphen_removed():
    assert fix_text("profes­ sional development") == "professional development"


def test_repeated_heading_kept():
    """A heading repeated on many pages (e.g. per-chapter 'Bibliography') is not a running header."""
    items = []
    for ch in range(1, 6):
        p = ch * 10
        items += [item("section_header", f"Chapter {ch} Topic {ch}", p),
                  item("text", f"Body text of chapter {ch}.", p),
                  item("section_header", "Bibliography", p + 5),
                  item("list_item", f"Author {ch} (2020) Some reference.", p + 5)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=60)
    assert len([b for b in blocks if b["type"] == "heading" and b["text"] == "Bibliography"]) == 5


def test_springer_per_chapter_bibliography_tagged():
    # 6 chapters, so "Bibliography" repeats on enough pages to look like a running header
    items = []
    for ch in range(1, 7):
        p = ch * 20
        items += [item("section_header", f"Chapter {ch}", p),
                  item("section_header", f"Topic number {ch}", p),
                  item("text", f"Substantive content of chapter {ch}.", p + 1),
                  item("section_header", "Bibliography", p + 9),
                  item("list_item", f"Smith, J. ({2000 + ch}) Title of work.", p + 9),
                  item("list_item", f"Doe, A. ({2010 + ch}) Another work.", p + 10)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=130)
    refs = [b for b in blocks if b["type"] == "list"]
    assert len(refs) == 12 and all(b["skip"] and b["skip_reason"] == "references" for b in refs)
    body = [b for b in blocks if b["text"].startswith("Substantive content")]
    assert len(body) == 6 and not any(b["skip"] for b in body)
    # chapter number merged with its title and used as the top of section_path
    assert body[1]["section_path"][0] == "Chapter 2: Topic number 2"


def test_front_matter_before_copyright_page_skipped():
    items = [item("text", "Sray Agarwal Shashin Mishra", 1),
             item("section_header", "Responsible AI", 1),
             item("text", "Implementing Ethical and Unbiased Algorithms", 3),
             item("text", "ISBN 978-3-030-76977-2 © Springer Nature 2021. All rights reserved.", 4),
             item("text", "The use of general descriptive names does not imply exemption.", 4),
             item("section_header", "Preface", 5),
             item("text", "This book is about fairness.", 5)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=200)
    reasons = {b["text"]: b["skip_reason"] for b in blocks}
    assert reasons["Sray Agarwal Shashin Mishra"] == "front_matter"
    assert reasons["Responsible AI"] == "front_matter"
    assert reasons["Implementing Ethical and Unbiased Algorithms"] == "front_matter"
    assert reasons["The use of general descriptive names does not imply exemption."] == "copyright"
    assert reasons["This book is about fairness."] == ""


def test_names_only_author_line_detected():
    raw = [item("text", "Sray Agarwal · Shashin Mishra", 3),
           item("section_header", "Responsible AI", 3),
           item("text", "Implementing Ethical and Unbiased Algorithms", 3)]
    meta = extract_metadata([], raw, BOOK, PCFG)
    assert meta["authors"] == "Sray Agarwal · Shashin Mishra"


def test_subtitle_not_mistaken_for_authors():
    raw = [item("text", "Implement Ethical AI Using Python", 2)]
    assert extract_metadata([], raw, BOOK, PCFG)["authors"] == ""
