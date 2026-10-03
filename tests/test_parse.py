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


def path_of(blocks, text):
    return next(b["section_path"] for b in blocks if b["text"] == text)


def test_edited_volume_chapters_from_numbering_restart():
    """Contributed chapters each number sections from 1; the unnumbered title before them is the chapter."""
    items = []
    for ch, title in ((1, "Digital Universities in Jordan"), (2, "AI Policy in the Gulf")):
        p = ch * 20
        items += [item("section_header", title, p),
                  item("text", f"Author Name {ch}", p),
                  item("section_header", "Abstract", p),
                  item("text", f"Abstract text {ch}.", p),
                  item("section_header", "1 Introduction", p + 1),
                  item("text", f"Intro text {ch}.", p + 1),
                  item("section_header", "2 Methods", p + 2),
                  item("text", f"Methods text {ch}.", p + 2),
                  item("section_header", "References", p + 5),
                  item("list_item", f"Ref {ch}.", p + 5)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=60)
    assert path_of(blocks, "Methods text 2.") == ["AI Policy in the Gulf", "2 Methods"]
    assert path_of(blocks, "Abstract text 2.") == ["AI Policy in the Gulf", "Abstract"]
    assert path_of(blocks, "Intro text 1.") == ["Digital Universities in Jordan", "1 Introduction"]


def test_monograph_numbered_chapters_and_unnumbered_subheadings():
    items = [item("section_header", "Generative AI and Higher Education", 20),
             item("section_header", "1.1 Introduction", 20),
             item("text", "Intro.", 20),
             item("section_header", "1.2 Knowledge Production", 22),
             item("section_header", "Prompt 1.1: Knowledge creator", 23),
             item("text", "Prompt text.", 23),
             item("section_header", "Lesson Preparation", 45),
             item("section_header", "2.1 Introduction", 45),
             item("text", "Chapter two intro.", 45)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=100)
    assert path_of(blocks, "Prompt text.") == [
        "Generative AI and Higher Education", "1.2 Knowledge Production", "Prompt 1.1: Knowledge creator"]
    assert path_of(blocks, "Chapter two intro.") == ["Lesson Preparation", "2.1 Introduction"]


def test_numbered_x_title_chapters():
    items = [item("section_header", "1 Introduction", 10), item("text", "One.", 10),
             item("section_header", "1.1 Background", 11), item("text", "One one.", 11),
             item("section_header", "2 Fairness", 20), item("text", "Two.", 20),
             item("section_header", "2.1 Proxy Features", 21), item("text", "Two one.", 21)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=40)
    assert path_of(blocks, "Two one.") == ["2 Fairness", "2.1 Proxy Features"]


def test_table_continued_heading_demoted_and_toc_heading_kept():
    items = [item("section_header", "Table of Contents", 3), item("text", "1 Intro .......... 5", 3),
             item("section_header", "1. Introduction", 5), item("text", "Body.", 5),
             item("section_header", "Table 4 (continued)", 6), item("text", "More body.", 6)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=40)
    assert next(b for b in blocks if b["text"] == "Table 4 (continued)")["type"] == "caption"
    assert next(b for b in blocks if b["text"] == "Table of Contents")["skip_reason"] == "toc"
    assert path_of(blocks, "More body.") == ["1. Introduction"]


def test_article_sections_stay_top_level():
    art = {"folder": "article", "path": "Research Articles_AI/JA_x.pdf", "in_bulk_download": "no"}
    items = [item("section_header", "ABSTRACT", 1), item("text", "Abstract body.", 1),
             item("section_header", "1. Introduction", 1), item("text", "Intro body.", 1),
             item("section_header", "1.1. Scope", 2), item("text", "Scope body.", 2),
             item("section_header", "2. Methods", 3), item("text", "Methods body.", 3)]
    blocks, _ = clean_items(items, art, PCFG, n_pages=10)
    assert path_of(blocks, "Abstract body.") == ["ABSTRACT"]
    assert path_of(blocks, "Scope body.") == ["1. Introduction", "1.1. Scope"]
    assert path_of(blocks, "Methods body.") == ["2. Methods"]


def test_back_matter_ends_at_appendix_and_references_end_at_next_chapter():
    art = {"folder": "article", "path": "Research Articles_AI/JA_x.pdf", "in_bulk_download": "no"}
    items = [item("section_header", "1. Introduction", 1), item("text", "Intro body.", 1),
             item("section_header", "4.2. Findings detail", 5), item("text", "Findings.", 5),
             item("section_header", "Acknowledgements", 9), item("text", "We thank X.", 9),
             item("section_header", "Appendix A. Concept Matrix", 10), item("text", "Matrix rows.", 10),
             item("section_header", "(Smith, 2024).", 10), item("text", "More matrix.", 10)]
    blocks, _ = clean_items(items, art, PCFG, n_pages=12)
    reasons = {b["text"]: b["skip_reason"] for b in blocks}
    assert reasons["We thank X."] == "back_matter"
    assert reasons["Matrix rows."] == "" and reasons["More matrix."] == ""
    assert next(b for b in blocks if b["text"] == "(Smith, 2024).")["type"] == "paragraph"

    # unnumbered chapter titles after a book's References are not swallowed
    items = [item("section_header", "Introduction", 10), item("text", "Book intro.", 10),
             item("section_header", "4 Introduction", 13), item("text", "Stray numbered.", 13),
             item("section_header", "References", 22), item("list_item", "Ref A.", 22),
             item("section_header", "Capitalist Universities, AI, and Value 2", 23),
             item("text", "Chapter two body.", 23)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=80)
    reasons = {b["text"]: b["skip_reason"] for b in blocks}
    assert reasons["Ref A."] == "references" and reasons["Chapter two body."] == ""


def test_toc_region_ends_at_prose_in_caps_styled_book():
    items = [item("section_header", "CONTENTS", 5), item("text", "The Rise of AI 1", 5),
             item("section_header", "List of Tables", 15), item("text", "Table 1.1 Adoption 12", 15),
             item("section_header", "The Rise of Artificial Intelligence in Latin America", 16),
             item("text", "However, this social trend about AI appeared in an era of rapid change, "
                          "when governments across the region began to adopt national strategies. "
                          "These strategies set out goals for research, talent and ethics, and they "
                          "were often modelled on European and North American examples.", 17)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=200)
    reasons = {b["text"][:20]: b["skip_reason"] for b in blocks}
    assert reasons["Table 1.1 Adoption 1"] == "toc"
    assert reasons["However, this social"] == ""


def test_merged_index_columns_tagged():
    filler = [item("text", f"Body paragraph {i} about responsible AI in universities.", i) for i in range(1, 90)]
    idx = ("Asimov, I. 174 assessment 10, 29-31, 36, 64-65, 69-72, 87-107 Google 44, 135, 177, "
           "181-182, 184, 187 privacy 7, 12, 45-47, 101 reflection 20, 41, 74-75, 79-80, 84")
    items = filler + [item("section_header", "Index", 95), item("text", idx, 95)]
    blocks, _ = clean_items(items, BOOK, PCFG, n_pages=100)
    assert next(b for b in blocks if b["text"].startswith("Asimov"))["skip_reason"] == "index"
