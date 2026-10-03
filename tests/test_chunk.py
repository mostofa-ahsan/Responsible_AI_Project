"""Tests for Phase 3 chunking guarantees (src/chunk.py), with a word-count tokenizer."""

from chunk import build_chunks

CCFG = {"min_tokens": 50, "max_tokens": 100, "hard_max_tokens": 130, "overlap_tokens": 12,
        "table_alone_tokens": 40}


class WordTok:
    def count(self, texts):
        return [len(t.split()) for t in texts]


def para(n_words, tag, section, i):
    text = " ".join(f"{tag}{k}" for k in range(n_words - 1)) + " end."
    return {"block_id": f"b{i:05d}", "type": "paragraph", "text": text, "section_path": section,
            "page": 1, "page_end": 1, "skip": False, "skip_reason": ""}


def doc(blocks):
    return {"doc_id": "d", "folder": "book", "doc_group": "d", "blocks": blocks}


def test_never_splits_paragraphs_or_crosses_top_level_sections():
    blocks, i = [], 0
    for sec in ("Chapter 1", "Chapter 2"):
        for k in range(6):
            blocks.append(para(35, f"{sec[-1]}p{k}w", [sec, f"{sec} s"], i))
            i += 1
    chunks = build_chunks(doc(blocks), {"title": "T"}, WordTok(), CCFG)
    paras = {b["text"] for b in blocks}
    for c in chunks:
        assert {b["section_path"][0] for b in blocks if b["block_id"] in c["block_ids"]} in (
            {"Chapter 1"}, {"Chapter 2"})
        body = c["text"].split("\n\n")
        # every body piece except a leading overlap is a whole paragraph
        assert all(piece in paras for piece in body[1 if c["overlap_tokens"] else 0:])
        assert c["n_tokens"] <= CCFG["hard_max_tokens"] + CCFG["overlap_tokens"]


def test_overlap_repeats_trailing_sentences():
    blocks = [para(45, f"p{k}w", ["Ch"], k) for k in range(5)]
    for k, b in enumerate(blocks):
        b["text"] = f"Sentence one of paragraph {k} is here. Short closing sentence number {k}."
        b["text"] = " ".join([b["text"]] * 5)
    chunks = build_chunks(doc(blocks), {"title": "T"}, WordTok(), CCFG)
    with_overlap = [c for c in chunks if c["overlap_tokens"]]
    assert with_overlap
    for c in with_overlap:
        i = chunks.index(c)
        assert c["text"].split("\n\n")[0] in chunks[i - 1]["text"]


def test_small_table_attached_large_table_own_chunk():
    small = {"block_id": "b00001", "type": "table", "text": "| a | b |\n|---|---|\n| 1 | 2 |",
             "section_path": ["Ch"], "page": 1, "page_end": 1, "skip": False, "skip_reason": ""}
    big = {**small, "block_id": "b00004",
           "text": "| x | y |\n" + "\n".join(f"| r{k} | v{k} |" for k in range(30))}
    cap = {**para(6, "cap", ["Ch"], 3), "type": "caption", "text": "Table 2 Large results table."}
    blocks = [para(20, "a", ["Ch"], 0), small, para(70, "b", ["Ch"], 2), cap, big, para(20, "c", ["Ch"], 5)]
    chunks = build_chunks(doc(blocks), {"title": "T"}, WordTok(), CCFG)
    text_with_small = [c for c in chunks if "b00001" in c["block_ids"]]
    assert text_with_small[0]["chunk_type"] == "text" and len(text_with_small[0]["block_ids"]) > 1
    table_chunk = next(c for c in chunks if "b00004" in c["block_ids"])
    assert table_chunk["chunk_type"] == "table"
    assert table_chunk["block_ids"] == ["b00003", "b00004"]   # caption pulled in with its table


def test_skip_blocks_excluded():
    blocks = [para(30, "keep", ["Ch"], 0), {**para(30, "ref", ["Ch", "References"], 1),
                                             "skip": True, "skip_reason": "references"}]
    chunks = build_chunks(doc(blocks), {"title": "T"}, WordTok(), CCFG)
    assert all("ref" not in c["text"] for c in chunks)
