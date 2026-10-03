"""Tests for the non-LLM parts of the Phase 4 QA pipeline."""

import csv

from qa_agreement import cohen_kappa
from qa_common import banned_regex, format_citation, write_review_csv
from qa_filter import sentences
from qa_generate import allocate_slots
from utils import load_config

CFG = load_config()


def test_banned_phrases_catch_unseen_context_but_allow_named_sources():
    rx = banned_regex(CFG)
    for q in ["What does the evidence say about AI literacy?",
              "According to this study, what drives adoption?",
              "What do the recommendations say should be prioritised?",
              "What did the authors find about grading?",
              "What does the chapter argue about rankings?"]:
        assert rx.search(q), q
    for q in ["What did a 2025 scoping review of ChatGPT in nursing education find?",
              "In the AIware competency model for K-12, what does the Tables skill involve?",
              "How does the THE World University Ranking distribute its top 300 universities?"]:
        assert not rx.search(q), q


def test_slot_allocation_matches_mixes():
    enabled = ["factual", "definition", "explanation", "application"]
    slots, tcounts, dcounts = allocate_slots(CFG, enabled, n_chunks=20, seed=13)
    assert len(slots) == 100
    assert tcounts == {"factual": 27, "definition": 27, "explanation": 26, "application": 20}
    assert dcounts == {"easy": 35, "medium": 45, "hard": 20}
    hard_types = [t for t, d in slots if d == "hard"]
    assert hard_types.count("application") == 20   # hard goes to application first


def test_sentence_counting_and_citation_format():
    assert sentences("One fact. Another fact here.") == 2
    assert sentences("A single sentence with 2.5 percent growth.") == 1
    assert format_citation("Book", [95, 96]) == "(Book, pp. 95-96)"
    assert format_citation("Book", [12]) == "(Book, p. 12)"


def test_review_csv_keeps_human_verdicts(tmp_path):
    path = tmp_path / "review.csv"
    header = ["qa_id", "question"]
    write_review_csv(path, header, [{"qa_id": "a", "question": "Q1"}, {"qa_id": "b", "question": "Q2"}])
    rows = list(csv.DictReader(path.open()))
    rows[0]["human_verdict"], rows[0]["human_notes"] = "reject", "unsupported claim"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    kept = write_review_csv(path, header, [{"qa_id": "a", "question": "Q1 revised"},
                                           {"qa_id": "b", "question": "Q2"}])
    rows = {r["qa_id"]: r for r in csv.DictReader(path.open())}
    assert kept == 1
    assert rows["a"]["human_verdict"] == "reject" and rows["a"]["question"] == "Q1 revised"
    assert rows["b"]["human_verdict"] == ""


def test_cohen_kappa():
    assert cohen_kappa(10, 0, 0, 10) == 1.0
    assert abs(cohen_kappa(5, 5, 5, 5)) < 1e-9


def test_v3_banned_phrases_and_meta_text():
    import re
    rx = banned_regex(CFG)
    for q in ["Why is AI literacy described as essential?", "What example is given of bias in grading?",
              "Which practices are recommended for course design?", "Which of the following risks apply?",
              "According to the review, what limits adoption?", "Which barriers are discussed in the chapter?"]:
        assert rx.search(q), q
    for q in ["Which practices does UNESCO recommend for course design?",
              "What practices are recommended by UNESCO for course design?"]:
        assert not rx.search(q), q
    meta = re.compile("|".join(f"(?:{x})" for x in CFG["qa"]["meta_text_patterns"]), re.I)
    assert meta.search("Teachers adapt (the quoted text does not say how).")
    assert meta.search("As unit u03 states, privacy matters.")
    assert not meta.search("Unit 3 of the course covers AI ethics.")


def test_single_sentence_evidence_relaxes_format():
    from qa_filter import evidence_sentences
    assert evidence_sentences("Only one sentence here.") == 1
    assert evidence_sentences("First span. Second sentence. … Another span.") == 3
