**Table 1. Dataset statistics and generation funnel**

| Statistic | Value |
|---|---|
| Documents (books / articles) | 100 (39 / 61) |
| Pages | 10,161 |
| Chunks (all / mineable) | 6,406 / 2,708 |
| Knowledge units extracted | 16,003 |
| QA pairs generated | 11,630 |
| QA pairs passing filters | 9,350 |
|   rejected: low_value (multi-label) | 1,637 |
|   rejected: not_standalone (multi-label) | 492 |
|   rejected: duplicate (multi-label) | 430 |
|   rejected: grounding (multi-label) | 149 |
|   rejected: garbled_question (multi-label) | 95 |
| Split train | 6,680 |
| Split val | 393 |
| Split test_indomain | 791 |
| Split test_heldout_docs | 1,486 |
| Split test_seen_facts | 279 |
| Evaluation subset (in-domain / held-out docs / seen facts) | 500 / 500 / 279 |
| By q type (train+val+test) | explanation 2,505, definition 2,480, factual 2,373, application 1,992 |
| By difficulty (train+val+test) | medium 4,200, easy 3,170, hard 1,980 |
| By dimension (train+val+test) | teaching_learning 1,794, governance 1,572, general 1,097, equity_accessibility 871, student_ai_literacy 808, privacy_security 639, monitoring_improvement 634, faculty_readiness 627, leadership_strategy 514, assessment 459, procurement_technology 335 |
| API spend so far (all LLM calls, USD) | 190.47 |

Generated with Claude Sonnet 5.5, filtered by Claude Opus 5.5 (full_v1); splits are document-level; spend from logs/llm_usage.jsonl (uncached calls).
