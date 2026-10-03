"""QLoRA fine-tune of the base model (config base_model, Qwen3-8B) on the QA splits.

Format: closed-book chat examples. user = question (each paraphrase is an extra example with the
same answer when train.use_paraphrases), assistant = answer WITHOUT the citation. The chat
template is applied with enable_thinking=False, matching how the model is used and evaluated.
--format raft instead puts the source evidence in the user turn and appends the citation to the
answer (for the RAG/RAFT arm); this needs retrieval contexts added in a later step.

Loss is computed on the assistant tokens only. The model is loaded in 4-bit NF4 with bf16
compute; LoRA adapters go on all attention and MLP projections (train.target_modules).

--dry-run builds and tokenizes the datasets on CPU and prints statistics and one example
without loading the model (safe while the GPU is busy).

Usage:
    python src/train_qlora.py --dry-run
    python src/train_qlora.py [--format closedbook] [--output models/qwen3-8b-qlora]
"""

import argparse
import json
import statistics

from qa_common import read_jsonl
from utils import get_logger, load_config, repo_path


def citation_text(c):
    pages = c["pages"]
    p = "" if not pages else (f", p. {pages[0]}" if len(pages) == 1 else f", pp. {pages[0]}-{pages[-1]}")
    return f"({c['title']}{p})"


def examples(rows, tcfg, fmt, with_paraphrases):
    out = []
    for r in rows:
        questions = [r["question"]] + (r["paraphrases"] if with_paraphrases else [])
        for q in questions:
            if fmt == "closedbook":
                user, answer = q, r["answer"]
            else:  # raft: evidence in context, cited answer
                user = f"Context:\n{r['evidence']}\n\nQuestion: {q}"
                answer = f"{r['answer']} {citation_text(r['citation'])}"
            out.append({"messages": [{"role": "system", "content": tcfg["system_prompt"]},
                                     {"role": "user", "content": user},
                                     {"role": "assistant", "content": answer}],
                        "qa_id": r["qa_id"]})
    return out


def tokenize(tok, ex, max_len):
    """input_ids with labels masked (-100) everywhere except the assistant answer."""
    prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    full = tok.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels[:len(f_ids)],
            "truncated": len(tok(full, add_special_tokens=False)["input_ids"]) > max_len}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", choices=["closedbook", "raft"], default="closedbook")
    ap.add_argument("--output")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="only use the first N training rows (smoke test)")
    args = ap.parse_args()
    cfg = load_config()
    tcfg = cfg["train"]
    log = get_logger("train", cfg)
    split_dir = repo_path(cfg["paths"]["splits"])
    train_rows = read_jsonl(split_dir / "train.jsonl")[:args.limit] if args.limit else read_jsonl(split_dir / "train.jsonl")
    val_rows = read_jsonl(split_dir / "val.jsonl")
    if not train_rows:
        raise SystemExit(f"no training data in {split_dir}; run src/split.py first")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tr = [tokenize(tok, e, tcfg["max_seq_length"]) for e in examples(train_rows, tcfg, args.format, tcfg["use_paraphrases"])]
    va = [tokenize(tok, e, tcfg["max_seq_length"]) for e in examples(val_rows, tcfg, args.format, False)]
    lens = [len(x["input_ids"]) for x in tr]
    steps = len(tr) * tcfg["epochs"] / (tcfg["per_device_batch_size"] * tcfg["grad_accum"])
    log.info(f"{args.format}: {len(train_rows)} train rows -> {len(tr)} examples; {len(va)} val examples; "
             f"tokens median {statistics.median(lens):.0f}, max {max(lens)}, truncated {sum(x['truncated'] for x in tr)}; "
             f"~{steps:.0f} optimizer steps")
    if args.dry_run:
        ex = examples(train_rows[:1], tcfg, args.format, False)[0]
        print(tok.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False))
        n_label = sum(t != -100 for t in tr[0]["labels"])
        print(f"[dry run] first example: {len(tr[0]['input_ids'])} tokens, {n_label} supervised (answer) tokens")
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb,
                                                 torch_dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"], lora_dropout=tcfg["lora_dropout"],
                      target_modules=tcfg["target_modules"], task_type="CAUSAL_LM")
    out_dir = repo_path(args.output or tcfg["output_dir"]) / args.format
    strip = lambda rows: Dataset.from_list([{k: v for k, v in x.items() if k != "truncated"} for x in rows])
    sft = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=tcfg["epochs"], learning_rate=tcfg["learning_rate"],
        per_device_train_batch_size=tcfg["per_device_batch_size"], gradient_accumulation_steps=tcfg["grad_accum"],
        warmup_ratio=tcfg["warmup_ratio"], lr_scheduler_type="cosine", bf16=True, logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        gradient_checkpointing=True, report_to=[], max_length=tcfg["max_seq_length"], dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False)
    trainer = SFTTrainer(model=model, args=sft, train_dataset=strip(tr), eval_dataset=strip(va), peft_config=lora,
                         processing_class=tok,
                         data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    (out_dir / "train_config.json").write_text(json.dumps({"format": args.format, **tcfg,
                                                           "base_model": cfg["base_model"]}, indent=2))
    log.info(f"saved adapter to {out_dir / 'final'}")


if __name__ == "__main__":
    main()
