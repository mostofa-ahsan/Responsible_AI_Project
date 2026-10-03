"""QLoRA fine-tune of a base model on the QA splits (same recipe for every model).

Format: closed-book chat examples. user = question (each paraphrase is an extra example with the
same answer when train.use_paraphrases), assistant = answer WITHOUT the citation. Each model's own
chat template is used; for Qwen3 it is applied with enable_thinking=False. --format raft instead
puts the evidence in the user turn and appends the citation (for the RAG/RAFT arm, later).

Recipe (config train): 4-bit NF4 load with bf16 compute, LoRA r/alpha/dropout on all language-model
linear layers (q/k/v/o/gate/up/down projections; vision/audio towers of multimodal models are
excluded), loss on answer tokens only, effective batch per_device_batch_size x grad_accum, cosine
schedule, seed train.seed. Val loss is evaluated every epoch and the best epoch's adapter is kept.

Outputs (in --output, default models/<model short name>):
  adapter/             LoRA adapter + tokenizer of the best epoch (nothing else is kept)
  train_loss.csv       step, epoch, train loss, learning rate, val loss
  train_summary.json   epochs, steps, time, peak VRAM, tokens/sec, adapter size, best val loss
Epoch checkpoints are deleted after training. --max-steps N runs a smoke test and writes
smoke_summary.json (loss trend, s/step, tokens/sec, peak VRAM, projected time per epoch).

Usage:
    python src/train_qlora.py --model Qwen/Qwen3-8B --dry-run
    python src/train_qlora.py --model Qwen/Qwen3-8B --max-steps 30 --output models/qwen3-8b/smoke
    python src/train_qlora.py --model Qwen/Qwen3-8B --epochs 1 --output models/qwen3-8b
"""

import argparse
import csv
import json
import shutil
import statistics
import time

from qa_common import read_jsonl
from utils import get_logger, load_config, repo_path

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_EXCLUDE = r".*(vision|audio|multi_modal_projector|embed_vision|embed_audio).*"


def short_name(model_id):
    return model_id.split("/")[-1].lower()


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


def chat(tok, messages, add_generation_prompt):
    # enable_thinking only affects templates that use it (Qwen3); others ignore the variable
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)


def tokenize(tok, ex, max_len, stats):
    """input_ids with labels masked (-100) everywhere except the assistant answer."""
    prompt = chat(tok, ex["messages"][:-1], True)
    full = chat(tok, ex["messages"], False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_all = tok(full, add_special_tokens=False)["input_ids"]
    if f_all[:len(p_ids)] != p_ids:
        # Template renders the prompt differently inside the full conversation: locate the answer.
        stats["prefix_mismatch"] += 1
        a_ids = tok(ex["messages"][-1]["content"], add_special_tokens=False)["input_ids"]
        start = next((i for i in range(len(f_all) - len(a_ids), -1, -1) if f_all[i:i + len(a_ids)] == a_ids),
                     len(p_ids))
        p_len = start
    else:
        p_len = len(p_ids)
    f_ids = f_all[:max_len]
    labels = [-100] * min(p_len, len(f_ids)) + f_ids[p_len:]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels[:len(f_ids)],
            "truncated": len(f_all) > max_len}


def progress_callback(log, tokens_per_step):
    """Logs tokens/sec, seconds per optimizer step and an ETA; records losses for the CSV."""
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.time()
            self.step_times = []
            self.rows = []

        def on_step_end(self, args, state, control, **kw):
            self.step_times.append(time.time())

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "eval_loss" in logs:
                self.rows.append({"step": state.global_step, "epoch": round(state.epoch or 0, 3), "train_loss": "",
                                  "learning_rate": "", "val_loss": round(logs["eval_loss"], 5)})
                log.info(f"eval at step {state.global_step} (epoch {state.epoch:.2f}): val loss {logs['eval_loss']:.4f}")
                return
            if not state.global_step or "loss" not in logs:
                return
            self.rows.append({"step": state.global_step, "epoch": round(state.epoch or 0, 3),
                              "train_loss": round(logs["loss"], 5), "learning_rate": logs.get("learning_rate", ""),
                              "val_loss": ""})
            el = time.time() - self.t0
            sps = el / state.global_step
            eta = (state.max_steps - state.global_step) * sps
            log.info(f"step {state.global_step}/{state.max_steps} epoch {state.epoch:.2f} loss {logs['loss']:.4f} "
                     f"lr {logs.get('learning_rate', 0):.2e} | {tokens_per_step * state.global_step / el:,.0f} tok/s, "
                     f"{sps:.1f} s/step, ETA {time.strftime('%H:%M', time.localtime(time.time() + eta))} "
                     f"({eta / 60:.0f} min)")
    return Progress()


def dir_size_mb(path):
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="HF model id (default: config base_model)")
    ap.add_argument("--format", choices=["closedbook", "raft"], default="closedbook")
    ap.add_argument("--output", help="output dir (default models/<model short name>)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="only use the first N training rows")
    ap.add_argument("--epochs", type=int, choices=[1, 2, 3], help="override train.epochs")
    ap.add_argument("--max-steps", type=int, help="stop after N optimizer steps (smoke test)")
    args = ap.parse_args()
    cfg = load_config()
    tcfg = dict(cfg["train"])
    if args.epochs:
        tcfg["epochs"] = args.epochs
    model_id = args.model or cfg["base_model"]
    out_dir = repo_path(args.output or f"models/{short_name(model_id)}")
    log = get_logger("train", cfg)
    split_dir = repo_path(cfg["paths"]["splits"])
    train_rows = read_jsonl(split_dir / "train.jsonl")
    train_rows = train_rows[:args.limit] if args.limit else train_rows
    val_rows = read_jsonl(split_dir / "val.jsonl")
    if not train_rows:
        raise SystemExit(f"no training data in {split_dir}; run src/split.py first")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stats = {"prefix_mismatch": 0}
    tr = [tokenize(tok, e, tcfg["max_seq_length"], stats)
          for e in examples(train_rows, tcfg, args.format, tcfg["use_paraphrases"])]
    va = [tokenize(tok, e, tcfg["max_seq_length"], stats) for e in examples(val_rows, tcfg, args.format, False)]
    lens = [len(x["input_ids"]) for x in tr]
    eff_bs = tcfg["per_device_batch_size"] * tcfg["grad_accum"]
    steps_per_epoch = -(-len(tr) // eff_bs)
    tokens_per_step = statistics.mean(lens) * eff_bs
    log.info(f"{model_id} {args.format}: {len(train_rows)} train rows -> {len(tr)} examples; {len(va)} val; "
             f"tokens median {statistics.median(lens):.0f}, max {max(lens)}, truncated {sum(x['truncated'] for x in tr)}; "
             f"prefix mismatches {stats['prefix_mismatch']}; {steps_per_epoch} steps/epoch x {tcfg['epochs']} epochs; "
             f"~{tokens_per_step:,.0f} tokens/step")
    if args.dry_run:
        ex = examples(train_rows[:1], tcfg, args.format, False)[0]
        print(chat(tok, ex["messages"], False))
        n_label = sum(t != -100 for t in tr[0]["labels"])
        print(f"[dry run] {model_id}: first example {len(tr[0]['input_ids'])} tokens, {n_label} supervised tokens; "
              f"{len(tr)} train / {len(va)} val examples; truncated {sum(x['truncated'] for x in tr)}; "
              f"prefix mismatches {stats['prefix_mismatch']}")
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(tcfg["seed"])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"], lora_dropout=tcfg["lora_dropout"],
                      target_modules=LORA_TARGETS, exclude_modules=LORA_EXCLUDE, task_type="CAUSAL_LM")
    ckpt_dir = out_dir / "checkpoints"
    strip = lambda rows: Dataset.from_list([{k: v for k, v in x.items() if k != "truncated"} for x in rows])
    sft = SFTConfig(
        output_dir=str(ckpt_dir), num_train_epochs=tcfg["epochs"], learning_rate=tcfg["learning_rate"],
        per_device_train_batch_size=tcfg["per_device_batch_size"], gradient_accumulation_steps=tcfg["grad_accum"],
        warmup_steps=tcfg["warmup_ratio"], lr_scheduler_type="cosine", bf16=True,  # float < 1 = ratio (transformers 5)
        logging_steps=1 if args.max_steps else 10, max_steps=args.max_steps or -1, seed=tcfg["seed"],
        data_seed=tcfg["seed"], eval_strategy="epoch", save_strategy="no" if args.max_steps else "epoch",
        save_total_limit=tcfg["epochs"] + 1, load_best_model_at_end=not args.max_steps,
        metric_for_best_model="eval_loss", greater_is_better=False,
        gradient_checkpointing=True, report_to=[], max_length=tcfg["max_seq_length"],
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False)
    progress = progress_callback(log, tokens_per_step)
    trainer = SFTTrainer(model=model, args=sft, train_dataset=strip(tr), eval_dataset=strip(va), peft_config=lora,
                         processing_class=tok,
                         data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                         callbacks=[progress])
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 1e9
    st = progress.step_times
    sps = (st[-1] - st[0]) / (len(st) - 1) if len(st) > 1 else float("nan")   # excludes the first step
    out_dir.mkdir(parents=True, exist_ok=True)
    train_losses = [r for r in progress.rows if r["train_loss"] != ""]
    trend = (f"{train_losses[0]['train_loss']:.3f} (step {train_losses[0]['step']}) -> "
             f"{train_losses[-1]['train_loss']:.3f} (step {train_losses[-1]['step']})") if train_losses else "n/a"
    if args.max_steps:
        summary = {"model": model_id, "steps": args.max_steps, "loss_trend": trend,
                   "loss_first": train_losses[0]["train_loss"] if train_losses else None,
                   "loss_last": train_losses[-1]["train_loss"] if train_losses else None,
                   "sec_per_step": round(sps, 2), "tokens_per_sec": round(tokens_per_step / sps) if sps == sps else None,
                   "peak_vram_gb": round(peak, 1), "steps_per_epoch": steps_per_epoch,
                   "projected_hours_per_epoch": round(steps_per_epoch * sps / 3600, 2),
                   "train_examples": len(tr), "val_examples": len(va), "trainable_params": n_trainable,
                   "prefix_mismatches": stats["prefix_mismatch"]}
        (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        log.info(f"SMOKE SUMMARY {model_id}: loss {trend}; {sps:.1f} s/step; ~{summary['tokens_per_sec']} tok/s; "
                 f"peak VRAM {peak:.1f} GB; {steps_per_epoch} steps/epoch ~ {summary['projected_hours_per_epoch']} h/epoch")
        return

    adapter_dir = out_dir / "adapter"
    shutil.rmtree(adapter_dir, ignore_errors=True)
    trainer.save_model(str(adapter_dir))           # best epoch (load_best_model_at_end): adapter only
    tok.save_pretrained(str(adapter_dir))
    with (out_dir / "train_loss.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "learning_rate", "val_loss"])
        w.writeheader()
        w.writerows(progress.rows)
    val = [r["val_loss"] for r in progress.rows if r["val_loss"] != ""]
    summary = {"model": model_id, "format": args.format, "epochs": tcfg["epochs"], "steps": trainer.state.global_step,
               "train_examples": len(tr), "val_examples": len(va), "train_minutes": round(train_min, 1),
               "sec_per_step": round(sps, 2), "tokens_per_sec": round(tokens_per_step / sps) if sps == sps else None,
               "peak_vram_gb": round(peak, 1), "best_val_loss": min(val) if val else None, "val_loss_by_epoch": val,
               "best_checkpoint": trainer.state.best_model_checkpoint, "loss_trend": trend,
               "adapter_size_mb": dir_size_mb(adapter_dir), "trainable_params": n_trainable,
               "hyperparameters": {k: tcfg[k] for k in ("lora_r", "lora_alpha", "lora_dropout", "learning_rate",
                                                        "per_device_batch_size", "grad_accum", "max_seq_length",
                                                        "warmup_ratio", "seed", "use_paraphrases")},
               "lora_targets": LORA_TARGETS, "lora_exclude": LORA_EXCLUDE}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    shutil.rmtree(ckpt_dir, ignore_errors=True)     # keep only the best adapter
    log.info(f"saved adapter ({summary['adapter_size_mb']} MB) to {adapter_dir}; {train_min:.0f} min; "
             f"peak VRAM {peak:.1f} GB; best val loss {summary['best_val_loss']}")


if __name__ == "__main__":
    main()
