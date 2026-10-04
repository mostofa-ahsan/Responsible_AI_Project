"""3-epoch QLoRA with an adapter saved after every epoch (one epoch per invocation, resumable).

Same recipe as run_2026-10-03 (src/train_qlora.py: data, hyperparameters, seed, chat templates,
closed-book format, LoRA targets) except num_train_epochs = 3, so the cosine schedule spans all three
epochs. Each invocation trains ONE epoch: it starts fresh, or resumes from the latest trainer checkpoint
(optimizer, scheduler, RNG and data position restored), evaluates val loss at the epoch end, saves a
checkpoint, stops, and saves the epoch's adapter + tokenizer to
models_epochs/<model>/epoch<N>/adapter. This releases the GPU between epochs so answers can be generated.
No epoch is "selected": all three are kept.

Checkpoints: only the latest is kept (it is needed to resume the next epoch); older ones are deleted once
the newer epoch's adapter is saved, and all are deleted after epoch 3.
Logs: models_epochs/<model>/train_loss.csv (appended per step/eval) and epochs.json (per epoch: train
loss, val loss, minutes, peak VRAM, adapter size).

Usage:
    python src/train_epochs.py --model meta-llama/Llama-3.1-8B-Instruct      # trains the next epoch
Exit codes: 0 = trained one epoch, 10 = all epochs already done.
"""

import argparse
import csv
import json
import shutil
import statistics
import sys
import time

from qa_common import read_jsonl
from train_qlora import (LORA_EXCLUDE, LORA_TARGETS, dir_size_mb, examples, progress_callback, short_name,
                         tokenize)
from utils import get_logger, load_config, repo_path

EPOCHS = 3


def model_dir(model_id, root="models_epochs"):
    return repo_path(root) / short_name(model_id)


def load_state(d):
    p = d / "epochs.json"
    return json.loads(p.read_text()) if p.exists() else {"epochs": []}


def latest_checkpoint(ckpt_dir):
    cks = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if ckpt_dir.exists() else []
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, help="test only: first N training rows")
    ap.add_argument("--root", default="models_epochs", help="test only: output root")
    args = ap.parse_args()
    cfg = load_config()
    tcfg = dict(cfg["train"])
    tcfg["epochs"] = EPOCHS
    log = get_logger("train_epochs", cfg)
    d = model_dir(args.model, args.root)
    d.mkdir(parents=True, exist_ok=True)
    state = load_state(d)
    done = len(state["epochs"])
    if done >= EPOCHS:
        log.info(f"{args.model}: all {EPOCHS} epochs done")
        sys.exit(10)
    ckpt_dir = d / "checkpoints"
    resume = latest_checkpoint(ckpt_dir)
    if done > 0 and resume is None:
        raise SystemExit(f"{args.model}: {done} epoch(s) recorded but no checkpoint to resume from")
    if done == 0 and resume is not None:          # crashed during epoch 1 after a mid-epoch save? start clean
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        resume = None

    split_dir = repo_path(cfg["paths"]["splits"])
    train_rows = read_jsonl(split_dir / "train.jsonl")[:args.limit] if args.limit else read_jsonl(split_dir / "train.jsonl")
    val_rows = read_jsonl(split_dir / "val.jsonl")[:40] if args.limit else read_jsonl(split_dir / "val.jsonl")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stats = {"prefix_mismatch": 0}
    tr = [tokenize(tok, e, tcfg["max_seq_length"], stats)
          for e in examples(train_rows, tcfg, "closedbook", tcfg["use_paraphrases"])]
    va = [tokenize(tok, e, tcfg["max_seq_length"], stats) for e in examples(val_rows, tcfg, "closedbook", False)]
    eff_bs = tcfg["per_device_batch_size"] * tcfg["grad_accum"]
    tokens_per_step = statistics.mean(len(x["input_ids"]) for x in tr) * eff_bs
    log.info(f"{args.model}: epoch {done + 1}/{EPOCHS} ({'resume ' + resume.name if resume else 'fresh start'}); "
             f"{len(tr)} train / {len(va)} val examples; prefix mismatches {stats['prefix_mismatch']}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, TrainerCallback,
                              set_seed)
    from trl import SFTConfig, SFTTrainer

    class KeepLoraBf16(TrainerCallback):
        """A fresh 4-bit run trains its LoRA weights in bf16 (TRL's cast, as in run_2026-10-03), but resuming
        reloads them through PEFT, which upcasts to fp32 by default. Cast back after the checkpoint and
        optimizer state are loaded so every epoch trains at the same precision."""
        def on_train_begin(self, args_, st, control, model=None, optimizer=None, **kw):
            n = 0
            for _, prm in model.named_parameters():
                if prm.requires_grad and prm.dtype != torch.bfloat16:
                    prm.data = prm.data.to(torch.bfloat16)
                    n += 1
            m = 0     # the optimizer state is upcast with the weights on reload: cast its moments back too
            opt = getattr(optimizer, "optimizer", optimizer)
            for state in (opt.state.values() if opt is not None else []):
                for k, v in state.items():
                    if torch.is_tensor(v) and v.is_floating_point() and v.dim() > 0 and v.dtype != torch.bfloat16:
                        state[k] = v.to(torch.bfloat16)
                        m += 1
            log.info(f"cast to bf16 at train begin: {n} LoRA tensors, {m} optimizer state tensors")

    class StopAtEpochEnd(TrainerCallback):
        """Stop after the first epoch boundary reached in this invocation (after eval + save)."""
        def on_epoch_end(self, args_, st, control, **kw):
            control.should_evaluate = True
            control.should_save = True
            control.should_training_stop = True

    set_seed(tcfg["seed"])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"], lora_dropout=tcfg["lora_dropout"],
                      target_modules=LORA_TARGETS, exclude_modules=LORA_EXCLUDE, task_type="CAUSAL_LM")
    strip = lambda rows: Dataset.from_list([{k: v for k, v in x.items() if k != "truncated"} for x in rows])
    sft = SFTConfig(
        output_dir=str(ckpt_dir), num_train_epochs=EPOCHS, learning_rate=tcfg["learning_rate"],
        per_device_train_batch_size=tcfg["per_device_batch_size"], gradient_accumulation_steps=tcfg["grad_accum"],
        warmup_steps=tcfg["warmup_ratio"], lr_scheduler_type="cosine", bf16=True, logging_steps=10,
        seed=tcfg["seed"], data_seed=tcfg["seed"], eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=None, load_best_model_at_end=False, gradient_checkpointing=True, report_to=[],
        max_length=tcfg["max_seq_length"], dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False)
    progress = progress_callback(log, tokens_per_step)
    trainer = SFTTrainer(model=model, args=sft, train_dataset=strip(tr), eval_dataset=strip(va), peft_config=lora,
                         processing_class=tok,
                         data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                         callbacks=[progress, KeepLoraBf16(), StopAtEpochEnd()])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    minutes = (time.time() - t0) / 60
    epoch = done + 1
    if round(trainer.state.epoch or 0) != epoch:
        raise SystemExit(f"expected to finish epoch {epoch}, trainer state epoch {trainer.state.epoch}")

    out = d / f"epoch{epoch}" / "adapter"
    shutil.rmtree(out, ignore_errors=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    rows = progress.rows
    new_file = not (d / "train_loss.csv").exists()
    with (d / "train_loss.csv").open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "learning_rate", "val_loss"])
        if new_file:
            w.writeheader()
        w.writerows(rows)
    tl = [r["train_loss"] for r in rows if r["train_loss"] != ""]
    vl = [r["val_loss"] for r in rows if r["val_loss"] != ""]
    state["epochs"].append({
        "epoch": epoch, "steps_end": trainer.state.global_step, "minutes": round(minutes, 1),
        "train_loss_last": tl[-1] if tl else None,
        "train_loss_mean": round(sum(tl) / len(tl), 4) if tl else None,
        "val_loss": vl[-1] if vl else None, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "adapter_size_mb": dir_size_mb(out), "resumed_from": resume.name if resume else None})
    state.update(model=args.model, recipe="run_2026-10-03 recipe, 3 epochs, cosine over 3 epochs",
                 steps_per_epoch=-(-len(tr) // eff_bs))
    (d / "epochs.json").write_text(json.dumps(state, indent=2))
    # keep only the newest checkpoint (needed to resume); none after the last epoch
    cks = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for c in (cks if epoch == EPOCHS else cks[:-1]):
        shutil.rmtree(c, ignore_errors=True)
    if epoch == EPOCHS:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    log.info(f"{args.model}: epoch {epoch} saved to {out} ({minutes:.0f} min, val loss {state['epochs'][-1]['val_loss']})")


if __name__ == "__main__":
    main()
