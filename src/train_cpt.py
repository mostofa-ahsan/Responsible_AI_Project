"""Arm C: mixed continued pretraining (raw text + QA) with QLoRA, one epoch per invocation (resumable).

Recipe: NF4 4-bit base (double quant, bf16 compute), LoRA r=128 / alpha=256 / dropout 0.05 on q,k,v,o,gate,up,down
(vision/audio towers excluded), lr 1e-4 cosine over ALL 3 epochs, warmup 3%, effective batch 16, max_seq 1024, seed 42,
gradient checkpointing. LoRA weights and optimizer state in bf16 (as in the QA-only arm).
Data: data/cpt/<model>/epoch{1,2,3}.npz (src/build_cpt_mix.py), concatenated and read IN ORDER (each epoch was shuffled
when built), so the cosine schedule spans the 3 epochs and epoch boundaries fall on optimizer steps.
Loss: token-level mean over every supervised token of the optimizer step (raw: all tokens; QA: answer tokens only);
RAW and QA losses are also logged separately.
Each epoch end: val QA loss (data/splits/val.jsonl) and perplexity on 200 sequences of the 14 held-out documents
(forgetting check); epoch 0 (= base model, LoRA B = 0) is measured before training starts.
Outputs: models_cpt/<model>/mixC/epoch<N>/adapter (bf16 safetensors + tokenizer), train_loss.csv, epochs.json.
Checkpoints every --save-steps (resume after a crash); only the latest is kept; all removed after the last epoch.
Exit codes: 0 one epoch trained, 10 all epochs done, 3 CUDA out of memory.

    python src/train_cpt.py --model meta-llama/Llama-3.1-8B-Instruct [--micro-batch 2] [--rank 128]
"""

import argparse
import csv
import json
import math
import shutil
import sys
import time

import numpy as np

from train_qlora import LORA_EXCLUDE, LORA_TARGETS, dir_size_mb, short_name
from utils import get_logger, load_config, repo_path

EPOCHS = 3
EFF_BATCH = 16


def load_npz(p, limit=None):
    z = np.load(p)
    off = z["offsets"]
    n = len(z["is_raw"]) if limit is None else min(limit, len(z["is_raw"]))
    return [{"input_ids": z["ids"][off[i]:off[i + 1]].tolist(), "is_raw": int(z["is_raw"][i]),
             "prompt_len": int(z["prompt_len"][i])} for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--alpha", type=int)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--data", default="data/cpt")
    ap.add_argument("--out", default="models_cpt")
    ap.add_argument("--smoke", type=int, default=0, help="test only: N optimizer steps on the first examples")
    args = ap.parse_args()
    alpha = args.alpha or 2 * args.rank
    cfg = load_config()
    log = get_logger("train_cpt", cfg)
    fam = short_name(args.model)
    d = repo_path(args.out) / fam / "mixC"
    d.mkdir(parents=True, exist_ok=True)
    sp = d / "epochs.json"
    state = json.loads(sp.read_text()) if sp.exists() else {"epochs": []}
    done = len(state["epochs"])
    if done >= EPOCHS and not args.smoke:
        log.info(f"{fam}: all {EPOCHS} epochs done")
        sys.exit(10)
    if state.get("rank") and state["rank"] != args.rank and done:
        raise SystemExit(f"{fam}: epochs trained with rank {state['rank']}; refusing to continue with {args.rank}")
    ckpt_dir = d / "checkpoints"
    cks = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if ckpt_dir.exists() else []
    resume = cks[-1] if cks else None
    if done > 0 and resume is None:
        raise SystemExit(f"{fam}: {done} epoch(s) recorded but no checkpoint to resume from")

    data_dir = repo_path(args.data) / fam
    lim = None
    if args.smoke:
        lim = args.smoke * EFF_BATCH
    epochs_data = [load_npz(data_dir / f"epoch{e}.npz", lim) for e in range(1, EPOCHS + 1)]
    bounds = np.cumsum([len(x) // EFF_BATCH for x in epochs_data]).tolist()      # optimizer step at each epoch end
    # Within each optimizer step (16 consecutive examples) put raw sequences first and QA examples after, so a micro-batch
    # rarely pads a ~130-token QA example to 1024 tokens. The examples of every step, and so its gradient, are unchanged.
    train_rows = []
    for x in epochs_data:
        for i in range(0, len(x), EFF_BATCH):
            train_rows += sorted(x[i:i + EFF_BATCH], key=lambda r: (-r["is_raw"], -len(r["input_ids"])))
    val_rows = load_npz(data_dir / "val_qa.npz", 48 if args.smoke else None)
    ho_rows = load_npz(data_dir / "heldout_ppl.npz", 16 if args.smoke else None)
    log.info(f"{fam}: epoch {done + 1}/{EPOCHS} ({'resume ' + resume.name if resume else 'fresh start'}); rank {args.rank} "
             f"alpha {alpha}; micro-batch {args.micro_batch} x accum {EFF_BATCH // args.micro_batch}; steps/epoch "
             f"{[bounds[0]] + [bounds[i] - bounds[i - 1] for i in range(1, len(bounds))]}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    set_seed(42)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                             exclude_modules=LORA_EXCLUDE, task_type="CAUSAL_LM"))
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        raw = torch.zeros(len(batch), dtype=torch.long)
        for i, b in enumerate(batch):
            x = torch.tensor(b["input_ids"], dtype=torch.long)
            ids[i, :len(x)] = x
            att[i, :len(x)] = 1
            lab[i, b["prompt_len"]:len(x)] = x[b["prompt_len"]:]
            raw[i] = b["is_raw"]
        return {"input_ids": ids, "attention_mask": att, "labels": lab, "is_raw": raw}

    class MixTrainer(Trainer):
        acc = {"raw": [0.0, 0], "qa": [0.0, 0]}

        def _get_train_sampler(self, *a, **k):
            return torch.utils.data.SequentialSampler(self.train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            is_raw = inputs.pop("is_raw")
            labels = inputs.pop("labels")
            out = model(**inputs)
            logits = out.logits[:, :-1, :]
            tgt = labels[:, 1:]
            ce = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                   ignore_index=-100, reduction="none").view(tgt.shape)
            mask = (tgt != -100)
            tot = (ce * mask).sum()
            ntok = mask.sum()
            if model.training:
                with torch.no_grad():
                    for key, sel in (("raw", is_raw == 1), ("qa", is_raw == 0)):
                        if sel.any():
                            self.acc[key][0] += float((ce[sel] * mask[sel]).sum())
                            self.acc[key][1] += int(mask[sel].sum())
            if model.training and num_items_in_batch is not None:
                loss = tot / num_items_in_batch          # token mean over the whole optimizer step
                if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                    loss = loss * self.current_gradient_accumulation_steps   # Trainer divides by accum again
            else:
                loss = tot / ntok.clamp(min=1)
            return (loss, out) if return_outputs else loss

        def log(self, logs, *a, **k):
            for key in ("raw", "qa"):
                if self.acc[key][1]:
                    logs[f"{key}_loss"] = round(self.acc[key][0] / self.acc[key][1], 4)
                self.acc[key] = [0.0, 0]
            super().log(logs, *a, **k)

    rows = []

    class Progress(TrainerCallback):
        def on_log(self, a, st, control, logs=None, **kw):
            if logs and "loss" in logs:
                rows.append({"step": st.global_step, "epoch": round(st.global_step / bounds[0], 3) if bounds[0] else 0,
                             "loss": logs.get("loss"), "raw_loss": logs.get("raw_loss", ""), "qa_loss": logs.get("qa_loss", ""),
                             "lr": logs.get("learning_rate")})
                if st.global_step % 50 == 0 or st.global_step < 20:
                    el = time.time() - t0
                    log.info(f"step {st.global_step}/{bounds[-1]} loss {logs.get('loss')} raw {logs.get('raw_loss')} "
                             f"qa {logs.get('qa_loss')} lr {logs.get('learning_rate'):.2e} | {el / 60:.1f} min")

        def on_step_end(self, a, st, control, **kw):
            flag = repo_path("results/.gpu_pause")
            while flag.exists():
                log.info(f"GPU pause flag set at step {st.global_step}: sleeping 5 min")
                time.sleep(300)
            target = bounds[done] if not args.smoke else args.smoke
            if st.global_step >= target:
                control.should_save = True
                control.should_training_stop = True

    class KeepBf16(TrainerCallback):
        """Resuming reloads LoRA weights / optimizer moments in fp32: cast back so every epoch trains in bf16."""
        def on_train_begin(self, a, st, control, model=None, optimizer=None, **kw):
            for _, p in model.named_parameters():
                if p.requires_grad and p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)
            opt = getattr(optimizer, "optimizer", optimizer)
            for s in (opt.state.values() if opt is not None else []):
                for k_, v in s.items():
                    if torch.is_tensor(v) and v.is_floating_point() and v.dim() > 0 and v.dtype != torch.bfloat16:
                        s[k_] = v.to(torch.bfloat16)

    total_steps = bounds[-1] if not args.smoke else args.smoke * EPOCHS
    targs = TrainingArguments(
        output_dir=str(ckpt_dir), max_steps=total_steps, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, per_device_train_batch_size=args.micro_batch, per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=EFF_BATCH // args.micro_batch, bf16=True, logging_steps=10, seed=42, data_seed=42,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=1, gradient_checkpointing=True,
        report_to=[], remove_unused_columns=False, eval_strategy="no", dataloader_num_workers=0,
        prediction_loss_only=True)
    ds = Dataset.from_list(train_rows)
    trainer = MixTrainer(model=model, args=targs, train_dataset=ds,
                         eval_dataset={"val_qa": Dataset.from_list(val_rows), "heldout_raw": Dataset.from_list(ho_rows)},
                         data_collator=collate, callbacks=[Progress(), KeepBf16()])

    # The token-level mean needs num_items_in_batch, which the Trainer only computes when it believes the model accepts
    # loss kwargs (True for Llama/Qwen, False for Gemma 4). compute_loss never forwards kwargs to the model, so force it:
    # every model then gets the same token-weighted loss (otherwise micro-batch 1 would weight every example equally).
    detected = trainer.model_accepts_loss_kwargs
    trainer.model_accepts_loss_kwargs = True
    log.info(f"{fam}: trainable params {n_train:,}; model_accepts_loss_kwargs detected={detected}, forced=True")

    def evaluate():
        r = trainer.evaluate()
        return {"val_qa_loss": round(r.get("eval_val_qa_loss", float("nan")), 4),
                "heldout_ppl": round(math.exp(r.get("eval_heldout_raw_loss", float("nan"))), 3)}
    try:
        if done == 0 and resume is None and "epoch0" not in state:
            state["epoch0"] = evaluate()
            log.info(f"{fam}: epoch 0 (base) {state['epoch0']}")
            sp.write_text(json.dumps(state, indent=2))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    except torch.cuda.OutOfMemoryError as e:
        log.info(f"{fam}: CUDA OOM ({str(e)[:200]})")
        sys.exit(3)
    minutes = (time.time() - t0) / 60
    epoch = done + 1
    ev = evaluate()
    out = d / f"epoch{epoch}" / "adapter"
    shutil.rmtree(out, ignore_errors=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    # store the adapter in bf16 (it is trained in bf16 already; make sure the file is too)
    from safetensors.torch import load_file, save_file
    f = out / "adapter_model.safetensors"
    sd = load_file(str(f))
    save_file({k: v.to(torch.bfloat16) for k, v in sd.items()}, str(f))
    with (d / "train_loss.csv").open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "epoch", "loss", "raw_loss", "qa_loss", "lr"])
        if fh.tell() == 0:
            w.writeheader()
        w.writerows(rows)

    def mean(key):
        v = [r[key] for r in rows if r[key] not in ("", None)]
        return round(sum(v) / len(v), 4) if v else None
    state["epochs"].append({"epoch": epoch, "steps_end": trainer.state.global_step, "minutes": round(minutes, 1),
                            "train_loss_mean": mean("loss"), "raw_loss_mean": mean("raw_loss"),
                            "qa_loss_mean": mean("qa_loss"), **ev,
                            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                            "adapter_size_mb": dir_size_mb(out), "resumed_from": resume.name if resume else None})
    state.update(model=args.model, rank=args.rank, alpha=alpha, micro_batch=args.micro_batch, trainable_params=n_train,
                 recipe="QLoRA NF4 (double quant) + bf16 LoRA, lr 1e-4 cosine over 3 epochs, warmup 3%, eff. batch 16, "
                        "max_seq 1024, seed 42, mixed raw + QA")
    if not args.smoke:
        sp.write_text(json.dumps(state, indent=2))
    cks = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for c in (cks if epoch == EPOCHS or args.smoke else cks[:-1]):
        shutil.rmtree(c, ignore_errors=True)
    log.info(f"{fam}: epoch {epoch} saved to {out} ({minutes:.0f} min; {state['epochs'][-1]})")


if __name__ == "__main__":
    main()
