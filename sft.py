"""LoRA SFT with Unsloth + TRL SFTTrainer. Pod only."""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

from transformers import TrainerCallback

import config
from config import (DATA_DIR, LOAD_IN_4BIT, MAX_PIXELS, MIN_PIXELS, MODEL_NAME, PROMPT,
                    RUNS_DIR, SEED)
from eval import image_path


class GuardCallback(TrainerCallback):
    # Time-based saves, wall-clock stop and KL stop.
    def __init__(self, save_every_min, max_hours=None, kl_max=None):
        self.save_every_min = save_every_min
        self.max_hours = max_hours
        self.kl_max = kl_max
        self.start = self.last_save = time.time()
        self.last_ckpt = None
        self.stopped_by_time = False
        self.stopped_by_kl = False

    def on_train_begin(self, args, state, control, **kw):
        self.start = self.last_save = time.time()

    def on_step_end(self, args, state, control, **kw):
        now = time.time()
        if (now - self.last_save) / 60 >= self.save_every_min:
            control.should_save = True
        if self.max_hours is not None and (now - self.start) / 3600 >= self.max_hours:
            control.should_save = True
            control.should_training_stop = True
            self.stopped_by_time = True
        return control

    def on_log(self, args, state, control, logs=None, **kw):
        # Runs before the save check of the same step, so the crossed step is not saved
        if self.kl_max is not None and logs and logs.get("kl", 0) > self.kl_max:
            control.should_training_stop = True
            control.should_save = False
            self.stopped_by_kl = True
        return control

    def on_save(self, args, state, control, **kw):
        self.last_save = time.time()
        self.last_ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        return control


def load_receipts(size):
    # Manifest receipts of the nested training subset only (never test or validation)
    with open(DATA_DIR / "manifest.json", encoding="utf-8") as f:
        rec = json.load(f)["receipts"]
    return [r for r in rec if r["split"] == "pool" and not r["val"]
            and r["subset"] is not None and r["subset"] <= size]


def oom_receipts():
    # Pool receipt with the most pixels and the one with the most Line_Items
    with open(DATA_DIR / "manifest.json", encoding="utf-8") as f:
        pool = [r for r in json.load(f)["receipts"] if r["split"] == "pool"]
    big = max(pool, key=lambda r: r["width"] * r["height"])
    many = max(pool, key=lambda r: r["n_items"])
    return [big] if big["base"] == many["base"] else [big, many]


def to_conversation(base):
    # User turn = image + PROMPT, assistant turn = Target_Text
    target = (DATA_DIR / "labels" / f"{base}.json").read_text(encoding="utf-8")
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "image": config.load_image(image_path(base))},
                                     {"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": target}]}]}


def save_loss_log(history, path):
    # One row per log entry of trainer.state.log_history
    keys = ["step", "epoch", "loss", "grad_norm", "learning_rate"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for h in history:
            if "loss" in h:
                w.writerow([h.get(k, "") for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, choices=[100, 250, 500, 1000], required=True)
    ap.add_argument("--layers", choices=["language+vision", "language-only"], default=config.LAYERS)
    ap.add_argument("--oom-check", action="store_true", help="2-step memory check (task 9.3)")
    ap.add_argument("--stop-pod", action="store_true", help="stop (not terminate) the pod at exit")
    args = ap.parse_args()

    config.check_settings()
    config.print_spend()
    if args.stop_pod and config.check_stop_pod():
        config.stop_pod_at_exit()
    start = datetime.now()

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME, load_in_4bit=LOAD_IN_4BIT, max_seq_length=4096,
        use_gradient_checkpointing="unsloth")
    tokenizer.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=(args.layers == "language+vision"),
        finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=SEED)
    FastVisionModel.for_training(model)

    hp = dict(num_train_epochs=3, per_device_train_batch_size=2, gradient_accumulation_steps=4,
              learning_rate=2e-4, lr_scheduler_type="linear", warmup_ratio=0.05,
              optim="adamw_8bit", bf16=True, seed=SEED, logging_steps=5,
              save_steps=50, save_total_limit=2, max_length=4096)
    common = dict(report_to="none", remove_unused_columns=False, dataset_text_field="",
                  dataset_kwargs={"skip_prepare_dataset": True})
    # resize="max" turns off Unsloth's default 512 px resize; MAX_PIXELS stays the only cap, same as eval
    collator = UnslothVisionDataCollator(model, tokenizer, resize="max")

    if args.oom_check:
        # 2 steps, batch 1: each worst-case receipt goes through one step. Nothing is saved.
        picked = oom_receipts()
        for r in picked:
            print(f"oom-check: {r['base']}  {r['width'] * r['height'] / 1e6:.2f} MP  "
                  f"{r['n_items']} items")
        oom_hp = {k: v for k, v in hp.items() if k not in ("save_steps", "save_total_limit")}
        oom_hp.update(max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=1,
                      logging_steps=1, save_strategy="no")
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, data_collator=collator,
            train_dataset=[to_conversation(r["base"]) for r in picked],
            args=SFTConfig(output_dir=str(RUNS_DIR / "oom_check"), **common, **oom_hp))
        torch.cuda.reset_peak_memory_stats()
        try:
            trainer.train()
        except torch.OutOfMemoryError:
            print("GPU out of memory. Set LOAD_IN_4BIT = True in config.py")
            sys.exit(1)
        print(f"oom-check passed, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        return

    tag = "lv" if args.layers == "language+vision" else "lo"
    name = f"sft_{tag}_{args.size}"
    run_dir = RUNS_DIR / name
    out = run_dir / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)

    receipts = load_receipts(args.size)
    assert len(receipts) == args.size, f"expected {args.size} receipts, got {len(receipts)}"
    print(f"{name}: {len(receipts)} training receipts, layers {args.layers}")
    train_data = [to_conversation(r["base"]) for r in receipts]

    sft_args = SFTConfig(output_dir=str(out), **common, **hp)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, data_collator=collator,
        train_dataset=train_data, args=sft_args, callbacks=[GuardCallback(save_every_min=30)])

    last = get_last_checkpoint(str(out))
    if last:
        print(f"resuming from {last}")
    try:
        trainer.train(resume_from_checkpoint=last is not None)
    except torch.OutOfMemoryError:
        print("GPU out of memory. Set LOAD_IN_4BIT = True in config.py")
        sys.exit(1)

    adapter = run_dir / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    save_loss_log(trainer.state.log_history, run_dir / "loss_log.csv")

    end = datetime.now()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    settings = dict(hp, lora=dict(r=16, lora_alpha=16, lora_dropout=0, attention=True, mlp=True,
                                  language_layers=True,
                                  vision_layers=args.layers == "language+vision"),
                    layers=args.layers, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS,
                    load_in_4bit=LOAD_IN_4BIT, model_name=MODEL_NAME, size=args.size,
                    n_train=len(receipts), trainable_params=trainable,
                    global_steps=trainer.state.global_step,
                    minutes=round((end - start).total_seconds() / 60, 2))
    with open(run_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    config.log_run(name, start, end)
    print(f"saved {adapter}, {trainable:,} trainable params, {settings['minutes']} min")


if __name__ == "__main__":
    main()
