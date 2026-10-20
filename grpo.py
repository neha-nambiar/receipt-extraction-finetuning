"""GRPO from the best SFT adapter (picked by Validation_Split F1). Pod only.

python grpo.py [--stop-pod]
torch/unsloth/trl are imported inside functions, so the file compiles locally.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import config
from config import (DATA_DIR, MAX_NEW_TOKENS, MAX_PIXELS, MIN_PIXELS, PROMPT, RESULTS_DIR, ROOT,
                    RUNS_DIR, SEED)
from scorer import reward

# GRPO settings
NUM_GENERATIONS = 4   # completions per prompt
KL_MAX = 0.5          # stop when logged KL goes above this; tune after first logs
MAX_HOURS = 4         # wall-clock cap, keeps GRPO within $10 project budget
LEARNING_RATE = 2e-5  # 5e-6 barely moved the LoRA (KL < 0.001 after 100 steps)
GRPO_DIR = RUNS_DIR / "grpo"


def pick_best_sft():
    # Highest overall val F1 among sft_lv_* runs
    paths = sorted(RESULTS_DIR.glob("sft_lv_*__val/scores.json"))
    lo = RESULTS_DIR / "sft_lo_500__val" / "scores.json"
    if lo.exists():
        paths.append(lo)
    cands = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        if s.get("partial"):
            print(f"skipping partial val run {p.parent.name}")
            continue
        cands.append((s["overall"]["f1"], s["model"]))
    if not cands:
        print("no full SFT val results in results/sft_*__val/; run eval.py --split val first")
        sys.exit(1)
    for f1, name in sorted(cands, reverse=True):
        print(f"  {name}: val F1 {f1:.3f}")
    f1, name = max(cands)
    adapter = RUNS_DIR / name / "adapter"
    if not adapter.is_dir():
        print(f"adapter directory not found: {adapter}")
        sys.exit(1)
    with open(RUNS_DIR / name / "settings.json", encoding="utf-8") as f:
        size = json.load(f)["size"]
    print(f"Best_SFT_Model: {name} (val F1 {f1:.3f}, trained on {size} receipts)")
    return name, adapter, size


def load_model(name, adapter):
    # TRL's reference model for a PEFT model is the model with the adapter switched off.
    # Training the SFT adapter directly would make that reference the *base* model, so the
    # KL penalty would pull back toward base (the smoke run logged KL 0.24 at step 1, lr 0).
    # So: merge the SFT adapter into the weights once, then train a fresh LoRA on top.
    # The reference is then the SFT model itself. No vLLM (no vision-LoRA support).
    import gc
    import torch
    from unsloth import FastVisionModel
    merged = RUNS_DIR / f"{name}_merged"
    if not (merged / "config.json").exists():
        m, t = FastVisionModel.from_pretrained(str(adapter), load_in_4bit=False, max_seq_length=4096)
        m.save_pretrained_merged(str(merged), t, save_method="merged_16bit")
        print(f"merged {name} into {merged}")
        del m, t
        gc.collect()
        torch.cuda.empty_cache()
    model, tokenizer = FastVisionModel.from_pretrained(
        str(merged), load_in_4bit=config.LOAD_IN_4BIT, max_seq_length=4096,
        fast_inference=False, use_gradient_checkpointing="unsloth")
    tokenizer.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    with open(RUNS_DIR / name / "settings.json", encoding="utf-8") as f:
        lora = json.load(f)["lora"]
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=lora["vision_layers"], finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=lora["r"], lora_alpha=lora["lora_alpha"], lora_dropout=0, bias="none", random_state=SEED)
    FastVisionModel.for_training(model)
    return model, tokenizer


def build_dataset(size):
    # Receipts of the 1,000 subset that the SFT adapter never saw (500 for sft_*_500), so
    # GRPO gets a real reward signal and uses the same 1,000 labels as sft_lv_1000.
    # Falls back to the adapter's own subset if it was trained on all 1,000.
    # Columns: prompt (user turn), image, target (Target_Text). Never test or validation.
    from datasets import Dataset
    from eval import image_path
    from sft import load_receipts
    seen = {r["base"] for r in load_receipts(size)}
    receipts = [r for r in load_receipts(1000) if r["base"] not in seen] or load_receipts(size)
    assert len(receipts) == (1000 - size if size < 1000 else size), f"got {len(receipts)} receipts"
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    rows = [dict(prompt=prompt,
                 image=config.load_image(image_path(r["base"])),
                 target=(DATA_DIR / "labels" / f"{r['base']}.json").read_text(encoding="utf-8"))
            for r in receipts]
    return Dataset.from_list(rows)


def reward_fn(completions, target, **kw):
    # TRL reward hook: one rule-based reward per completion
    return [reward(c[0]["content"] if isinstance(c, list) else c, json.loads(t))
            for c, t in zip(completions, target)]


def save_log(history, path):
    # step, reward, reward_std, kl from trainer.state.log_history
    keys = ["step", "reward", "reward_std", "kl"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for h in history:
            if "reward" in h:
                w.writerow([h.get(k, "") for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-pod", action="store_true", help="stop (not terminate) the pod at exit")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only N steps into runs/grpo_smoke/ (timing and log check)")
    args = ap.parse_args()
    global GRPO_DIR
    if args.smoke:
        GRPO_DIR = RUNS_DIR / "grpo_smoke"

    config.check_settings()
    config.print_spend()
    if args.stop_pod and config.check_stop_pod():
        config.stop_pod_at_exit()
    if not 4 <= NUM_GENERATIONS <= 8:
        print(f"invalid setting NUM_GENERATIONS={NUM_GENERATIONS}: must be 4..8")
        sys.exit(1)
    start = datetime.now()

    name, adapter, size = pick_best_sft()
    model, tokenizer = load_model(name, adapter)
    dataset = build_dataset(size)
    print(f"GRPO dataset: {len(dataset)} receipts not seen by {name}")

    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer
    from sft import GuardCallback

    out = GRPO_DIR / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    hp = dict(num_generations=NUM_GENERATIONS,
              per_device_train_batch_size=NUM_GENERATIONS,  # one prompt per step
              gradient_accumulation_steps=1,
              learning_rate=LEARNING_RATE, lr_scheduler_type="cosine", warmup_ratio=0.1,
              beta=0.04, temperature=0.9,
              max_prompt_length=None,  # never cut image tokens
              max_completion_length=MAX_NEW_TOKENS, max_steps=args.smoke or 300,
              logging_steps=1, save_steps=20, optim="adamw_8bit", bf16=True, seed=SEED)
    guard = GuardCallback(save_every_min=30, max_hours=MAX_HOURS, kl_max=KL_MAX)
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, reward_funcs=[reward_fn],
        train_dataset=dataset, callbacks=[guard],
        args=GRPOConfig(output_dir=str(out), report_to="none",
                        remove_unused_columns=False, **hp))  

    last = get_last_checkpoint(str(out))
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)

    # Adapter selection: last checkpoint before the KL crossing, else the final weights
    if guard.stopped_by_kl:
        selected = guard.last_ckpt or get_last_checkpoint(str(out)) or str(adapter)
        print(f"stopped by KL > {KL_MAX}; selected {selected}")
    else:
        selected = str(GRPO_DIR / "adapter")
        model.save_pretrained(selected)
        tokenizer.save_pretrained(selected)
        if guard.stopped_by_time:
            print(f"stopped by the {MAX_HOURS} h wall-clock limit")
    selected = Path(selected).resolve()
    try:
        selected = selected.relative_to(ROOT)
    except ValueError:
        pass
    (GRPO_DIR / "selected.txt").write_text(selected.as_posix() + "\n", encoding="utf-8")
    save_log(trainer.state.log_history, GRPO_DIR / "log.csv")

    end = datetime.now()
    minutes = round((end - start).total_seconds() / 60, 2)
    settings = dict(hp, kl_max=KL_MAX, max_hours=MAX_HOURS, arith_bonus=config.ARITH_BONUS,
                    from_sft=name, sft_size=size, n_train=len(dataset), max_pixels=MAX_PIXELS,
                    min_pixels=MIN_PIXELS, load_in_4bit=config.LOAD_IN_4BIT, seed=SEED,
                    global_steps=trainer.state.global_step, stopped_by_kl=guard.stopped_by_kl,
                    stopped_by_time=guard.stopped_by_time, selected=selected.as_posix(),
                    minutes=minutes)
    with open(GRPO_DIR / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    config.log_run("grpo_smoke" if args.smoke else "grpo", start, end)
    print(f"GRPO done: {trainer.state.global_step} steps, total run time {minutes} min, "
          f"selected {selected.as_posix()}")


if __name__ == "__main__":
    main()
