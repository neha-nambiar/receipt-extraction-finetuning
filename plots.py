"""Charts from results/*/scores.json and runs/sft_*/loss_log.csv -> plots/*.png.

python plots.py
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from config import PLOTS_DIR, RESULTS_DIR, RUNS_DIR, SCORED_FIELDS

SIZES = [100, 500, 1000]  


def load_scores(tag, split):
    # scores.json for results/<tag>__<split>/
    p = RESULTS_DIR / f"{tag}__{split}" / "scores.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def label(name, s):
    return f"{name} (partial)" if s.get("partial") else name


def best_sft_tag():
    # Best_SFT_Model by overall val F1
    tags = [p.name[:-len("__val")] for p in RESULTS_DIR.glob("sft_lv_*__val")]
    if (RESULTS_DIR / "sft_lo_500__val").exists():
        tags.append("sft_lo_500")
    best, best_f1 = None, -1.0
    for t in sorted(tags):
        s = load_scores(t, "val")
        if s and s["overall"]["f1"] > best_f1:
            best, best_f1 = t, s["overall"]["f1"]
    return best


def plot_f1_vs_size():
    # Test_Set overall F1 vs training size
    xs, ys = [], []
    for n in SIZES:
        s = load_scores(f"sft_lv_{n}", "test")
        if s is None:
            print(f"f1_vs_size: skipped sft_lv_{n} (no test results)")
            continue
        xs.append(n)
        ys.append(s["overall"]["f1"])
    baselines = []
    for p in sorted(RESULTS_DIR.glob("*__test")):
        tag = p.name[:-len("__test")]
        if tag == "base" or tag.startswith("api_"):
            s = load_scores(tag, "test")
            if s:
                baselines.append((label(tag, s), s["overall"]["f1"]))
    if load_scores("base", "test") is None:
        print("f1_vs_size: skipped base (no test results)")
    if not xs and not baselines:
        print("f1_vs_size: nothing to plot, chart skipped")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if xs:
        ax.plot(xs, ys, "o-", color="tab:blue", label="SFT (language+vision)")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    colors = ["tab:gray", "tab:red", "tab:green", "tab:purple"]
    for (name, f1), c in zip(baselines, colors):
        ax.axhline(f1, ls="--", color=c, label=f"{name}: {f1:.3f}")
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlim(70, 1400)
    ax.set_ylim(0, 1)
    ax.set_xlabel("training examples")
    ax.set_ylabel("overall F1 (Test_Set)")
    ax.set_title("F1 vs training-set size")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, "f1_vs_size.png")


def plot_ablation():
    # Per-field + overall F1, language+vision vs language-only, 500 examples
    variants = [("language+vision", "sft_lv_500"), ("language-only", "sft_lo_500")]
    have = []
    for name, tag in variants:
        s = load_scores(tag, "test")
        if s is None:
            print(f"ablation: skipped {tag} (no test results)")
        else:
            have.append((label(name, s), s))
    if not have:
        print("ablation: nothing to plot, chart skipped")
        return

    cats = SCORED_FIELDS + ["overall"]
    w = 0.8 / len(have)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (name, s) in enumerate(have):
        vals = [s["fields"][f]["f1"] for f in SCORED_FIELDS] + [s["overall"]["f1"]]
        ax.bar([j + (i - (len(have) - 1) / 2) * w for j in range(len(cats))], vals, w, label=name)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (Test_Set)")
    ax.set_title("Vision-layer ablation (500 examples)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save(fig, "ablation.png")


def plot_errors():
    # Stacked Missing / Hallucination / Misread per field for base, Best_SFT, GRPO
    best = best_sft_tag()
    models = [("base", "base"), (f"Best_SFT ({best})" if best else "Best_SFT", best), ("GRPO", "grpo")]
    have = []
    for name, tag in models:
        s = load_scores(tag, "test") if tag else None
        if s is None:
            print(f"errors: skipped {name} (no test results)")
        else:
            have.append((label(name, s), s))
    if not have:
        print("errors: nothing to plot, chart skipped")
        return

    kinds = [("missing", "Missing", "tab:orange"), ("hallucination", "Hallucination", "tab:red"),
             ("misread", "Misread", "tab:purple")]
    fig, axes = plt.subplots(1, len(have), figsize=(5.5 * len(have), 5), sharey=True, sharex=True, squeeze=False)
    for ax, (name, s) in zip(axes[0], have):
        bottom = [0] * len(SCORED_FIELDS)
        for key, kname, c in kinds:
            vals = [s["fields"][f][key] for f in SCORED_FIELDS]
            ax.barh(SCORED_FIELDS, vals, left=bottom, color=c, label=kname)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f"{name} (n={s['n_receipts']})")
        ax.set_xlabel("receipts")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    axes[0][0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.suptitle("Errors per field on the Test_Set")
    save(fig, "errors.png")


def plot_loss_curves():
    # SFT training loss per run from runs/sft_*/loss_log.csv
    files = sorted(RUNS_DIR.glob("sft_*/loss_log.csv"))
    if not files:
        print("loss_curves: skipped (no runs/sft_*/loss_log.csv)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in files:
        steps, losses = [], []
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("loss") not in (None, ""):
                    steps.append(int(float(r["step"])))
                    losses.append(float(r["loss"]))
        if steps:
            ax.plot(steps, losses, label=p.parent.name)
        else:
            print(f"loss_curves: skipped {p.parent.name} (no loss values)")
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title("SFT loss curves")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "loss_curves.png")


def plot_leaderboard():
    # Overall Test_Set F1 for every evaluated model, sorted
    names = {"base": "Qwen3-VL-4B base (zero-shot)", "api_gpt-4.1-mini": "gpt-4.1-mini (zero-shot)",
             "sft_lv_100": "SFT, 100 receipts", "sft_lv_500": "SFT, 500 receipts",
             "sft_lv_1000": "SFT, 1,000 receipts", "sft_lo_500": "SFT, 500, language layers only",
             "grpo": "SFT 500 + GRPO on 500 more"}
    rows = [(names[t], load_scores(t, "test")["overall"]["f1"]) for t in names if load_scores(t, "test")]
    if not rows:
        print("leaderboard: nothing to plot, chart skipped")
        return
    rows.sort(key=lambda r: r[1])
    colors = ["tab:gray" if "zero-shot" in n else "tab:orange" if "GRPO" in n else "tab:blue" for n, _ in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([n for n, _ in rows], [f for _, f in rows], color=colors)
    for i, (_, f) in enumerate(rows):
        ax.text(f + 0.005, i, f"{f:.3f}", va="center")
    ax.set_xlim(0.5, 0.82)
    ax.set_xlabel("overall F1 on 300 held-out receipts")
    ax.set_title("Fine-tuned 4B model vs zero-shot baselines")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "leaderboard.png")


def plot_grpo():
    # GRPO reward and KL per step, 20-step moving average, for each run's log.csv
    runs = [("lr 2e-5 (300 steps)", RUNS_DIR / "grpo" / "log.csv"),
            ("lr 5e-6 (stopped at 115)", RUNS_DIR / "grpo_lr5e-6" / "log.csv")]
    have = []
    for name, p in runs:
        if not p.exists():
            print(f"grpo: skipped {name} (no {p.name})")
            continue
        with open(p, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("reward")]
        have.append((name, [float(r["reward"]) for r in rows], [float(r["kl"]) for r in rows]))
    if not have:
        print("grpo: nothing to plot, chart skipped")
        return

    k = 20

    def smooth(v):
        return [sum(v[i - k:i]) / k for i in range(k, len(v) + 1)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, rew, kl in have:
        a1.plot(range(k, len(rew) + 1), smooth(rew), label=name)
        a2.plot(range(k, len(kl) + 1), smooth(kl), label=name)
    a1.set_title("Reward (20-step moving average)")
    a2.set_title("KL to the SFT model (20-step moving average)")
    for a in (a1, a2):
        a.set_xlabel("step")
        a.grid(alpha=0.3)
        a.legend()
    save(fig, "grpo.png")


def save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=150)
    plt.close(fig)
    print(f"wrote plots/{name}")


def main():
    plot_f1_vs_size()
    plot_ablation()
    plot_errors()
    plot_loss_curves()
    plot_leaderboard()
    plot_grpo()


if __name__ == "__main__":
    main()
