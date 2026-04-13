"""
comparison.py — Multi-seed study: baseline QANet vs Inception QANet.

Runs 3 paired training runs (same seeds across conditions), records per-run
and aggregated statistics for EM, F1, loss, convergence, parameter count,
and training time per checkpoint.  Saves incremental progress to disk so
interrupted runs can be resumed.

Hypotheses:
  H1: Inception increases F1 and EM on the test set
  H2: Inception accelerates convergence (lower loss at each epoch)
  H3: Inception increases compute cost and parameter count
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from TrainTools.train import train
from EvaluateTools.evaluate import evaluate
from Data import load_word_char_mats
from Models import QANet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 456]

COMMON_ARGS = dict(
    train_npz       = "_data/train.npz",
    dev_npz         = "_data/dev.npz",
    word_emb_json   = "_data/word_emb.json",
    char_emb_json   = "_data/char_emb.json",
    train_eval_json = "_data/train_eval.json",
    dev_eval_json   = "_data/dev_eval.json",
    save_dir        = "_model",
    log_dir         = "_log",
    num_steps       = 2000,
    batch_size      = 32,
    optimizer_name  = "adam",
    scheduler_name  = "cosine",
    loss_name       = "qa_nll",
)

RESULTS_DIR = "_results"
PROGRESS_FILE = os.path.join(RESULTS_DIR, "progress.json")


# ── Helpers ──────────────────────────────────────────────────────────────────


def count_parameters(use_inception: bool, epoch_based: bool,
    epoch_amount:  int, norm_name="group_norm") -> dict:
    """Instantiate a model and count trainable / total parameters."""
    args_dict = {
        **COMMON_ARGS,
        "use_inception": use_inception,
        "epoch_based": epoch_based,
        "epoch_amount":  epoch_amount,
        "seed": 0,
        "para_limit": 400, "ques_limit": 50, "char_limit": 16,
        "d_model": 96, "num_heads": 8, "glove_dim": 300, "char_dim": 64,
        "dropout": 0.1, "dropout_char": 0.05, "pretrained_char": False,
        "norm_name": norm_name, "norm_groups": 8,
        "activation": "relu", "init_name": "kaiming", 
    }
    args = argparse.Namespace(**args_dict)
    word_mat, char_mat = load_word_char_mats(args)
    model = QANet(word_mat, char_mat, args)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_progress() -> dict:
    """Load previously saved progress, or return empty state."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_runs": [], "param_counts": {}}


def save_progress(progress: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def is_run_completed(progress: dict, name: str, seed: int) -> bool:
    return any(r["name"] == name and r["seed"] == seed
               for r in progress["completed_runs"])


# ── Single run ───────────────────────────────────────────────────────────────


def run_single(name: str, seed: int, use_inception: bool, epoch_based: bool,
    epoch_amount:  int, norm_name="group_norm",inverse_encoder=False) -> dict:
    """Train + evaluate one configuration with one seed."""
    args = {**COMMON_ARGS, "use_inception": use_inception, "seed": seed, "epoch_based": epoch_based,
    "epoch_amount":  epoch_amount, "norm_name": norm_name, "inverseencoder": inverse_encoder}
    args["save_dir"] = os.path.join("_model", name, f"seed_{seed}")
    args["log_dir"]  = os.path.join("_log", name, f"seed_{seed}")

    print(f"\n{'='*60}")
    print(f"  TRAINING: {name}  seed={seed}")
    print(f"{'='*60}\n")

    t0 = time.time()
    train_results = train(**args)
    train_time = time.time() - t0

    num_checkpoints = len(train_results["history"])
    time_per_checkpoint = train_time / max(num_checkpoints, 1)

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {name}  seed={seed}")
    print(f"{'='*60}\n")
    eval_metrics = evaluate(
        save_dir      = args["save_dir"],
        ckpt_name     = "model.pt",
        dev_npz       = args["dev_npz"],
        word_emb_json = args["word_emb_json"],
        char_emb_json = args["char_emb_json"],
        dev_eval_json = args["dev_eval_json"],
        loss_name     = args["loss_name"],
    )

    return {
        "name":                name,
        "seed":                seed,
        "use_inception":       use_inception,
        "epoch_based":         epoch_based,
        "epoch_amount":        epoch_amount,
        "history":             train_results["history"],
        "best_dev_f1":         train_results["best_f1"],
        "best_dev_em":         train_results["best_em"],
        "test_f1":             eval_metrics["f1"],
        "test_em":             eval_metrics["exact_match"],
        "test_loss":           eval_metrics["loss"],
        "total_train_time":    train_time,
        "time_per_checkpoint": time_per_checkpoint,
        "num_checkpoints":     num_checkpoints,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────


def _stats(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "values": vals}


def aggregate_stats(runs: list) -> dict:
    """Mean ± std across seeds for scalar metrics."""
    if not runs:
        return {}
    return {
        "test_f1":             _stats([r["test_f1"] for r in runs]),
        "test_em":             _stats([r["test_em"] for r in runs]),
        "test_loss":           _stats([r["test_loss"] for r in runs]),
        "best_dev_f1":         _stats([r["best_dev_f1"] for r in runs]),
        "best_dev_em":         _stats([r["best_dev_em"] for r in runs]),
        "time_per_checkpoint": _stats([r["time_per_checkpoint"] for r in runs]),
        "total_train_time":    _stats([r["total_train_time"] for r in runs]),
    }


def aggregate_histories(runs: list) -> dict:
    """Per-step mean ± std for learning curves across seeds."""
    if not runs:
        return {}
    min_len = min(len(r["history"]) for r in runs)
    keys = ["dev_f1", "dev_em", "dev_loss", "train_f1", "train_em", "train_loss"]
    steps = [runs[0]["history"][i]["step"] for i in range(min_len)]
    result = {"steps": steps}
    for key in keys:
        result[key] = [
            {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for vals in (
                [r["history"][i][key] for r in runs]
                for i in range(min_len)
            )
        ]
    return result


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_results(progress: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    baseline_runs  = [r for r in progress["completed_runs"] if r["name"] == "baseline"]
    inception_runs = [r for r in progress["completed_runs"] if r["name"] == "experimental"]
    if not baseline_runs or not inception_runs:
        print("Not enough data to plot.")
        return

    bl_agg  = aggregate_histories(baseline_runs)
    inc_agg = aggregate_histories(inception_runs)
    bl_stats  = aggregate_stats(baseline_runs)
    inc_stats = aggregate_stats(inception_runs)

    # ── 1. Per-seed learning curves ──────────────────────────────────────────
    metrics = [
        ("dev_f1", "Dev F1"), ("dev_em", "Dev EM"), ("dev_loss", "Dev Loss"),
        ("train_f1", "Train F1"), ("train_em", "Train EM"), ("train_loss", "Train Loss"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (key, title) in zip(axes.flatten(), metrics):
        for run in baseline_runs:
            s = [h["step"] for h in run["history"]]
            v = [h[key]    for h in run["history"]]
            ax.plot(s, v, color="tab:blue", alpha=0.35, linewidth=1)
        for run in inception_runs:
            s = [h["step"] for h in run["history"]]
            v = [h[key]    for h in run["history"]]
            ax.plot(s, v, color="tab:orange", alpha=0.35, linewidth=1)
        if bl_agg:
            ax.plot(bl_agg["steps"], [p["mean"] for p in bl_agg[key]],
                    color="tab:blue", linewidth=2, label="baseline (mean)")
        if inc_agg:
            ax.plot(inc_agg["steps"], [p["mean"] for p in inc_agg[key]],
                    color="tab:orange", linewidth=2, label="experimental (mean)")
        ax.set_xlabel("Step"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Baseline vs Experimental — Learning Curves (3 seeds)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "learning_curves.png"), dpi=150)
    plt.close(fig)

    # ── 2. Convergence: mean ± std band for val F1 and val loss ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [(axes[0], "dev_f1", "Validation F1"),
                           (axes[1], "dev_loss", "Validation Loss")]:
        for agg, color, label in [(bl_agg, "tab:blue", "baseline"),
                                  (inc_agg, "tab:orange", "experimental")]:
            if not agg:
                continue
            steps = agg["steps"]
            means = np.array([p["mean"] for p in agg[key]])
            stds  = np.array([p["std"]  for p in agg[key]])
            ax.plot(steps, means, color=color, linewidth=2, label=label)
            ax.fill_between(steps, means - stds, means + stds,
                            color=color, alpha=0.2)
        ax.set_xlabel("Step"); ax.set_ylabel(title)
        ax.set_title(f"{title} (mean ± 1 std)")
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "convergence.png"), dpi=150)
    plt.close(fig)

    # ── 3. Bar charts: test metrics, params, timing ──────────────────────────
    bl_params  = progress["param_counts"].get("baseline", {})
    inc_params = progress["param_counts"].get("experimental", {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Test F1 & EM
    ax = axes[0]
    x = np.arange(2); w = 0.35
    bl_v  = [bl_stats["test_f1"]["mean"],  bl_stats["test_em"]["mean"]]
    bl_e  = [bl_stats["test_f1"]["std"],   bl_stats["test_em"]["std"]]
    inc_v = [inc_stats["test_f1"]["mean"], inc_stats["test_em"]["mean"]]
    inc_e = [inc_stats["test_f1"]["std"],  inc_stats["test_em"]["std"]]
    ax.bar(x - w/2, bl_v,  w, yerr=bl_e,  label="baseline",  capsize=4)
    ax.bar(x + w/2, inc_v, w, yerr=inc_e, label="experimental", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(["F1", "EM"])
    ax.set_title("Test F1 & EM"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # Trainable parameters
    ax = axes[1]
    counts = [bl_params.get("trainable", 0), inc_params.get("trainable", 0)]
    bars = ax.bar(["baseline", "experimental"], counts,
                  color=["tab:blue", "tab:orange"])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Trainable Parameters"); ax.grid(True, alpha=0.3, axis="y")

    # Time per checkpoint
    ax = axes[2]
    vals = [bl_stats["time_per_checkpoint"]["mean"],
            inc_stats["time_per_checkpoint"]["mean"]]
    errs = [bl_stats["time_per_checkpoint"]["std"],
            inc_stats["time_per_checkpoint"]["std"]]
    ax.bar(["baseline", "experimental"], vals, yerr=errs,
           color=["tab:blue", "tab:orange"], capsize=4)
    ax.set_title("Time per Checkpoint (s)"); ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Performance & Efficiency Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "comparison_bars.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {RESULTS_DIR}/")


# ── Summary ──────────────────────────────────────────────────────────────────


def print_summary(progress: dict):
    baseline_runs  = [r for r in progress["completed_runs"] if r["name"] == "baseline"]
    inception_runs = [r for r in progress["completed_runs"] if r["name"] == "experimental"]
    bl  = aggregate_stats(baseline_runs)
    inc = aggregate_stats(inception_runs)
    bl_p  = progress["param_counts"].get("baseline", {})
    inc_p = progress["param_counts"].get("experimental", {})

    def _fmt(s):
        return f"{s['mean']:.4f} +/- {s['std']:.4f}"

    print(f"\n{'='*70}")
    print(f"  AGGREGATED RESULTS  ({len(SEEDS)} seeds: {SEEDS})")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Baseline':>20} {'Experimental':>20}")
    print("-" * 65)
    for label, bk, ik in [
        ("Test F1",              "test_f1",             "test_f1"),
        ("Test EM",              "test_em",             "test_em"),
        ("Test Loss",            "test_loss",           "test_loss"),
        ("Best Dev F1",          "best_dev_f1",         "best_dev_f1"),
        ("Best Dev EM",          "best_dev_em",         "best_dev_em"),
        ("Time/Ckpt (s)",        "time_per_checkpoint", "time_per_checkpoint"),
        ("Total Train Time (s)", "total_train_time",    "total_train_time"),
    ]:
        print(f"{label:<25} {_fmt(bl[bk]):>20} {_fmt(inc[ik]):>20}")

    print(f"\n{'Trainable Params':<25} {bl_p.get('trainable', 'N/A'):>20,}"
          f" {inc_p.get('trainable', 'N/A'):>20,}")
    print(f"{'Total Params':<25} {bl_p.get('total', 'N/A'):>20,}"
          f" {inc_p.get('total', 'N/A'):>20,}")

    # ── Hypothesis evaluation ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  HYPOTHESIS EVALUATION")
    print(f"{'='*70}")

    f1_d = inc["test_f1"]["mean"] - bl["test_f1"]["mean"]
    em_d = inc["test_em"]["mean"] - bl["test_em"]["mean"]
    print(f"\n  H1: Experimental increases F1 and EM on the test set")
    print(f"      dF1 = {f1_d:+.4f},  dEM = {em_d:+.4f}")
    print(f"      -> {'SUPPORTED' if f1_d > 0 and em_d > 0 else 'NOT SUPPORTED'}")

    bl_hist  = aggregate_histories(baseline_runs)
    inc_hist = aggregate_histories(inception_runs)
    if bl_hist and inc_hist:
        lower = sum(1 for b, i in zip(bl_hist["dev_loss"], inc_hist["dev_loss"])
                    if i["mean"] < b["mean"])
        total = len(bl_hist["dev_loss"])
        bl_final  = bl_hist["dev_loss"][-1]["mean"]
        inc_final = inc_hist["dev_loss"][-1]["mean"]
        print(f"\n  H2: Experimental accelerates convergence (lower loss per epoch)")
        print(f"      Experimental lower dev loss at {lower}/{total} checkpoints")
        print(f"      Final dev loss: baseline={bl_final:.4f}, "
              f"experimental={inc_final:.4f}")
        print(f"      -> {'SUPPORTED' if lower > total / 2 else 'NOT SUPPORTED'}")

    param_d = inc_p.get("trainable", 0) - bl_p.get("trainable", 0)
    time_d  = inc["time_per_checkpoint"]["mean"] - bl["time_per_checkpoint"]["mean"]
    print(f"\n  H3: Experimental increases compute cost and parameter count")
    print(f"      dParams = {param_d:+,}")
    print(f"      dTime/Ckpt = {time_d:+.2f}s")
    print(f"      -> {'SUPPORTED' if param_d > 0 and time_d > 0 else 'NOT SUPPORTED'}")


# ── Final results JSON ───────────────────────────────────────────────────────


def save_final_results(progress: dict):
    baseline_runs  = [r for r in progress["completed_runs"] if r["name"] == "baseline"]
    inception_runs = [r for r in progress["completed_runs"] if r["name"] == "experimental"]
    final = {
        "seeds": SEEDS,
        "num_runs_per_condition": len(SEEDS),
        "baseline": {
            "aggregated":          aggregate_stats(baseline_runs),
            "param_counts":        progress["param_counts"].get("baseline", {}),
            "per_seed":            baseline_runs,
            "history_aggregated":  aggregate_histories(baseline_runs),
        },
        "experimental": {
            "aggregated":          aggregate_stats(inception_runs),
            "param_counts":        progress["param_counts"].get("experimental", {}),
            "per_seed":            inception_runs,
            "history_aggregated":  aggregate_histories(inception_runs),
        },
    }
    path = os.path.join(RESULTS_DIR, "inception_study.json")
    with open(path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFinal results saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    progress = load_progress()

    # Count parameters once
    for name, use_inc in [("baseline", False), ("experimental", True)]:
        if name not in progress["param_counts"]:
            print(f"Counting {name} parameters...")
            progress["param_counts"][name] = count_parameters(use_inception=use_inc)
            save_progress(progress)

    print("\nParameter counts:")
    for name, counts in progress["param_counts"].items():
        print(f"  {name}: {counts['trainable']:,} trainable / "
              f"{counts['total']:,} total")

    # Paired runs: same seed for baseline & inception
    for seed in SEEDS:
        for name, use_inception in [("baseline", False), ("experimental", True)]:
            if is_run_completed(progress, name, seed):
                print(f"\nSkipping {name} seed={seed} (already completed)")
                continue
            result = run_single(name, seed, use_inception)
            progress["completed_runs"].append(result)
            save_progress(progress)
            print(f"\nProgress saved after {name} seed={seed}")

    # Final outputs
    save_final_results(progress)
    plot_results(progress)
    print_summary(progress)