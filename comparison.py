"""
inception_study.py — Train baseline vs inception QANet, evaluate both, save & plot results.
"""

import json
import os

import matplotlib.pyplot as plt

from TrainTools.train import train
from EvaluateTools.evaluate import evaluate


COMMON_ARGS = dict(
    train_npz       = "_data/train.npz",
    dev_npz         = "_data/dev.npz",
    word_emb_json   = "_data/word_emb.json",
    char_emb_json   = "_data/char_emb.json",
    train_eval_json = "_data/train_eval.json",
    dev_eval_json   = "_data/dev_eval.json",
    save_dir        = "_model",
    log_dir         = "_log",
    num_steps       = 3000,
    batch_size      = 16,
    seed            = 42,
    optimizer_name  = "adam",
    scheduler_name  = "lambda",
    loss_name       = "qa_nll",
)

RESULTS_DIR = "_results"


def run_experiment(name, extra_args=None):
    """Train, evaluate, and return combined results for one configuration."""
    args = {**COMMON_ARGS}
    if extra_args:
        args.update(extra_args)

    # Use separate save/log dirs so checkpoints don't overwrite each other
    args["save_dir"] = os.path.join("_model", name)
    args["log_dir"] = os.path.join("_log", name)

    print(f"\n{'='*60}")
    print(f"  TRAINING: {name}")
    print(f"{'='*60}\n")
    train_results = train(**args)

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {name}")
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
        "name":         name,
        "history":      train_results["history"],
        "best_f1":      train_results["best_f1"],
        "best_em":      train_results["best_em"],
        "test_f1":      eval_metrics["f1"],
        "test_em":      eval_metrics["exact_match"],
        "test_loss":    eval_metrics["loss"],
    }


def save_results(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "inception_study.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {path}")


def plot_results(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics = [
        ("dev_f1",   "Dev F1"),
        ("dev_em",   "Dev EM"),
        ("dev_loss", "Dev Loss"),
        ("train_f1", "Train F1"),
        ("train_em", "Train EM"),
        ("train_loss", "Train Loss"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        for result in all_results:
            steps = [h["step"] for h in result["history"]]
            values = [h[key] for h in result["history"]]
            ax.plot(steps, values, label=result["name"], marker="o", markersize=3)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Baseline vs Inception", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "inception_study.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


def print_summary(all_results):
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    header = f"{'Name':<20} {'Best F1':>8} {'Best EM':>8} {'Test F1':>8} {'Test EM':>8} {'Test Loss':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['name']:<20} {r['best_f1']:>8.2f} {r['best_em']:>8.2f} "
              f"{r['test_f1']:>8.2f} {r['test_em']:>8.2f} {r['test_loss']:>10.4f}")


if __name__ == "__main__":
    results = []

    results.append(run_experiment("baseline"))
    results.append(run_experiment("inception", extra_args={"use_inception": True}))

    save_results(results)
    plot_results(results)
    print_summary(results)
