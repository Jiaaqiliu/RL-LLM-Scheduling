"""
Result visualization script for RL-LLM-Scheduling experiments.
Generates paper-quality plots matching Llumnix paper Figure 11 format.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import argparse

matplotlib.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'Round-Robin': '#888888',
    'INFaaS++': '#ff7f0e',
    'Llumnix-Heuristic': '#2ca02c',
    'Llumnix-RL': '#1f77b4',
}

MARKERS = {
    'Round-Robin': 's',
    'INFaaS++': '^',
    'Llumnix-Heuristic': 'o',
    'Llumnix-RL': 'D',
}

LINESTYLES = {
    'Round-Robin': ':',
    'INFaaS++': '--',
    'Llumnix-Heuristic': '-.',
    'Llumnix-RL': '-',
}


def load_results(results_dir):
    """Load benchmark results from a directory."""
    results_path = Path(results_dir)
    summary_file = results_path / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)

    latency_file = results_path / "latencies.csv"
    if latency_file.exists():
        df = pd.read_csv(latency_file)
        return {
            "request_p99": np.percentile(df["end_to_end"], 99),
            "request_mean": np.mean(df["end_to_end"]),
            "prefill_p99": np.percentile(df["prefill"], 99),
            "prefill_mean": np.mean(df["prefill"]),
            "decode_p99": np.percentile(df["decode"], 99),
            "decode_mean": np.mean(df["decode"]),
            "preemption_loss": df["preemption_loss"].sum() if "preemption_loss" in df else 0,
        }

    return None


def plot_serving_comparison(results_dict, x_values, x_label, output_path, title=""):
    """
    Plot 7-column comparison matching Llumnix paper Figure 11.

    results_dict: {policy_name: {x_value: metrics_dict}}
    """
    metrics = [
        ("request_p99", "Request P99 (s)"),
        ("request_mean", "Request Mean (s)"),
        ("prefill_p99", "Prefill P99 (s)"),
        ("prefill_mean", "Prefill Mean (s)"),
        ("decode_p99", "Decode P99 (s)"),
        ("decode_mean", "Decode Mean (s)"),
        ("preemption_loss", "Preemption Loss (s)"),
    ]

    fig, axes = plt.subplots(1, 7, figsize=(24, 3))
    if title:
        fig.suptitle(title, fontsize=13, y=1.05)

    for col, (metric_key, ylabel) in enumerate(metrics):
        ax = axes[col]
        for policy_name, policy_results in results_dict.items():
            y_values = [policy_results.get(x, {}).get(metric_key, 0) for x in x_values]
            ax.plot(
                x_values, y_values,
                color=COLORS.get(policy_name, '#333333'),
                marker=MARKERS.get(policy_name, 'o'),
                linestyle=LINESTYLES.get(policy_name, '-'),
                label=policy_name,
                markersize=5,
                linewidth=1.5,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        if col == 0:
            ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_curves(log_dir, output_path):
    """Plot RL training curves (reward vs. timesteps)."""
    # Try loading from TensorBoard logs
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()

        if 'rollout/ep_rew_mean' in ea.scalars.Keys():
            events = ea.scalars.Items('rollout/ep_rew_mean')
            steps = [e.step for e in events]
            rewards = [e.value for e in events]

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(steps, rewards, color='#1f77b4', linewidth=1.5)
            ax.set_xlabel('Training Timesteps')
            ax.set_ylabel('Mean Episode Reward')
            ax.set_title('RL Agent Training Curve')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")
            return
    except (ImportError, Exception) as e:
        print(f"Could not load TensorBoard logs: {e}")

    print("No training data found. Generate after training.")


def plot_ablation_rewards(ablation_results, output_path):
    """
    Plot reward weight ablation study.

    ablation_results: {config_name: {metric: value}}
    """
    configs = list(ablation_results.keys())
    metrics = ["request_p99", "request_mean", "decode_p99", "preemption_loss"]
    metric_labels = ["Request P99", "Request Mean", "Decode P99", "Preemption Loss"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))

    x = np.arange(len(configs))
    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [ablation_results[c].get(metric, 0) for c in configs]
        axes[col].bar(x, values, color='#1f77b4', alpha=0.8)
        axes[col].set_xticks(x)
        axes[col].set_xticklabels(configs, rotation=45, ha='right', fontsize=7)
        axes[col].set_ylabel(label)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--results_dir", type=str, default="./results/")
    parser.add_argument("--output_dir", type=str, default="./results/figures/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Result plotting utility ready.")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print("")
    print("Usage examples:")
    print("  After running experiments, place results in:")
    print("    results/exp1/sharegpt_rl_qps7.5/summary.json")
    print("  Then run this script to generate figures.")


if __name__ == "__main__":
    main()
