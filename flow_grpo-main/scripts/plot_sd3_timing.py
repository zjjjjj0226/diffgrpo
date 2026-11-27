"""Plot the durable timing logs; run from repository root."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/sd3")
    args = parser.parse_args()
    root = Path(args.input)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    epochs = pd.read_json(root / "epochs_rank0.jsonl", lines=True)
    rewards = pd.read_json(root / "rewards_rank0.jsonl", lines=True)
    rewards = rewards.dropna(subset=["reward_ocr"]).copy()
    phases = pd.read_csv(root / "phases_rank0.csv")
    assert not epochs.epoch.duplicated().any(), "Multiple runs found: split logs before plotting"
    interrupted = set(phases.loc[~phases.completed.astype(str).str.lower().eq("true"), "epoch"])
    seconds = pd.DataFrame([{key: value["seconds"] for key, value in row.items()} for row in epochs.phases]).fillna(0)
    seconds.index = epochs.epoch
    steady_ids = epochs.loc[(epochs.epoch > 0) & ~epochs.epoch.isin(interrupted), "epoch"]
    steady_ids = [i for i in steady_ids if seconds.loc[i].get("evaluation", 0) < 1 and seconds.loc[i].get("checkpoint", 0) < 1]
    steady = seconds.loc[steady_ids]
    summary = pd.DataFrame({"mean_seconds": steady.mean(), "total_seconds": steady.sum()})
    duration = epochs.set_index("epoch").loc[steady_ids, "epoch_seconds"]
    summary["percent"] = summary.total_seconds / duration.sum() * 100
    summary.sort_values("mean_seconds", ascending=False).to_csv(out / "phase_summary.csv")
    rewards.to_csv(out / "training_rewards.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(rewards.elapsed_seconds / 60, rewards.reward_ocr, alpha=.45, label="Training OCR reward")
    ax.plot(rewards.elapsed_seconds / 60, rewards.reward_ocr.rolling(5, min_periods=1).mean(), label="5-epoch rolling mean")
    ax.set(xlabel="Elapsed wall time (min, includes initial evaluation)", ylabel="Reward", title="Training reward vs time")
    ax.legend()
    ax = axes[0, 1]
    ordered = summary[summary.percent > .05].sort_values("percent")
    ax.barh(ordered.index, ordered.percent)
    ax.set(xlabel="Share of steady epoch wall time (%)", title="Steady training bottlenecks (excluding evaluation epochs)")
    ax = axes[1, 0]
    for key in ["sampling_including_vae", "backward_call", "train_forward", "text_encoding", "reward_wait_and_transfer"]:
        ax.plot(steady.index, steady[key], label=key)
    ax.set(xlabel="Epoch", ylabel="Seconds", title="Phase duration per steady epoch")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    by_step = phases[phases.epoch.isin(steady_ids) & phases.phase.isin(["train_forward", "backward_call", "optimizer_calls"])].groupby(["timestep", "phase"]).seconds.mean().unstack()
    by_step.plot.bar(ax=ax)
    ax.set(xlabel="Training timestep index", ylabel="Mean seconds per call", title="Training call time by timestep")
    ax.legend(fontsize=8)
    fig.savefig(out / "sd3_analysis.png", dpi=180)
    fig.savefig(out / "sd3_analysis.pdf")

    # When distributed logs are present, visualize rank balance separately.
    rank1_path = root / "epochs_rank1.jsonl"
    rank_stats = None
    if rank1_path.exists():
        rank1 = pd.read_json(rank1_path, lines=True)
        rank_compare = epochs[["epoch", "epoch_seconds"]].merge(
            rank1[["epoch", "epoch_seconds"]], on="epoch", suffixes=("_rank0", "_rank1")
        )
        rank_compare["rank1_minus_rank0_seconds"] = (
            rank_compare.epoch_seconds_rank1 - rank_compare.epoch_seconds_rank0
        )
        rank_compare.to_csv(out / "rank_comparison.csv", index=False)
        rfig, raxes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        raxes[0].plot(rank_compare.epoch, rank_compare.epoch_seconds_rank0, "o-", ms=3, label="rank 0")
        raxes[0].plot(rank_compare.epoch, rank_compare.epoch_seconds_rank1, ".--", ms=4, label="rank 1")
        raxes[0].set(xlabel="Epoch", ylabel="Seconds", title="Epoch wall time by rank")
        raxes[0].legend()
        raxes[1].bar(rank_compare.epoch, rank_compare.rank1_minus_rank0_seconds)
        raxes[1].axhline(0, color="black", lw=.8)
        raxes[1].set(xlabel="Epoch", ylabel="rank 1 - rank 0 (seconds)", title="Cross-rank timing imbalance")
        rfig.savefig(out / "rank_comparison.png", dpi=180)
        rfig.savefig(out / "rank_comparison.pdf")
        delta = rank_compare.rank1_minus_rank0_seconds
        rank_stats = {"mean_delta_seconds": delta.mean(), "std_delta_seconds": delta.std(),
                      "max_abs_delta_seconds": delta.abs().max()}
    total = epochs.epoch_seconds.sum()
    stats = {"epochs": len(epochs), "interrupted_epochs": sorted(interrupted), "steady_epochs": len(steady_ids),
             "logged_epoch_minutes": total / 60, "elapsed_minutes": epochs.elapsed_seconds.iloc[-1] / 60,
             "evaluation_minutes": seconds.evaluation.sum() / 60,
             "evaluation_percent_of_epochs": seconds.evaluation.sum() / total * 100,
             "steady_epoch_mean_seconds": duration.mean(), "steady_epoch_std_seconds": duration.std(),
             "first_reward": rewards.reward_ocr.iloc[0], "last_reward": rewards.reward_ocr.iloc[-1],
             "first_10_reward_mean": rewards.reward_ocr.head(10).mean(), "last_10_reward_mean": rewards.reward_ocr.tail(10).mean(),
             "best_reward": rewards.reward_ocr.max(), "best_reward_epoch": int(rewards.loc[rewards.reward_ocr.idxmax(), "epoch"]),
             "phase_summary": summary.to_dict(orient="index"), "rank_comparison": rank_stats}
    (out / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
