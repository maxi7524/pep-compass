from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_results(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("agent_*_results.json"))
    if not files:
        files = sorted(results_dir.glob("*_results.json"))
    runs: list[dict] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_file"] = path.name
        runs.append(data)
    return runs


def plot_losses_and_returns(results_dir: str = "results\\basic_eps_greedy_rl") -> None:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_results(out_dir)
    if not runs:
        raise FileNotFoundError(f"No result JSON files in {out_dir}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    loss_ax, ret_ax = axes

    for idx, run in enumerate(runs):
        name = run.get("run_name", f"agent_{idx + 1}")
        losses = np.asarray(run.get("epoch_losses", []), dtype=float)
        returns = np.asarray(run.get("epoch_returns", []), dtype=float)
        x_loss = np.arange(1, len(losses) + 1)
        x_ret = np.arange(1, len(returns) + 1)
        loss_ax.plot(x_loss, losses, linewidth=1.2, alpha=0.9, label=name)
        ret_ax.plot(x_ret, returns, linewidth=1.2, alpha=0.9, label=name)

    loss_ax.set_title("Basic epsilon-greedy RL: epoch loss")
    loss_ax.set_ylabel("Loss = -mean(step_rewards)")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend(loc="best", fontsize=8)

    ret_ax.set_title("Basic epsilon-greedy RL: epoch return")
    ret_ax.set_xlabel("Epoch")
    ret_ax.set_ylabel("Return = sum(step_rewards)")
    ret_ax.grid(alpha=0.25)
    ret_ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "losses_returns.png", dpi=180)
    plt.close(fig)

    summary_lines = []
    for run in runs:
        best = float(run.get("best_log2mic", float("nan")))
        name = run.get("run_name", run.get("_file", "run"))
        summary_lines.append(f"{name}: best_log2mic={best:.6f}")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results\\basic_eps_greedy_rl")
    args = parser.parse_args()
    plot_losses_and_returns(args.results_dir)
