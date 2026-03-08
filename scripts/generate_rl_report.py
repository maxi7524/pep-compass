"""
Generate a multi-page PDF report for the RL peptide optimisation runs.

Usage
-----
    python scripts/generate_rl_report.py
    python scripts/generate_rl_report.py --results_dir results --output results/rl_report.pdf

Requirements: matplotlib, numpy  (already in the project venv)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    "#2274A5", "#E84855", "#3B1F2B", "#F9A620", "#54C6EB",
    "#2DC653", "#9B59B6", "#E67E22", "#1ABC9C", "#C0392B",
]

SEED_ORDER = [
    "middle-1", "jurand-4", "jurand-2", "jurand-7",
    "mammuthusin-3", "hydrodamin-2",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_run(data: dict, fpath: str) -> dict:
    """Normalize a single run's data, adding defaults and computed fields."""
    data["_file"] = os.path.basename(fpath)
    # normalise key names (support both old and new JSON schema)
    data.setdefault("start_log2mic",       data.get("start_score", 0.0))
    data.setdefault("best_log2mic",        data.get("best_score",  0.0))
    data.setdefault("start_mic_uM",        2 ** data["start_log2mic"])
    data.setdefault("best_mic_uM",         2 ** data["best_log2mic"])
    data.setdefault("improvement_log2mic", data.get("improvement", 0.0))
    data.setdefault("fold_improvement",
                    data["start_mic_uM"] / max(data["best_mic_uM"], 1e-9))
    data.setdefault("algorithm", "DQN")  # default for legacy files
    return data


def _get_seed_name_from_run(run: dict) -> str | None:
    """Extract the seed peptide name from a run (stripping td3_ prefix if present)."""
    name = run.get("run_name", run.get("_file", ""))
    # Strip td3_ prefix for TD3 runs
    if name.startswith("td3_"):
        name = name[4:]
    # Check if it matches a known seed
    for seed in SEED_ORDER:
        if seed in name:
            return seed
    return None


def load_results(results_dir: str) -> list[dict]:
    """Load DQN results (excluding TD3 runs) for backward compatibility."""
    pattern = os.path.join(results_dir, "*_results.json")
    files = sorted(glob.glob(pattern))
    # keep only the 6 benchmark runs (exclude TD3 runs and old unlabelled runs)
    files = [f for f in files if any(
        n in os.path.basename(f)
        for n in ["middle-1", "jurand-2", "jurand-4", "jurand-7",
                  "mammuthusin-3", "hydrodamin-2"]
    ) and "td3_" not in os.path.basename(f)
        and "_a2c_" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(
            f"No benchmark result JSON files found in {results_dir!r}. "
            "Run the optimiser first."
        )
    runs: list[dict] = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as fh:
            data = json.load(fh)
        runs.append(_normalize_run(data, fpath))

    # sort by SEED_ORDER
    def sort_key(r):
        name = r.get("run_name", r["_file"])
        try:
            return SEED_ORDER.index(name)
        except ValueError:
            return 99

    runs.sort(key=sort_key)
    return runs


def load_td3_results(results_dir: str) -> list[dict]:
    """Load TD3 results (files with td3_ prefix in run_name or filename)."""
    pattern = os.path.join(results_dir, "*_results.json")
    files = sorted(glob.glob(pattern))
    # keep only TD3 benchmark runs
    files = [f for f in files if "td3_" in os.path.basename(f) and any(
        n in os.path.basename(f)
        for n in ["middle-1", "jurand-2", "jurand-4", "jurand-7",
                  "mammuthusin-3", "hydrodamin-2"]
    )]
    runs: list[dict] = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as fh:
            data = json.load(fh)
        data = _normalize_run(data, fpath)
        data["algorithm"] = "TD3"
        runs.append(data)

    # sort by SEED_ORDER (stripping td3_ prefix for matching)
    def sort_key(r):
        seed = _get_seed_name_from_run(r)
        if seed:
            try:
                return SEED_ORDER.index(seed)
            except ValueError:
                pass
        return 99

    runs.sort(key=sort_key)
    return runs


def load_a2c_results(results_dir: str) -> list[dict]:
    """Load A2C results (files with a2c_ in filename or model_type == 'A2C' in JSON)."""
    pattern = os.path.join(results_dir, "*_results.json")
    files = sorted(glob.glob(pattern))
    runs: list[dict] = []
    
    for fpath in files:
        basename = os.path.basename(fpath)
        # Check if it's a benchmark seed
        is_benchmark = any(
            n in basename
            for n in ["middle-1", "jurand-2", "jurand-4", "jurand-7",
                      "mammuthusin-3", "hydrodamin-2"]
        )
        if not is_benchmark:
            continue
        
        with open(fpath, encoding="utf-8") as fh:
            data = json.load(fh)
        
        # Check if it's an A2C run (either by filename prefix or model_type field)
        is_a2c = "a2c_" in basename or data.get("model_type") == "A2C"
        if not is_a2c:
            continue
        
        data = _normalize_run(data, fpath)
        data["algorithm"] = "A2C"
        runs.append(data)

    # sort by SEED_ORDER
    def sort_key(r):
        seed = _get_seed_name_from_run(r)
        if seed:
            try:
                return SEED_ORDER.index(seed)
            except ValueError:
                pass
        return 99

    runs.sort(key=sort_key)
    return runs


def load_all_results(results_dir: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Load DQN, TD3, and A2C results.
    
    Returns
    -------
    dqn_runs : list[dict]
        DQN benchmark runs.
    td3_runs : list[dict]
        TD3 benchmark runs (may be empty if not yet run).
    a2c_runs : list[dict]
        A2C benchmark runs (may be empty if not yet run).
    """
    try:
        dqn_runs = load_results(results_dir)
    except FileNotFoundError:
        dqn_runs = []
    
    td3_runs = load_td3_results(results_dir)
    a2c_runs = load_a2c_results(results_dir)
    
    if not dqn_runs and not td3_runs and not a2c_runs:
        raise FileNotFoundError(
            f"No benchmark result JSON files found in {results_dir!r}. "
            "Run the optimiser first."
        )
    
    return dqn_runs, td3_runs, a2c_runs


# ─────────────────────────────────────────────────────────────────────────────
# Page helpers
# ─────────────────────────────────────────────────────────────────────────────

def _new_page(pdf: PdfPages, title: str = "", figsize=(11.69, 8.27)) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
    return fig


def _footer(fig: plt.Figure, text: str = "") -> None:
    fig.text(0.5, 0.01, text or f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             ha="center", fontsize=7, color="grey")


def _save(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Cover / title
# ─────────────────────────────────────────────────────────────────────────────

def page_cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#1a1a2e", transform=ax.transAxes))

    ax.text(0.5, 0.72, "RL Peptide Optimizer",
            ha="center", va="center", fontsize=34, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.62,
            "DQN-based Antimicrobial Peptide Optimisation\nin the HydrAMP Latent Space",
            ha="center", va="center", fontsize=18, color="#aaaacc",
            transform=ax.transAxes, linespacing=1.5)
    ax.text(0.5, 0.48,
            "Objective: minimise mean log₂ MIC against E. coli\n"
            "via Deep Q-Network + MUTANG++ + APEX",
            ha="center", va="center", fontsize=13, color="#ccccee",
            transform=ax.transAxes, linespacing=1.6)
    ax.text(0.5, 0.33,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ha="center", va="center", fontsize=11, color="#888888",
            transform=ax.transAxes)
    ax.text(0.5, 0.20,
            "Runs: 6 benchmark seed peptides  ·  100 episodes each  ·  GPU: NVIDIA RTX 5000 Ada",
            ha="center", va="center", fontsize=10, color="#666688",
            transform=ax.transAxes)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Methodology
# ─────────────────────────────────────────────────────────────────────────────

METHODOLOGY_TEXT = """\
PROBLEM
Given a seed antimicrobial peptide (AMP), find a mutant sequence with lower Minimum Inhibitory
Concentration (MIC) against Escherichia coli.  Lower MIC = more potent antimicrobial activity.
Objective score:  score(p) = (1/3) Σᵢ log₂(MICᵢ(p))  over E. coli strains i ∈ {1, 2, 3}.

COMPONENTS
  HydrAMP encoder–decoder (VAE):
    Maps peptides ↔ 64-dimensional latent vectors.  Used to encode the current state and to
    represent candidate actions.  Latent dim = 64, ambient dim = 525 (25 positions × 21 tokens).

  MUTANG++ (decoder Jacobian SVD):
    Computes the decoder Jacobian at the current latent point, decomposes it via SVD to obtain
    local tangent directions, and enumerates per-position amino-acid substitutions.
    Candidates with softmax-normalised probability > 1/k (uniform baseline) are kept.

  DecoderLogProbPotential:
    Scores each mutation by its log-probability under the HydrAMP decoder.
    The parent sequence is explicitly removed before softmax to prevent trivial no-op actions.

  APEX MIC predictor:
    Black-box oracle predicting MIC (µM) for 11 pathogens.  We use mean log₂ over three
    E. coli strains (APEX indices 1, 2, 3: ATCC 11775, AIG221, AIG222).

  DQN Agent:
    Standard Deep Q-Network with experience replay (capacity 10 000) and a hard-updated
    target network.  Q-Network: [s‖a] (128) → Linear(256) → ReLU → Linear(128) → ReLU
    → Linear(64) → ReLU → Linear(1).

MDP FORMULATION
  State  sₜ: 64-D HydrAMP latent vector of the current peptide.
  Action aₜ: latent vector of a chosen mutant candidate (≤ max_candidates = 40 per step).
  Reward rₜ: max(0, ep_best_so_far − score(aₜ))  — positive only when setting a new
              episode-best.  Episode return = start_score − min(trajectory_scores).
  Horizon: 20 steps per episode.

EXPLORATION
  ε reset at episode start with a linear schedule: ε = 1.0 → 0.05 over 100 episodes.
  This prevents premature convergence to a local mutation after early episodes.

HYPERPARAMETERS
  Episodes: 100   Steps/ep: 20   Candidates/step: 40   lr: 1e-3   γ: 0.99
  Batch: 32   Buffer: 10 000   Target-net sync: every 50 updates   Device: NVIDIA RTX 5000 Ada
"""


def page_methodology(pdf: PdfPages) -> None:
    fig = _new_page(pdf, "Methodology")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88])
    ax.set_axis_off()
    ax.text(0.0, 1.0, METHODOLOGY_TEXT,
            va="top", ha="left", fontsize=8.2,
            fontfamily="monospace",
            transform=ax.transAxes, wrap=True,
            linespacing=1.45)
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Summary results table
# ─────────────────────────────────────────────────────────────────────────────

def page_summary_table(pdf: PdfPages, runs: list[dict]) -> None:
    fig = _new_page(pdf, "Summary Results — All 6 Benchmark Seed Peptides")
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.82])
    ax.set_axis_off()

    col_labels = [
        "Name", "Start peptide", "Start\nlog₂MIC", "Start\nMIC (µM)",
        "Best peptide", "Best\nlog₂MIC", "Best\nMIC (µM)",
        "Δ log₂MIC", "Fold ↓",
    ]
    table_data = []
    row_colors = []
    for r in runs:
        name = r.get("run_name", r["_file"])
        imp = r["improvement_log2mic"]
        fold = r["fold_improvement"]
        table_data.append([
            name,
            r["start_peptide"],
            f"{r['start_log2mic']:.3f}",
            f"{r['start_mic_uM']:.1f}",
            r["best_peptide"],
            f"{r['best_log2mic']:.3f}",
            f"{r['best_mic_uM']:.1f}",
            f"{imp:+.3f}",
            f"{fold:.1f}×",
        ])
        # green rows = improved, grey = no improvement
        c = "#d4edda" if imp > 0.01 else "#f5f5f5"
        row_colors.append([c] * len(col_labels))

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)

    # style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2274A5")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Scores are mean log₂(MIC) over 3 E. coli strains (APEX).  "
        "Δ log₂MIC = start − best (positive = improvement).  "
        "Green rows = improvement found.",
        fontsize=8, style="italic", pad=10,
    )
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — MIC before / after bar chart
# ─────────────────────────────────────────────────────────────────────────────

def page_mic_comparison(pdf: PdfPages, runs: list[dict]) -> None:
    fig = _new_page(pdf, "E. coli MIC: Start vs Best (log₂ scale)")
    ax = fig.add_axes([0.1, 0.18, 0.85, 0.72])

    names        = [r.get("run_name", r["_file"]) for r in runs]
    start_log2   = [r["start_log2mic"] for r in runs]
    best_log2    = [r["best_log2mic"]  for r in runs]

    x = np.arange(len(names))
    w = 0.35

    bars1 = ax.bar(x - w / 2, start_log2, w, label="Start log₂MIC", color="#aec6cf", edgecolor="grey", linewidth=0.6)
    bars2 = ax.bar(x + w / 2, best_log2,  w, label="Best log₂MIC",  color="#77dd77", edgecolor="grey", linewidth=0.6)

    # annotate bars with µM values
    for bar, r in zip(bars1, runs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{r['start_mic_uM']:.0f}µM",
                ha="center", va="bottom", fontsize=6.5, color="#444444")
    for bar, r in zip(bars2, runs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{r['best_mic_uM']:.1f}µM",
                ha="center", va="bottom", fontsize=6.5, color="#1a6b2f")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Mean log₂(MIC) over E. coli strains", fontsize=11)
    ax.set_title("Lower = more potent.  Numbers above bars show MIC in µM.", fontsize=9, style="italic")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Global best score trajectory (all 6 on one plot)
# ─────────────────────────────────────────────────────────────────────────────

def page_global_best_trajectories(pdf: PdfPages, runs: list[dict]) -> None:
    fig = _new_page(pdf, "Global Best Score Trajectory per Seed Peptide")
    ax = fig.add_axes([0.1, 0.15, 0.85, 0.74])

    for i, r in enumerate(runs):
        name = r.get("run_name", r["_file"])
        best_scores = r["all_best_scores"]
        global_min = np.minimum.accumulate(best_scores)
        eps = np.arange(1, len(best_scores) + 1)
        ax.plot(eps, global_min, label=name, color=PALETTE[i % len(PALETTE)], lw=2)
        ax.axhline(r["start_log2mic"], color=PALETTE[i % len(PALETTE)],
                   ls=":", lw=0.9, alpha=0.5)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Global best log₂MIC", fontsize=11)
    ax.set_title("Dotted lines = starting log₂MIC for each seed.", fontsize=9, style="italic")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.35)
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Per-run reward curves (2 × 3 grid)
# ─────────────────────────────────────────────────────────────────────────────

def page_reward_curves(pdf: PdfPages, runs: list[dict]) -> None:
    n = min(len(runs), 6)  # cap at 6 subplots per page
    runs = runs[:n]
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig = _new_page(pdf, "Per-Episode Reward (= episode log₂MIC improvement)")
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.40,
                           left=0.08, right=0.97, top=0.88, bottom=0.08)
    for idx, r in enumerate(runs):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        name = r.get("run_name", r["_file"])
        ep_rewards  = r["episode_rewards"]
        best_scores = r["all_best_scores"]
        global_min  = np.minimum.accumulate(best_scores)
        eps = np.arange(1, len(ep_rewards) + 1)

        ax2 = ax.twinx()
        ax.bar(eps, ep_rewards, color="#2274A5", alpha=0.5, width=0.8)
        ax2.plot(eps, global_min, color="#E84855", lw=1.6, label="global best")
        ax2.axhline(r["start_log2mic"], color="grey", ls="--", lw=0.9)

        ax.set_title(f"{name}\n{r['start_peptide']} → {r['best_peptide']}",
                     fontsize=7, pad=3)
        ax.set_xlabel("Episode", fontsize=7)
        ax.set_ylabel("Reward (Δlog₂MIC)", fontsize=6.5, color="#2274A5")
        ax2.set_ylabel("Best log₂MIC", fontsize=6.5, color="#E84855")
        ax.tick_params(axis="both", labelsize=6)
        ax2.tick_params(axis="both", labelsize=6)
        ax.tick_params(axis="y", labelcolor="#2274A5")
        ax2.tick_params(axis="y", labelcolor="#E84855")

    # hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 7 — Epsilon decay illustration
# ─────────────────────────────────────────────────────────────────────────────

def page_epsilon_schedule(pdf: PdfPages) -> None:
    fig = _new_page(pdf, "Exploration Schedule (ε-greedy)")
    ax = fig.add_axes([0.12, 0.18, 0.82, 0.68])

    eps_start, eps_end, n_episodes = 1.0, 0.05, 100
    episodes = np.arange(1, n_episodes + 1)
    frac = (episodes - 1) / (n_episodes - 1)
    epsilon = eps_start + frac * (eps_end - eps_start)

    ax.plot(episodes, epsilon, lw=2.5, color="#2274A5")
    ax.fill_between(episodes, epsilon, alpha=0.15, color="#2274A5")
    ax.axhline(eps_end, color="grey", ls="--", lw=1)
    ax.text(n_episodes * 0.6, eps_end + 0.02, f"ε_min = {eps_end}", fontsize=9, color="grey")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("ε (exploration probability)", fontsize=11)
    ax.set_title(
        "Per-episode linear schedule: ε is reset at the start of each episode.\n"
        "Ensures the agent remains explorative throughout training.",
        fontsize=9, style="italic",
    )
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 8 — Top peptides per run
# ─────────────────────────────────────────────────────────────────────────────

def page_top_peptides(pdf: PdfPages, runs: list[dict]) -> None:
    fig = _new_page(pdf, "Top Discovered Peptides per Seed")
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])
    ax.set_axis_off()

    lines = []
    for r in runs:
        name = r.get("run_name", r["_file"])
        lines.append(f"{'─'*80}")
        lines.append(
            f"  {name:20s}  "
            f"Start: {r['start_peptide']:22s}  "
            f"{r['start_log2mic']:6.3f} log₂MIC  ({r['start_mic_uM']:6.1f} µM)"
        )
        # deduplicated top-5
        scores_peps = sorted(
            zip(r["all_best_scores"], r["all_best_peptides"]), key=lambda x: x[0]
        )
        seen: set[str] = set()
        top: list[tuple[float, str]] = []
        for sc, pep in scores_peps:
            if pep not in seen:
                top.append((sc, pep))
                seen.add(pep)
            if len(top) == 5:
                break

        for rank, (sc, pep) in enumerate(top, 1):
            mic = 2 ** sc
            lines.append(
                f"    #{rank}  {pep:22s}  "
                f"{sc:6.3f} log₂MIC  ({mic:6.1f} µM)"
            )

    text = "\n".join(lines)
    ax.text(0.0, 1.0, text, va="top", ha="left",
            fontsize=8, fontfamily="monospace",
            transform=ax.transAxes, linespacing=1.5)
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 9 — Known issues / limitations
# ─────────────────────────────────────────────────────────────────────────────

PROBLEMS_TEXT = """\
KNOWN ISSUES AND LIMITATIONS

1. mammuthusin-3 (KTLKIIRLLF) — No improvement found
   The agent explored actively across all 100 episodes (unique peptides visited in every
   trajectory), but every mutation proposed by MUTANG++ was scored higher (worse) by APEX
   than the starting sequence.  KTLKIIRLLF appears to sit at a local minimum of the APEX
   E. coli surface within the region reachable by single tangent-space mutations.
   Possible remedies: multi-step look-ahead (MCTS / beam search), larger max_candidates,
   or starting from a different seed in the same peptide family.

2. Action space collapse (fixed)
   Earlier versions suffered from the parent peptide dominating the softmax (prob ≈ 0.9996)
   because the HydrAMP decoder assigns highest log-prob to parent residues.  Fixed by
   explicitly stripping the parent sequence before computing softmax.

3. Premature convergence (fixed)
   Multiplicative per-step ε decay caused ε to hit its minimum after ~29 episodes, leaving
   the agent in pure-exploitation mode.  Fixed by a per-episode linear ε reset schedule.

4. Short peptide / low diversity for mammuthusin-3
   KTLKIIRLLF is only 10 AA long.  MUTANG++ generates fewer candidate mutations for short
   peptides (smaller tangent space), limiting the action space.

5. Score units
   All scores reported are mean log₂(MIC in µM) over three E. coli APEX indices (1,2,3).
   Equivalent µM values are computed as 2^score.  These are APEX model predictions, not
   experimental measurements.

6. Reward sparsity for near-optimal seeds
   Seeds already close to APEX's E. coli optimum (jurand-4, log₂MIC ≈ 2.7 ≈ 6.6 µM)
   receive near-zero rewards, making Q-value learning noisy.

7. No sequence-validity constraint
   MUTANG++ + HydrAMP decoder occasionally produces sequences that differ substantially from
   typical AMPs.  No hard validity filter is applied beyond the softmax threshold.

FUTURE DIRECTIONS
  • Replace DQN with a continuous actor–critic (DDPG / SAC) acting directly in latent space.
  • Use multi-objective reward including HydrAMP AMP probability and hemolytic toxicity.
  • Integrate beam-search lookahead for sparse-reward seeds.
  • Benchmark against LEBO and LPBeBo on the same seeds with the same APEX oracle.
"""


def page_problems(pdf: PdfPages) -> None:
    fig = _new_page(pdf, "Known Issues, Limitations & Future Work")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88])
    ax.set_axis_off()
    ax.text(0.0, 1.0, PROBLEMS_TEXT,
            va="top", ha="left", fontsize=8.5,
            fontfamily="monospace",
            transform=ax.transAxes, linespacing=1.5)
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 10 — How to run
# ─────────────────────────────────────────────────────────────────────────────

HOW_TO_RUN_TEXT = """\
HOW TO RUN THE CODE
═══════════════════

PREREQUISITES
  • Python ≥ 3.10 with project venv installed:
        cd pep-compass
        uv sync              # or:  pip install -e .
  • GPU recommended (CUDA).  CPU is supported but slow (~3× per episode).
  • Models auto-download on first run (HydrAMP weights, APEX weights).

─────────────────────────────────────────────────────────────────────────────
1. SMOKE TEST — verify all components work
─────────────────────────────────────────────────────────────────────────────
  python scripts/rl_peptide_optimizer.py test_components --device cuda

  Expected output: "All component tests passed ✓"

─────────────────────────────────────────────────────────────────────────────
2. SINGLE PEPTIDE RUN
─────────────────────────────────────────────────────────────────────────────
  python scripts/rl_peptide_optimizer.py run_rl_optimization \\
      --start_peptide KTLKIIRLLF \\
      --n_episodes 100 \\
      --max_steps 20 \\
      --max_candidates 40 \\
      --device cuda \\
      --output_dir results \\
      --run_name mammuthusin-3

  Output files (in results/):
    mammuthusin-3_rl_<timestamp>_results.json   ← full results (JSON)
    mammuthusin-3_rl_<timestamp>_log.csv        ← per-episode CSV log

  Key JSON fields:
    start_peptide / start_log2mic / start_mic_uM
    best_peptide  / best_log2mic  / best_mic_uM
    improvement_log2mic          / fold_improvement
    episode_rewards              (list, len = n_episodes)
    all_best_scores              (list of per-episode best log2MIC)
    trajectories                 (list of peptide-sequence lists)

─────────────────────────────────────────────────────────────────────────────
3. ALL 6 BENCHMARK PEPTIDES
─────────────────────────────────────────────────────────────────────────────
  python scripts/rl_peptide_optimizer.py run_all_peptides \\
      --n_episodes 100 --max_steps 20 --device cuda --output_dir results

─────────────────────────────────────────────────────────────────────────────
4. SLURM CLUSTER (one job per peptide, RTX 5000 Ada)
─────────────────────────────────────────────────────────────────────────────
  # Submit one peptide:
  sbatch scripts/rl_job_mammuthusin-3.sh

  # Submit all (run sequentially due to QOS limit of 4 concurrent jobs):
  for P in middle-1 jurand-4 jurand-2 jurand-7 mammuthusin-3 hydrodamin-2; do
      sbatch scripts/rl_job_${P}.sh
      sleep 1
  done

  # Monitor:
  squeue -u $USER

─────────────────────────────────────────────────────────────────────────────
5. REPORT NOTEBOOK
─────────────────────────────────────────────────────────────────────────────
  jupyter notebook scripts/rl_exploration.ipynb

─────────────────────────────────────────────────────────────────────────────
6. REGENERATE THIS PDF REPORT
─────────────────────────────────────────────────────────────────────────────
  python scripts/generate_rl_report.py --results_dir results \\
      --output results/rl_report.pdf
"""


def page_how_to_run(pdf: PdfPages) -> None:
    fig = _new_page(pdf, "How to Run the Code")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88])
    ax.set_axis_off()
    ax.text(0.0, 1.0, HOW_TO_RUN_TEXT,
            va="top", ha="left", fontsize=8,
            fontfamily="monospace",
            transform=ax.transAxes, linespacing=1.45)
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 11 — DQN vs TD3 Comparison
# ─────────────────────────────────────────────────────────────────────────────

def page_dqn_vs_td3_comparison(
    pdf: PdfPages, dqn_runs: list[dict], td3_runs: list[dict]
) -> None:
    """Generate a comparison page showing DQN vs TD3 best scores per seed."""
    fig = _new_page(pdf, "DQN vs TD3 Comparison — Best log₂MIC per Seed")
    
    # Build comparison data aligned by seed
    dqn_by_seed: dict[str, dict] = {}
    for r in dqn_runs:
        seed = _get_seed_name_from_run(r)
        if seed and seed not in dqn_by_seed:
            dqn_by_seed[seed] = r
    
    td3_by_seed: dict[str, dict] = {}
    for r in td3_runs:
        seed = _get_seed_name_from_run(r)
        if seed and seed not in td3_by_seed:
            td3_by_seed[seed] = r
    
    # Get list of seeds that have at least one result
    all_seeds = [s for s in SEED_ORDER if s in dqn_by_seed or s in td3_by_seed]
    
    if not all_seeds:
        # No data to compare
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No TD3 results available yet.\n\n"
                "Run TD3 optimization first with:\n"
                "  python scripts/rl_continuous_optimizer.py run_td3_all_peptides",
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes)
        _footer(fig)
        _save(pdf, fig)
        return
    
    # Create bar chart
    ax = fig.add_axes([0.12, 0.20, 0.83, 0.65])
    
    x = np.arange(len(all_seeds))
    w = 0.35
    
    # Get scores (use start score if algorithm wasn't run for that seed)
    dqn_scores = []
    td3_scores = []
    start_scores = []
    
    for seed in all_seeds:
        if seed in dqn_by_seed:
            dqn_scores.append(dqn_by_seed[seed]["best_log2mic"])
            start_scores.append(dqn_by_seed[seed]["start_log2mic"])
        elif seed in td3_by_seed:
            # DQN not run, use start score
            dqn_scores.append(td3_by_seed[seed]["start_log2mic"])
            start_scores.append(td3_by_seed[seed]["start_log2mic"])
        else:
            dqn_scores.append(np.nan)
            start_scores.append(np.nan)
        
        if seed in td3_by_seed:
            td3_scores.append(td3_by_seed[seed]["best_log2mic"])
        elif seed in dqn_by_seed:
            # TD3 not run, use start score
            td3_scores.append(dqn_by_seed[seed]["start_log2mic"])
        else:
            td3_scores.append(np.nan)
    
    # Plot bars
    dqn_bars = ax.bar(x - w/2, dqn_scores, w, label="DQN Best", 
                      color="#2274A5", edgecolor="grey", linewidth=0.6)
    td3_bars = ax.bar(x + w/2, td3_scores, w, label="TD3 Best",
                      color="#E84855", edgecolor="grey", linewidth=0.6)
    
    # Plot start score reference line for each seed
    for i, (seed, start) in enumerate(zip(all_seeds, start_scores)):
        ax.hlines(start, i - 0.45, i + 0.45, colors="grey", 
                  linestyles="--", linewidth=1, alpha=0.7)
    
    # Annotate bars with µM values
    for bar, score in zip(dqn_bars, dqn_scores):
        if not np.isnan(score):
            mic = 2 ** score
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{mic:.1f}µM", ha="center", va="bottom", fontsize=6.5,
                    color="#1a4a6b")
    for bar, score in zip(td3_bars, td3_scores):
        if not np.isnan(score):
            mic = 2 ** score
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{mic:.1f}µM", ha="center", va="bottom", fontsize=6.5,
                    color="#8b2233")
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_seeds, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Best log₂(MIC) achieved", fontsize=11)
    ax.set_title("Lower = more potent.  Dashed lines = start scores.  "
                 "Numbers above bars show MIC in µM.", fontsize=9, style="italic")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Add summary table below
    ax_tbl = fig.add_axes([0.05, 0.02, 0.90, 0.12])
    ax_tbl.set_axis_off()
    
    # Compute summary stats
    dqn_wins = sum(1 for i, s in enumerate(all_seeds) 
                   if s in dqn_by_seed and s in td3_by_seed
                   and dqn_by_seed[s]["best_log2mic"] < td3_by_seed[s]["best_log2mic"])
    td3_wins = sum(1 for i, s in enumerate(all_seeds)
                   if s in dqn_by_seed and s in td3_by_seed
                   and td3_by_seed[s]["best_log2mic"] < dqn_by_seed[s]["best_log2mic"])
    ties = sum(1 for s in all_seeds 
               if s in dqn_by_seed and s in td3_by_seed
               and abs(dqn_by_seed[s]["best_log2mic"] - td3_by_seed[s]["best_log2mic"]) < 0.01)
    
    summary_text = (
        f"Head-to-head comparison (seeds with both algorithms run):  "
        f"DQN wins: {dqn_wins}  |  TD3 wins: {td3_wins}  |  Ties: {ties}\n"
        f"DQN: Discrete actions via MUTANG++ mutation enumeration  |  "
        f"TD3: Continuous actions directly in latent space"
    )
    ax_tbl.text(0.5, 0.5, summary_text, ha="center", va="center", fontsize=9,
                transform=ax_tbl.transAxes)
    
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 12 — DQN vs A2C Comparison
# ─────────────────────────────────────────────────────────────────────────────

def page_dqn_vs_a2c_comparison(
    pdf: PdfPages, dqn_runs: list[dict], a2c_runs: list[dict]
) -> None:
    """Generate a comparison page showing DQN vs A2C best scores per seed."""
    fig = _new_page(pdf, "DQN vs A2C Comparison — Best log₂MIC per Seed")
    
    # Build comparison data aligned by seed
    dqn_by_seed: dict[str, dict] = {}
    for r in dqn_runs:
        seed = _get_seed_name_from_run(r)
        if seed and seed not in dqn_by_seed:
            dqn_by_seed[seed] = r
    
    a2c_by_seed: dict[str, dict] = {}
    for r in a2c_runs:
        seed = _get_seed_name_from_run(r)
        if seed and seed not in a2c_by_seed:
            a2c_by_seed[seed] = r
    
    # Get list of seeds that have at least one result
    all_seeds = [s for s in SEED_ORDER if s in dqn_by_seed or s in a2c_by_seed]
    
    if not all_seeds:
        # No data to compare
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No A2C results available yet.\n\n"
                "Run A2C optimization first with:\n"
                "  python scripts/rl_actor_critic_optimizer.py run_a2c_all_peptides",
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes)
        _footer(fig)
        _save(pdf, fig)
        return
    
    # Create bar chart
    ax = fig.add_axes([0.12, 0.20, 0.83, 0.65])
    
    x = np.arange(len(all_seeds))
    w = 0.35
    
    # Get scores (use start score if algorithm wasn't run for that seed)
    dqn_scores = []
    a2c_scores = []
    start_scores = []
    
    for seed in all_seeds:
        if seed in dqn_by_seed:
            dqn_scores.append(dqn_by_seed[seed]["best_log2mic"])
            start_scores.append(dqn_by_seed[seed]["start_log2mic"])
        elif seed in a2c_by_seed:
            # DQN not run, use start score
            dqn_scores.append(a2c_by_seed[seed]["start_log2mic"])
            start_scores.append(a2c_by_seed[seed]["start_log2mic"])
        else:
            dqn_scores.append(np.nan)
            start_scores.append(np.nan)
        
        if seed in a2c_by_seed:
            a2c_scores.append(a2c_by_seed[seed]["best_log2mic"])
        elif seed in dqn_by_seed:
            # A2C not run, use start score
            a2c_scores.append(dqn_by_seed[seed]["start_log2mic"])
        else:
            a2c_scores.append(np.nan)
    
    # Plot bars
    dqn_bars = ax.bar(x - w/2, dqn_scores, w, label="DQN Best", 
                      color="#2274A5", edgecolor="grey", linewidth=0.6)
    a2c_bars = ax.bar(x + w/2, a2c_scores, w, label="A2C Best",
                      color="#2DC653", edgecolor="grey", linewidth=0.6)
    
    # Plot start score reference line for each seed
    for i, (seed, start) in enumerate(zip(all_seeds, start_scores)):
        ax.hlines(start, i - 0.45, i + 0.45, colors="grey", 
                  linestyles="--", linewidth=1, alpha=0.7)
    
    # Annotate bars with µM values
    for bar, score in zip(dqn_bars, dqn_scores):
        if not np.isnan(score):
            mic = 2 ** score
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{mic:.1f}µM", ha="center", va="bottom", fontsize=6.5,
                    color="#1a4a6b")
    for bar, score in zip(a2c_bars, a2c_scores):
        if not np.isnan(score):
            mic = 2 ** score
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{mic:.1f}µM", ha="center", va="bottom", fontsize=6.5,
                    color="#1a6b2f")
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_seeds, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Best log₂(MIC) achieved", fontsize=11)
    ax.set_title("Lower = more potent.  Dashed lines = start scores.  "
                 "Numbers above bars show MIC in µM.", fontsize=9, style="italic")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    # Add summary table below
    ax_tbl = fig.add_axes([0.05, 0.02, 0.90, 0.12])
    ax_tbl.set_axis_off()
    
    # Compute summary stats
    dqn_wins = sum(1 for i, s in enumerate(all_seeds) 
                   if s in dqn_by_seed and s in a2c_by_seed
                   and dqn_by_seed[s]["best_log2mic"] < a2c_by_seed[s]["best_log2mic"])
    a2c_wins = sum(1 for i, s in enumerate(all_seeds)
                   if s in dqn_by_seed and s in a2c_by_seed
                   and a2c_by_seed[s]["best_log2mic"] < dqn_by_seed[s]["best_log2mic"])
    ties = sum(1 for s in all_seeds 
               if s in dqn_by_seed and s in a2c_by_seed
               and abs(dqn_by_seed[s]["best_log2mic"] - a2c_by_seed[s]["best_log2mic"]) < 0.01)
    
    summary_text = (
        f"Head-to-head comparison (seeds with both algorithms run):  "
        f"DQN wins: {dqn_wins}  |  A2C wins: {a2c_wins}  |  Ties: {ties}\n"
        f"DQN: Q-value scoring of MUTANG++ candidates  |  "
        f"A2C: Continuous actor direction guides MUTANG++ selection"
    )
    ax_tbl.text(0.5, 0.5, summary_text, ha="center", va="center", fontsize=9,
                transform=ax_tbl.transAxes)
    
    _footer(fig)
    _save(pdf, fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RL peptide optimiser PDF report")
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing *_results.json files")
    parser.add_argument("--output", default=None,
                        help="Output PDF path (default: <results_dir>/rl_report.pdf)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_path = args.output or os.path.join(results_dir, "rl_report.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Loading results from: {results_dir!r}")
    dqn_runs, td3_runs, a2c_runs = load_all_results(results_dir)
    print(f"Found {len(dqn_runs)} DQN run(s): {[r.get('run_name') for r in dqn_runs]}")
    print(f"Found {len(td3_runs)} TD3 run(s): {[r.get('run_name') for r in td3_runs]}")
    print(f"Found {len(a2c_runs)} A2C run(s): {[r.get('run_name') for r in a2c_runs]}")
    
    # For backward compatibility, use DQN runs as the primary runs
    runs = dqn_runs if dqn_runs else (a2c_runs if a2c_runs else td3_runs)

    print(f"Writing PDF → {output_path!r}")
    with PdfPages(output_path) as pdf:
        # metadata
        d = pdf.infodict()
        d["Title"]   = "RL Peptide Optimizer — Results Report"
        d["Author"]  = "pep-compass / generate_rl_report.py"
        d["Subject"] = "DQN, TD3, and A2C AMP optimisation in HydrAMP latent space"
        d["CreationDate"] = datetime.now()

        page_cover(pdf)
        page_methodology(pdf)
        if runs:
            page_summary_table(pdf, runs)
            page_mic_comparison(pdf, runs)
            page_global_best_trajectories(pdf, runs)
            page_reward_curves(pdf, runs)
        page_epsilon_schedule(pdf)
        if runs:
            page_top_peptides(pdf, runs)
        page_problems(pdf)
        page_how_to_run(pdf)
        # DQN vs TD3 comparison page (always added, shows message if no TD3 data)
        page_dqn_vs_td3_comparison(pdf, dqn_runs, td3_runs)
        # DQN vs A2C comparison page (always added, shows message if no A2C data)
        page_dqn_vs_a2c_comparison(pdf, dqn_runs, a2c_runs)

    print(f"\nDone!  {output_path}  ({os.path.getsize(output_path) // 1024} KB)")


if __name__ == "__main__":
    main()
