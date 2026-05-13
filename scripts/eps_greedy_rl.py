"""Epsilon-greedy RL for antimicrobial peptide optimization via MUTANG++ action space.

Episode: exactly n_steps (default 20) from a starting peptide.
Action space: neighbor peptides produced by MUTANG++ potential-filtered mutations
              at the current peptide's latent-space Jacobian.
Reward: log2MIC_t - log2MIC_{t+1}  (positive = MIC decreased = better antimicrobial activity)
Goal:   minimize APEX MIC score (E. coli) over n_steps moves from the starting peptide.

Action-space filtering modes (--action_filter):
  top_p  – nucleus sampling: softmax of MUTANG++ potentials, keep cumulative mass ≤ top_p (default 0.90)
  top_n  – keep top N candidates by potential score (default 20)

Usage:
  python scripts/eps_greedy_rl.py run  [options]   # single agent
  python scripts/eps_greedy_rl.py multi [options]  # multiple agents (models built once)
"""

from __future__ import annotations

import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

LATENT_DIM: int = 64
MAX_PEPTIDE_LEN: int = 25
ALPHABET: list[str] = list(" ACDEFGHIKLMNPQRSTVWY")
ECOLI_INDICES: list[int] = [1, 2, 3]

STARTING_PEPTIDES: dict[str, str] = {
    # validated in prior experiments (fallback when no file is provided)
    "middle-1": "FLYKWWIRIGRLKL",
    "jurand-4": "KYCRRFRWLTFRWL",
    "jurand-2": "KFRNRHRWKFKLIFRN",
    "jurand-7": "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF",
    "hydrodamin-2": "RMARNLVRYVQGLKKKKVI",
    # from peptides_from_paper.csv (base variants)
    "equusin-4": "CVLLFSQLPAVKARGTKHRIKWNRK",
    "arctoterin-1": "GHLLIHLIGKATLAL",
    "lophiosin-1": "HWITINTIKLSISLKI",
    "hesperelin-3": "RQKNHGIHFRVLAKALR",
}

DEFAULT_START_PEPTIDES_FILE = str(
    REPO_ROOT / "basic_eps_greedy_rl" / "inputs" / "sampled_500_peptides.txt"
)


# ---------------------------------------------------------------------------
# Peptide pool loading
# ---------------------------------------------------------------------------

def _load_peptide_pool(start_peptides_file: str) -> list[str]:
    """Load peptides from a plain-text file (one per line) or return STARTING_PEPTIDES values."""
    valid_aas = set(ALPHABET[1:])
    path = Path(start_peptides_file)
    if not path.exists():
        return list(STARTING_PEPTIDES.values())

    peptides: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        pep = line.strip().upper()
        if pep and set(pep).issubset(valid_aas) and len(pep) <= MAX_PEPTIDE_LEN:
            peptides.append(pep)

    if not peptides:
        return list(STARTING_PEPTIDES.values())
    return peptides


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_encoder_decoder(device: str = "cpu"):
    from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import HydrAMPEncoderDecoder

    return HydrAMPEncoderDecoder(
        jacobian_mode="approx",
        device=device,
        default_condition=torch.tensor([1.0, 1.0]),
        temp=1.0,
        jacobian_eps=0.05,
        field_eps=0.05,
    )


def build_apex_predictor(device: str = "cpu"):
    from pep_compass.models.apex.APEX_predictor import PredictorAPEX

    return PredictorAPEX(device=device)


def build_mutation_enumerator():
    from pep_compass.local_enumeration.mutation.mutation_enumerator import (
        MutationEnumerationInTangentSpace,
    )

    return MutationEnumerationInTangentSpace(
        max_len=MAX_PEPTIDE_LEN,
        direction_significance_threshold=1e-3,
        min_number_of_directions=5,
        token_threshold=0.1,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_peptides(
    apex_predictor,
    peptides: list[str],
    ecoli_indices: list[int] = ECOLI_INDICES,
) -> np.ndarray:
    with torch.no_grad():
        mic: np.ndarray = apex_predictor.predict(peptides)
    mic_ecoli = mic[:, ecoli_indices]
    mic_ecoli = np.clip(mic_ecoli, a_min=1e-6, a_max=None)
    return np.log2(mic_ecoli).mean(axis=1)


# ---------------------------------------------------------------------------
# Action-space helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max()
    e = np.exp(shifted)
    return e / e.sum()


def _top_p_filter(log_potentials: np.ndarray, top_p: float) -> np.ndarray:
    """Return boolean mask keeping candidates whose cumulative softmax mass ≤ top_p.

    Candidates are sorted descending before cumsum so the highest-potential
    sequences are included first. At least one candidate is always kept.
    """
    probs = _softmax(log_potentials)
    order = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[order])
    keep_sorted = cumsum <= top_p
    # always keep at least the first (highest-probability) candidate
    if not keep_sorted.any():
        keep_sorted[0] = True
    mask = np.zeros(len(log_potentials), dtype=bool)
    mask[order[keep_sorted]] = True
    return mask


def _normalise_mutations(mutations: dict[int, list[int]]) -> dict[int, list[int]]:
    normalised: dict[int, list[int]] = {}
    for pos, aa_indices in mutations.items():
        unique = sorted({int(idx) for idx in aa_indices})
        if unique:
            normalised[int(pos)] = unique
    return normalised


def generate_candidates_for_rl(
    peptide: str,
    z_np: np.ndarray,
    encoder_decoder,
    mutation_enumerator,
    action_filter: str = "top_p",
    top_p: float = 0.90,
    top_n: int = 20,
) -> tuple[list[str], np.ndarray]:
    """Return (candidate_sequences, softmax_probs) via MUTANG++ at current latent point.

    Filtering modes:
      top_p – nucleus: keep highest-potential candidates up to cumulative mass top_p
      top_n – hard cap: keep at most top_n candidates by potential score

    Returns the stuck sentinel ([peptide], [1.0]) when no mutations are found.
    """
    from pep_compass.local_enumeration.mutation.mutation_potentials import (
        ProjectedDirectionPairwiseSimilarityPotential,
        compose_mutant_distribution,
    )
    from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace

    z_tensor = torch.tensor(z_np, dtype=torch.float32, device=encoder_decoder.device)
    z_2d = z_tensor.unsqueeze(0) if z_tensor.ndim == 1 else z_tensor

    with torch.no_grad():
        jacobian = encoder_decoder.decoder_jacobian(z_2d)
    if jacobian.ndim == 3:
        jacobian = jacobian.squeeze(0)

    u, s, vh = torch.linalg.svd(jacobian, full_matrices=False)
    v = vh.transpose(0, 1)

    raw_mutations = mutation_enumerator.get_mutations_from_s_u(
        s.detach().cpu().numpy(),
        u.detach().cpu().numpy(),
    )
    mutations = _normalise_mutations(raw_mutations)
    if not mutations:
        return [peptide], np.array([1.0])

    tangent_space = SubRiemannianTangentSpace(
        U=u.detach().cpu(),
        S=s.detach().cpu(),
        V=v.detach().cpu(),
        horizontal_threshold=1e-3,
        device="cpu",
    )
    potential = ProjectedDirectionPairwiseSimilarityPotential(
        tangent_space=tangent_space,
        alphabet=ALPHABET,
    )

    # top_k caps the Cartesian product before nucleus filtering to avoid OOM
    top_k_arg = top_n if action_filter == "top_n" else 200
    mutant_dist = compose_mutant_distribution(
        parent_peptide=peptide,
        mutations=mutations,
        potential=potential,
        alphabet=ALPHABET,
        max_len=MAX_PEPTIDE_LEN,
        include_parent_residue=False,
        top_k=top_k_arg,
    )

    if not mutant_dist.sequences:
        return [peptide], np.array([1.0])

    seqs = mutant_dist.sequences
    log_pots = mutant_dist.log_potentials

    # apply the requested filter
    if action_filter == "top_n":
        # compose_mutant_distribution already returns sorted by potential desc;
        # top_k_arg == top_n so no further trimming needed
        mask = np.ones(len(seqs), dtype=bool)
    else:
        mask = _top_p_filter(log_pots, top_p)

    # exclude parent sequence
    parent_stripped = peptide.strip()
    kept = [
        (s, lp)
        for s, lp, m in zip(seqs, log_pots, mask)
        if m and s.strip() != parent_stripped
    ]

    if not kept:
        return [peptide], np.array([1.0])

    kept_seqs = [s for s, _ in kept]
    kept_lp = np.array([lp for _, lp in kept], dtype=float)
    probs = _softmax(kept_lp)
    return kept_seqs, probs


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(
    start_peptide: str,
    epsilon: float,
    encoder_decoder,
    apex_predictor,
    mutation_enumerator,
    latent_cache: dict[str, np.ndarray],
    score_cache: dict[str, float],
    candidate_cache: dict[str, tuple[list[str], np.ndarray]],
    action_filter: str,
    top_p: float,
    top_n: int,
    n_steps: int = 20,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    def get_latent(pep: str) -> np.ndarray:
        if pep not in latent_cache:
            with torch.no_grad():
                latent_cache[pep] = (
                    encoder_decoder.encode_peptides([pep]).detach().cpu().numpy()[0]
                )
        return latent_cache[pep]

    def get_score(pep: str) -> float:
        if pep not in score_cache:
            score_cache[pep] = float(score_peptides(apex_predictor, [pep])[0])
        return score_cache[pep]

    def get_candidates(pep: str, z: np.ndarray) -> tuple[list[str], np.ndarray]:
        if pep not in candidate_cache:
            candidate_cache[pep] = generate_candidates_for_rl(
                peptide=pep,
                z_np=z,
                encoder_decoder=encoder_decoder,
                mutation_enumerator=mutation_enumerator,
                action_filter=action_filter,
                top_p=top_p,
                top_n=top_n,
            )
        return candidate_cache[pep]

    current_peptide = start_peptide
    current_z = get_latent(current_peptide).copy()
    current_score = get_score(current_peptide)

    best_peptide = current_peptide
    best_score = current_score

    trajectory: list[dict] = []
    episode_return = 0.0
    n_stuck = 0

    for step in range(n_steps):
        cand_seqs, cand_probs = get_candidates(current_peptide, current_z)
        is_stuck = len(cand_seqs) == 1 and cand_seqs[0].strip() == current_peptide.strip()

        if is_stuck:
            trajectory.append(
                {
                    "step": step,
                    "peptide": current_peptide,
                    "score": current_score,
                    "action_type": "stuck",
                    "next_peptide": current_peptide,
                    "next_score": current_score,
                    "reward": 0.0,
                }
            )
            n_stuck += 1
            continue

        if rng.random() < epsilon:
            action_idx = int(rng.choice(len(cand_seqs), p=cand_probs))
            action_type = "explore"
        else:
            candidate_scores = score_peptides(apex_predictor, cand_seqs)
            action_idx = int(np.argmin(candidate_scores))
            action_type = "exploit"

        next_peptide = cand_seqs[action_idx]
        next_z = get_latent(next_peptide).copy()
        next_score = get_score(next_peptide)

        reward = float(current_score - next_score)
        if not np.isfinite(reward):
            reward = 0.0

        trajectory.append(
            {
                "step": step,
                "peptide": current_peptide,
                "score": current_score,
                "action_type": action_type,
                "next_peptide": next_peptide,
                "next_score": next_score,
                "reward": reward,
            }
        )
        episode_return += reward

        current_peptide = next_peptide
        current_z = next_z
        current_score = next_score

        if current_score < best_score:
            best_score = current_score
            best_peptide = current_peptide

    return {
        "start_peptide": start_peptide,
        "start_score": float(get_score(start_peptide)),
        "best_peptide": best_peptide,
        "best_score": float(best_score),
        "trajectory": trajectory,
        "episode_return": float(episode_return),
        "n_stuck_steps": n_stuck,
    }


# ---------------------------------------------------------------------------
# Inner runner (accepts pre-built models so multi-agent reuses them)
# ---------------------------------------------------------------------------

def _run_with_prebuilt_models(
    encoder_decoder,
    apex_predictor,
    mutation_enumerator,
    *,
    start_pool: list[str],
    n_episodes: int,
    n_steps: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    action_filter: str,
    top_p: float,
    top_n: int,
    output_dir: str,
    run_name: str,
    seed: int,
    start_selection: str,
    verbose: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # pre-encode and pre-score all starting peptides
    with torch.no_grad():
        start_z = encoder_decoder.encode_peptides(start_pool).detach().cpu().numpy()
    start_scores = score_peptides(apex_predictor, start_pool)

    latent_cache: dict[str, np.ndarray] = {
        p: start_z[i].copy() for i, p in enumerate(start_pool)
    }
    score_cache: dict[str, float] = {
        p: float(start_scores[i]) for i, p in enumerate(start_pool)
    }
    candidate_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    global_best_peptide = start_pool[int(np.argmin(start_scores))]
    global_best_score = float(np.min(start_scores))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{run_name + '_' if run_name else ''}eps_{int(time.time())}"
    csv_path = out_dir / f"{run_id}_log.csv"
    json_path = out_dir / f"{run_id}_results.json"

    episode_returns: list[float] = []
    episode_best_scores: list[float] = []
    episode_best_peptides: list[str] = []
    all_trajectories: list[dict] = []

    epsilon = epsilon_start
    selection = start_selection.strip().lower()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "epsilon",
                "start_peptide",
                "start_log2mic",
                "best_peptide",
                "best_log2mic",
                "global_best_peptide",
                "global_best_log2mic",
                "episode_return",
                "n_stuck_steps",
                "action_filter",
            ]
        )
        f.flush()

        for ep in range(n_episodes):
            if selection == "cycle":
                start_peptide = start_pool[ep % len(start_pool)]
            else:
                start_peptide = random.choice(start_pool)

            ep_result = run_episode(
                start_peptide=start_peptide,
                epsilon=epsilon,
                encoder_decoder=encoder_decoder,
                apex_predictor=apex_predictor,
                mutation_enumerator=mutation_enumerator,
                latent_cache=latent_cache,
                score_cache=score_cache,
                candidate_cache=candidate_cache,
                action_filter=action_filter,
                top_p=top_p,
                top_n=top_n,
                n_steps=n_steps,
                rng=rng,
            )

            if ep_result["best_score"] < global_best_score:
                global_best_score = ep_result["best_score"]
                global_best_peptide = ep_result["best_peptide"]

            episode_returns.append(ep_result["episode_return"])
            episode_best_scores.append(ep_result["best_score"])
            episode_best_peptides.append(ep_result["best_peptide"])
            all_trajectories.append(ep_result)

            writer.writerow(
                [
                    ep + 1,
                    f"{epsilon:.6f}",
                    start_peptide,
                    f"{ep_result['start_score']:.6f}",
                    ep_result["best_peptide"],
                    f"{ep_result['best_score']:.6f}",
                    global_best_peptide,
                    f"{global_best_score:.6f}",
                    f"{ep_result['episode_return']:.6f}",
                    ep_result["n_stuck_steps"],
                    action_filter,
                ]
            )
            f.flush()

            # epsilon decays once per episode, after the episode ends
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if verbose and (ep + 1) % 50 == 0:
                print(
                    f"[{run_name or 'run'}] ep {ep + 1:4d}/{n_episodes}"
                    f"  ret={ep_result['episode_return']:+.4f}"
                    f"  best={global_best_score:.4f}"
                    f"  eps={epsilon:.4f}"
                    f"  stuck={ep_result['n_stuck_steps']}"
                )

    results = {
        "run_name": run_name or run_id,
        "run_id": run_id,
        "algorithm": "eps_greedy_rl",
        "seed": seed,
        "n_episodes": n_episodes,
        "n_steps": n_steps,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "action_filter": action_filter,
        "top_p": top_p,
        "top_n": top_n,
        "start_selection": selection,
        "start_pool_size": len(start_pool),
        "best_peptide": global_best_peptide,
        "best_log2mic": float(global_best_score),
        "episode_returns": episode_returns,
        "episode_best_scores": episode_best_scores,
        "episode_best_peptides": episode_best_peptides,
        "all_trajectories": all_trajectories,
        "reward_formula": "log2MIC_t - log2MIC_{t+1}",
    }
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if verbose:
        print(
            f"[{run_name or 'run'}] done — best: {global_best_peptide}"
            f"  log2MIC={global_best_score:.4f}"
        )
    return results


# ---------------------------------------------------------------------------
# Public entry points (Fire-callable)
# ---------------------------------------------------------------------------

def run_eps_greedy_rl(
    n_episodes: int = 500,
    n_steps: int = 20,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9940,
    action_filter: str = "top_p",
    top_p: float = 0.90,
    top_n: int = 20,
    device: str = "cuda",
    output_dir: str = "results/eps_greedy_rl",
    run_name: str = "",
    seed: int = 0,
    start_selection: str = "random",
    start_peptides_file: str = DEFAULT_START_PEPTIDES_FILE,
    verbose: bool = True,
) -> dict:
    """Single-agent epsilon-greedy RL run.

    start_peptides_file: path to a plain-text file with one peptide per line.
        Defaults to the pre-sampled 500-peptide pool. Pass "" to use the
        built-in 10-peptide STARTING_PEPTIDES fallback.
    """
    if action_filter not in {"top_p", "top_n"}:
        raise ValueError("action_filter must be 'top_p' or 'top_n'")
    if start_selection not in {"cycle", "random"}:
        raise ValueError("start_selection must be 'cycle' or 'random'")

    start_pool = _load_peptide_pool(start_peptides_file)
    if verbose:
        print(f"Loaded {len(start_pool)} starting peptides from '{start_peptides_file or 'built-in'}'")

    enc_dec = build_encoder_decoder(device)
    apex = build_apex_predictor(device)
    enumerator = build_mutation_enumerator()

    return _run_with_prebuilt_models(
        enc_dec,
        apex,
        enumerator,
        start_pool=start_pool,
        n_episodes=n_episodes,
        n_steps=n_steps,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        action_filter=action_filter,
        top_p=top_p,
        top_n=top_n,
        output_dir=output_dir,
        run_name=run_name,
        seed=seed,
        start_selection=start_selection,
        verbose=verbose,
    )


def run_multi_agent(
    n_agents: int = 5,
    n_episodes: int = 500,
    n_steps: int = 20,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9940,
    action_filter: str = "top_p",
    top_p: float = 0.90,
    top_n: int = 20,
    device: str = "cuda",
    output_dir: str = "results/eps_greedy_rl",
    start_selection: str = "random",
    start_peptides_file: str = DEFAULT_START_PEPTIDES_FILE,
    verbose: bool = True,
) -> list[dict]:
    """Run multiple independent agents sequentially, building models only once.

    start_peptides_file: path to a plain-text file with one peptide per line.
        Defaults to the pre-sampled 500-peptide pool. Pass "" to use the
        built-in 10-peptide STARTING_PEPTIDES fallback.
    """
    if action_filter not in {"top_p", "top_n"}:
        raise ValueError("action_filter must be 'top_p' or 'top_n'")
    if start_selection not in {"cycle", "random"}:
        raise ValueError("start_selection must be 'cycle' or 'random'")

    start_pool = _load_peptide_pool(start_peptides_file)
    if verbose:
        print(f"Loaded {len(start_pool)} starting peptides from '{start_peptides_file or 'built-in'}'")

    enc_dec = build_encoder_decoder(device)
    apex = build_apex_predictor(device)
    enumerator = build_mutation_enumerator()

    all_results: list[dict] = []
    for i in range(n_agents):
        run_name = f"agent_{i + 1}"
        if verbose:
            print(f"\n=== Starting {run_name} (seed={17 + i}) ===")
        result = _run_with_prebuilt_models(
            enc_dec,
            apex,
            enumerator,
            start_pool=start_pool,
            n_episodes=n_episodes,
            n_steps=n_steps,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            action_filter=action_filter,
            top_p=top_p,
            top_n=top_n,
            output_dir=output_dir,
            run_name=run_name,
            seed=17 + i,
            start_selection=start_selection,
            verbose=verbose,
        )
        all_results.append(result)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return all_results


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run_eps_greedy_rl, "multi": run_multi_agent})
