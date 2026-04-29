"""
Minimal epsilon-greedy peptide policy.

This module replaces the previous RL policies with a simple epsilon-greedy
policy: at each step we choose among local mutation candidates, exploring with
probability epsilon and otherwise exploiting the currently best APEX score.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

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

SEED_PEPTIDES: dict[str, str] = {
    "middle-1": "FLYKWWIRIGRLKL",
    "jurand-4": "KYCRRFRWLTFRWL",
    "jurand-2": "KFRNRHRWKFKLIFRN",
    "jurand-7": "KKYWLIRKWIRLWFLT",
    "mammuthusin-3": "KTLKIIRLLF",
    "hydrodamin-2": "RMARNLVRYVQGLKKKKVI",
}


def build_encoder_decoder(device: str = "cpu"):
    from pep_compass.models.encoder_decoder.hydramp_encoder_decoder import (
        HydrAMPEncoderDecoder,
    )

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


def _iter_text_chunks(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="ignore")]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        chunks: list[str] = []
        for item in value:
            chunks.extend(_iter_text_chunks(item))
        return chunks
    return [str(value)]


def _tokenize_peptide_text(text: str) -> list[str]:
    peptides: list[str] = []
    valid_aas = set(ALPHABET[1:])
    skip_tokens = {"peptide", "sequence", "seq"}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.replace(",", " ").split():
            peptide = token.strip().upper()
            if not peptide or peptide.lower() in skip_tokens:
                continue
            if set(peptide).issubset(valid_aas):
                peptides.append(peptide)
    return peptides


def resolve_start_peptides(
    start_peptide: str,
    start_peptides: object = "",
    start_peptides_file: object = "",
) -> list[str]:
    resolved: list[str] = []
    for chunk in _iter_text_chunks(start_peptides):
        resolved.extend(_tokenize_peptide_text(chunk))
    for path_chunk in _iter_text_chunks(start_peptides_file):
        if not path_chunk:
            continue
        file_text = Path(path_chunk).read_text(encoding="utf-8")
        resolved.extend(_tokenize_peptide_text(file_text))
    if not resolved and start_peptide:
        resolved.append(start_peptide.strip().upper())

    deduped: list[str] = []
    seen: set[str] = set()
    for peptide in resolved:
        if peptide and peptide not in seen:
            deduped.append(peptide)
            seen.add(peptide)
    if not deduped:
        raise ValueError("No valid start peptides were provided.")
    return deduped


def _normalise_mutations(mutations: dict[int, list[int]]) -> dict[int, list[int]]:
    normalised: dict[int, list[int]] = {}
    for pos, aa_indices in mutations.items():
        unique = sorted({int(idx) for idx in aa_indices})
        if unique:
            normalised[int(pos)] = unique
    return normalised


def score_peptides(
    apex_predictor,
    peptides: list[str],
    ecoli_indices: list[int] = ECOLI_INDICES,
) -> np.ndarray:
    mic: np.ndarray = apex_predictor.predict(peptides)
    mic_ecoli = mic[:, ecoli_indices]
    mic_ecoli = np.clip(mic_ecoli, a_min=1e-6, a_max=None)
    return np.log2(mic_ecoli).mean(axis=1)


def generate_candidates(
    peptide: str,
    z_np: np.ndarray,
    encoder_decoder,
    mutation_enumerator,
    max_candidates: int = 40,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    from pep_compass.local_enumeration.mutation.mutation_potentials import (
        ProjectedDirectionPairwiseSimilarityPotential,
        compose_mutant_distribution,
    )
    from pep_compass.local_enumeration.sampling.sorbes import SubRiemannianTangentSpace

    z_tensor = torch.tensor(z_np, dtype=torch.float32, device=encoder_decoder.device)
    z_2d = z_tensor.unsqueeze(0) if z_tensor.ndim == 1 else z_tensor
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
        return [peptide], np.array([1.0]), z_np[np.newaxis, :]

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
    mutant_dist = compose_mutant_distribution(
        parent_peptide=peptide,
        mutations=mutations,
        potential=potential,
        alphabet=ALPHABET,
        max_len=MAX_PEPTIDE_LEN,
        include_parent_residue=False,
        top_k=max_candidates,
    )
    if not mutant_dist.sequences:
        return [peptide], np.array([1.0]), z_np[np.newaxis, :]

    seqs = mutant_dist.sequences
    log_pots = mutant_dist.log_potentials
    shifted = log_pots - log_pots.max()
    probs = np.exp(shifted)
    probs = probs / probs.sum()

    # Requested threshold: keep actions with probability > 2/N.
    n = len(seqs)
    threshold = 2.0 / max(n, 1)
    keep = probs > threshold
    if keep.sum() == 0:
        keep[np.argmax(probs)] = True

    kept_pairs = [
        (s, float(p))
        for s, p, m in zip(seqs, probs, keep)
        if m and s.strip() != peptide.strip()
    ]
    if not kept_pairs:
        return [peptide], np.array([1.0]), z_np[np.newaxis, :]

    seqs = [s for s, _p in kept_pairs]
    probs = np.array([p for _s, p in kept_pairs], dtype=float)
    probs = probs / probs.sum()
    with torch.no_grad():
        z_batch = encoder_decoder.encode_peptides(seqs)
    candidate_zs = z_batch.detach().cpu().numpy()
    return seqs, probs, candidate_zs


def run_basic_epsilon_greedy(
    start_peptide: str = "FLPKKVIPLL",
    start_peptides: str = "",
    start_peptides_file: str = "",
    start_selection: str = "random",
    n_epochs: int = 1500,
    max_steps: int = 200,
    epsilon_start: float = 0.30,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.9995,
    max_candidates: int = 40,
    device: str = "cpu",
    output_dir: str = "results/basic_eps_greedy_rl",
    run_name: str = "",
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    from pep_compass.local_enumeration.mutation.mutation_enumerator import (
        MutationEnumerationInTangentSpace,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    selection = start_selection.strip().lower()
    if selection not in {"cycle", "random"}:
        raise ValueError("start_selection must be one of: 'cycle', 'random'")

    encoder_decoder = build_encoder_decoder(device)
    apex = build_apex_predictor(device)
    mutation_enumerator = MutationEnumerationInTangentSpace(
        max_len=MAX_PEPTIDE_LEN,
        direction_significance_threshold=1e-3,
        min_number_of_directions=5,
        token_threshold=0.1,
    )

    start_pool = resolve_start_peptides(start_peptide, start_peptides, start_peptides_file)
    with torch.no_grad():
        start_z = encoder_decoder.encode_peptides(start_pool).detach().cpu().numpy()
    start_scores = score_peptides(apex, start_pool)

    latent_cache: dict[str, np.ndarray] = {
        p: start_z[i].copy() for i, p in enumerate(start_pool)
    }
    score_cache: dict[str, float] = {
        p: float(start_scores[i]) for i, p in enumerate(start_pool)
    }
    candidate_cache: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}

    def get_latent(peptide: str) -> np.ndarray:
        if peptide not in latent_cache:
            with torch.no_grad():
                latent_cache[peptide] = encoder_decoder.encode_peptides([peptide]).detach().cpu().numpy()[0]
        return latent_cache[peptide]

    def get_score(peptide: str) -> float:
        if peptide not in score_cache:
            score_cache[peptide] = float(score_peptides(apex, [peptide])[0])
        return score_cache[peptide]

    def get_candidates(peptide: str, z: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        if peptide not in candidate_cache:
            candidate_cache[peptide] = generate_candidates(
                peptide=peptide,
                z_np=z,
                encoder_decoder=encoder_decoder,
                mutation_enumerator=mutation_enumerator,
                max_candidates=max_candidates,
            )
        return candidate_cache[peptide]

    best_idx = int(np.argmin(start_scores))
    global_best = start_pool[best_idx]
    global_best_score = float(start_scores[best_idx])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{run_name + '_' if run_name else ''}eps_{int(time.time())}"
    csv_path = out_dir / f"{run_id}_log.csv"
    json_path = out_dir / f"{run_id}_results.json"

    epoch_losses: list[float] = []
    epoch_returns: list[float] = []
    epoch_best_scores: list[float] = []
    epoch_best_peptides: list[str] = []
    no_mutation_steps_per_epoch: list[int] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "epsilon",
                "start_peptide",
                "start_log2mic",
                "best_peptide",
                "best_log2mic",
                "epoch_return",
                "epoch_loss",
                "no_mutation_steps",
            ]
        )

        epsilon = epsilon_start
        for epoch in range(n_epochs):
            if selection == "random":
                current_peptide = random.choice(start_pool)
            else:
                current_peptide = start_pool[epoch % len(start_pool)]

            epoch_start_peptide = current_peptide
            current_z = get_latent(current_peptide).copy()
            current_score = get_score(current_peptide)
            start_score = current_score
            epoch_best = current_peptide
            epoch_best_score = current_score
            rewards: list[float] = []
            no_mutation_steps = 0

            for _step in range(max_steps):
                cand_seqs, cand_probs, cand_zs = get_candidates(current_peptide, current_z)
                if len(cand_seqs) == 1 and cand_seqs[0] == current_peptide:
                    no_mutation_steps += 1
                    rewards.append(0.0)
                    continue

                if random.random() < epsilon:
                    action_idx = int(np.random.choice(len(cand_seqs), p=cand_probs))
                else:
                    candidate_scores = score_peptides(apex, cand_seqs)
                    action_idx = int(np.argmin(candidate_scores))

                next_peptide = cand_seqs[action_idx]
                next_z = cand_zs[action_idx].copy()
                next_score = get_score(next_peptide)
                reward = float(current_score - next_score)
                if not np.isfinite(reward):
                    reward = 0.0
                rewards.append(reward)

                current_peptide = next_peptide
                current_z = next_z
                current_score = next_score
                latent_cache[next_peptide] = next_z

                if current_score < epoch_best_score:
                    epoch_best_score = current_score
                    epoch_best = current_peptide
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best = current_peptide

            epoch_return = float(np.sum(rewards))
            epoch_loss = float(-np.mean(rewards)) if rewards else 0.0
            epoch_returns.append(epoch_return)
            epoch_losses.append(epoch_loss)
            epoch_best_scores.append(epoch_best_score)
            epoch_best_peptides.append(epoch_best)
            no_mutation_steps_per_epoch.append(no_mutation_steps)

            writer.writerow(
                [
                    epoch + 1,
                    f"{epsilon:.6f}",
                    epoch_start_peptide,
                    f"{start_score:.6f}",
                    epoch_best,
                    f"{epoch_best_score:.6f}",
                    f"{epoch_return:.6f}",
                    f"{epoch_loss:.6f}",
                    no_mutation_steps,
                ]
            )

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            if verbose and (epoch + 1) % 50 == 0:
                print(
                    f"[{run_name or 'run'}] epoch {epoch + 1:4d}/{n_epochs} "
                    f"ret={epoch_return:+.4f} loss={epoch_loss:+.4f} "
                    f"best={global_best_score:.4f} eps={epsilon:.4f}"
                )

    results = {
        "run_name": run_name or run_id,
        "run_id": run_id,
        "algorithm": "basic_epsilon_greedy",
        "seed": seed,
        "n_epochs": n_epochs,
        "max_steps": max_steps,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "start_selection": selection,
        "start_peptides": start_pool,
        "start_scores_log2mic": [float(x) for x in start_scores.tolist()],
        "best_peptide": global_best,
        "best_log2mic": float(global_best_score),
        "epoch_returns": epoch_returns,
        "epoch_losses": epoch_losses,
        "epoch_best_scores": epoch_best_scores,
        "epoch_best_peptides": epoch_best_peptides,
        "no_mutation_steps_per_epoch": no_mutation_steps_per_epoch,
        "reward_formula": "log2MIC_t - log2MIC_t+1",
        "loss_formula": "-mean(step_rewards)",
    }
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_five_agents(
    start_peptides: str = "",
    start_peptides_file: str = "",
    n_agents: int = 5,
    n_epochs: int = 1500,
    max_steps: int = 200,
    device: str = "cpu",
    output_dir: str = "results/basic_eps_greedy_rl",
    start_selection: str = "random",
    verbose: bool = True,
) -> list[dict]:
    if not start_peptides and not start_peptides_file:
        start_peptides = "\n".join(SEED_PEPTIDES.values())

    results: list[dict] = []
    for i in range(n_agents):
        run_name = f"agent_{i + 1}"
        res = run_basic_epsilon_greedy(
            start_peptide="",
            start_peptides=start_peptides,
            start_peptides_file=start_peptides_file,
            start_selection=start_selection,
            n_epochs=n_epochs,
            max_steps=max_steps,
            device=device,
            output_dir=output_dir,
            run_name=run_name,
            seed=17 + i,
            verbose=verbose,
        )
        results.append(res)
        
        # Memory cleanup: explicitly delete model instances between agents to prevent OOM
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "run_basic_epsilon_greedy": run_basic_epsilon_greedy,
            "run_five_agents": run_five_agents,
        }
    )
