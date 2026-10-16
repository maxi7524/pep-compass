"""Convenient compositions of real scoring and selection objects."""

from .base import Ranked
from .scoring.latent_geometry.lams import LamsScore
from .scoring.latent_geometry.move import MoveScore
from .scoring.latent_geometry.tandem import TandemScore
from .scoring.model_scores.decoder_likelihood import DecoderLikelihoodScore
from .scoring.model_scores.esm import ESM2PPLScorer
from .selection.nucleus import NucleusSelection
from .selection.threshold import ThresholdSelection
from .selection.top_k import TopKSelection


def Lams(
    *,
    similarity_threshold=0.15,
    horizontal_threshold=0.1,
    maximum_candidates=30000,
    alphabet=None,
):
    """Compose LAMS scoring and threshold selection without a marker subclass."""
    return Ranked(
        LamsScore(
            horizontal_threshold=horizontal_threshold,
            maximum_candidates=maximum_candidates,
            alphabet=alphabet,
        ),
        ThresholdSelection(similarity_threshold),
    )


def Tandem(
    *,
    top_p=0.9,
    temperature=1.0,
    horizontal_threshold=0.1,
    maximum_candidates=30000,
    alphabet=None,
):
    """Compose TANDEM scoring with nucleus selection."""
    return Ranked(
        TandemScore(
            horizontal_threshold=horizontal_threshold,
            maximum_candidates=maximum_candidates,
            alphabet=alphabet,
        ),
        NucleusSelection(top_p, temperature),
    )


def Move(*, top_p=0.6, temperature=1.0, maximum_candidates=30000, alphabet=None):
    """Compose MOVE scoring with nucleus selection."""
    return Ranked(
        MoveScore(maximum_candidates=maximum_candidates, alphabet=alphabet),
        NucleusSelection(top_p, temperature),
    )


def LpBebo(*, top_p=0.9, temperature=1.0, maximum_candidates=30000, alphabet=None):
    """Compose decoder likelihood with nucleus selection."""
    return Ranked(
        DecoderLikelihoodScore(
            maximum_candidates=maximum_candidates, alphabet=alphabet
        ),
        NucleusSelection(top_p, temperature),
    )


def ESM(*, threshold=None, top_k=None, model_name="esm2_t6_8M_UR50D", device="cpu"):
    """Compose lazy ESM plausibility scoring with one explicit selection policy."""
    if (threshold is None) == (top_k is None):
        raise ValueError("Select threshold or top_k.")
    return Ranked(
        ESM2PPLScorer(model_name, device),
        ThresholdSelection(threshold) if top_k is None else TopKSelection(top_k),
    )
