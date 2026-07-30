"""Built-in walker strategies."""

from pep_compass.walkers.strategies.sorbes import SorbesWalker
from pep_compass.walkers.strategies.subriemannian import (
    SecondOrderRiemannianBrownianEfficientSampling,
    SubRiemannianTangentSpace,
)
from pep_compass.walkers.manager import WalkerManager


@WalkerManager.register("sorbes")
def build_sorbes(encoder_decoder, **parameters):
    """Build SORBES from the core-owned encoder-decoder service."""
    sampling_walker = SecondOrderRiemannianBrownianEfficientSampling(
        encoder_decoder=encoder_decoder,
        **parameters,
    )
    return SorbesWalker(sampling_walker)

__all__ = [
    "SecondOrderRiemannianBrownianEfficientSampling",
    "SorbesWalker",
    "SubRiemannianTangentSpace",
]
