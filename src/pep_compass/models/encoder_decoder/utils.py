import torch
from einops import rearrange
from typing import Callable, Optional, Literal


def decoder_jacobian(
    decoder_forward: Callable[torch.Tensor, torch.Tensor],
    x: torch.Tensor,
    jacobian_fn_mode: Literal["strict", "approx"],
    jacobian_fn_kwargs: Optional[dict[str, any]] = None,
) -> torch.Tensor:
    r"""input shape: (batch_dim, latent_dim), output shape: (batch_dim, ambient_dim, latent_dim)"""
    assert x.ndim == 2, ValueError(f"x should be 2D, got {x.ndim}D instead.")
    jacobian_fn_kwargs = jacobian_fn_kwargs if jacobian_fn_kwargs is not None else {}
    if jacobian_fn_mode == "strict":
        return decoder_jacobian_strict(decoder_forward, x, **jacobian_fn_kwargs)
    elif jacobian_fn_mode == "approx":
        return decoder_jacobian_approx(decoder_forward, x, **jacobian_fn_kwargs)


def decoder_jacobian_strict(
    decoder_forward: Callable[torch.Tensor, torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    return rearrange(
        torch.autograd.functional.jacobian(decoder_forward, x)[0],
        "a b d -> b a d",
    )


def decoder_jacobian_approx(
    decoder_forward: Callable[torch.Tensor, torch.Tensor],
    x: torch.Tensor,
    jacobian_eps: float,
) -> torch.Tensor:
    assert x.ndim == 2, ValueError(f"x should be 2D, got {x.ndim}D instead.")
    latent_dim = x.shape[1]
    decoder_input_delta = torch.cat(
        [
            torch.eye(latent_dim, device=x.device) * jacobian_eps,
            torch.zeros((1, latent_dim), device=x.device),
        ],
        dim=0,
    )
    decoder_input = rearrange(
        rearrange(x, "b d -> b 1 d")
        + rearrange(decoder_input_delta, "d_plus_1 d -> 1 d_plus_1 d"),
        "b d_plus_1 d -> (b d_plus_1) d",
    )
    with torch.no_grad():
        decoder_output = decoder_forward(decoder_input)

    decoder_output = rearrange(
        decoder_output, "(b d_plus_1) d -> b d_plus_1 d", b=x.shape[0]
    )
    approx_jac = (decoder_output[:, :-1, :] - decoder_output[:, [-1], :]) / jacobian_eps

    return rearrange(approx_jac, "b d a -> b a d")
