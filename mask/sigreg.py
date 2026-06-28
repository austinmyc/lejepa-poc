"""
SIGReg: Slice-based Isotropy Regularizer from the LeJEPA paper.

Identical maths to root isotropy.py, renamed sigreg_loss for the mask/ build.
Here it is applied to PROJECTION-space embeddings z = proj(encoder(x)), not the
raw encoder output — so isotropy is enforced on z while the encoder stays free
to be anisotropic.

For each of `num_slices` random unit directions it compares the empirical
characteristic function (ECF) of the 1D projection against the N(0,1) CF via the
Epps-Pulley test, integrated over a grid of t.

Reference: LeJEPA paper, Section 4.3.
"""

import torch


def sigreg_loss(
    x: torch.Tensor,
    num_slices: int = 512,
    num_points: int = 17,
    global_step: int = 0,
) -> torch.Tensor:
    """
    Args:
        x:           (N, P) float tensor — projection-space embeddings (full grad)
        num_slices:  number of random 1D projections (M in the paper).
        num_points:  integration grid size for the CF distance.
        global_step: seeds the slice directions so they are reproducible per step
                     and agree across devices. Resampled each step.

    Returns:
        Scalar loss. Zero when x is a perfectly isotropic standard Gaussian.
    """
    N, D = x.shape

    # Random unit projection directions — seeded by step so all ranks agree.
    g = torch.Generator(device=x.device)
    g.manual_seed(global_step)
    A = torch.randn((D, num_slices), generator=g, device=x.device, dtype=x.dtype)
    A = A / A.norm(p=2, dim=0)          # (D, M) unit columns

    # Integration points for the weighted L2 distance in CF space.
    t = torch.linspace(-5, 5, num_points, device=x.device, dtype=x.dtype)  # (T,)

    # Theoretical CF of N(0,1): phi(t) = exp(-0.5 t^2), used as reference and as
    # the Gaussian window weighting the error.
    exp_f = torch.exp(-0.5 * t ** 2)    # (T,)

    # Empirical CF: E[exp(i t <x, a>)] per slice a and per t.
    x_proj = x @ A                      # (N, M)
    x_t = x_proj.unsqueeze(2) * t       # (N, M, T)
    ecf_real = torch.cos(x_t).mean(0)   # (M, T)
    ecf_imag = torch.sin(x_t).mean(0)   # (M, T)

    # Weighted squared distance between empirical and theoretical CF.
    err = ((ecf_real - exp_f) ** 2 + ecf_imag ** 2) * exp_f  # (M, T)

    # Trapezoidal integration over t, averaged over slices. The * N scaling
    # matches the paper's Epps-Pulley statistic (proportional to sample size).
    T = torch.trapezoid(err, t, dim=1)  # (M,)
    return (T * N).mean()
