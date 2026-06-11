"""
SIGReg: Slice-based Isotropy Regularizer from the LeJEPA paper.

Measures how far the projected embeddings deviate from an isotropic Gaussian
using the Epps-Pulley characteristic function test. For each of `num_slices`
random unit directions, it computes the empirical characteristic function (ECF)
of the 1D projection and penalizes its distance from the N(0,1) CF.

Reference: LeJEPA paper, Section 4.3.
"""

import torch


def isotropy_loss(
    x: torch.Tensor,
    global_step: int = 0,
    num_slices: int = 256,
    num_points: int = 17,
) -> torch.Tensor:
    """
    SIGReg loss from the LeJEPA paper.

    Args:
        x:           (N, D) float tensor — projected embeddings (detached from encoder)
        global_step: training step, used to seed the slice directions so they are
                     consistent across devices / calls at the same step.
        num_slices:  number of random 1D projections (M in the paper).

    Returns:
        Scalar loss. Zero when x is perfectly isotropic Gaussian.
    """
    N, D = x.shape

    # Random unit projection directions — seeded by step so all ranks agree.
    g = torch.Generator(device=x.device)
    g.manual_seed(global_step)
    A = torch.randn((D, num_slices), generator=g, device=x.device, dtype=x.dtype)
    A = A / A.norm(p=2, dim=0)          # (D, M) unit columns

    # Integration points for the weighted L2 distance in CF space.
    t = torch.linspace(-5, 5, num_points, device=x.device, dtype=x.dtype)  # (T,)

    # Theoretical CF of N(0,1): phi(t) = exp(-0.5 * t^2), used as both
    # the reference and the Gaussian window weighting the error.
    exp_f = torch.exp(-0.5 * t ** 2)    # (T,)

    # Empirical CF: E[exp(i*t*<x, a>)] for each slice direction a and each t.
    # x @ A → (N, M); multiply by t → (N, M, T); take exp(i·…) and average over N.
    x_proj = x @ A                      # (N, M)
    x_t = x_proj.unsqueeze(2) * t       # (N, M, T)
    # Complex exponential via Euler: exp(i*u) = cos(u) + i*sin(u).
    # We only need the real part of |ECF - exp_f|^2 because exp_f is real and
    # the imaginary part of the ECF is zero for symmetric distributions at convergence,
    # but we track both for correctness during training.
    ecf_real = torch.cos(x_t).mean(0)   # (M, T)
    ecf_imag = torch.sin(x_t).mean(0)   # (M, T)

    # Weighted squared distance between empirical and theoretical CF.
    err = ((ecf_real - exp_f) ** 2 + ecf_imag ** 2) * exp_f  # (M, T)

    # Integrate over t using the trapezoidal rule; average over slices.
    # The * N scaling matches the paper's Epps-Pulley statistic — it makes
    # SIGReg proportional to sample size, which balances against the
    # mean-reduced pred loss when lam ≈ 0.05 and embeddings are near-Gaussian.
    T = torch.trapezoid(err, t, dim=1)  # (M,)
    return (T * N).mean()
