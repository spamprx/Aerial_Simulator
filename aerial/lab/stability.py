"""Step-size and stability checks for second-order consensus.

For consensus mode lam (an eigenvalue of the Laplacian), one step of the simulator's integrator is

    M(lam) = [[1 - k_p*lam*dt^2/2,  dt - k_v*lam*dt^2/2],
              [   -k_p*lam*dt,         1 - k_v*lam*dt   ]]

Jury's test on its characteristic polynomial gives the whole stability region in closed form, for
k_p > 0 and real lam > 0:

    stable  <=>  k_v * lam * dt < 2   and   k_p * dt / 2 < k_v

The first bound is why summed gains blow up as a fleet grows, and it depends only on the largest
lam.  The second is independent of lam: with k_v = 0 no step size is stable, because the zero-order
hold pumps energy into an undamped oscillator.

Summed gains see the eigenvalues of L = D - A.  Normalised gains (D11) see those of I - D^-1 A,
which lie in [0, 2] whatever the graph, so they get one check that holds for every fleet size.

Damping with the agent's own velocity (DP-MSR, the default under W-MSR) takes lam out of the velocity
term, and the same test then gives: stable  <=>  k_v * dt < 2  and  k_p * lam * dt / 2 < k_v.
"""
from __future__ import annotations

import warnings

import numpy as np

from .coordination.consensus import SecondOrderConsensus
from .network.comm import FixedComm
from .network.graph import laplacian

NORMALISED_LAMBDA_MAX = 2.0


def update_matrix(k_p: float, k_v: float, dt: float, lam, own_damping: bool = False) -> np.ndarray:
    """M(lam), the exact one-step update of a consensus mode.  ``lam`` may be complex."""
    damping = k_v if own_damping else k_v * lam
    return np.array([[1.0 - k_p * lam * dt ** 2 / 2.0, dt - damping * dt ** 2 / 2.0],
                     [-k_p * lam * dt, 1.0 - damping * dt]])


def spectral_radius(k_p: float, k_v: float, dt: float, lam, own_damping: bool = False) -> float:
    return float(np.abs(np.linalg.eigvals(update_matrix(k_p, k_v, dt, lam, own_damping))).max())


def is_stable(k_p: float, k_v: float, dt: float, lam_max: float, grid: int = 200, own_damping: bool = False) -> bool:
    """Whether no consensus mode with lam in (0, lam_max] grows.

    With k_p = 0, positions have nothing to pull them together, so one eigenvalue is exactly 1: they
    hold still rather than converge.  That counts as stable, since nothing diverges.  So does
    ``lam_max = 0``: a lone agent has no consensus modes at all.
    """
    if lam_max <= 0.0:
        return True
    limit = 1.0 + 1e-12 if k_p == 0.0 else 1.0
    radii = [spectral_radius(k_p, k_v, dt, lam, own_damping) for lam in np.linspace(lam_max / grid, lam_max, grid)]
    return max(radii) <= limit if k_p == 0.0 else max(radii) < limit


def is_stable_normalised(k_p: float, k_v: float, dt: float) -> bool:
    """``is_stable`` for controllers with mean neighbour weights (D11), at any fleet size."""
    return is_stable(k_p, k_v, dt, NORMALISED_LAMBDA_MAX)


def max_stable_dt(k_p: float, k_v: float, lam_max: float) -> float:
    """The step size at which stability is lost.  Stable for every dt below this, not at it."""
    if lam_max <= 0.0:
        return float("inf")
    if k_v <= 0.0:
        return 0.0 if k_p > 0.0 else float("inf")
    bound = 2.0 / (k_v * lam_max)
    return bound if k_p == 0.0 else min(bound, 2.0 * k_v / k_p)


def predicted_rate(k_p: float, k_v: float, dt: float, lam) -> float:
    """Decay rate per second of consensus mode ``lam``, negative if it grows.

    Pass lambda_2 to get the rate that lambda_2 alone implies.  This need not be the fleet's rate:
    with k_v^2 * lam > 4 * k_p a mode is overdamped, and its rate then falls towards k_p / k_v as
    lam grows.  So a larger lam can be slower, and ``fleet_rate`` checks every mode.
    """
    return -float(np.log(spectral_radius(k_p, k_v, dt, lam))) / dt


def fleet_rate(k_p: float, k_v: float, dt: float, adjacency: np.ndarray, scale: float = 1.0) -> float:
    """Decay rate per second of the slowest non-trivial mode of this graph; 0 if it is disconnected.

    ``scale`` multiplies the Laplacian: with independent packet loss p, L is on average (1 - p) * L.
    """
    eigenvalues = np.linalg.eigvals(laplacian(adjacency))
    eigenvalues = np.delete(eigenvalues, np.argmin(np.abs(eigenvalues)))  # the agreement mode itself
    if len(eigenvalues) == 0 or np.abs(eigenvalues).min() < 1e-9:
        return 0.0
    return min(predicted_rate(k_p, k_v, dt, scale * lam) for lam in eigenvalues)


def worst_case_lambda(comm, n: int) -> float:
    """The largest lam of L the fleet can meet under this communication model.

    A fixed graph with no packet loss has a known spectrum.  Otherwise the bound is min(2 * d_max, n):
    2 * d_max is Gershgorin's, and n is exact for the complete graph, which the Laplacian of any
    symmetric graph on n nodes cannot exceed.  Gershgorin alone would call N = 20 at k_v = 1.5
    unstable (lam <= 38), and it is not: a complete graph has lam = 20.
    """
    if isinstance(comm, FixedComm):
        base = comm.base
        if comm.drop_prob <= 0.0 and np.array_equal(base, base.T):
            return float(np.linalg.eigvalsh(laplacian(base)).max())
        return float(min(2 * base.sum(axis=1).max(), n))
    return float(min(2 * (n - 1), n))


def check_controller_stability(controller, cfg, comm) -> None:
    """Warn if this controller's consensus gains are unstable in the worst case for this setup."""
    if isinstance(controller, SecondOrderConsensus) and controller.filter is None:
        gains, lam_max, kind = (controller.k_p, controller.k_v), worst_case_lambda(comm, cfg.n_agents), "summed"
    else:
        gains, lam_max, kind = controller.normalised_gains(), NORMALISED_LAMBDA_MAX, "normalised"
    own_damping = getattr(controller, "damping", None) == "own"
    if gains is None or is_stable(gains[0], gains[1], cfg.dt, lam_max, own_damping=own_damping):
        return
    hint = "k_v * dt < 2 and k_p * lambda * dt / 2 < k_v" if own_damping \
        else f"dt < {max_stable_dt(gains[0], gains[1], lam_max):.4g}"
    warnings.warn(f"{type(controller).__name__} with {kind} gains k_p={gains[0]}, k_v={gains[1]} is unstable at "
                  f"dt={cfg.dt} for lambda up to {lam_max:g}: it needs {hint}")
