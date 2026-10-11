"""Mean-field / Holstein-Primakoff treatment for large N."""
import numpy as np


def structure_factor_lattice(N: int, dq: float, a: float) -> complex:
    """
    Analytic structure factor for a 1D lattice of N sites, spacing a:
        S(dq) = (1/N) * sin(N dq a / 2) / sin(dq a / 2)
                * exp(i (N-1) dq a / 2)
    """
    s = np.sin(dq * a / 2.0)
    if abs(s) < 1e-30:
        return 1.0 + 0j
    return (np.sin(N * dq * a / 2.0) / (N * s)) * np.exp(1j * (N - 1) * dq * a / 2.0)


def hp_spectrum(q_axis, omega_0: float, hopping: float, a: float):
    """Linear spin-wave spectrum: omega(q) = omega_0 + 2 J cos(q a)."""
    return omega_0 + 2.0 * hopping * np.cos(np.asarray(q_axis) * a)


def self_consistent_magnetization(J: float, T: float,
                                  tol: float = 1e-8,
                                  max_iter: int = 5000) -> float:
    m = 0.5
    if T <= 0:
        return 1.0
    for _ in range(max_iter):
        m_new = np.tanh(J * m / T)
        if abs(m_new - m) < tol:
            break
        m = m_new
    return float(m)


def mode_orthogonality_scaling(N_values, delta_q: float, a: float = 1.0):
    """Overlap magnitudes |S(dq)| for the analytic Dirichlet kernel."""
    return np.array([abs(structure_factor_lattice(N, delta_q, a))
                     for N in N_values])


def capacity_estimate(N: int, q_max: float, a: float,
                      threshold: float = 0.1) -> int:
    """
    Estimate the number of distinguishable modes by counting q's on a
    uniform grid whose mutual overlap stays below threshold.
    """
    qs = np.linspace(-q_max, q_max, max(N, 100))
    kept = []
    for q in qs:
        ok = True
        for qk in kept:
            if abs(structure_factor_lattice(N, q - qk, a)) > threshold:
                ok = False
                break
        if ok:
            kept.append(q)
    return len(kept)
