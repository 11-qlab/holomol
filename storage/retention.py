"""Decoherence and retention models."""
import numpy as np

KB = 1.380649e-23  # J/K


def t2_from_density(n_density: float, sigma_collision: float,
                    v_thermal: float) -> float:
    """Collisional decoherence: 1/T2 = n * sigma * v."""
    if n_density <= 0 or sigma_collision <= 0 or v_thermal <= 0:
        return np.inf
    rate = n_density * sigma_collision * v_thermal
    return 1.0 / rate if rate > 0 else np.inf


def phonon_decoherence_rate(T: float, omega_phonon: float,
                            coupling: float) -> float:
    """Bose-enhanced phonon scattering rate."""
    if omega_phonon <= 0 or T <= 0:
        return 0.0
    x = omega_phonon / (KB * T)
    if x > 700:
        return float(coupling ** 2)
    n_B = 1.0 / (np.exp(x) - 1.0)
    return float(coupling ** 2 * (2.0 * n_B + 1.0))


def retention_probability(t: float, T2: float) -> float:
    if not np.isfinite(T2) or T2 <= 0:
        return 1.0
    return float(np.exp(-t / T2))


def inhomogeneous_broadening(sigma_freq: float) -> float:
    """Static Gaussian disorder std (rad/s)."""
    return float(sigma_freq)


def decay_probability(gamma: float, t: float) -> float:
    return float(np.exp(-gamma * t))
