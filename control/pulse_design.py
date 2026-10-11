"""Raman pulse envelopes for target q-vectors."""
import numpy as np


def gaussian_envelope(t, t0, sigma, amplitude=1.0):
    return amplitude * np.exp(-((t - t0) ** 2) / (2.0 * sigma ** 2))


def sech_envelope(t, t0, width, amplitude=1.0):
    return amplitude / np.cosh((t - t0) / width)


def drag_envelope(t, t0, sigma, amplitude=1.0, beta=0.0, alpha=0.0):
    """I/Q DRAG pair. Returns (I, Q)."""
    I = gaussian_envelope(t, t0, sigma, amplitude)
    dI = -(t - t0) / (sigma ** 2) * I
    Q = beta * dI + alpha * I
    return I, Q


def raman_pi_pulse(t_axis, Omega_eff, shape: str = "gaussian"):
    """Generate a pi-pulse envelope of the requested shape."""
    T = np.pi / abs(Omega_eff) if Omega_eff != 0 else np.inf
    if not np.isfinite(T):
        return np.zeros_like(t_axis)
    t0 = t_axis[-1] / 2.0
    sigma = T / 4.0
    if shape == "gaussian":
        return gaussian_envelope(t_axis, t0, sigma, amplitude=Omega_eff)
    if shape == "sech":
        return sech_envelope(t_axis, t0, T / 4.0, amplitude=Omega_eff)
    if shape == "square":
        env = np.zeros_like(t_axis)
        mask = np.abs(t_axis - t0) <= T / 2.0
        env[mask] = Omega_eff
        return env
    raise ValueError(f"unknown shape: {shape}")


def q_addressing_pulse(t_axis, q_target, k1_mag, Omega_eff,
                       theta_beam: float = 0.0):
    """
    Return (env1, env2, k1, k2) that address a target q = k1 - k2.
    k1 is fixed along +z. Beam geometry can add a tilt theta_beam.
    """
    k1 = np.array([0.0, 0.0, float(k1_mag)])
    if theta_beam != 0.0:
        k1 = np.array([np.sin(theta_beam), 0.0, np.cos(theta_beam)]) * k1_mag
    k2 = k1 - np.asarray(q_target, dtype=float)
    env = raman_pi_pulse(t_axis, Omega_eff)
    return env, env, k1, k2


def sample_rate_for(duration: float, min_samples: int = 1024) -> float:
    return max(min_samples / duration, 1e9)


def time_axis(duration: float, n_samples: int) -> np.ndarray:
    return np.linspace(0.0, duration, n_samples)
