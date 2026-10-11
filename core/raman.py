"""Raman coupling and effective q-vector addressing."""
import numpy as np


class RamanCoupling:
    """
    Two classical optical fields (k1, Omega1) and (k2, Omega2) drive a
    far-detuned Raman transition |g> -> |e> -> |f>.

    Effective two-photon Rabi frequency: Omega_eff = Omega1 * Omega2 / (2 Delta)
    Effective wavevector:                 q = k1 - k2
    """

    def __init__(self, k1, k2, Omega1, Omega2, detuning):
        self.k1 = np.asarray(k1, dtype=float)
        self.k2 = np.asarray(k2, dtype=float)
        self.Omega1 = complex(Omega1)
        self.Omega2 = complex(Omega2)
        self.detuning = float(detuning)

    @property
    def q(self):
        return self.k1 - self.k2

    @property
    def Omega_eff(self):
        denom = 2.0 * self.detuning if self.detuning != 0 else 1.0
        return self.Omega1 * self.Omega2 / denom

    def rabi_frequency(self) -> float:
        return float(np.abs(self.Omega_eff))

    def phase_pattern(self, positions) -> np.ndarray:
        return np.exp(1j * np.asarray(positions) @ self.q)


def effective_q(k1, k2):
    return np.asarray(k1, dtype=float) - np.asarray(k2, dtype=float)


def raman_pulse_area(Omega_eff, duration) -> float:
    return float(abs(Omega_eff) * duration)


def pi_pulse_duration(Omega_eff) -> float:
    a = abs(Omega_eff)
    return float(np.pi / a) if a > 0 else float("inf")


def two_photon_detuning(freq_g_e, laser1_freq, laser2_freq):
    return freq_g_e - (laser1_freq - laser2_freq)
