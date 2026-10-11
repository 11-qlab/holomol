"""Stripline cavity + Cooper pair box transfer."""
import numpy as np


class StriplineCavity:
    """Coplanar stripline cavity with mode omega_c and quality factor Q."""

    def __init__(self, freq_c: float, Q: float, coupling_g: float):
        self.freq_c = float(freq_c)
        self.Q = float(Q)
        self.g = float(coupling_g)

    @property
    def kappa(self) -> float:
        return self.freq_c / self.Q

    def photon_lifetime(self) -> float:
        return 1.0 / self.kappa if self.kappa > 0 else np.inf

    def cooperativity(self, gamma: float) -> float:
        return self.g ** 2 / (self.kappa * gamma) if self.kappa * gamma > 0 else np.inf


class CooperPairBox:
    """Transmon-like qubit used for state transfer and readout."""

    def __init__(self, freq_q: float, anharmonicity: float, coupling_g: float):
        self.freq_q = float(freq_q)
        self.alpha = float(anharmonicity)
        self.g = float(coupling_g)

    def dispersive_shift(self, n_photons: float, detuning: float) -> float:
        chi = self.g ** 2 / detuning if detuning != 0 else 0.0
        return chi * n_photons


def transfer_fidelity(g: float, kappa: float, T: float) -> float:
    """Population transfer fidelity ensemble -> cavity (simple model)."""
    if kappa <= 0:
        return 1.0
    return float(1.0 - np.exp(-(g ** 2) * T / kappa))


def readout_snr(n_photons: float, kappa: float, T_meas: float) -> float:
    return float(np.sqrt(max(kappa * T_meas * n_photons, 0.0)))


def cavity_q_required(T2_target: float, freq_c: float) -> float:
    """Q needed for a cavity photon lifetime >= T2_target."""
    return float(freq_c * T2_target)
