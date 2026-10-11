"""Small-N exact diagonalization in the single-excitation subspace.

Basis: { |all g>, |f_1>, ..., |f_N> }.  Dimension N+1.

Convention: q = k1 - k2, so H[0, j+1] = Omega_eff * exp(-i q.x_j).
The bright collective mode is then |f, q> = (1/sqrt(N)) sum_j exp(i q.x_j) |f_j>.
"""
import numpy as np
from scipy.linalg import eigh


def effective_raman_hamiltonian(ensemble, omega_gf, Omega_eff, q):
    N = ensemble.N
    H = np.zeros((N + 1, N + 1), dtype=complex)
    q = np.asarray(q, dtype=float)
    for j in range(N):
        phase = np.exp(-1j * np.dot(q, ensemble.positions[j]))
        cpl = Omega_eff * phase
        H[0, j + 1] = cpl
        H[j + 1, 0] = np.conj(cpl)
        H[j + 1, j + 1] = omega_gf
    return H


def diagonalize(H):
    E, V = eigh(H)
    return E, V


def collective_mode_vector(ensemble, q):
    N = ensemble.N
    v = np.zeros(N + 1, dtype=complex)
    v[1:] = np.exp(1j * ensemble.positions @ np.asarray(q, dtype=float)) \
            / np.sqrt(N)
    return v


def all_ground_vector(ensemble):
    v = np.zeros(ensemble.N + 1, dtype=complex)
    v[0] = 1.0
    return v


def bright_coupling_magnitude(ensemble, Omega_eff, q):
    """|<f,q|H|all g>| = |Omega_eff| * sqrt(N)."""
    return abs(Omega_eff) * np.sqrt(ensemble.N)
