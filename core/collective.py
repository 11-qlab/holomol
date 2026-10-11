"""Collective phase modes |f,q> and their orthogonality."""
import numpy as np


def collective_mode(ensemble, q) -> np.ndarray:
    """
    Return |f,q> as an N-vector of complex amplitudes:

        |f,q> = (1/sqrt(N)) sum_j exp(i q . x_j) |g...f_j...g>

    So that |f,q> = sum_j c_j |g_1 ... f_j ... g_N>.
    """
    q = np.asarray(q, dtype=float)
    phases = np.exp(1j * ensemble.positions @ q)
    return phases / np.sqrt(ensemble.N)


def overlap(ensemble, q1, q2) -> complex:
    """<f,q1|f,q2> = S(q2 - q1), the static structure factor."""
    dq = np.asarray(q2) - np.asarray(q1)
    return np.sum(np.exp(1j * ensemble.positions @ dq)) / ensemble.N


def structure_factor(ensemble, dq) -> complex:
    dq = np.asarray(dq, dtype=float)
    return np.sum(np.exp(1j * ensemble.positions @ dq)) / ensemble.N


def orthogonality_matrix(ensemble, q_list) -> np.ndarray:
    Nq = len(q_list)
    M = np.zeros((Nq, Nq), dtype=complex)
    for i, qi in enumerate(q_list):
        for j, qj in enumerate(q_list):
            M[i, j] = overlap(ensemble, qi, qj)
    return M


def mode_gram_schmidt(ensemble, q_list, tol: float = 1e-6):
    """Return orthogonalized modes and the q-vectors that survived."""
    A = np.column_stack([collective_mode(ensemble, q) for q in q_list])
    Q, R = np.linalg.qr(A)
    diag = np.abs(np.diag(R))
    keep = diag > tol
    return Q[:, keep], [q_list[i] for i, k in enumerate(keep) if k]


def dark_state(Omega1: complex, Omega2: complex):
    """
    Single-molecule dark state under Raman coupling:

        |D> = (Omega2 |g> - Omega1 |f>) / sqrt(|Omega1|^2 + |Omega2|^2)

    Returns (amp_g, amp_f).
    """
    norm = np.sqrt(abs(Omega1) ** 2 + abs(Omega2) ** 2)
    if norm == 0:
        return 1.0 + 0j, 0.0 + 0j
    return Omega2 / norm, -Omega1 / norm


def bright_state(Omega1: complex, Omega2: complex):
    norm = np.sqrt(abs(Omega1) ** 2 + abs(Omega2) ** 2)
    if norm == 0:
        return 0.0 + 0j, 1.0 + 0j
    return np.conj(Omega1) / norm, np.conj(Omega2) / norm
