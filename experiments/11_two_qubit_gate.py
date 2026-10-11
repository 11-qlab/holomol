"""Cavity-mediated ZZ / CZ gate between two collective modes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIGMA_Z = np.diag([1.0, -1.0])


def zz_unitary(J: float, t: float) -> np.ndarray:
    """U_ZZ(t) = exp(-i J t sigma_z sigma_z)."""
    H = J * np.kron(SIGMA_Z, SIGMA_Z)
    return expm(-1j * H * t)


def zz_fidelity(J: float, t: float) -> float:
    """|<ideal ZZ at pi/4|U_actual>|/4.  Ideal ZZ: theta = pi/4."""
    U = zz_unitary(J, t)
    U_ideal = zz_unitary(1.0, np.pi / 4.0)
    return float(abs(np.trace(U_ideal.conj().T @ U)) / 4.0)


def cz_from_zz(U_zz: np.ndarray) -> np.ndarray:
    """Convert ZZ(pi/4) to CZ via single-qubit Z rotations + global phase."""
    # CZ = e^{-i pi/4} * exp(i pi/4 sigma_z x I) * exp(i pi/4 I x sigma_z)
    #      * exp(-i pi/4 sigma_z x sigma_z)
    from scipy.linalg import expm as E
    Za = E(1j * np.pi / 4 * np.kron(SIGMA_Z, np.eye(2)))
    Zb = E(1j * np.pi / 4 * np.kron(np.eye(2), SIGMA_Z))
    phase = np.exp(-1j * np.pi / 4)
    return phase * Za @ Zb @ U_zz


def cz_fidelity(U_actual: np.ndarray) -> float:
    CZ = np.diag([1.0, 1.0, 1.0, -1.0])
    return float(abs(np.trace(CZ.conj().T @ U_actual)) / 4.0)


def main():
    print("=" * 60)
    print("Experiment 11: Two-Qubit ZZ / CZ Gate")
    print("=" * 60)

    J_values = np.logspace(3, 8, 40)
    t_cz = np.pi / (4 * J_values)

    print(f"Coupling range: {J_values.min():.0f} to {J_values.max():.0f} Hz")
    print(f"CZ time range:  {t_cz.min()*1e9:.3f} to {t_cz.max()*1e9:.3f} ns")

    T2 = 20.0
    gamma = 1.0 / T2

    F_zz = np.array([zz_fidelity(J, t) for J, t in zip(J_values, t_cz)])
    U_zz_best = zz_unitary(1.0, np.pi / 4.0)
    F_cz_at_opt = cz_fidelity(cz_from_zz(U_zz_best))

    F_decoh = F_zz * np.exp(-t_cz / T2)

    print(f"\nAt exact ZZ angle (theta = pi/4):")
    print(f"  ZZ gate fidelity: {F_zz[0]:.6f}")
    print(f"  CZ (with 1Q Z rotations): {F_cz_at_opt:.6f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    ax[0].semilogx(J_values, F_zz, "o-")
    ax[0].set_xlabel("J (Hz)"); ax[0].set_ylabel("ZZ fidelity")
    ax[0].set_title("Ideal ZZ gate fidelity")
    ax[0].set_ylim(0.9, 1.01); ax[0].grid(alpha=0.3, which="both")

    ax[1].loglog(t_cz, np.maximum(1.0 - F_zz, 1e-16), "o-",
                 label="ideal gate error")
    ax[1].loglog(t_cz, np.maximum(1.0 - F_decoh, 1e-16), "s-",
                 color="red", label="decoherence-limited")
    ax[1].axhline(1e-3, color="g", ls=":", alpha=0.5, label="10^-3")
    ax[1].axhline(1e-4, color="orange", ls=":", alpha=0.5, label="10^-4")
    ax[1].set_xlabel("gate time (s)")
    ax[1].set_ylabel("gate error")
    ax[1].set_title("Gate error vs gate time")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "11_two_qubit_gate.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")

    print("\nZZ gate at selected J:")
    print(f"{'J (Hz)':>12} {'t_ZZ (ns)':>12} {'F_ZZ':>10} {'F_decoh':>10}")
    for J in [1e4, 1e5, 1e6, 1e7]:
        t = np.pi / (4 * J)
        F_i = zz_fidelity(J, t)
        F_d = F_i * np.exp(-t / T2)
        print(f"{J:>12.1e} {t*1e9:>12.3f} {F_i:>10.6f} {F_d:>10.6f}")


if __name__ == "__main__":
    main()
