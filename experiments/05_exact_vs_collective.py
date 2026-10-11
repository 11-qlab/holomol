"""Exact diagonalization vs collective-mode prediction (corrected)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import make_1d_lattice
from holomol.sim.exact import (
    effective_raman_hamiltonian, diagonalize, collective_mode_vector,
    all_ground_vector
)


def main():
    print("=" * 60)
    print("Experiment 05: Exact vs Collective (N+1 basis)")
    print("=" * 60)

    a = 0.5e-6
    Ns = [4, 8, 16, 32, 64, 128]
    Omega_eff = 1.0
    omega_gf = 0.0

    print(f"\n{'N':>5} {'dim':>5} {'|<allg|GS>|^2':>14} "
          f"{'|<f,q0|GS>|^2':>14} {'E0 exact':>12} "
          f"{'-sqrt(N)|Om|':>16} {'n_dark':>8}")

    ov_g_list, ov_f_list, E0_list, dark_list = [], [], [], []
    for N in Ns:
        ens = make_1d_lattice(N, a)
        q0 = np.array([2 * np.pi / (N * a), 0.0, 0.0])

        H = effective_raman_hamiltonian(ens, omega_gf, Omega_eff, q0)
        E, V = diagonalize(H)
        gs = V[:, 0]

        ov_g = abs(np.vdot(all_ground_vector(ens), gs)) ** 2
        ov_f = abs(np.vdot(collective_mode_vector(ens, q0), gs)) ** 2
        E0 = E[0]
        pred = -np.sqrt(N) * abs(Omega_eff) / 2.0
        n_dark = int(np.sum(np.abs(E) < 1e-8))

        ov_g_list.append(ov_g); ov_f_list.append(ov_f)
        E0_list.append(E0); dark_list.append(n_dark)

        print(f"{N:>5d} {N+1:>5d} {ov_g:>14.6f} {ov_f:>14.6f} "
              f"{E0:>12.6f} {pred:>16.6f} {n_dark:>8d}")

    # Off-resonant probes on GS (N=16)
    print("\nOff-resonant probe on GS (N=16, expect near 0):")
    N = 16
    ens = make_1d_lattice(N, a)
    q0 = np.array([2 * np.pi / (N * a), 0.0, 0.0])
    H = effective_raman_hamiltonian(ens, omega_gf, Omega_eff, q0)
    E, V = diagonalize(H)
    gs = V[:, 0]
    for k in [1, 2, 3, 5, 10]:
        q_probe = np.array([2 * np.pi * (1 + k) / (N * a), 0.0, 0.0])
        ov = abs(np.vdot(collective_mode_vector(ens, q_probe), gs)) ** 2
        print(f"  k={k:2d}  |<f,q_probe|GS>|^2 = {ov:.3e}")

    # Full spectrum structure for N=16
    print("\nEnergy spectrum structure (N=16):")
    print(f"  total eigenvalues: {len(E)}  (expect N+1 = {N+1})")
    print(f"  |E| < 1e-8 (dark manifold): {int(np.sum(np.abs(E) < 1e-8))}"
          f"  (expect N-1 = {N-1})")
    print(f"  bright doublet: E = {E[0]:.6f}, {E[-1]:.6f}")
    print(f"  predicted:     {-np.sqrt(N)*abs(Omega_eff):.6f}, "
          f"{np.sqrt(N)*abs(Omega_eff):.6f}")

    # Plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].semilogx(Ns, ov_g_list, "o-", label=r"$|\langle all\,g|GS\rangle|^2$")
    ax[0].semilogx(Ns, ov_f_list, "s-", label=r"$|\langle f,q_0|GS\rangle|^2$")
    ax[0].axhline(0.5, color="k", ls=":", alpha=0.5)
    ax[0].set_xlabel("N"); ax[0].set_ylabel("overlap$^2$")
    ax[0].set_title("GS structure: 50/50 split")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].semilogx(Ns, np.abs(E0_list), "o-", label="|E0| exact")
    ax[1].semilogx(Ns, [np.sqrt(N) * abs(Omega_eff) / 2 for N in Ns],
                   "r--", label=r"$\sqrt{N}\,|\Omega|/2$")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("|E0|")
    ax[1].set_title("Superradiant ground-state energy")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    ax[2].plot(Ns, dark_list, "o-")
    ax[2].plot(Ns, [N - 1 for N in Ns], "r--",
               label=r"$N-1$")
    ax[2].set_xlabel("N"); ax[2].set_ylabel("# zero eigenvalues")
    ax[2].set_title("Dark manifold dimension")
    ax[2].legend(); ax[2].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "05_exact_vs_collective.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
