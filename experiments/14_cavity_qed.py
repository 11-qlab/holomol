"""Cavity transfer efficiency with realistic kappa, gamma, G.

Model: two-mode beam-splitter between collective bright mode and cavity,
plus cavity decay kappa and ensemble dephasing gamma.

Transfer efficiency:  eta = G^2 / (G^2 + (kappa+gamma)^2/4)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def transfer_efficiency(G, kappa, gamma):
    return G**2 / (G**2 + (kappa + gamma)**2 / 4.0)


def cooperativity(G, kappa, gamma):
    return 4 * G**2 / (kappa * gamma)


def collective_coupling(g0, N, mode_volume_ratio=1.0):
    """G = g0 * sqrt(N) for the symmetric bright mode."""
    return g0 * np.sqrt(N) * np.sqrt(mode_volume_ratio)


def main():
    print("=" * 60)
    print("Experiment 14: Full Cavity QED Transfer")
    print("=" * 60)

    # Realistic stripline cavity parameters (from hybrid atom-cavity literature)
    # Coupling: g0 ~ 2pi * 10 kHz per molecule
    # Cavity decay: kappa/2pi ~ 100 kHz to 10 MHz
    # Ensemble dephasing: gamma/2pi ~ 1 kHz
    g0 = 2 * np.pi * 1e4           # single-molecule coupling
    Ns = np.array([10, 100, 1000, 10000, 100000])
    kappas = np.array([2*np.pi*1e5, 2*np.pi*1e6, 2*np.pi*1e7])
    gamma = 2 * np.pi * 1e3

    print(f"\nSingle-molecule g0/2pi: {g0/(2*np.pi):.3e} Hz")
    print(f"Ensemble gamma/2pi:     {gamma/(2*np.pi):.3e} Hz")
    print(f"\n{'N':>8} {'G/2pi (Hz)':>14} {'kappa/2pi':>12} "
          f"{'C':>10} {'eta':>10}")

    for kappa in kappas:
        for N in Ns:
            G = collective_coupling(g0, N)
            C = cooperativity(G, kappa, gamma)
            eta = transfer_efficiency(G, kappa, gamma)
            print(f"{N:>8d} {G/(2*np.pi):>14.3e} {kappa/(2*np.pi):>12.2e} "
                  f"{C:>10.2f} {eta:>10.6f}")

    # --- Parameter space map ---
    G_range = np.logspace(4, 9, 60)
    kappa_range = np.logspace(4, 9, 60)
    G_grid, K_grid = np.meshgrid(G_range, kappa_range)
    eta_grid = transfer_efficiency(G_grid, K_grid, gamma)
    C_grid = cooperativity(G_grid, K_grid, gamma)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    im0 = ax[0].pcolormesh(kappa_range, G_range, eta_grid.T,
                           shading="auto", cmap="viridis", vmin=0, vmax=1)
    ax[0].contour(kappa_range, G_range, C_grid.T,
                  levels=[1, 10, 100, 1000], colors="white", linewidths=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\kappa/2\pi$ (Hz)")
    ax[0].set_ylabel(r"$G/2\pi$ (Hz)")
    ax[0].set_title("Transfer efficiency")
    plt.colorbar(im0, ax=ax[0])
    for N, lab in zip(Ns[1:], ["N=1e2", "N=1e3", "N=1e4", "N=1e5"]):
        G = collective_coupling(g0, N)
        ax[0].axhline(G, color="red", ls=":", alpha=0.5)
        ax[0].text(2e4, G*1.05, lab, color="red", fontsize=7)

    # Efficiency vs N for three kappas
    for kappa in kappas:
        etas = np.array([transfer_efficiency(collective_coupling(g0, N),
                                             kappa, gamma) for N in Ns])
        ax[1].semilogx(Ns, etas, "o-", label=f"kappa/2pi={kappa/(2*np.pi):.1e}")
    ax[1].axhline(0.99, color="g", ls=":", alpha=0.5)
    ax[1].set_xlabel("N"); ax[1].set_ylabel("transfer efficiency")
    ax[1].set_title("Efficiency vs ensemble size")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

    # Cooperativity heat map
    im2 = ax[2].pcolormesh(kappa_range, G_range, np.log10(C_grid.T + 1e-3),
                           shading="auto", cmap="plasma")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$\kappa/2\pi$ (Hz)")
    ax[2].set_ylabel(r"$G/2\pi$ (Hz)")
    ax[2].set_title(r"$\log_{10}$ cooperativity")
    plt.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "14_cavity_qed.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
