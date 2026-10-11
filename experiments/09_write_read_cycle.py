"""Write -> hold -> read cycle with realistic fidelities.

Effective 2-level collective model:
    |G> = |all g>,   |F> = |f, q0>
    Rabi frequency  Omega_R = Omega_eff * sqrt(N)     (superradiant)
    pi-pulse time   t_pi = pi / Omega_R
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_fidelity(delta_rabi: float) -> float:
    """F_write = sin^2(pi/2 * (1 + delta)) for over/under-rotation delta."""
    return float(np.sin(np.pi / 2.0 * (1.0 + delta_rabi)) ** 2)


def hold_fidelity(t_hold: float, T2: float) -> float:
    if T2 <= 0 or not np.isfinite(T2):
        return 1.0
    return float(np.exp(-t_hold / T2))


def read_fidelity(n_photons: float, kappa_t: float) -> float:
    """Simple readout model: F_read = 1 - exp(-kappa * t * n)."""
    return float(1.0 - np.exp(-kappa_t * n_photons))


def ber_from_fidelity(F: float) -> float:
    return 0.5 * (1.0 - F)


def main():
    print("=" * 60)
    print("Experiment 09: Write / Hold / Read Cycle")
    print("=" * 60)

    N = 1000
    Omega_eff = 2 * np.pi * 1e6        # 1 MHz single-molecule Rabi
    Omega_R = Omega_eff * np.sqrt(N)   # superradiant
    t_pi = np.pi / Omega_R
    print(f"N = {N}")
    print(f"Single-molecule Rabi: {Omega_eff/(2*np.pi):.3e} Hz")
    print(f"Collective Rabi:      {Omega_R/(2*np.pi):.3e} Hz  (sqrt(N) = {np.sqrt(N):.1f}x)")
    print(f"Pi-pulse duration:    {t_pi*1e9:.3f} ns")

    # Species-dependent T2 from Exp 06
    T2 = {"KRb": 15.0, "NaK": 18.0, "SrF": 35.5}

    # Sweep over hold times
    t_holds = np.logspace(-3, 2.5, 60)     # 1 ms to ~300 s
    deltas = [0.0, 0.01, 0.05]             # Rabi pulse error
    read_kappa_t = 10.0                    # kappa * t
    n_photons = 5.0
    F_read = read_fidelity(n_photons, read_kappa_t)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: fidelity vs hold time per species (ideal write/read)
    for sp, T2v in T2.items():
        F_hold = np.array([hold_fidelity(t, T2v) for t in t_holds])
        F_total = F_read * F_hold
        ax[0].semilogx(t_holds, F_total, "-", label=f"{sp} (T2={T2v}s)")
    ax[0].axhline(0.5, color="k", ls=":", alpha=0.5, label="BER = 25%")
    ax[0].axhline(0.99, color="g", ls=":", alpha=0.5, label="99%")
    ax[0].set_xlabel("hold time (s)")
    ax[0].set_ylabel("total fidelity")
    ax[0].set_title("Fidelity vs hold time")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    # Panel 2: total fidelity vs write error
    dvec = np.linspace(0, 0.2, 100)
    for sp, T2v in T2.items():
        for th in [0.1, 1.0, 10.0]:
            F_w = np.array([write_fidelity(d) for d in dvec])
            F_h = hold_fidelity(th, T2v)
            F = F_w * F_h * F_read
            ax[1].semilogx(np.maximum(dvec, 1e-4), F, "-",
                           label=f"{sp}, hold={th}s", alpha=0.7)
    ax[1].set_xlabel("Rabi pulse error delta")
    ax[1].set_ylabel("total fidelity")
    ax[1].set_title("Fidelity vs write error")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3, which="both")

    # Panel 3: BER
    for sp, T2v in T2.items():
        F_hold = np.array([hold_fidelity(t, T2v) for t in t_holds])
        F_total = F_read * F_hold
        BER = 0.5 * (1.0 - F_total)
        ax[2].loglog(t_holds, BER, "-", label=sp)
    ax[2].axhline(0.5, color="r", ls="--", alpha=0.5, label="random guess")
    ax[2].axhline(0.01, color="g", ls=":", alpha=0.5, label="1% BER")
    ax[2].set_xlabel("hold time (s)")
    ax[2].set_ylabel("bit-error rate")
    ax[2].set_title("Bit-error rate vs hold time")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "09_write_read_cycle.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")

    print("\nFidelity at 1 s hold:")
    for sp, T2v in T2.items():
        F = F_read * hold_fidelity(1.0, T2v)
        print(f"  {sp:>4s}  F_total = {F:.4f}   BER = {ber_from_fidelity(F):.4e}")


if __name__ == "__main__":
    main()
