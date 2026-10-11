"""Adiabatic storage fidelity: dark-state evolution |g> -> |f>."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.collective import dark_state


def main():
    print("=" * 60)
    print("Experiment 02: Dark-State Adiabatic Storage |g> -> |f>")
    print("=" * 60)

    t = np.linspace(0.0, 1.0, 1000)

    # STIRAP: pump Omega1 goes 0 -> 1, Stokes Omega2 goes 1 -> 0
    # so the dark state rotates from |g> at t=0 to |f> at t=1.
    Omega1 = np.sin(np.pi * t / 2.0)   # pump
    Omega2 = np.cos(np.pi * t / 2.0)   # Stokes

    a_g = np.zeros_like(t)
    a_f = np.zeros_like(t)
    for i in range(len(t)):
        ag, af = dark_state(Omega1[i], Omega2[i])
        a_g[i] = abs(ag) ** 2
        a_f[i] = abs(af) ** 2

    print(f"t=0:  |<g|D>|^2 = {a_g[0]:.6f}   |<f|D>|^2 = {a_f[0]:.6f}")
    print(f"t=1:  |<g|D>|^2 = {a_g[-1]:.6f}   |<f|D>|^2 = {a_f[-1]:.6f}")

    # Landau-Zener: fidelity of adiabatic transfer vs sweep rate
    Omega = 1.0
    sweep = np.logspace(-2, 1, 40)
    P_diabatic = np.exp(-np.pi * Omega ** 2 / (2.0 * sweep))
    fidelity = 1.0 - P_diabatic

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(t, a_g, "b-", lw=1.8, label=r"$|g\rangle$")
    ax[0].plot(t, a_f, "r-", lw=1.8, label=r"$|f\rangle$")
    ax[0].set_xlabel("normalized time"); ax[0].set_ylabel("population")
    ax[0].set_title("STIRAP dark-state evolution")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].loglog(sweep, fidelity, "o-")
    ax[1].axhline(0.99, color="g", ls=":", label="99%")
    ax[1].set_xlabel("sweep rate (arb)")
    ax[1].set_ylabel("transfer fidelity")
    ax[1].set_title("Landau-Zener adiabatic fidelity")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "02_dark_state.png")
    plt.savefig(out, dpi=120)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
