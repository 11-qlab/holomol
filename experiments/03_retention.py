"""Retention time vs density and temperature."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.storage.retention import (
    t2_from_density, phonon_decoherence_rate
)


def main():
    print("=" * 60)
    print("Experiment 03: Storage Retention")
    print("=" * 60)

    densities = np.logspace(18, 22, 60)
    sigma = 1e-18
    v_th = 1e-2
    T2_coll = np.array([t2_from_density(n, sigma, v_th) for n in densities])

    T_list = np.logspace(-3, 0, 60)
    omega_ph = 2 * np.pi * 1e5
    coupling = 1e3
    rates = np.array([phonon_decoherence_rate(T, omega_ph, coupling)
                      for T in T_list])
    T2_ph = 1.0 / np.maximum(rates, 1e-30)

    print(f"T2 at n=1e18 m^-3, collisional:  {t2_from_density(1e18, sigma, v_th):.3e} s")
    print(f"T2 at n=1e22 m^-3, collisional:  {t2_from_density(1e22, sigma, v_th):.3e} s")
    print(f"T2 at T=1 mK, phonon-limited:    {1.0/max(phonon_decoherence_rate(1e-3, omega_ph, coupling),1e-30):.3e} s")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].loglog(densities, T2_coll, "b-")
    ax[0].set_xlabel(r"density (m$^{-3}$)")
    ax[0].set_ylabel(r"$T_2$ (s)")
    ax[0].set_title("Collisional decoherence")
    ax[0].grid(alpha=0.3, which="both")

    ax[1].loglog(T_list, T2_ph, "r-")
    ax[1].set_xlabel("temperature (K)")
    ax[1].set_ylabel(r"$T_2$ (s)")
    ax[1].set_title("Phonon-limited decoherence")
    ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "03_retention.png")
    plt.savefig(out, dpi=120)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
