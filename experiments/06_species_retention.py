"""Retention times for KRb, NaK, SrF at realistic densities."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from holomol.data.species import (
    SPECIES, thermal_velocity, rotational_temperature
)
from holomol.storage.retention import (
    t2_from_density, phonon_decoherence_rate
)


def main():
    print("=" * 60)
    print("Experiment 06: Species-Specific Retention")
    print("=" * 60)

    T_gas = 1e-6
    densities = np.logspace(16, 22, 60)
    omega_ph = 2 * np.pi * 1e5
    T_phonon = np.array([1e-3, 1e-2, 1e-1, 1.0])

    print(f"\nGas temperature T = {T_gas:.0e} K")
    print(f"Trap frequency   = {omega_ph/(2*np.pi):.1e} Hz")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, s in SPECIES.items():
        v_th = thermal_velocity(T_gas, s["mass_amu"])
        sigma = s["sigma_collision_m2"]
        T2 = np.array([t2_from_density(n, sigma, v_th) for n in densities])
        ax[0].loglog(densities, T2, "-", label=name)

        T2_ph = []
        for T in T_phonon:
            rate = phonon_decoherence_rate(T, omega_ph, 1e3)
            T2_ph.append(1.0 / max(rate, 1e-30))
        ax[1].semilogy(T_phonon, T2_ph, "o-", label=name)

    ax[0].set_xlabel(r"density (m$^{-3}$)")
    ax[0].set_ylabel(r"$T_2$ collisional (s)")
    ax[0].set_title(f"Collisional decoherence (T={T_gas:.0e} K)")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

    ax[1].set_xlabel("temperature (K)")
    ax[1].set_ylabel(r"$T_2$ phonon-limited (s)")
    ax[1].set_title("Phonon decoherence")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "06_species_retention.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")

    print("\nT2 collisional at n = 1e18 m^-3, T = 1 uK:")
    print(f"{'species':>6} {'v_th (m/s)':>12} {'sigma (m^2)':>14} {'T2 (s)':>12}")
    for name, s in SPECIES.items():
        v_th = thermal_velocity(T_gas, s["mass_amu"])
        T2 = t2_from_density(1e18, s["sigma_collision_m2"], v_th)
        print(f"{name:>6} {v_th:>12.4e} {s['sigma_collision_m2']:>14.2e} "
              f"{T2:>12.4e}")


if __name__ == "__main__":
    main()
