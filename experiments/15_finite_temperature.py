"""Temperature-dependent retention: collisions + Boltzmann population."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.data.species import SPECIES, thermal_velocity, rotational_temperature


def boltzmann_ground_fraction(T, T_rot):
    """Fraction in rotational ground state for a linear rotor (J=0)."""
    if T <= 0 or T_rot <= 0:
        return 1.0
    # Truncate at J=10 (higher states negligible below T ~ 10*T_rot)
    Js = np.arange(0, 15)
    energies = Js * (Js + 1) * T_rot
    degeneracies = 2 * Js + 1
    weights = degeneracies * np.exp(-energies / T)
    return float(weights[0] / weights.sum())


def effective_fidelity(T, T_rot, mass_amu, sigma, n_density, t_hold):
    v_th = thermal_velocity(T, mass_amu)
    T2_max = 1000.0
    if n_density * sigma * v_th <= 0:
        T2 = np.inf
    else:
        T2 = min(1.0 / (n_density * sigma * v_th), T2_max)
    P0 = boltzmann_ground_fraction(T, T_rot)
    return P0 * np.exp(-t_hold / T2), T2, P0


def main():
    print("=" * 60)
    print("Experiment 15: Finite-Temperature Retention")
    print("=" * 60)

    T = np.logspace(-9, -6, 100)     # 1 nK to 1 uK (relevant range)
    n = 1e18
    t_hold = 1.0

    print(f"n = {n:.1e} m^-3, hold = {t_hold} s")
    print(f"\n{'Species':>8} {'T_rot (mK)':>12} {'T_opt (mK)':>12} "
          f"{'F_max':>10} {'T2 @ opt':>12}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, s in SPECIES.items():
        T_rot = rotational_temperature(s["B_rot_GHz"])
        F_list, T2_list, P0_list = [], [], []
        for t in T:
            F, T2, P0 = effective_fidelity(t, T_rot, s["mass_amu"],
                                           s["sigma_collision_m2"],
                                           n, t_hold)
            F_list.append(F); T2_list.append(T2); P0_list.append(P0)
        F_arr = np.array(F_list)
        idx_opt = np.argmax(F_arr)
        T_opt = T[idx_opt]
        F_opt = F_arr[idx_opt]
        print(f"{name:>8} {T_rot*1e3:>12.3f} {T_opt*1e3:>12.4f} "
              f"{F_opt:>10.6f} {T2_list[idx_opt]:>12.3e}")

        ax[0].loglog(T, F_arr, "-", label=f"{name}", alpha=0.8)
        ax[1].loglog(T, P0_list, "--", label=f"{name} P0", alpha=0.6)
        ax[1].loglog(T, T2_list, ":", label=f"{name} T2", alpha=0.6)

    ax[0].set_xlabel("T (K)"); ax[0].set_ylabel("fidelity at 1 s hold")
    ax[0].set_title("Optimal temperature window")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[0].axvline(1e-6, color="k", ls=":", alpha=0.5, label="1 uK")

    ax[1].set_xlabel("T (K)"); ax[1].set_ylabel("value")
    ax[1].set_title("Population P0 vs coherence T2")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "15_finite_temperature.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
