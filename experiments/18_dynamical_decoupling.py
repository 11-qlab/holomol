"""CPMG dynamical decoupling: extension of T2 vs pulse error."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def T2_with_dd(T2_free, n_pulses, noise_exponent=1.0):
    """CPMG extends T2 by n_pulses^alpha for 1/f noise (alpha=1)."""
    if n_pulses <= 0:
        return T2_free
    return T2_free * n_pulses ** noise_exponent


def total_fidelity(t_hold, T2_free, n_pulses, eps_pulse, alpha=1.0):
    """Fidelity including DD extension and cumulative pulse errors."""
    T2_dd = T2_with_dd(T2_free, n_pulses, alpha)
    F_coherence = np.exp(-t_hold / T2_dd)
    F_pulses = (1 - eps_pulse) ** n_pulses
    return F_coherence * F_pulses


def main():
    print("=" * 60)
    print("Experiment 18: Dynamical Decoupling")
    print("=" * 60)

    T2_free = 15.0           # KRb
    eps_pulse = 1e-3         # 0.1% per DD pulse
    t_hold = 1.0             # 1 s target hold

    n_pulses = np.logspace(0, 5, 100).astype(int)
    alphas = [0.5, 1.0, 1.5, 2.0]

    print(f"T2_free = {T2_free} s, target hold = {t_hold} s")
    print(f"per-pulse error = {eps_pulse:.1e}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    for alpha in alphas:
        F = np.array([total_fidelity(t_hold, T2_free, int(n), eps_pulse, alpha)
                      for n in n_pulses])
        i_opt = int(np.argmax(F))
        print(f"alpha={alpha:.1f}: n_opt = {n_pulses[i_opt]:>6d}, "
              f"F_max = {F[i_opt]:.6f}")
        ax[0].semilogx(n_pulses, F, "-", label=f"alpha={alpha}")

    ax[0].axhline(0.99, color="g", ls=":", alpha=0.5, label="99%")
    ax[0].set_xlabel("number of DD pulses")
    ax[0].set_ylabel("total fidelity at 1 s")
    ax[0].set_title("DD optimization")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    # Compare species
    from holomol.data.species import SPECIES
    T2_species = {"KRb": 15.0, "NaK": 18.0, "SrF": 35.5}
    for name, T2v in T2_species.items():
        F = np.array([total_fidelity(t_hold, T2v, int(n), eps_pulse, 1.0)
                      for n in n_pulses])
        ax[1].semilogx(n_pulses, F, "-", label=name)
    ax[1].set_xlabel("number of DD pulses")
    ax[1].set_ylabel("total fidelity at 1 s")
    ax[1].set_title("DD across species")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    # Effect on effective T2
    for alpha in alphas:
        T2_eff = np.array([T2_with_dd(T2_free, int(n), alpha)
                           for n in n_pulses])
        ax[2].loglog(n_pulses, T2_eff, "-", label=f"alpha={alpha}")
    ax[2].set_xlabel("number of DD pulses")
    ax[2].set_ylabel(r"$T_2^{DD}$ (s)")
    ax[2].set_title(r"Effective $T_2$ extension")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "18_dynamical_decoupling.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
