"""Dark manifold as a write-isolation resource.

Setup: a multiplexed memory with two logical qubits:
    q_A stored in |f, q0>  (bright at q0, the write channel)
    q_B stored in |f, q1>  (dark at q0, protected)

Test: apply K write pulses on the q0 channel. Each pulse is imperfect
(small off-resonant tail). Measure how much of q_A leaks out vs q_B.

Model: leakage per pulse = epsilon * S(q1 - q0), where S is structure
factor. For bright-bright pairs, S is large. For dark-bright pairs,
S is small at large q separation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import make_1d_lattice


def S_lattice(N, a, dq):
    dq = np.asarray(dq, dtype=float)
    half = dq * a / 2.0
    denom = N * np.sin(half)
    numer = np.sin(N * half)
    mask = np.abs(denom) < 1e-12
    out = np.ones_like(numer, dtype=float)
    np.divide(np.abs(numer), np.abs(denom), out=out, where=~mask)
    return out


def simulate(N, a, K_pulses, epsilon=0.01, delta_q_frac=0.1):
    """Return (F_bright, F_dark) after K pulses.

    F_bright: pair separated by delta_q_frac * BZ (small separation)
    F_dark:   pair separated by 0.9 * BZ (large separation)
    """
    q_BZ = 2 * np.pi / a
    S_bright = S_lattice(N, a, delta_q_frac * q_BZ)
    S_dark = S_lattice(N, a, 0.9 * q_BZ)
    F_b = (1 - epsilon * S_bright) ** K_pulses
    F_d = (1 - epsilon * S_dark) ** K_pulses
    return F_b, F_d


def main():
    print("=" * 60)
    print("Experiment 13b: Dark Manifold as Write-Isolation Resource")
    print("=" * 60)

    a = 0.5e-6
    Ns = [100, 1000, 10000]
    K_vals = np.arange(0, 1001, 10)

    print(f"\nPer-pulse leakage epsilon = 1e-2")
    print(f"\n{'N':>6} {'K=10':>14} {'K=100':>14} {'K=1000':>14}")
    print(f"{'':>6} {'F_bright F_dark':>28} ")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for N in Ns:
        Fb, Fd = [], []
        for K in K_vals:
            b, d = simulate(N, a, K)
            Fb.append(b); Fd.append(d)
        print(f"{N:>6d}  Fb={Fb[1]:.6f} Fd={Fd[1]:.6f}  "
              f"Fb={Fb[10]:.6f} Fd={Fd[10]:.6f}  "
              f"Fb={Fb[-1]:.6f} Fd={Fd[-1]:.6f}")
        ax[0].semilogy(K_vals, Fb, "--", label=f"bright N={N}", alpha=0.6)
        ax[0].semilogy(K_vals, Fd, "-",  label=f"dark   N={N}", alpha=0.9)

    ax[0].set_xlabel("number of write pulses K")
    ax[0].set_ylabel("survival fidelity")
    ax[0].set_title("Write-isolation: dark vs bright channel")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which="both")
    ax[0].set_ylim(0.5, 1.02)

    # Encode 2 logical bits per memory cell using bright + dark channels
    print(f"\n{'N':>6} {'K_pulses':>10} {'(F_bright, F_dark)':>30} "
          f"{'advantage':>14}")
    for N in Ns:
        for K in [10, 100, 1000]:
            b, d = simulate(N, a, K)
            print(f"{N:>6d} {K:>10d} ({b:.6f}, {d:.6f}) {d/b:>14.3e}")

    # Advantage vs separation
    print("\nFidelity advantage F_dark / F_bright vs mode separation:")
    N = 1000
    K = 100
    fracs = np.linspace(0.05, 0.95, 30)
    ratios = []
    for fr in fracs:
        q_BZ = 2 * np.pi / a
        S_b = S_lattice(N, a, fr * q_BZ)
        S_d = S_lattice(N, a, (1 - fr) * q_BZ)  # symmetric separation
        Fb = (1 - 0.01 * S_b) ** K
        Fd = (1 - 0.01 * S_d) ** K
        ratios.append(Fd / Fb)
    ax[1].semilogx(fracs, ratios, "o-")
    ax[1].set_xlabel("mode separation as fraction of BZ")
    ax[1].set_ylabel("F_dark / F_bright")
    ax[1].set_title(f"Fidelity advantage (N={N}, K={K})")
    ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "13b_dark_isolation.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
