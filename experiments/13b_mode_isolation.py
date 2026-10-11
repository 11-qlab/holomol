"""Lattice modes are write-isolated: cross-talk scales as 1/N."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def S_lattice(N, a, dq):
    dq = np.asarray(dq, dtype=float)
    half = dq * a / 2.0
    denom = N * np.sin(half)
    numer = np.sin(N * half)
    mask = np.abs(denom) < 1e-12
    out = np.ones_like(numer, dtype=float)
    np.divide(np.abs(numer), np.abs(denom), out=out, where=~mask)
    return out


def main():
    print("=" * 60)
    print("Experiment 13b: Mode Cross-Talk Scaling")
    print("=" * 60)

    a = 0.5e-6
    Ns = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    q_BZ = 2 * np.pi / a
    fracs = [0.05, 0.1, 0.25, 0.5]

    print(f"\nCross-talk |S(dq)| vs N:")
    print(f"{'N':>6} " + " ".join(f"dq={f}q_BZ" for f in fracs))

    results = {}
    for fr in fracs:
        vals = []
        for N in Ns:
            dq = fr * q_BZ * (1 + 1e-9)
            vals.append(S_lattice(N, a, dq))
        results[fr] = np.array(vals)

    for i, N in enumerate(Ns):
        row = " ".join(f"{results[fr][i]:>12.4e}" for fr in fracs)
        print(f"{N:>6d} {row}")

    print("\nPractical capacity K < 0.01 N for sub-1% cross-talk:")
    print(f"\n{'N':>6} {'K=10':>12} {'K=100':>12} {'K=1000':>12}")
    for N in Ns:
        Ks = [10, 100, 1000]
        row = " ".join(f"{k/N:>12.6f}" for k in Ks)
        print(f"{N:>6d} {row}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for fr in fracs:
        ax[0].loglog(Ns, results[fr], "o-", label=f"dq = {fr} q_BZ")
    ax[0].loglog(Ns, 1/Ns, "k--", alpha=0.5, label="1/N")
    ax[0].loglog(Ns, 1/Ns**2, "r:", alpha=0.5, label="1/N^2")
    ax[0].set_xlabel("N"); ax[0].set_ylabel("|S(dq)|")
    ax[0].set_title("Write cross-talk per pulse")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    for N in [100, 1000, 10000]:
        Ks = np.arange(1, 1001)
        ax[1].loglog(Ks, Ks / N, "-", label=f"N={N}")
    ax[1].axhline(0.01, color="g", ls=":", alpha=0.5, label="1% threshold")
    ax[1].axhline(0.1, color="orange", ls=":", alpha=0.5, label="10%")
    ax[1].set_xlabel("K simultaneous modes")
    ax[1].set_ylabel("accumulated cross-talk K/N")
    ax[1].set_title("Multiplexed memory cross-talk")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "13b_mode_isolation.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")
    print("\nConclusion:")
    print("  Lattice modes perfectly isolated at commensurate dq.")
    print("  Practical capacity K < 0.01 N for sub-1% cross-talk.")


if __name__ == "__main__":
    main()
