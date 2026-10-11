"""Storage capacity: orthogonal modes vs N. Analytic Dirichlet kernel."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def S_lattice(N, a, dq):
    """
    Analytic structure factor for 1D perfect lattice of N sites, spacing a.
    Returns |S(dq)| = |sin(N dq a/2) / (N sin(dq a/2))|, with proper limit at dq=0.
    """
    dq = np.asarray(dq, dtype=float)
    half = dq * a / 2.0
    # Safe denominator: where sin(half) ~ 0, S -> 1
    denom = N * np.sin(half)
    numer = np.sin(N * half)
    # Mask out the removable singularity
    out = np.where(np.abs(denom) < 1e-12, 1.0, np.abs(numer / denom))
    return out


def count_modes(N, a, q_max, threshold, oversample=4):
    """
    Greedy pick of q-points in [-q_max, q_max] such that pairwise
    |S(dq)| stays below threshold. Uses the analytic kernel — no N x N matrix.
    """
    n_q = max(400, oversample * N)
    qs = np.linspace(-q_max, q_max, n_q)
    kept = np.empty(n_q, dtype=float)
    k = 0
    for q in qs:
        if k == 0:
            kept[0] = q
            k = 1
            continue
        dq = q - kept[:k]
        if S_lattice(N, a, dq).max() <= threshold:
            kept[k] = q
            k += 1
    return k, kept[:k]


def main():
    print("=" * 60)
    print("Experiment 04: Storage Capacity (analytic kernel)")
    print("=" * 60)

    a = 0.5e-6
    Ns = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    q_max = np.pi / a
    thresholds = [0.10, 0.05, 0.02]

    results = {}
    for thr in thresholds:
        counts = []
        for N in Ns:
            c, _ = count_modes(N, a, q_max, thr)
            counts.append(c)
            print(f"  thr={thr:.2f}  N={N:5d} -> {c:5d} modes")
        results[thr] = np.array(counts)

    # Fit scaling exponent
    print("\nScaling exponents (log-log fit):")
    for thr, counts in results.items():
        mask = counts > 0
        slope = np.polyfit(np.log(Ns[mask]), np.log(counts[mask]), 1)[0]
        print(f"  thr={thr:.2f}  slope = {slope:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for thr, counts in results.items():
        ax[0].loglog(Ns, counts, "o-", label=f"thr={thr}")
    ax[0].loglog(Ns, Ns, "k--", alpha=0.4, label="N (guide)")
    ax[0].loglog(Ns, Ns / 2, "k:", alpha=0.4, label="N/2")
    ax[0].loglog(Ns, Ns / 4, "k-.", alpha=0.4, label="N/4")
    ax[0].set_xlabel("N"); ax[0].set_ylabel("# orthogonal modes")
    ax[0].set_title("Distinguishable modes in Brillouin zone")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

    for thr, counts in results.items():
        ax[1].loglog(Ns, counts / Ns, "o-", label=f"thr={thr}")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("# modes / N")
    ax[1].set_title("Mode packing fraction")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "04_storage_capacity.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
