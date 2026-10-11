"""Simultaneous storage of k modes: cross-talk and BER."""
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


def crosstalk_matrix(N, a, q_list):
    k = len(q_list)
    M = np.eye(k)
    for i in range(k):
        for j in range(k):
            if i != j:
                M[i, j] = S_lattice(N, a, q_list[i] - q_list[j])
    return M


def write_read_cycle(N, a, k, reps=50, seed=0):
    """Write random k-bit word, read out via cross-talk, count errors."""
    rng = np.random.default_rng(seed)
    # Distribute k modes uniformly in BZ
    q_max = np.pi / a
    q_list = np.linspace(-q_max * 0.9, q_max * 0.9, k)
    M = crosstalk_matrix(N, a, q_list)

    total_errors = 0
    total_bits = 0
    for _ in range(reps):
        bits = rng.integers(0, 2, k)
        signal = M @ bits
        # Threshold at 0.5
        decoded = (signal > 0.5).astype(int)
        total_errors += int(np.sum(bits != decoded))
        total_bits += k
    return total_errors / total_bits


def main():
    print("=" * 60)
    print("Experiment 17: Multi-Mode Simultaneous Storage")
    print("=" * 60)

    a = 0.5e-6
    Ns = [100, 1000, 10000]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    print(f"\n{'N':>6} {'k':>6} {'k/N':>10} {'BER':>12}")
    for N in Ns:
        ks = np.unique(np.round(np.logspace(0, np.log10(N), 20)).astype(int))
        bers = []
        for k in ks:
            if k < 2:
                bers.append(0.0)
                continue
            ber = write_read_cycle(N, a, int(k), reps=30)
            bers.append(ber)
            if k in [ks[0], ks[len(ks)//2], ks[-1]]:
                print(f"{N:>6d} {k:>6d} {k/N:>10.4f} {ber:>12.6f}")
        ax[0].loglog(ks, np.array(bers) + 1e-8, "o-", label=f"N={N}")

    ax[0].axhline(0.5, color="r", ls="--", alpha=0.5, label="random")
    ax[0].axhline(0.01, color="g", ls=":", alpha=0.5, label="1% BER")
    ax[0].set_xlabel("number of modes k")
    ax[0].set_ylabel("bit-error rate")
    ax[0].set_title("BER vs mode count")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

    # Cross-talk matrix for a specific N, k
    N = 1000
    k = 50
    q_max = np.pi / a
    q_list = np.linspace(-q_max * 0.9, q_max * 0.9, k)
    M = crosstalk_matrix(N, a, q_list)
    im = ax[1].imshow(np.abs(M), cmap="hot", vmin=0, vmax=0.5)
    ax[1].set_xlabel("mode index j")
    ax[1].set_ylabel("mode index i")
    ax[1].set_title(f"Cross-talk matrix (N={N}, k={k})")
    plt.colorbar(im, ax=ax[1])

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "17_multi_mode.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
