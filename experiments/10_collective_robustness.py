"""Robustness of collective modes to site-local errors.

Encode a qubit in {|all g>, |f, q0>}.  Apply k random site-local
sigma_x flips.  Measure fidelity F(k, N) = |<psi_0 | psi_k>|^2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import make_1d_lattice


def collective_state(N, q0, a, k_flips, rng):
    """Start from |f,q0> and apply k random single-site sigma_x flips."""
    x = np.arange(N) * a
    phases = np.exp(1j * q0 * x) / np.sqrt(N)     # amplitude on |f_j>
    g_amp = 0.0 + 0j                              # amplitude on |all g>
    # Each flip on site j swaps |f_j> <-> |g_j> for that site's amplitude
    flip_sites = rng.choice(N, size=k_flips, replace=False)
    flipped_phases = phases.copy()
    flip_weight = 0.0 + 0j
    for j in flip_sites:
        flip_weight += phases[j]
        flipped_phases[j] = 0.0
    # Final state: (g_amp + flip_weight)|all g> + sum remaining phases |f_j>
    g_final = g_amp + flip_weight
    f_final = flipped_phases
    return np.concatenate([[g_final], f_final])


def reference_state(N, q0, a):
    x = np.arange(N) * a
    v = np.zeros(N + 1, dtype=complex)
    v[1:] = np.exp(1j * q0 * x) / np.sqrt(N)
    return v


def main():
    print("=" * 60)
    print("Experiment 10: Collective Robustness to Local Errors")
    print("=" * 60)

    a = 0.5e-6
    Ns = [16, 64, 256, 1024]
    rng = np.random.default_rng(1)

    print(f"\n{'N':>5} {'k/N':>8} {'F(k,N)':>12} {'1-2k/N':>12} "
          f"{'1-2k/N+...':>12}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    for N in Ns:
        q0 = 2 * np.pi / (N * a)
        ref = reference_state(N, q0, a)
        ks = np.arange(0, max(2, N // 4) + 1, max(1, N // 40))
        Fs = []
        for k in ks:
            reps = 20 if N <= 256 else 5
            vals = []
            for _ in range(reps):
                psi = collective_state(N, q0, a, int(k), rng)
                vals.append(abs(np.vdot(ref, psi)) ** 2)
            Fs.append(float(np.mean(vals)))
        Fs = np.array(Fs)
        for k, F in zip(ks, Fs):
            print(f"{N:>5d} {k/N:>8.4f} {F:>12.6f} {1-2*k/N:>12.6f} "
                  f"{(1-k/N)**2:>12.6f}")
        ax[0].plot(ks / N, Fs, "-", label=f"N={N}", alpha=0.8)
        ax[1].plot(ks / N, Fs, "-", label=f"N={N}", alpha=0.8)
    ax[1].plot(np.linspace(0, 0.25, 50),
               1.0 - 2.0 * np.linspace(0, 0.25, 50),
               "k--", alpha=0.6, label=r"$1-2k/N$")

    ax[0].set_xlabel("k/N (fraction of flipped sites)")
    ax[0].set_ylabel("fidelity")
    ax[0].set_title("Fidelity vs error fraction")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].set_xlabel("k/N"); ax[1].set_ylabel("fidelity")
    ax[1].set_title("Collapse: universality in k/N")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0, 0.25)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "10_collective_robustness.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
