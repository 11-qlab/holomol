"""Does the dark manifold provide error protection?

Compare two encodings under k random single-site sigma_x errors:
  - Bright:  |0_L> = |all g>,  |1_L> = |f, q0>
  - Dark:    |0_L> = |f, q1>,  |1_L> = |f, q2>   (both dark w.r.t. laser at q0)

Also test collective errors (all sites flipped coherently), where the
dark subspace should act as a decoherence-free subspace.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import make_1d_lattice


def state_from_amps(N, amp_g, amp_f):
    v = np.zeros(N + 1, dtype=complex)
    v[0] = amp_g
    v[1:] = amp_f
    return v


def bright_ref(N, a, q0):
    x = np.arange(N) * a
    v = np.zeros(N + 1, dtype=complex)
    v[1:] = np.exp(1j * q0 * x) / np.sqrt(N)
    return v


def dark_ref(N, a, q1, q2):
    """(|f,q1> + |f,q2>)/sqrt(2)"""
    x = np.arange(N) * a
    v = np.zeros(N + 1, dtype=complex)
    v[1:] = (np.exp(1j * q1 * x) + np.exp(1j * q2 * x)) / np.sqrt(2 * N)
    return v


def apply_local_flips(N, a, q1, q2, k, rng, collective=False):
    """Apply k random (or all-collective) sigma_x flips."""
    x = np.arange(N) * a
    amp_f = (np.exp(1j * q1 * x) + np.exp(1j * q2 * x)) / np.sqrt(2 * N)
    if collective:
        # Flip all sites: moves every |f_j> amplitude to |all g>
        amp_g = amp_f.sum()
        return state_from_amps(N, amp_g, np.zeros(N, dtype=complex))
    flip_sites = rng.choice(N, size=k, replace=False)
    amp_g = 0.0 + 0j
    for j in flip_sites:
        amp_g += amp_f[j]
        amp_f[j] = 0.0
    return state_from_amps(N, amp_g, amp_f)


def apply_local_flips_bright(N, a, q0, k, rng):
    """Same error model on bright encoding."""
    x = np.arange(N) * a
    amp_f = np.exp(1j * q0 * x) / np.sqrt(N)
    flip_sites = rng.choice(N, size=k, replace=False)
    amp_g = 0.0 + 0j
    for j in flip_sites:
        amp_g += amp_f[j]
        amp_f[j] = 0.0
    return state_from_amps(N, amp_g, amp_f)


def main():
    print("=" * 60)
    print("Experiment 13: Dark Manifold Error Correction")
    print("=" * 60)

    a = 0.5e-6
    Ns = [16, 64, 256]
    rng = np.random.default_rng(11)
    reps = 30

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Local errors: bright vs dark ---
    print(f"\n{'N':>5} {'k/N':>7} {'F_bright':>12} {'F_dark':>12} "
          f"{'(1-k/N)^2':>12}")

    for N in Ns:
        q0 = 2 * np.pi / (N * a)
        q1 = 2 * np.pi * 2 / (N * a)
        q2 = 2 * np.pi * 3 / (N * a)
        bright = bright_ref(N, a, q0)
        dark = dark_ref(N, a, q1, q2)

        ks = np.arange(0, max(2, N // 3) + 1, max(1, N // 30))
        Fb, Fd = [], []
        for k in ks:
            vb = [abs(np.vdot(bright,
                              apply_local_flips_bright(N, a, q0, int(k), rng))) ** 2
                  for _ in range(reps)]
            vd = [abs(np.vdot(dark,
                              apply_local_flips(N, a, q1, q2, int(k), rng))) ** 2
                  for _ in range(reps)]
            Fb.append(float(np.mean(vb)))
            Fd.append(float(np.mean(vd)))
            if N == 16 or k == 0 or k == ks[-1]:
                print(f"{N:>5d} {k/N:>7.4f} {Fb[-1]:>12.6f} {Fd[-1]:>12.6f} "
                      f"{(1-k/N)**2:>12.6f}")
        ax[0].plot(ks / N, Fb, "o-", label=f"bright N={N}", alpha=0.7)
        ax[0].plot(ks / N, Fd, "s--", label=f"dark N={N}", alpha=0.7)

    ax[0].plot(np.linspace(0, 0.33, 50),
               (1 - np.linspace(0, 0.33, 50)) ** 2, "k:", alpha=0.5,
               label=r"$(1-k/N)^2$")
    ax[0].set_xlabel("k / N")
    ax[0].set_ylabel("fidelity")
    ax[0].set_title("Local errors: bright vs dark encoding")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    # --- Collective errors: dark subspace should be protected ---
    print("\nCollective error (all sites flipped coherently):")
    print(f"{'N':>5} {'F_bright_coll':>16} {'F_dark_coll':>16}")
    Ns2 = [4, 8, 16, 32, 64, 128]
    Fb_coll, Fd_coll = [], []
    for N in Ns2:
        q0 = 2 * np.pi / (N * a)
        q1 = 2 * np.pi * 2 / (N * a)
        q2 = 2 * np.pi * 3 / (N * a)
        bright = bright_ref(N, a, q0)
        dark = dark_ref(N, a, q1, q2)
        # Collective DE-PHASING (sigma_z on all sites): adds phase phi_j
        x = np.arange(N) * a
        phi = 0.3
        amp_f_b = np.exp(1j * q0 * x) / np.sqrt(N)
        amp_f_d = (np.exp(1j * q1 * x) + np.exp(1j * q2 * x)) / np.sqrt(2 * N)
        phases = np.exp(1j * phi * np.ones(N))
        vb = state_from_amps(N, 0.0 + 0j, phases * amp_f_b)
        vd = state_from_amps(N, 0.0 + 0j, phases * amp_f_d)
        Fb_coll.append(abs(np.vdot(bright, vb)) ** 2)
        Fd_coll.append(abs(np.vdot(dark, vd)) ** 2)
        print(f"{N:>5d} {Fb_coll[-1]:>16.8f} {Fd_coll[-1]:>16.8f}")

    ax[1].semilogx(Ns2, Fb_coll, "o-", label="bright")
    ax[1].semilogx(Ns2, Fd_coll, "s-", label="dark")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("fidelity under collective error")
    ax[1].set_title("Collective error: DFS protection")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "13_dark_manifold_qec.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
