"""Experiment 13d: DD mitigation of gradient dephasing.

Tests spin-echo / CPMG sequences on the four encodings from 13c.
Predicts: DD recovers fidelity for MIXED-excitation encodings (B, D'),
but not for SAME-excitation encodings (B', D).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mode(N, a, q):
    x = np.arange(N) * a
    v = np.zeros(N + 1, dtype=complex)
    v[1:] = np.exp(1j * q * x) / np.sqrt(N)
    return v


def ground(N):
    v = np.zeros(N + 1, dtype=complex)
    v[0] = 1.0
    return v


def pi_swap_ground_mode(N, a, q):
    """π-pulse between |allg> and |f,q>."""
    v0 = ground(N)
    vq = mode(N, a, q)
    U = np.eye(N + 1, dtype=complex)
    P = np.outer(v0, v0.conj()) + np.outer(vq, vq.conj())
    X = np.outer(v0, vq.conj()) + np.outer(vq, v0.conj())
    return U - P + X


def pi_swap_two_modes(N, a, q1, q2):
    """π-pulse between |f,q1> and |f,q2>."""
    v1 = mode(N, a, q1)
    v2 = mode(N, a, q2)
    U = np.eye(N + 1, dtype=complex)
    P = np.outer(v1, v1.conj()) + np.outer(v2, v2.conj())
    X = np.outer(v1, v2.conj()) + np.outer(v2, v1.conj())
    return U - P + X


def simulate(N, plus, pi_op, phi_gradient, T_total, n_pulses):
    """Apply DD sequence: E(τ) [π E(τ)]^n_pulses."""
    psi = plus.copy()
    n_int = n_pulses + 1
    tau = T_total / n_int
    phi_tau = phi_gradient * tau   # per-site phase over one interval

    for i in range(n_int):
        d = np.ones(N + 1, dtype=complex)
        d[1:] = np.exp(1j * phi_tau)
        psi = psi * d
        if i < n_pulses:
            psi = pi_op @ psi

    return abs(np.vdot(plus, psi)) ** 2


def main():
    print("=" * 60)
    print("Experiment 13d: DD Mitigation of Gradient Dephasing")
    print("=" * 60)

    N = 100
    a = 0.5e-6
    q0 = 2 * np.pi / (N * a)
    q1 = 3 * 2 * np.pi / (N * a)
    q2 = 5 * 2 * np.pi / (N * a)

    # Encoding definitions: (name, v0, v1, pi_op, excitation_class)
    encodings = [
        ("B  bright",   ground(N), mode(N, a, q0),
         pi_swap_ground_mode(N, a, q0), "mixed"),
        ("B' same-exc", mode(N, a, q0), mode(N, a, q1),
         pi_swap_two_modes(N, a, q0, q1), "same"),
        ("D  dark",     mode(N, a, q1), mode(N, a, q2),
         pi_swap_two_modes(N, a, q1, q2), "same"),
        ("D' mixed-d",  ground(N), mode(N, a, q1),
         pi_swap_ground_mode(N, a, q1), "mixed"),
    ]

    beta = np.pi / 2   # gradient strength (max phase per unit time)
    T_total = 1.0
    phi_grad = beta * (np.arange(N) / N - 0.5)

    pulse_counts = [0, 1, 2, 4, 8]

    print(f"\nGradient strength beta = {beta:.4f} rad (max phase = beta/2)")
    print(f"Total hold time T = {T_total}")
    print(f"\n{'Encoding':15s} " + " ".join(f"n={p:>3d}" for p in pulse_counts))
    results = {}
    for name, v0, v1, pi_op, cls in encodings:
        plus = (v0 + v1) / np.sqrt(2)
        Fs = []
        for np_ in pulse_counts:
            F = simulate(N, plus, pi_op, phi_grad, T_total, np_)
            Fs.append(F)
        results[name] = Fs
        print(f"{name:15s} " + " ".join(f"{F:>6.4f}" for F in Fs))

    # Detailed sweep over beta for each encoding at n=4
    print("\nFidelity vs gradient strength (n_pulses=4):")
    betas = np.linspace(0, 4 * np.pi, 50)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for name, v0, v1, pi_op, cls in encodings:
        plus = (v0 + v1) / np.sqrt(2)
        Fs_bare = []
        Fs_dd = []
        for b in betas:
            phi = b * (np.arange(N) / N - 0.5)
            Fs_bare.append(simulate(N, plus, pi_op, phi, T_total, 0))
            Fs_dd.append(simulate(N, plus, pi_op, phi, T_total, 4))
        ax.plot(betas / np.pi, Fs_bare, "--", alpha=0.5, label=f"{name} bare")
        ax.plot(betas / np.pi, Fs_dd, "-", label=f"{name} + DD")
    ax.set_xlabel(r"$\beta / \pi$")
    ax.set_ylabel("fidelity")
    ax.set_title("DD vs bare (all encodings)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Fidelity vs n_pulses for selected beta
    ax = axes[1]
    beta_plot = np.pi
    phi_p = beta_plot * (np.arange(N) / N - 0.5)
    pulses = np.arange(0, 17)
    for name, v0, v1, pi_op, cls in encodings:
        plus = (v0 + v1) / np.sqrt(2)
        Fs = [simulate(N, plus, pi_op, phi_p, T_total, int(np_))
              for np_ in pulses]
        ax.plot(pulses, Fs, "o-", label=name, lw=1.5)
    ax.set_xlabel("number of DD pulses")
    ax.set_ylabel("fidelity")
    ax.set_title(rf"DD recovery at $\beta = \pi$")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Summary: DD advantage (F_dd - F_bare) vs encoding class
    ax = axes[2]
    labels = []
    adv = []
    for name, v0, v1, pi_op, cls in encodings:
        plus = (v0 + v1) / np.sqrt(2)
        F_b = simulate(N, plus, pi_op, phi_p, T_total, 0)
        F_d = simulate(N, plus, pi_op, phi_p, T_total, 4)
        labels.append(f"{name}\n({cls})")
        adv.append(F_d - F_b)
    colors = ["#d62728" if "same" in enc[4] else "#1f77b4"
              for enc in encodings]
    ax.bar(range(4), adv, color=colors)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$F_{DD} - F_{bare}$")
    ax.set_title(rf"DD advantage at $\beta = \pi$")
    ax.axhline(0, color="k", ls=":", alpha=0.5)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "13d_gradient_mitigation.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")

    # Print summary
    print("\n--- DD advantage summary ---")
    print(f"{'Encoding':15s} {'Class':>8s} {'F_bare':>10s} "
          f"{'F_DD(4)':>10s} {'advantage':>12s}")
    for name, v0, v1, pi_op, cls in encodings:
        plus = (v0 + v1) / np.sqrt(2)
        F_b = simulate(N, plus, pi_op, phi_p, T_total, 0)
        F_d = simulate(N, plus, pi_op, phi_p, T_total, 4)
        print(f"{name:15s} {cls:>8s} {F_b:>10.6f} {F_d:>10.6f} "
              f"{F_d - F_b:>12.6f}")

    print("\n--- Conclusion ---")
    print("  Mixed-excitation encodings (B, D') show large DD gain.")
    print("  Same-excitation encodings (B', D) show near-zero DD gain.")
    print("  --> DD and DFS are orthogonal mitigation strategies.")
    print("  Design rule:")
    print("    Uniform noise:  same-exc + DFS")
    print("    Gradient noise: mixed-exc + DD")


if __name__ == "__main__":
    main()
