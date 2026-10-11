"""Experiment 13c: Deep DFS analysis — light-shift dephasing.

Tests four encodings under four dephasing models. The DFS mechanism
is: codewords with the SAME excitation number are protected against
uniform dephasing. Dark-ness (q != q0) is orthogonal — it provides
write isolation, not DFS protection.
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


def apply_dephasing(psi, phi):
    """E = diag(1, e^{i phi_1}, ..., e^{i phi_N}).

    This models a site-dependent light shift on the |f_j> amplitudes.
    |allg> is unaffected (no f-excitation).
    """
    d = np.ones(len(psi), dtype=complex)
    d[1:] = np.exp(1j * phi)
    return psi * d


def logical_fidelity(v0, v1, phi):
    """|<+_L|E|+_L>|^2 for the +_L state (v0+v1)/sqrt(2)."""
    plus = (v0 + v1) / np.sqrt(2)
    err = apply_dephasing(plus, phi)
    return abs(np.vdot(plus, err)) ** 2


def main():
    print("=" * 60)
    print("Experiment 13c: DFS Deep Dive")
    print("=" * 60)

    N = 100
    a = 0.5e-6
    q0 = 2 * np.pi / (N * a)          # write mode
    q1 = 3 * 2 * np.pi / (N * a)      # dark mode 1
    q2 = 5 * 2 * np.pi / (N * a)      # dark mode 2

    encodings = {
        "B  bright      {|g>, |f,q0>}":   (ground(N), mode(N, a, q0)),
        "B' single-exc  {|f,q0>,|f,q1>}": (mode(N, a, q0), mode(N, a, q1)),
        "D  dark        {|f,q1>,|f,q2>}": (mode(N, a, q1), mode(N, a, q2)),
        "D' mixed-dark  {|g>, |f,q1>}":   (ground(N), mode(N, a, q1)),
    }

    # ---- TEST 1: uniform dephasing ----
    print("\n--- Test 1: uniform dephasing phi_j = phi ---")
    print(f"{'encoding':40s}  F(phi=pi/2)   F(phi=pi)")
    for name, (v0, v1) in encodings.items():
        F1 = logical_fidelity(v0, v1, np.full(N, np.pi/2))
        F2 = logical_fidelity(v0, v1, np.full(N, np.pi))
        print(f"{name:40s}  {F1:>10.6f}   {F2:>10.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    phis = np.linspace(0, np.pi, 60)
    ax = axes[0]
    for name, (v0, v1) in encodings.items():
        Fs = [logical_fidelity(v0, v1, np.full(N, phi)) for phi in phis]
        ax.plot(phis / np.pi, Fs, "-", label=name, lw=1.5)
    ax.set_xlabel(r"$\phi / \pi$")
    ax.set_ylabel("fidelity")
    ax.set_title("Test 1: uniform dephasing")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # ---- TEST 2: linear gradient ----
    print("\n--- Test 2: linear gradient phi_j = beta*(j/N - 1/2) ---")
    print(f"{'encoding':40s}  F(beta=pi/4)  F(beta=pi)")
    for name, (v0, v1) in encodings.items():
        for beta_lab, beta in [("pi/4", np.pi/4), ("pi", np.pi)]:
            phi = beta * (np.arange(N) / N - 0.5)
            F = logical_fidelity(v0, v1, phi)
        F1 = logical_fidelity(v0, v1, np.pi/4 * (np.arange(N)/N - 0.5))
        F2 = logical_fidelity(v0, v1, np.pi * (np.arange(N)/N - 0.5))
        print(f"{name:40s}  {F1:>11.6f}   {F2:>8.6f}")

    betas = np.linspace(0, np.pi, 60)
    ax = axes[1]
    for name, (v0, v1) in encodings.items():
        Fs = []
        for beta in betas:
            phi = beta * (np.arange(N) / N - 0.5)
            Fs.append(logical_fidelity(v0, v1, phi))
        ax.plot(betas / np.pi, Fs, "-", label=name, lw=1.5)
    ax.set_xlabel(r"$\beta / \pi$")
    ax.set_ylabel("fidelity")
    ax.set_title("Test 2: linear gradient")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # ---- TEST 3: sinusoidal dephasing (resonant with some mode) ----
    print("\n--- Test 3: sinusoidal phi_j = gamma*sin(2pi*k*j/N) ---")
    print(f"{'encoding':40s}  k=2 (col.)   k=4 (col.)")
    for name, (v0, v1) in encodings.items():
        Fs = []
        for k in [2, 4]:
            j = np.arange(N)
            phi = np.pi/4 * np.sin(2*np.pi*k*j/N)
            Fs.append(logical_fidelity(v0, v1, phi))
        print(f"{name:40s}  {Fs[0]:>10.6f}   {Fs[1]:>10.6f}")

    kvals = np.arange(1, 30)
    ax = axes[2]
    for name, (v0, v1) in encodings.items():
        Fs = []
        for k in kvals:
            j = np.arange(N)
            phi = np.pi/4 * np.sin(2*np.pi*k*j/N)
            Fs.append(logical_fidelity(v0, v1, phi))
        ax.plot(kvals, Fs, "o-", label=name, alpha=0.7)
    ax.set_xlabel("spatial frequency k")
    ax.set_ylabel("fidelity")
    ax.set_title(r"Test 3: sinusoidal dephasing at $\gamma=\pi/4$")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "13c_dfs_deep.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")

    print("\n--- Conclusion ---")
    print("  D  (dark, single-exc) and B' (non-dark, single-exc) behave")
    print("     identically: DFS protection comes from same-excitation")
    print("     encoding, NOT from the dark-manifold structure.")
    print("  B  (bright) and D' (mixed-dark) both fail under uniform")
    print("     dephasing because their codewords have different n.")
    print("  The dark-manifold property provides write isolation (Exp 13b),")
    print("  orthogonal to DFS protection.")


if __name__ == "__main__":
    main()
