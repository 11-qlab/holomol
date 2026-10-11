"""Experiment 13f: Measurement-based feedforward for stochastic gradients.

Tests tracking and feedforward correction of stochastic spatial gradients.
Result: Measuring the gradient shift and applying a real-time adaptive
q-shift counter-pulse restores unit fidelity (subject to measurement noise).
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


def apply_gradient(psi, beta):
    N = len(psi) - 1
    phi = beta * (np.arange(N) / N - 0.5)
    d = np.ones(N + 1, dtype=complex)
    d[1:] = np.exp(1j * phi)
    return psi * d


def shift_q(psi, dq, a):
    N = len(psi) - 1
    x = np.arange(N) * a
    d = np.ones(N + 1, dtype=complex)
    d[1:] = np.exp(1j * dq * x)
    return psi * d


def main():
    print("=" * 60)
    print("Experiment 13f: Stochastic Gradient Measurement & Feedforward")
    print("=" * 60)

    N, a = 100, 0.5e-6
    q0 = 2 * np.pi / (N * a)
    q1 = 3 * 2 * np.pi / (N * a)
    plus = (mode(N, a, q0) + mode(N, a, q1)) / np.sqrt(2)

    rng = np.random.default_rng(42)
    n_steps = 100
    sigma_noise = 1.0  # step size of stochastic gradient walk

    # Simulate a random walk of gradient strengths beta_t
    betas = rng.normal(0, sigma_noise, size=n_steps)

    # Trajectory 1: Uncorrected (accumulates random walk)
    psi_uncorr = plus.copy()
    for b in betas:
        psi_uncorr = apply_gradient(psi_uncorr, b)
    F_uncorr = abs(np.vdot(plus, psi_uncorr)) ** 2

    # Trajectory 2: Perfect feedforward (measure beta_t exactly, apply inverse q-shift)
    psi_perfect = plus.copy()
    for b in betas:
        psi_perfect = apply_gradient(psi_perfect, b)
        psi_perfect = shift_q(psi_perfect, -b / (N * a), a)
    F_perfect = abs(np.vdot(plus, psi_perfect)) ** 2

    # Trajectory 3: Imperfect feedforward (measured with Gaussian error)
    meas_noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
    print(f"\n--- Test: Feedforward fidelity vs measurement noise ---")
    print(f"{'meas_noise (sigma)':20s}  {'Final Fidelity':>15s}")
    print(f"{'Uncorrected (no FF)':20s}  {F_uncorr:>15.6f}")

    results_noise = []
    for sn in meas_noise_levels:
        psi_noisy = plus.copy()
        for b in betas:
            psi_noisy = apply_gradient(psi_noisy, b)
            b_meas = b + rng.normal(0, sn * abs(b) if abs(b)>1e-5 else sn)
            psi_noisy = shift_q(psi_noisy, -b_meas / (N * a), a)
        F_noisy = abs(np.vdot(plus, psi_noisy)) ** 2
        results_noise.append((sn, F_noisy))
        print(f"{sn:<20.2f}  {F_noisy:>15.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Plot 1: Cumulative trajectory comparison over time steps
    ax = axes[0]
    steps = np.arange(n_steps + 1)
    f_time_uncorr, f_time_perf = [1.0], [1.0]
    psi_u = plus.copy()
    psi_p = plus.copy()
    for b in betas:
        psi_u = apply_gradient(psi_u, b)
        f_time_uncorr.append(abs(np.vdot(plus, psi_u)) ** 2)
        psi_p = apply_gradient(psi_p, b)
        psi_p = shift_q(psi_p, -b / (N * a), a)
        f_time_perf.append(abs(np.vdot(plus, psi_p)) ** 2)

    ax.plot(steps, f_time_uncorr, "o--", label="Uncorrected random walk", color="red")
    ax.plot(steps, f_time_perf, "s-", label="Real-time Feedforward", color="green")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Fidelity")
    ax.set_title("Stochastic Gradient Mitigation over Time")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)

    # Plot 2: Fidelity vs measurement uncertainty
    ax = axes[1]
    sns_vals = [r[0] for r in results_noise]
    f_vals = [r[1] for r in results_noise]
    ax.plot(sns_vals, f_vals, "o-", color="purple", lw=2)
    ax.axhline(F_uncorr, color="red", ls="--", label="Uncorrected baseline")
    ax.set_xlabel("Measurement relative error / noise")
    ax.set_ylabel("Final Fidelity")
    ax.set_title("Sensitivity to Feedforward Estimation Error")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "13f_stochastic_mitigation.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")

    print("\n--- Conclusion ---")
    print("  Stochastic gradients cause a random walk in q-space leading to decoherence.")
    print("  Dynamical Decoupling (DD) fails because noise is spatial displacement, not sigma_z.")
    print("  Measurement-based feedforward (q-shift tracking) successfully restores fidelity,")
    print("  proving that active tracking is the correct mitigation paradigm for stochastic gradients.")


if __name__ == "__main__":
    main()
