"""STIRAP vs Raman for collective state transfer under spontaneous emission.

STIRAP (counter-intuitive pulses) follows the dark state, which has
zero |e> population. Under spontaneous emission Gamma, STIRAP is
essentially lossless. Raman (intuitive) transfers through |e> and
suffers from Gamma.

This quantifies the STIRAP advantage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gaussian(t, t0, sigma):
    return np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))


def solve(T_total, Omega, Delta, Gamma, scheme):
    """scheme: 'stirap' (counter-intuitive) or 'raman' (intuitive)."""
    if scheme == "stirap":
        t1, t2 = 0.4 * T_total, 0.6 * T_total   # Omega1 later
    else:  # raman
        t1, t2 = 0.4 * T_total, 0.6 * T_total
    sigma = 0.18 * T_total

    def rhs(t, y):
        rho = (y[:9] + 1j * y[9:]).reshape(3, 3)
        if scheme == "stirap":
            O1 = Omega * gaussian(t, t2, sigma)   # pump last
            O2 = Omega * gaussian(t, t1, sigma)   # Stokes first
        else:
            O1 = Omega * gaussian(t, t1, sigma)   # pump first
            O2 = Omega * gaussian(t, t2, sigma)   # Stokes last

        H = np.array([[0, O1/2, 0],
                      [O1/2, Delta, O2/2],
                      [0, O2/2, 0]], dtype=complex)
        drho = -1j * (H @ rho - rho @ H)

        L1 = np.zeros((3, 3), dtype=complex); L1[0, 1] = np.sqrt(Gamma/2)
        L2 = np.zeros((3, 3), dtype=complex); L2[2, 1] = np.sqrt(Gamma/2)
        for L in (L1, L2):
            drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho
                                                 + rho @ L.conj().T @ L)
        return np.concatenate([drho.real.flatten(), drho.imag.flatten()])

    rho0 = np.zeros((3, 3), dtype=complex); rho0[0, 0] = 1.0
    y0 = np.concatenate([rho0.real.flatten(), rho0.imag.flatten()])
    sol = solve_ivp(rhs, (0, T_total), y0, method="DOP853",
                    rtol=1e-10, atol=1e-12, max_step=T_total/2000)
    yf = sol.y[:, -1]
    rf = (yf[:9] + 1j*yf[9:]).reshape(3, 3)
    tr = np.real(np.trace(rf))
    if tr > 0:
        rf = rf / tr
    return float(np.real(rf[2, 2]))


def main():
    print("=" * 60)
    print("Experiment 16b: STIRAP vs Raman (with spontaneous emission)")
    print("=" * 60)

    Omega = 2 * np.pi * 1e6        # 1 MHz peak Rabi
    Delta = 0.0
    T_vals = np.logspace(-8, np.log10(5e-5), 30)
    Gammas = [2*np.pi*1e3, 2*np.pi*1e4, 2*np.pi*1e5]

    print(f"Omega/2pi = {Omega/(2*np.pi):.1e} Hz\n")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    for Gamma in Gammas:
        F_stirap = np.array([solve(T, Omega, Delta, Gamma, "stirap")
                             for T in T_vals])
        F_raman = np.array([solve(T, Omega, Delta, Gamma, "raman")
                            for T in T_vals])
        i_opt_s = int(np.argmax(F_stirap))
        i_opt_r = int(np.argmax(F_raman))
        print(f"Gamma/2pi = {Gamma/(2*np.pi):>8.0f} Hz: "
              f"STIRAP F_max = {F_stirap[i_opt_s]:.6f} "
              f"(T = {T_vals[i_opt_s]*1e9:.1f} ns) | "
              f"Raman F_max = {F_raman[i_opt_r]:.6f}")

        ax[0].semilogx(T_vals*1e9, F_stirap, "o-",
                       label=f"STIRAP Gamma/2pi={Gamma/(2*np.pi):.0f}")
        ax[0].semilogx(T_vals*1e9, F_raman, "s--",
                       label=f"Raman  Gamma/2pi={Gamma/(2*np.pi):.0f}",
                       alpha=0.6)

    ax[0].set_xlabel("pulse duration (ns)")
    ax[0].set_ylabel("population |f>")
    ax[0].set_title("STIRAP vs Raman")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which="both")

    # Advantage F_stirap - F_raman
    for Gamma in Gammas:
        F_s = np.array([solve(T, Omega, Delta, Gamma, "stirap")
                        for T in T_vals])
        F_r = np.array([solve(T, Omega, Delta, Gamma, "raman")
                        for T in T_vals])
        ax[1].semilogx(T_vals*1e9, F_s - F_r, "o-",
                       label=f"Gamma/2pi={Gamma/(2*np.pi):.0f}")
    ax[1].axhline(0, color="k", ls=":", alpha=0.5)
    ax[1].set_xlabel("pulse duration (ns)")
    ax[1].set_ylabel("F_STIRAP - F_Raman")
    ax[1].set_title("STIRAP advantage")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

    # Best-case for each
    for Gamma in Gammas:
        F_s = np.array([solve(T, Omega, Delta, Gamma, "stirap")
                        for T in T_vals])
        F_r = np.array([solve(T, Omega, Delta, Gamma, "raman")
                        for T in T_vals])
        ax[2].semilogx(T_vals*1e9, 1 - F_s, "-",
                       label=f"STIRAP {Gamma/(2*np.pi):.0f}")
        ax[2].semilogx(T_vals*1e9, 1 - F_r, "--", alpha=0.6,
                       label=f"Raman  {Gamma/(2*np.pi):.0f}")
    ax[2].set_xlabel("pulse duration (ns)")
    ax[2].set_ylabel("infidelity 1 - F")
    ax[2].set_title("Infidelity vs duration")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "16b_stirap.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
