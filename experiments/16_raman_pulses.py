"""3-level Raman transfer with finite pulses + spontaneous emission.

Levels: |g>, |e>, |f>.  Pulse Omega1(t) drives g-e, Omega2(t) drives e-f.
Includes detuning Delta and spontaneous emission from |e> with rate Gamma.
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


def raman_rhs(t, rho_flat, params):
    Omega0, Delta, Gamma, t0, sigma, delay = params
    rho = rho_flat.reshape(3, 3, 2).astype(complex)
    rho = rho[:, :, 0] + 1j * rho[:, :, 1]

    O1 = Omega0 * gaussian(t, t0 - delay, sigma)
    O2 = Omega0 * gaussian(t, t0 + delay, sigma)

    # Hamiltonian (rotating frame)
    H = np.array([
        [0,     O1/2,           0],
        [O1/2,  Delta,          O2/2],
        [0,     O2/2,          0],
    ], dtype=complex)

    drho = -1j * (H @ rho - rho @ H)

    # Spontaneous emission from |e> -> |g>, |f> (equal branching)
    L_e = np.zeros((3, 3), dtype=complex)
    L_e[0, 1] = np.sqrt(Gamma / 2)
    L_e[2, 1] = np.sqrt(Gamma / 2)
    drho += L_e @ rho @ L_e.conj().T - 0.5 * (L_e.conj().T @ L_e @ rho
                                              + rho @ L_e.conj().T @ L_e)
    return np.concatenate([drho.real.flatten(), drho.imag.flatten()])


def simulate(Omega0, Gamma, T_total, delay_frac=0.5):
    t0 = T_total / 2.0
    sigma = T_total / 6.0
    delay = delay_frac * T_total / 2.0
    params = (Omega0, 0.0, Gamma, t0, sigma, delay)

    rho0 = np.zeros((3, 3), dtype=complex)
    rho0[0, 0] = 1.0
    y0 = np.concatenate([rho0.real.flatten(), rho0.imag.flatten()])

    t_span = (0.0, T_total)
    sol = solve_ivp(raman_rhs, t_span, y0, args=(params,),
                    method="DOP853", rtol=1e-11, atol=1e-13,
                    dense_output=True)
    rho_f = sol.y[:, -1]
    rho_f = rho_f[:9] + 1j * rho_f[9:]
    rho_f = rho_f.reshape(3, 3)
    return float(np.real(rho_f[2, 2]))   # population in |f>


def main():
    print("=" * 60)
    print("Experiment 16: Realistic Raman Pulse Fidelity")
    print("=" * 60)

    Omega0 = 2 * np.pi * 1e6     # 1 MHz peak Rabi
    Gamma_vals = [0.0, 2*np.pi*1e3, 2*np.pi*1e4]   # 0, 1 kHz, 10 kHz
    T_pulse_vals = np.logspace(-8, np.log10(5e-6), 40)  # 10 ns to 5 us

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    print(f"\nOmega0/2pi = {Omega0/(2*np.pi):.3e} Hz")
    print(f"\n{'T (ns)':>10} " + " ".join(f"F(G={G/(2*np.pi):.0f})"
                                        for G in Gamma_vals))

    curves = {G: [] for G in Gamma_vals}
    for T in T_pulse_vals:
        row = []
        for G in Gamma_vals:
            try:
                F = simulate(Omega0, G, float(T))
            except Exception:
                F = 0.0
            curves[G].append(F)
            row.append(F)
        if T in [T_pulse_vals[0], T_pulse_vals[len(T_pulse_vals)//2],
                 T_pulse_vals[-1]]:
            print(f"{T*1e9:>10.2f} " + " ".join(f"{f:>10.6f}" for f in row))

    for G in Gamma_vals:
        ax[0].semilogx(T_pulse_vals * 1e9, curves[G], "-",
                       label=f"Gamma/2pi = {G/(2*np.pi):.0f} Hz")
    ax[0].set_xlabel("pulse duration (ns)")
    ax[0].set_ylabel("population transfer |g> -> |f>")
    ax[0].set_title("Pulse duration vs fidelity")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
    ax[0].set_ylim(0, 1.05)

    # Optimal pulse duration
    print("\nOptimal pulse durations:")
    for G in Gamma_vals:
        Fs = np.array(curves[G])
        i = int(np.argmax(Fs))
        print(f"  Gamma/2pi = {G/(2*np.pi):>8.0f} Hz: "
              f"T_opt = {T_pulse_vals[i]*1e9:.2f} ns, "
              f"F_max = {Fs[i]:.6f}")

    # Analytic Rabi model for reference
    T_analytic = np.pi / Omega0
    ax[1].semilogx(T_pulse_vals * 1e9, curves[Gamma_vals[-1]], "-",
                   label="full simulation")
    ax[1].axvline(T_analytic * 1e9, color="r", ls="--",
                  label=f"pi-pulse = {T_analytic*1e9:.1f} ns")
    ax[1].set_xlabel("pulse duration (ns)")
    ax[1].set_ylabel("population")
    ax[1].set_title("Optimal vs analytic pi-pulse")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "16_raman_pulses.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
