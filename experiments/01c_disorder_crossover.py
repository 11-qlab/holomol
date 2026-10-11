"""Where does lattice scaling break under positional disorder?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import Ensemble


def make_disordered_lattice(N, a, sigma, seed):
    rng = np.random.default_rng(seed)
    x = np.arange(N) * a + rng.normal(0, sigma, N)
    y = rng.normal(0, sigma, N)
    z = rng.normal(0, sigma, N)
    return Ensemble(np.column_stack([x, y, z]), species="disordered")


def rms_overlap(ens, q0, dq_list):
    phases0 = np.exp(1j * ens.positions @ q0) / np.sqrt(ens.N)
    vals = []
    for dq in dq_list:
        phases1 = np.exp(1j * ens.positions @ (q0 + np.array([dq, 0.0, 0.0]))) \
                  / np.sqrt(ens.N)
        vals.append(abs(np.vdot(phases0, phases1)))
    return float(np.sqrt(np.mean(np.square(vals))))


def main():
    N = 1000
    a = 0.5e-6
    rng = np.random.default_rng(0)
    dq_list = 2 * np.pi * rng.uniform(0.1, 0.9, 100) / a
    q0 = np.array([2 * np.pi / (N * a), 0.0, 0.0])

    sigmas = np.array([0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6])  # meters
    rms = []
    for s in sigmas:
        vals = [rms_overlap(make_disordered_lattice(N, a, s, sd), q0, dq_list)
                for sd in range(6)]
        rms.append(float(np.mean(vals)))
    rms = np.array(rms)

    print("=" * 60)
    print("Experiment 01c: Positional Disorder Crossover")
    print("=" * 60)
    print(f"N={N}, a={a*1e9:.1f} nm, perfect-lattice RMS = {rms[0]:.4e}")
    print(f"{'sigma (m)':>12} {'sigma/a':>10} {'RMS':>14} {'ratio to perfect':>18}")
    for s, r in zip(sigmas, rms):
        print(f"{s:>12.1e} {s/a:>10.3f} {r:>14.4e} {r/rms[0]:>18.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = sigmas > 0
    ax.loglog(sigmas[mask] / a, rms[mask] / rms[0], "o-")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel(r"disorder $\sigma / a$")
    ax.set_ylabel("RMS / perfect-lattice RMS")
    ax.set_title(f"Disorder-induced mode degradation (N={N})")
    ax.grid(alpha=0.3, which="both")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "01c_disorder_crossover.png")
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
