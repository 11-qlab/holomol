"""Clean 1/N vs 1/sqrt(N) scaling via RMS over many Delta-q."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import make_1d_lattice, make_random_cloud
from holomol.core.collective import collective_mode


PHI = (1 + np.sqrt(5)) / 2


def rms_overlap(ens, q0, dq_list):
    psi0 = collective_mode(ens, q0)
    vals = [abs(np.vdot(psi0,
                        collective_mode(ens, q0 + np.array([dq, 0.0, 0.0]))))
            for dq in dq_list]
    return float(np.sqrt(np.mean(np.square(vals))))


def main():
    print("=" * 60)
    print("Experiment 01b: Clean Scaling (RMS over many Delta-q)")
    print("=" * 60)

    a = 0.5e-6
    Ns = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    n_dq = 200

    rng = np.random.default_rng(42)
    # Many irrational-fraction dq values uniformly covering the BZ
    fracs = rng.uniform(0.1, 0.9, n_dq)
    dq_list = 2 * np.pi * fracs / a

    ov_lat, ov_cld = [], []
    for N in Ns:
        ens_l = make_1d_lattice(N, a)
        q0 = np.array([2 * np.pi / (N * a), 0.0, 0.0])
        ov_lat.append(rms_overlap(ens_l, q0, dq_list))

        vals = []
        for _ in range(8):
            ens_r = make_random_cloud(N, radius=N * a / 2,
                                      seed=int(rng.integers(1 << 30)))
            vals.append(rms_overlap(ens_r, q0, dq_list[:40]))
        ov_cld.append(np.sqrt(np.mean(np.square(vals))))
    ov_lat = np.array(ov_lat); ov_cld = np.array(ov_cld)

    print(f"\n{'N':>6} {'lattice RMS':>14} {'1/N':>12} {'cloud RMS':>12} {'1/sqrt(N)':>12}")
    for N, ol, oc in zip(Ns, ov_lat, ov_cld):
        print(f"{N:>6d} {ol:>14.4e} {1.0/N:>12.4e} {oc:>12.4e} {1.0/np.sqrt(N):>12.4e}")

    # Fit slopes in log-log
    sl_lat = np.polyfit(np.log(Ns), np.log(ov_lat), 1)[0]
    sl_cld = np.polyfit(np.log(Ns), np.log(ov_cld), 1)[0]
    print(f"\nLattice log-log slope: {sl_lat:+.3f}   (expect -1)")
    print(f"Cloud   log-log slope: {sl_cld:+.3f}   (expect -0.5)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(Ns, ov_lat, "o-", label=f"lattice (slope {sl_lat:.2f})")
    ax.loglog(Ns, 1.0 / Ns, "r--", alpha=0.6, label=r"$1/N$")
    ax.loglog(Ns, ov_cld, "s-", color="purple",
              label=f"random cloud (slope {sl_cld:.2f})")
    ax.loglog(Ns, 1.0 / np.sqrt(Ns), color="orange", ls="--",
              alpha=0.6, label=r"$1/\sqrt{N}$")
    ax.set_xlabel("N"); ax.set_ylabel("RMS overlap")
    ax.set_title("Lattice (1/N) vs disordered cloud (1/sqrt(N))")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "01b_scaling_clean.png")
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
