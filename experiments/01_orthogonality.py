"""Verify <f,q|f,q'> ~ delta_qq' for a molecular ensemble."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import make_1d_lattice, make_random_cloud
from holomol.core.collective import collective_mode


PHI = (1 + np.sqrt(5)) / 2      # 1.618...; PHI-1 = 0.618... irrational


def main():
    print("=" * 60)
    print("Experiment 01: Collective Mode Orthogonality")
    print("=" * 60)

    N, a = 500, 0.5e-6
    ens = make_1d_lattice(N, a)
    q0 = np.array([2 * np.pi / (N * a), 0.0, 0.0])

    # --- Zoom near first zero ---
    dq_zoom = np.linspace(-4 * np.pi / (N * a), 4 * np.pi / (N * a), 800)
    psi0 = collective_mode(ens, q0)
    ov_zoom = np.array([
        abs(np.vdot(psi0, collective_mode(ens, q0 + np.array([dq, 0.0, 0.0]))))
        for dq in dq_zoom
    ])

    # --- Lattice scaling with INCOMMENSURATE dq ---
    Ns = np.array([50, 100, 200, 500, 1000, 2000])
    dq_a = 2 * np.pi * (PHI - 1) / 4.0       # irrational fraction of BZ
    dq_fixed = dq_a / a
    print(f"Fixed dq*a/(2pi) = {dq_a/(2*np.pi):.6f}  (irrational)")
    ov_lattice = np.array([
        abs(np.vdot(collective_mode(make_1d_lattice(Nv, a), q0),
                    collective_mode(make_1d_lattice(Nv, a),
                                    q0 + np.array([dq_fixed, 0.0, 0.0]))))
        for Nv in Ns
    ])

    # --- Random cloud RMS scaling ---
    rng = np.random.default_rng(0)
    ov_cloud = []
    for Nv in Ns:
        vals = []
        for _ in range(20):
            ens_r = make_random_cloud(Nv, radius=Nv * a / 2,
                                      seed=int(rng.integers(1 << 30)))
            p0 = collective_mode(ens_r, q0)
            p1 = collective_mode(ens_r, q0 + np.array([dq_fixed, 0.0, 0.0]))
            vals.append(abs(np.vdot(p0, p1)))
        ov_cloud.append(np.sqrt(np.mean(np.array(vals) ** 2)))
    ov_cloud = np.array(ov_cloud)

    print("\nLattice scaling (incommensurate dq):")
    for Nv, ov in zip(Ns, ov_lattice):
        print(f"  N={Nv:5d}  |ov| = {ov:.3e}   1/N = {1/Nv:.3e}")
    print("\nRandom cloud RMS scaling:")
    for Nv, ov in zip(Ns, ov_cloud):
        print(f"  N={Nv:5d}  RMS = {ov:.3e}   1/sqrt(N) = {1/np.sqrt(Nv):.3e}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(dq_zoom * N * a / (2 * np.pi), ov_zoom, "b-", lw=1.5)
    ax[0].axvline(1, color="r", ls="--", alpha=0.6, label="2pi/(Na)")
    ax[0].set_xlabel(r"$\Delta q \cdot N a / 2\pi$")
    ax[0].set_ylabel(r"$|\langle f,q | f,q{+}\Delta q\rangle|$")
    ax[0].set_title(f"Lattice overlap (N={N})")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].loglog(Ns, ov_lattice, "o-", label="lattice, incomm. Δq")
    ax[1].loglog(Ns, 1.0 / Ns, "r--", label=r"$1/N$")
    ax[1].loglog(Ns, ov_cloud, "s-", color="purple", label="random cloud RMS")
    ax[1].loglog(Ns, 1.0 / np.sqrt(Ns), color="orange", ls="--",
                 label=r"$1/\sqrt{N}$")
    ax[1].set_xlabel("N"); ax[1].set_ylabel("overlap")
    ax[1].set_title("Lattice (1/N) vs cloud (1/sqrt(N))")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "01_orthogonality.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
