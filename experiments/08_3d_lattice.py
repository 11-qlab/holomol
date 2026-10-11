"""3D lattice: axial, face, and body-diagonal orthogonality scaling."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import Ensemble


def S_1d(N, a, dq):
    dq = np.asarray(dq, dtype=float)
    half = dq * a / 2.0
    denom = N * np.sin(half)
    numer = np.sin(N * half)
    mask = np.abs(denom) < 1e-12
    out = np.ones_like(numer, dtype=float)
    np.divide(np.abs(numer), np.abs(denom), out=out, where=~mask)
    return out


def S_3d(L, a, dqx, dqy, dqz):
    return S_1d(L, a, dqx) * S_1d(L, a, dqy) * S_1d(L, a, dqz)


def make_disordered_3d(L, a, sigma, seed):
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(np.arange(L)*a, np.arange(L)*a, np.arange(L)*a,
                          indexing="ij")
    pos = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    pos += rng.normal(0, sigma, pos.shape)
    return Ensemble(pos, species="disordered_3d")


def main():
    print("=" * 60)
    print("Experiment 08: 3D Lattice Extension")
    print("=" * 60)

    a = 0.5e-6
    Ls = np.array([2, 4, 8, 16, 32])
    Ns = Ls ** 3
    rng = np.random.default_rng(7)

    print(f"\n{'L':>4} {'N':>8} {'axial (1/q)':>14} {'1/N^(1/3)':>12} "
          f"{'face (2/q)':>14} {'1/N^(2/3)':>12} "
          f"{'body (3/q)':>14} {'1/N':>12}")

    axial, face, body = [], [], []
    for L in Ls:
        N = L ** 3
        ax = np.mean([S_3d(L, a, 2*np.pi*rng.uniform(0.1, 0.9)/a, 0.0, 0.0)
                      for _ in range(40)])
        fc = np.mean([S_3d(L, a, 2*np.pi*rng.uniform(0.1, 0.9)/a,
                           2*np.pi*rng.uniform(0.1, 0.9)/a, 0.0)
                      for _ in range(40)])
        bd = np.mean([S_3d(L, a, 2*np.pi*rng.uniform(0.1, 0.9)/a,
                           2*np.pi*rng.uniform(0.1, 0.9)/a,
                           2*np.pi*rng.uniform(0.1, 0.9)/a)
                      for _ in range(40)])
        axial.append(ax); face.append(fc); body.append(bd)
        print(f"{L:>4d} {N:>8d} {ax:>14.4e} {1/N**(1/3):>12.4e} "
              f"{fc:>14.4e} {1/N**(2/3):>12.4e} "
              f"{bd:>14.4e} {1/N:>12.4e}")

    sl = lambda y: np.polyfit(np.log(Ns), np.log(y), 1)[0]
    print(f"\nAxial slope:  {sl(axial):+.3f}  (expect -1/3 = -0.333)")
    print(f"Face slope:   {sl(face):+.3f}   (expect -2/3 = -0.667)")
    print(f"Body slope:   {sl(body):+.3f}   (expect -1.0)")

    # 3D disorder crossover at L=8
    print("\n3D disorder crossover (L=8, N=512):")
    L = 8
    q0 = np.array([2*np.pi/(L*a), 2*np.pi/(L*a), 2*np.pi/(L*a)])
    dq_test = [np.array([2*np.pi*rng.uniform(0.1, 0.9)/a,
                         2*np.pi*rng.uniform(0.1, 0.9)/a,
                         2*np.pi*rng.uniform(0.1, 0.9)/a])
               for _ in range(20)]
    sigmas = np.array([0, 1e-10, 1e-9, 1e-8, 1e-7])
    rms = []
    for s in sigmas:
        vals = []
        for sd in range(4):
            ens = make_disordered_3d(L, a, s, sd)
            p0 = np.exp(1j * ens.positions @ q0) / np.sqrt(ens.N)
            ov = [abs(np.vdot(p0, np.exp(1j * ens.positions @ (q0 + dq)) / np.sqrt(ens.N)))
                  for dq in dq_test]
            vals.append(np.sqrt(np.mean(np.square(ov))))
        rms.append(float(np.mean(vals)))
    rms = np.array(rms)
    for s, r in zip(sigmas, rms):
        print(f"  sigma/a = {s/a:.4f}   RMS = {r:.4e}   ratio = {r/rms[0]:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].loglog(Ns, axial, "o-", label=f"axial ({sl(axial):+.2f})")
    ax[0].loglog(Ns, face,  "s-", label=f"face ({sl(face):+.2f})")
    ax[0].loglog(Ns, body,  "^-", label=f"body ({sl(body):+.2f})")
    ax[0].loglog(Ns, Ns**(-1/3), "r--", alpha=0.5, label=r"$N^{-1/3}$")
    ax[0].loglog(Ns, Ns**(-2/3), "g--", alpha=0.5, label=r"$N^{-2/3}$")
    ax[0].loglog(Ns, 1/Ns,       "b--", alpha=0.5, label=r"$N^{-1}$")
    ax[0].set_xlabel("N"); ax[0].set_ylabel("mean |S|")
    ax[0].set_title("3D directional scaling")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    ax[1].semilogx(sigmas[1:]/a, rms[1:]/rms[0], "o-")
    ax[1].axhline(1.0, color="k", ls=":", alpha=0.5)
    ax[1].set_xlabel(r"$\sigma/a$"); ax[1].set_ylabel("RMS / perfect")
    ax[1].set_title(f"3D disorder crossover (L={L})")
    ax[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "08_3d_lattice.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
