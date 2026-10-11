"""2D lattice extension: directional orthogonality, disorder, capacity."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomol.core.ensemble import Ensemble, make_2d_lattice


def S_1d(N, a, dq):
    dq = np.asarray(dq, dtype=float)
    half = dq * a / 2.0
    denom = N * np.sin(half)
    numer = np.sin(N * half)
    mask = np.abs(denom) < 1e-12
    out = np.ones_like(numer, dtype=float)
    np.divide(np.abs(numer), np.abs(denom), out=out, where=~mask)
    return out


def S_2d(L, a, dqx, dqy):
    return S_1d(L, a, dqx) * S_1d(L, a, dqy)


def make_disordered_2d(L, a, sigma, seed):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(L) * a, np.arange(L) * a)
    pos = np.column_stack([x.ravel(), y.ravel(), np.zeros(L * L)])
    pos += rng.normal(0, sigma, pos.shape)
    return Ensemble(pos, species="disordered_2d")


def rms_overlap(ens, q0, dq_list):
    p0 = np.exp(1j * ens.positions @ q0) / np.sqrt(ens.N)
    vals = []
    for dq in dq_list:
        p1 = np.exp(1j * ens.positions @ (q0 + dq)) / np.sqrt(ens.N)
        vals.append(abs(np.vdot(p0, p1)))
    return float(np.sqrt(np.mean(np.square(vals))))


def count_modes_2d(L, a, q_max, threshold, n_side=40):
    qs = np.linspace(-q_max, q_max, n_side)
    QX, QY = np.meshgrid(qs, qs)
    qxf = QX.ravel(); qyf = QY.ravel()
    kx = np.empty(len(qxf)); ky = np.empty(len(qyf)); k = 0
    for i in range(len(qxf)):
        if k == 0:
            kx[0] = qxf[i]; ky[0] = qyf[i]; k = 1; continue
        S = S_2d(L, a, qxf[i] - kx[:k], qyf[i] - ky[:k])
        if S.max() <= threshold:
            kx[k] = qxf[i]; ky[k] = qyf[i]; k += 1
    return k


def main():
    print("=" * 60)
    print("Experiment 07: 2D Lattice Extension")
    print("=" * 60)

    a = 0.5e-6
    Ls = np.array([4, 8, 16, 32, 64])
    Ns = Ls ** 2
    q_max = np.pi / a
    rng = np.random.default_rng(42)

    # --- Directional scaling (analytic) ---
    print("\nDirectional scaling for perfect 2D lattice:")
    print(f"{'L':>4} {'N':>6} {'axial':>12} {'1/sqrt(N)':>12} "
          f"{'diagonal':>12} {'1/N':>12}")
    axial, diag = [], []
    for L in Ls:
        N = L * L
        ax = np.mean([S_2d(L, a, 2*np.pi*rng.uniform(0.1, 0.9)/a, 0.0)
                      for _ in range(40)])
        di = np.mean([S_2d(L, a, 2*np.pi*rng.uniform(0.1, 0.9)/a,
                           2*np.pi*rng.uniform(0.1, 0.9)/a)
                      for _ in range(40)])
        axial.append(ax); diag.append(di)
        print(f"{L:>4d} {N:>6d} {ax:>12.4e} {1/np.sqrt(N):>12.4e} "
              f"{di:>12.4e} {1/N:>12.4e}")

    slopes_ax = np.polyfit(np.log(Ns), np.log(axial), 1)[0]
    slopes_di = np.polyfit(np.log(Ns), np.log(diag), 1)[0]
    print(f"\nAxial slope: {slopes_ax:+.3f}  (expect -0.5)")
    print(f"Diagonal slope: {slopes_di:+.3f}  (expect -1.0)")

    # --- 2D disorder crossover ---
    print("\n2D disorder crossover (L=16):")
    L = 16
    q0 = np.array([2*np.pi/(L*a), 2*np.pi/(L*a), 0.0])
    dq_test = [np.array([2*np.pi*rng.uniform(0.1, 0.9)/a,
                         2*np.pi*rng.uniform(0.1, 0.9)/a, 0.0])
               for _ in range(30)]
    sigmas = np.array([0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6])
    rms_2d = []
    for s in sigmas:
        vals = [rms_overlap(make_disordered_2d(L, a, s, sd), q0, dq_test)
                for sd in range(5)]
        rms_2d.append(float(np.mean(vals)))
    rms_2d = np.array(rms_2d)
    for s, r in zip(sigmas, rms_2d):
        print(f"  sigma/a = {s/a:.4f}  RMS = {r:.4e}  ratio = {r/rms_2d[0]:.2f}")

    # --- 2D capacity ---
    print("\n2D mode capacity:")
    thresholds = [0.10, 0.05]
    cap = {}
    for thr in thresholds:
        counts = []
        for L in Ls:
            c = count_modes_2d(L, a, q_max, thr, n_side=max(20, 3*L))
            counts.append(c)
            print(f"  thr={thr:.2f}  L={L:3d}  N={L*L:5d} -> {c:5d} modes")
        cap[thr] = np.array(counts)

    # --- Plot ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].loglog(Ns, axial, "o-", label=f"2D axial (slope {slopes_ax:.2f})")
    ax[0].loglog(Ns, diag, "s-", label=f"2D diagonal (slope {slopes_di:.2f})")
    ax[0].loglog(Ns, 1/np.sqrt(Ns), "r--", alpha=0.6, label=r"$1/\sqrt{N}$")
    ax[0].loglog(Ns, 1/Ns, "b:", alpha=0.6, label=r"$1/N$")
    ax[0].set_xlabel("N"); ax[0].set_ylabel("mean |S|")
    ax[0].set_title("Perfect 2D lattice: directional scaling")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

    ax[1].semilogx(sigmas[1:]/a, rms_2d[1:]/rms_2d[0], "o-")
    ax[1].axhline(1.0, color="k", ls=":", alpha=0.5)
    ax[1].set_xlabel(r"$\sigma/a$"); ax[1].set_ylabel("RMS / perfect")
    ax[1].set_title(f"2D disorder crossover (L={L})")
    ax[1].grid(alpha=0.3, which="both")

    for thr, counts in cap.items():
        ax[2].loglog(Ns, counts, "o-", label=f"thr={thr}")
    ax[2].loglog(Ns, Ns, "k--", alpha=0.4, label="N")
    ax[2].set_xlabel("N"); ax[2].set_ylabel("# modes")
    ax[2].set_title("2D mode capacity")
    ax[2].legend(); ax[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "07_2d_lattice.png")
    plt.savefig(out, dpi=120)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
