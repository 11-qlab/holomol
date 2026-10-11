"""Molecular ensemble: positions, internal states, trapping geometry."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Ensemble:
    """Ensemble of N trapped polar molecules (3-level: |g>, |f>, |e>)."""
    positions: np.ndarray              # (N, 3) in meters
    species: str = "generic"
    temperature: float = 1e-6          # K
    states: Optional[np.ndarray] = None  # 0=|g>, 1=|f>, 2=|e>

    def __post_init__(self):
        N = len(self.positions)
        if self.states is None:
            self.states = np.zeros(N, dtype=int)

    @property
    def N(self) -> int:
        return len(self.positions)

    @property
    def density(self) -> float:
        """Number density (m^-3), assuming a bounding box."""
        if self.N < 2:
            return 0.0
        extent = self.positions.max(axis=0) - self.positions.min(axis=0)
        vol = np.prod(extent + 1e-30)
        return self.N / vol

    def average_spacing(self) -> float:
        """Mean nearest-neighbor spacing (m)."""
        if self.N < 2:
            return 0.0
        from scipy.spatial import cKDTree
        tree = cKDTree(self.positions)
        d, _ = tree.query(self.positions, k=2)
        return float(np.mean(d[:, 1]))

    def copy(self) -> "Ensemble":
        e = Ensemble(self.positions.copy(), self.species, self.temperature)
        e.states = self.states.copy()
        return e


def make_1d_lattice(N: int, spacing: float) -> Ensemble:
    x = np.arange(N) * spacing
    return Ensemble(np.column_stack([x, np.zeros(N), np.zeros(N)]),
                    species="1d_lattice")


def make_2d_lattice(Nx: int, Ny: int, spacing: float) -> Ensemble:
    x, y = np.meshgrid(np.arange(Nx) * spacing, np.arange(Ny) * spacing)
    pos = np.column_stack([x.ravel(), y.ravel(), np.zeros(Nx * Ny)])
    return Ensemble(pos, species="2d_lattice")


def make_3d_lattice(Nx: int, Ny: int, Nz: int, spacing: float) -> Ensemble:
    x, y, z = np.meshgrid(np.arange(Nx) * spacing,
                          np.arange(Ny) * spacing,
                          np.arange(Nz) * spacing)
    pos = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return Ensemble(pos, species="3d_lattice")


def make_random_cloud(N: int, radius: float, seed: int = 0) -> Ensemble:
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    r = radius * rng.random(N) ** (1.0 / 3.0)
    pos = pts * r[:, None]
    return Ensemble(pos, species="random_cloud")
