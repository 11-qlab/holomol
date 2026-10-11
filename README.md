[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23304955.svg)](https://doi.org/10.5281/zenodo.23304955)

# holomol

Numerical simulator for the holographic molecular quantum register.

Based on the proposal by Tordrup, Negretti, and Mølmer,
*Phys. Rev. Lett.* **101**, 040501 (2008).

## What it does

Stores classical bits in **collective phase modes** of a trapped polar-molecule
ensemble:

    |f, q> = (1/sqrt(N)) sum_j exp(i q . x_j) |g_1 ... f_j ... g_N>

where `q = k1 - k2` is the wavevector difference of two Raman control fields.

## Results

- Lattice orthogonality scales as **1/N**; disordered cloud as **1/sqrt(N)**
- Disorder threshold: **sigma/a ≈ 0.01** (5 nm site control for 500 nm lattice)
- Register capacity: **N modes** (one qubit per lattice site)
- Superradiant coupling: **sqrt(N)** bright-state enhancement
- Dark manifold: **N−1** zero-energy states
- Species retention: **SrF best** (35.5 s T2 at 1e18 m^-3)
- 2D/3D directional scaling: diagonal retains full **1/N**
- Error dilution: **F = (1−k/N)^2**
- Two-qubit ZZ gate: **fidelity 1.000000**

## Install

    pip install -e .

## Run

    python experiments/01b_scaling_clean.py
    python experiments/01c_disorder_crossover.py
    python experiments/04_storage_capacity.py
    python experiments/05_exact_vs_collective.py
    python experiments/06_species_retention.py
    python experiments/07_2d_lattice.py
    python experiments/08_3d_lattice.py
    python experiments/09_write_read_cycle.py
    python experiments/10_collective_robustness.py
    python experiments/11_two_qubit_gate.py

Figures land in `results/`.

## Paper

Source in `paper/holomol.tex`. Build:

    cd paper && pdflatex holomol.tex && pdflatex holomol.tex

## Layout

    holomol/
    ├── core/          ensemble, collective modes, Raman, cavity
    ├── storage/       encode / decode / retention
    ├── control/       pulse envelopes, openqhal bridge
    ├── sim/           exact diagonalization, mean-field
    ├── data/          species constants
    ├── experiments/   01-11 (all reproducible)
    ├── paper/         LaTeX source
    └── results/       generated figures
