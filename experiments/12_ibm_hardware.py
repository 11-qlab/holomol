"""Run holomol validation on IBM Quantum hardware using SamplerV2.

Requires: export QISKIT_IBM_TOKEN=<your IBM Quantum API key>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from holomol.core.ensemble import make_1d_lattice
from holomol.sim.exact import effective_raman_hamiltonian, diagonalize


def main():
    # 1. Exact diagonalization reference
    N, a, Omega_eff = 4, 0.5e-6, 1.0
    ens = make_1d_lattice(N, a)
    q0 = np.array([2 * np.pi / (N * a), 0.0, 0.0])
    H = effective_raman_hamiltonian(ens, 0.0, Omega_eff, q0)
    E, V = diagonalize(H)
    print(f"N = {N}")
    print(f"Exact ground-state energy: {E[0]:.6f}")
    print(f"Exact bright doublet: {E[0]:.6f}, {E[-1]:.6f}")
    print(f"Predicted sqrt(N)*|Omega|: {-np.sqrt(N)*abs(Omega_eff):.6f}")

    # 2. GHZ circuit encoding the collective bright mode
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure_all()
    print(f"\nCircuit depth (logical): {qc.depth()}")
    print(f"Circuit size: {qc.size()}")

    # 3. Get backend and transpile
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend("ibm_kingston")
    print(f"\nSubmitting to backend: {backend.name}")

    isa_qc = transpile(qc, backend, optimization_level=3)
    print(f"ISA circuit depth: {isa_qc.depth()}")

    # 4. Run via SamplerV2 (replaces backend.run)
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 4096

    job = sampler.run([isa_qc])
    print(f"Job ID: {job.job_id()}")
    print(f"Job status: {job.status()}")

    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()

    print(f"\nCounts (4096 shots):")
    total = sum(counts.values())
    for state, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {state}: {n} ({100*n/total:.2f}%)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "12_ibm_hardware.npz")
    np.savez(out, E_exact=E, counts=counts, backend=backend.name,
             job_id=job.job_id(), shots=4096, circuit_depth=isa_qc.depth())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
