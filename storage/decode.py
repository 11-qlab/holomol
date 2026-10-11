"""Decode q-vector measurements back into bits."""
import numpy as np


def measure_q_populations(state_vector, q_list, ensemble):
    """
    Given a stored state |psi> (length-N complex vector in the single-f
    excitation subspace), return the population on each |f,q>.
    """
    pop = np.zeros(len(q_list))
    for i, q in enumerate(q_list):
        phases = np.exp(1j * ensemble.positions @ np.asarray(q))
        amp = np.vdot(phases / np.sqrt(ensemble.N), state_vector)
        pop[i] = abs(amp) ** 2
    return pop


def populations_to_bits(pop, threshold: float = 0.5):
    return [1 if p > threshold else 0 for p in pop]


def decode_with_confidence(pop, threshold: float = 0.5):
    bits = populations_to_bits(pop, threshold)
    conf = [abs(float(p) - threshold) for p in pop]
    return bits, conf


def bit_error_rate(true_bits, decoded_bits) -> float:
    t = np.asarray(true_bits)
    d = np.asarray(decoded_bits)
    n = min(len(t), len(d))
    if n == 0:
        return 0.0
    return float(np.mean(t[:n] != d[:n]))


def reconstruct_bytes(bits):
    if len(bits) % 8 != 0:
        bits = list(bits) + [0] * (8 - len(bits) % 8)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | int(b)
        out.append(byte)
    return bytes(out)
