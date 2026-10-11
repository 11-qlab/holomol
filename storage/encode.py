"""Map classical bits onto a set of q-vectors."""
import numpy as np


def bits_to_index(bits) -> int:
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    return idx


def index_to_bits(idx: int, width: int):
    return [(idx >> (width - 1 - i)) & 1 for i in range(width)]


def bytes_to_bits(data: bytes):
    bits = []
    for byte in data:
        bits.extend(index_to_bits(byte, 8))
    return bits


def make_q_grid(n_modes: int, q_max: float, dim: int = 1) -> np.ndarray:
    """Uniform grid of q-vectors. For dim=1 a line; for dim=2 a square grid."""
    if dim == 1:
        qs = np.linspace(-q_max, q_max, n_modes)
        return np.column_stack([qs, np.zeros(n_modes), np.zeros(n_modes)])
    if dim == 2:
        side = int(np.ceil(np.sqrt(n_modes)))
        qs = np.linspace(-q_max, q_max, side)
        QX, QY = np.meshgrid(qs, qs)
        return np.column_stack([QX.ravel(), QY.ravel(), np.zeros(side * side)])
    raise ValueError("dim must be 1 or 2")


def encode_message(message_bits, q_list):
    """
    Simplest encoding: one q-vector per '1' bit. More elaborate schemes
    assign multi-bit symbols to each q.
    """
    bits = np.asarray(message_bits, dtype=int)
    n = min(len(bits), len(q_list))
    return [q_list[i] for i in range(n) if bits[i] == 1]


def encode_bytes(data: bytes, q_list):
    return encode_message(bytes_to_bits(data), q_list)


def capacity_bits(n_modes: int) -> float:
    """Log2 of the number of distinct q-subset patterns."""
    return float(n_modes)  # one bit per mode in this simple scheme
