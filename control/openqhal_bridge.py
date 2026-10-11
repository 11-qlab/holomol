"""
Optional bridge to openqhal for waveform rendering and IBM backend submission.

If openqhal (github.com/11-qlab/openqhal) is not installed, this module
degrades gracefully: is_available() returns False and any function that
would need it raises ImportError with a helpful message.

The core holomol simulator does NOT depend on this module.
"""
import numpy as np

_OPENQHAL_AVAILABLE = False
try:
    import qhal  # noqa: F401
    _OPENQHAL_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    return _OPENQHAL_AVAILABLE


def describe() -> dict:
    return {
        "available": _OPENQHAL_AVAILABLE,
        "purpose": "Optional waveform rendering and IBM backend submission",
        "required_for_core": False,
        "repo": "https://github.com/11-qlab/openqhal",
    }


def render_envelope_to_iq(t_axis, I, Q, sample_rate: float):
    """
    Placeholder interface: openqhal exposes its own envelope/compiler pipeline.
    This just returns the caller's samples for now.
    """
    if not _OPENQHAL_AVAILABLE:
        raise ImportError(
            "openqhal not installed. See https://github.com/11-qlab/openqhal"
        )
    return t_axis, I, Q


def compile_raman_to_qhal(t_axis, I, Q, sample_rate: float,
                          channel_ids=(0, 1)):
    """Compile a Raman I/Q pair into an openqhal channel program (stub)."""
    if not _OPENQHAL_AVAILABLE:
        raise ImportError("openqhal not installed.")
    return {
        "channels": list(channel_ids),
        "sample_rate": sample_rate,
        "I": np.asarray(I),
        "Q": np.asarray(Q),
        "n_samples": len(t_axis),
    }


def submit_to_ibm(program, backend_name: str, token: str = None):
    """Submit an openqhal program to IBM. Requires qhal + a valid token."""
    if not _OPENQHAL_AVAILABLE:
        raise ImportError("openqhal not installed.")
    from qhal.backends import ibm  # type: ignore
    return ibm.submit(program, backend_name, token=token)
