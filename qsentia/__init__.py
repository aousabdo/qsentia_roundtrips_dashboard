"""
Core package for the QSentia Roundtrips dashboard.

Modules
-------
io
    Data loading helpers with schema validation and caching-friendly interfaces.
compute
    Aggregations and analytics derived from the original roundtrips_deepdive_v1 script.
viz
    Plotly-based figures used across the Streamlit UI.
utils
    Formatting helpers, constants, and shared utilities.
"""

from . import io, compute, viz, utils  # noqa: F401

__all__ = ["io", "compute", "viz", "utils"]
