"""HoRUS offline core engine.

Plain-Python port of the decision engine from ``horus3_1_streamlit.py``. Nothing here
imports Streamlit, Gemini or GitHub, and nothing reaches the network: the dataset, the
embedding model and (later) the language model are all read from local folders.

Typical use::

    from horus import HorusEngine
    engine = HorusEngine.from_defaults()
    cat = engine.finalise(structured_case, raw_text)     # or engine.categorise(raw_text) with an LLM
    report = engine.report("PT-2026-001", cat)
"""

from .engine import HorusEngine

__all__ = ["HorusEngine"]
__version__ = "0.1.0"
