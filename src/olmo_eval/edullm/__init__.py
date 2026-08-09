"""EduLLM's reusable judging, calibration, and adaptive-evaluation library.

The submodules are intentionally not imported here.  In particular, the
calibration and CAT numerical modules require the optional ``edullm``
dependency group, while bank inspection and configuration tooling should
remain importable without eagerly importing SciPy.
"""

__all__: tuple[str, ...] = ()
