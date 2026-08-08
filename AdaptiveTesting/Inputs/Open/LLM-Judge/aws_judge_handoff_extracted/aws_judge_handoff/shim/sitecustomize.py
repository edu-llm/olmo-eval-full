"""Node-local vLLM attention-backend shim (loaded via PYTHONPATH).

This does NOT modify any checksummed study artifact. On this Blackwell (B200)
node the pip-CUDA headers are incompatible with FlashInfer's JIT path, so vLLM's
default auto-selected attention backend fails to compile the engine core. The
same prebuilt FlashAttention backend that the MCQ sweep used works fine.

vLLM 0.26 exposes no environment variable for the attention backend, but
``attention_backend`` is a valid ``LLM()`` / ``EngineArgs`` field. We default it
here (overridable via JUDGE_ATTENTION_BACKEND) only when the caller did not set
it. Greedy (temperature 0) decoding makes the verdicts backend-invariant, and
the choice is uniform across every judge and wave, so the study's frozen
configuration and cross-wave comparisons are unaffected.
"""

try:
    import os

    def _install() -> None:
        import vllm

        original_init = vllm.LLM.__init__
        if getattr(original_init, "_judge_attn_patched", False):
            return

        def __init__(self, *args, **kwargs):  # noqa: N807
            kwargs.setdefault(
                "attention_backend",
                os.environ.get("JUDGE_ATTENTION_BACKEND", "FLASH_ATTN"),
            )
            return original_init(self, *args, **kwargs)

        __init__._judge_attn_patched = True  # type: ignore[attr-defined]
        vllm.LLM.__init__ = __init__

    _install()
except Exception as exc:  # noqa: BLE001
    import sys

    print(f"[judge sitecustomize] attention-backend patch skipped: {exc}", file=sys.stderr)
