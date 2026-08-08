"""Inference engine wrapper.

Backends:
  * ``vllm``  - production path (Linux/CUDA), continuous batching + prefix cache.
  * ``hf``    - transformers fallback for architectures vLLM cannot load.
  * ``mock``  - deterministic, no weights; for local smoke tests / CI.

The engine exposes two primitives used by the rest of the pipeline:
  * :meth:`loglikelihood` - length-normalized logprob of each (context, continuation) pair.
  * :meth:`generate`      - free-form completion for a batch of prompts.

Use :func:`build_engine` rather than the constructor: it adds the load-robustness
respgen was built around - capability routing, retries, a smoke generation that
must produce output, and a vLLM -> transformers fallback so a model that vLLM
cannot serve still answers instead of dead-lettering every item.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from models_registry import ModelSpec, resolve_max_model_len

# Load robustness (respgen parity): retry a construction a few times, since
# transient Hub 429s / network blips resolve on retry, before giving up on a
# backend and falling back.
LOAD_ATTEMPTS = 3
LOAD_RETRY_DELAY = 5.0


@dataclass
class GenParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 1234
    stop: list[str] = field(default_factory=list)
    # Kept low + uniform for FRQ generation; still changes the argmax at
    # temperature 0, so it is recorded per output row. Default 1.0 is a no-op,
    # leaving MCQ scoring and the judge unchanged.
    repetition_penalty: float = 1.0


class Engine:
    """Unified inference engine over a single resident model."""

    def __init__(
        self,
        spec: ModelSpec,
        backend: str = "vllm",
        vllm_cfg: dict[str, Any] | None = None,
    ):
        self.spec = spec
        self.backend = backend
        self.vllm_cfg = vllm_cfg or {}
        self._llm = None
        self._tok = None
        self._hf_model = None
        if backend == "vllm":
            self._init_vllm()
        elif backend == "hf":
            self._init_hf()
        elif backend == "mock":
            pass
        else:
            raise ValueError(f"unknown backend {backend!r}")

    # ------------------------------------------------------------------ init
    def _init_vllm(self) -> None:
        import dataclasses

        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs

        kwargs = {
            "model": self.spec.id,
            "revision": getattr(self.spec, "revision", None),
            "tokenizer": getattr(self.spec, "tokenizer_id", None),
            "dtype": self.vllm_cfg.get("dtype", self.spec.dtype),
            "trust_remote_code": self.spec.trust_remote_code,
            "tensor_parallel_size": self.spec.tp,
            "max_model_len": self.spec.max_model_len,
            "gpu_memory_utilization": self.vllm_cfg.get("gpu_memory_utilization", 0.30),
            "enable_prefix_caching": self.vllm_cfg.get("enable_prefix_caching", True),
            "max_num_seqs": self.vllm_cfg.get("max_num_seqs", 256),
            "max_num_batched_tokens": self.vllm_cfg.get("max_num_batched_tokens", 8192),
            "swap_space": self.vllm_cfg.get("swap_space_gb", 4),
        }
        # Engine arguments drift between vLLM releases; only pass what this
        # installed version actually accepts.
        accepted = {f.name for f in dataclasses.fields(EngineArgs)}
        kwargs = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
        self._llm = LLM(**kwargs)
        self._tok = self._llm.get_tokenizer()

    def _init_hf(self) -> None:
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Some repos ship no tokenizer (e.g. apple/OpenELM expects Llama-2's). When
        # borrowing another repo's tokenizer, don't reuse this model's revision SHA
        # - it doesn't exist in the tokenizer repo.
        revision = getattr(self.spec, "revision", None)
        tok_src = getattr(self.spec, "tokenizer_id", None) or self.spec.id
        tok_rev = revision if tok_src == self.spec.id else None
        self._tok = AutoTokenizer.from_pretrained(
            tok_src, revision=tok_rev, trust_remote_code=self.spec.trust_remote_code
        )
        # FRQ prompts are fit to the window by keeping their TAIL (the student's
        # latest turn + the generation cue). HF defaults to truncation_side
        # "right", which would drop exactly that tail if a decode -> re-encode
        # drift pushes the prompt one token over the limit.
        try:
            self._tok.truncation_side = "left"
        except Exception:
            pass
        # FORCE_CPU=1 / device_map=cpu for multi-machine CPU sweeps; otherwise auto.
        force_cpu = os.environ.get("FORCE_CPU", "").strip() in {"1", "true", "yes"}
        device_map = "cpu" if force_cpu else "auto"
        if force_cpu:
            dtype = torch.float32
        elif self.spec.dtype == "bfloat16" and torch.cuda.is_available():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.spec.id,
            revision=revision,
            trust_remote_code=self.spec.trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self._hf_model.eval()

    # --------------------------------------------------------------- prompts
    @property
    def tokenizer(self):
        """The live tokenizer (None for the mock backend). FRQ generation needs it
        to count prompt tokens and size each item's generation budget."""
        return self._tok

    @property
    def has_chat_template(self) -> bool:
        return getattr(self._tok, "chat_template", None) is not None

    def format_prompt(self, prompt: str) -> str:
        """Apply the chat template for instruct models; raw text otherwise."""
        if not self.spec.apply_chat_template or self._tok is None:
            return prompt
        try:
            return self._tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def render_chat(
        self, messages: list[dict[str, str]], enable_thinking: bool | None = None
    ) -> tuple[str, bool]:
        """Render a full multi-turn message list, returning (text, applied).

        Unlike :meth:`format_prompt` (single user turn) this preserves the FRQ
        system turn and conversation history. ``applied`` is False when the model
        is not chat-tuned or ships no template, so the caller can fall back to the
        flat base rendering and record ``Chat Template Applied = 0``.
        """
        if not self.spec.apply_chat_template or self._tok is None or not self.has_chat_template:
            return "", False
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return self._tok.apply_chat_template(messages, **kwargs), True
        except TypeError:
            kwargs.pop("enable_thinking", None)  # tokenizer without that kwarg
            try:
                return self._tok.apply_chat_template(messages, **kwargs), True
            except Exception:
                return "", False
        except Exception:
            return "", False

    # ---------------------------------------------------------- loglikelihood
    def loglikelihood(self, contexts: list[str], continuations: list[str]) -> list[float]:
        assert len(contexts) == len(continuations)
        if self.backend == "vllm":
            return self._loglik_vllm(contexts, continuations)
        if self.backend == "hf":
            return self._loglik_hf(contexts, continuations)
        return self._loglik_mock(contexts, continuations)

    def _loglik_vllm(self, contexts, continuations) -> list[float]:
        from vllm import SamplingParams

        full_ids: list[list[int]] = []
        ctx_lens: list[int] = []
        for ctx, cont in zip(contexts, continuations, strict=True):
            c_ids = self._tok(ctx, add_special_tokens=True).input_ids
            f_ids = self._tok(ctx + cont, add_special_tokens=True).input_ids
            # guard against non-prefix tokenization
            clen = len(c_ids)
            if f_ids[:clen] != c_ids:
                clen = _common_prefix_len(c_ids, f_ids)
            ctx_lens.append(min(clen, len(f_ids) - 1))
            full_ids.append(f_ids)
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        prompts = [{"prompt_token_ids": ids} for ids in full_ids]
        outs = self._llm.generate(prompts, sp)
        scores: list[float] = []
        for out, ids, clen in zip(outs, full_ids, ctx_lens, strict=True):
            plp = out.prompt_logprobs or []
            total, n = 0.0, 0
            for pos in range(clen, len(ids)):
                if pos >= len(plp) or plp[pos] is None:
                    continue
                entry = plp[pos]
                tid = ids[pos]
                lp = entry.get(tid)
                if lp is None:
                    continue
                total += lp.logprob
                n += 1
            scores.append(total / max(1, n))
        return scores

    def _loglik_hf(self, contexts, continuations) -> list[float]:
        import torch

        scores: list[float] = []
        device = next(self._hf_model.parameters()).device
        for ctx, cont in zip(contexts, continuations, strict=True):
            c_ids = self._tok(ctx, return_tensors="pt", add_special_tokens=True).input_ids
            f_ids = self._tok(ctx + cont, return_tensors="pt", add_special_tokens=True).input_ids
            clen = c_ids.shape[1]
            f_ids = f_ids.to(device)
            with torch.no_grad():
                logits = self._hf_model(f_ids).logits
            logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
            targets = f_ids[0, 1:]
            tok_lp = logprobs[torch.arange(targets.shape[0]), targets]
            cont_lp = tok_lp[clen - 1 :]
            scores.append(float(cont_lp.mean().item()) if cont_lp.numel() else 0.0)
        return scores

    def _loglik_mock(self, contexts, continuations) -> list[float]:
        return [
            -_hash_unit(c + "||" + k) * 10.0 for c, k in zip(contexts, continuations, strict=True)
        ]

    # -------------------------------------------------------------- generate
    def generate(
        self,
        prompts: list[str],
        params: GenParams,
        *,
        pre_rendered: bool = False,
        max_tokens_per_prompt: list[int] | None = None,
    ) -> list[str]:
        """Free-form completion for a batch of prompts.

        ``pre_rendered`` skips :meth:`format_prompt` - the FRQ path renders the
        full multi-turn prompt itself (system turn + history) and must not have it
        re-wrapped. ``max_tokens_per_prompt`` gives each item its own generation
        budget so a short prompt gets the full budget while a long prompt on a
        small context window still keeps room to answer.
        """
        formatted = prompts if pre_rendered else [self.format_prompt(p) for p in prompts]
        if self.backend == "vllm":
            return self._generate_vllm(formatted, params, max_tokens_per_prompt)
        if self.backend == "hf":
            return self._generate_hf(formatted, params, max_tokens_per_prompt)
        return self._generate_mock(formatted, params)

    def _generate_vllm(
        self, prompts, params: GenParams, max_tokens_per_prompt: list[int] | None = None
    ) -> list[str]:
        from vllm import SamplingParams

        def _sp(max_tokens: int):
            return SamplingParams(
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=max_tokens,
                seed=params.seed,
                stop=params.stop or None,
                repetition_penalty=params.repetition_penalty,
            )

        # vLLM accepts a list of SamplingParams aligned to prompts, so each item
        # decodes with its own budget.
        sp = _sp(params.max_tokens) if max_tokens_per_prompt is None else [
            _sp(mt) for mt in max_tokens_per_prompt
        ]
        outs = self._llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    def _generate_hf(
        self, prompts, params: GenParams, max_tokens_per_prompt: list[int] | None = None
    ) -> list[str]:
        import torch

        device = next(self._hf_model.parameters()).device
        results: list[str] = []
        for i, p in enumerate(prompts):
            mt = params.max_tokens if max_tokens_per_prompt is None else max_tokens_per_prompt[i]
            ids = self._tok(p, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = self._hf_model.generate(
                    ids,
                    max_new_tokens=mt,
                    do_sample=params.temperature > 0,
                    temperature=max(params.temperature, 1e-5),
                    top_p=params.top_p,
                    repetition_penalty=params.repetition_penalty,
                )
            text = self._tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
            results.append(text)
        return results

    def _generate_mock(self, prompts, params: GenParams) -> list[str]:
        out = []
        for p in prompts:
            if "[RESULT]" in p:
                # judge-style prompt: emit parseable feedback + score
                out.append("Feedback: mock judgement based on the rubric. [RESULT] 4")
            else:
                out.append(f"[mock:{self.spec.id}] answer to: {p[:60].strip()}")
        return out

    # ------------------------------------------------------------------ misc
    def smoke_test(self) -> None:
        """Tiny generation that must yield output; folds a health check into the
        load. A model that constructs fine but generates nothing is a failure -
        catching it here means the caller can retry or fall back instead of
        writing empty rows for every item."""
        out = self.generate(["Say OK."], GenParams(max_tokens=8, temperature=0.0, seed=0))
        if not out or not (out[0] or "").strip():
            raise RuntimeError(f"smoke test produced no output for {self.spec.id}")

    def close(self) -> None:
        """Drop model references and best-effort reclaim GPU memory. Never raises.

        Models are resident here (one process walks the whole model list), so a
        vLLM engine that is merely dereferenced - KV-cache blocks, CUDA graphs,
        NCCL/tensor-parallel state that plain GC does not reclaim - accumulates
        and makes *later* models OOM at engine init. respgen avoided this by
        using a fresh process per model; the resident loop must free explicitly.
        """
        used_vllm = self.backend == "vllm" and self._llm is not None
        self._llm = None
        self._hf_model = None
        self._tok = None
        _reclaim_gpu(used_vllm)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _hash_unit(s: str) -> float:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def probe_backend(spec: ModelSpec, requested: str) -> str:
    """Resolve the effective backend for a model (capability routing).

    ``hf_fallback`` models are forced to the HF path (unless mock is requested).
    """
    if requested == "mock":
        return "mock"
    if spec.backend == "hf_fallback" and requested == "vllm":
        return "hf"
    return requested


# ---------------------------------------------------------------------------
# Load robustness: retry, smoke test, vLLM -> transformers fallback
# ---------------------------------------------------------------------------


def _reclaim_gpu(destroy_vllm_state: bool) -> None:
    """Best-effort GPU-memory reclamation. Never raises."""
    try:
        import gc

        if destroy_vllm_state:
            try:  # vLLM's own teardown of the tensor-parallel / device state
                from vllm.distributed.parallel_state import destroy_model_parallel

                destroy_model_parallel()
            except Exception:
                pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


@contextlib.contextmanager
def _capture_fd_stderr():
    """Redirect OS-level fd 2 to a temp file for the duration of the block, then
    replay it to the real stderr on exit.

    vLLM V1 runs its engine core in a SUBPROCESS that inherits fd 2; when it
    crashes at init the parent only sees ``Failed core proc(s): {}`` while the
    real traceback goes to that inherited stderr and is lost. Redirecting the fd
    (not ``sys.stderr``, which the subprocess doesn't share) captures it so the
    fallback can report why vLLM actually failed. Yields the temp file, or None
    if fd 2 can't be duplicated (then capture is skipped).
    """
    try:
        saved = os.dup(2)
    except Exception:
        yield None
        return
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 2)
        yield tmp
    finally:
        try:
            os.dup2(saved, 2)
        finally:
            os.close(saved)
        try:  # replay captured bytes so vLLM's load logs aren't swallowed
            tmp.seek(0)
            data = tmp.read()
            if data:
                os.write(2, data)
        except Exception:
            pass
        tmp.close()


def _try_backend(make, attempts: int, delay: float, sleep=time.sleep) -> Engine:
    """Construct + smoke-test an engine, retrying transient failures. Frees any
    partially-built engine between attempts so a retry starts from clean memory.
    Raises the last error if every attempt fails."""
    last: Exception | None = None
    for i in range(attempts):
        engine: Engine | None = None
        try:
            engine = make()
            engine.smoke_test()
            return engine
        except Exception as exc:  # noqa: BLE001 - decide retry vs give up below
            last = exc
            if engine is not None:
                engine.close()
            else:
                _reclaim_gpu(True)
            if i < attempts - 1 and delay > 0:
                print(
                    f"  [load retry {i + 1}/{attempts - 1}] {type(exc).__name__}: {exc}",
                    flush=True,
                )
                sleep(delay)
    raise last if last is not None else RuntimeError("engine construction failed")


def _stderr_tail(errbuf, limit: int = 3000) -> str:
    if errbuf is None:
        return ""
    try:
        errbuf.seek(0)
        return errbuf.read().decode("utf-8", "replace")[-limit:]
    except Exception:
        return ""


def build_engine(
    spec: ModelSpec,
    requested: str = "vllm",
    vllm_cfg: dict[str, Any] | None = None,
    *,
    sleep=time.sleep,
    resolve_window: bool = True,
) -> Engine:
    """Load one model with respgen's protections, returning a live engine.

    * capability routing first (:func:`probe_backend`) - ``hf_fallback`` models go
      straight to transformers, no wasted vLLM attempt;
    * each construction is retried ``LOAD_ATTEMPTS`` times and must pass a smoke
      generation;
    * a vLLM engine that still won't come up falls back to transformers, so the
      model produces output instead of every item dead-lettering. ``mock`` is
      never faked into another backend.

    ``spec.max_model_len`` is resolved here (see
    :func:`models_registry.resolve_max_model_len`) so the engine, prompt fitting
    and the recorded ``Max Model Len`` all agree on one window.
    """
    backend = probe_backend(spec, requested)
    if resolve_window:
        # mock stays offline: resolve from the static table only.
        spec.max_model_len = resolve_max_model_len(spec, offline=(backend == "mock"))

    if backend == "mock":
        return Engine(spec, backend="mock", vllm_cfg=vllm_cfg)
    if backend == "hf":
        return _try_backend(
            lambda: Engine(spec, backend="hf", vllm_cfg=vllm_cfg),
            LOAD_ATTEMPTS,
            LOAD_RETRY_DELAY,
            sleep,
        )

    tail = ""
    with _capture_fd_stderr() as errbuf:
        try:
            return _try_backend(
                lambda: Engine(spec, backend="vllm", vllm_cfg=vllm_cfg),
                LOAD_ATTEMPTS,
                LOAD_RETRY_DELAY,
                sleep,
            )
        except Exception as exc:  # noqa: BLE001 - fall back to transformers below
            primary = exc
            tail = _stderr_tail(errbuf)

    print(
        f"  [engine] vLLM load failed for {spec.id}: {primary!r}\n"
        f"  [engine] falling back to transformers (hf)",
        flush=True,
    )
    try:
        return _try_backend(
            lambda: Engine(spec, backend="hf", vllm_cfg=vllm_cfg),
            2,
            LOAD_RETRY_DELAY,
            sleep,
        )
    except Exception as hf_err:  # noqa: BLE001
        suffix = f"\n--- vLLM stderr tail ---\n{tail}" if tail else ""
        raise RuntimeError(
            f"vLLM load failed: {primary!r}; transformers fallback also failed: "
            f"{hf_err!r}{suffix}"
        ) from primary
