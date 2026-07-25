"""Inference engine wrapper.

Backends:
  * ``vllm``  - production path (Linux/CUDA), continuous batching + prefix cache.
  * ``hf``    - transformers fallback for architectures vLLM cannot load.
  * ``mock``  - deterministic, no weights; for local smoke tests / CI.

The engine exposes two primitives used by the rest of the pipeline:
  * :meth:`loglikelihood` - length-normalized logprob of each (context, continuation) pair.
  * :meth:`generate`      - free-form completion for a batch of prompts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from models_registry import ModelSpec


@dataclass
class GenParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 1234
    stop: list[str] = field(default_factory=list)


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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(
            self.spec.id, trust_remote_code=self.spec.trust_remote_code
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.spec.id,
            trust_remote_code=self.spec.trust_remote_code,
            torch_dtype=torch.bfloat16 if self.spec.dtype == "bfloat16" else torch.float16,
            device_map="auto",
        )
        self._hf_model.eval()

    # --------------------------------------------------------------- prompts
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
    def generate(self, prompts: list[str], params: GenParams) -> list[str]:
        formatted = [self.format_prompt(p) for p in prompts]
        if self.backend == "vllm":
            return self._generate_vllm(formatted, params)
        if self.backend == "hf":
            return self._generate_hf(formatted, params)
        return self._generate_mock(prompts, params)

    def _generate_vllm(self, prompts, params: GenParams) -> list[str]:
        from vllm import SamplingParams

        sp = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            seed=params.seed,
            stop=params.stop or None,
        )
        outs = self._llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs]

    def _generate_hf(self, prompts, params: GenParams) -> list[str]:
        import torch

        device = next(self._hf_model.parameters()).device
        results: list[str] = []
        for p in prompts:
            ids = self._tok(p, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = self._hf_model.generate(
                    ids,
                    max_new_tokens=params.max_tokens,
                    do_sample=params.temperature > 0,
                    temperature=max(params.temperature, 1e-5),
                    top_p=params.top_p,
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
    def close(self) -> None:
        self._llm = None
        self._hf_model = None
        self._tok = None


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
