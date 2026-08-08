# Torch / OLMo-core TODOs

Deferred deliberately, with the evidence, so a later reader inherits the measurement
rather than the guess.

## KV caching is off on the native generative path

**Status: pinned 2026-08-08. Not blocking — generative answers are short enough that the
cacheless path is affordable.**

`use_cache=False` on every native generation today, so each decode step re-forwards the
whole prefix. Cost is quadratic in sequence length instead of linear.

Why it is off, in two layers:

- OLMo-core's default `torch` attention backend does not implement KV caching and refuses
  outright — `assert_supports_kv_cache` raises at
  `OLMo-core/src/olmo_core/nn/attention/backend.py:288-289`. Only the flash backends
  override it as `pass` (`flash_2` at 479, `flash_3` at 703, `flash_4` at 899); `te`
  refuses too. Probe `run_019fe2e6-9461` died on exactly this before emitting a token.
- Asking for `flash_2` then fails for a different reason: `'FlashAttention2Backend' is
  missing the flash-attn package or is not supported on this platform`. flash-attn is a
  binary wheel and `pip install ".[hf,s3]"` does not pull it. Measured in probe
  `run_019fe316-0e47`.

**Measured cost of leaving it off:** 64 tokens in ~2.0s on an L4 for the 135M model, over
three prompts (ifeval 2.20s, leaderboard_math 1.99s, gpqa 1.67s). About 32s per
1024-token item, so a 40-item bank is roughly 20 minutes of pure decode against the 1h
`olmo-core-check` bound. Tight but viable, which is why this is pinned rather than fixed.

**Two ways to fix it when it matters:**
1. Put flash-attn in the image — a binary wheel matching CUDA 12, the pinned torch, CPython
   3.12 and the CXX11 ABI. `OLMo-core/.edullm/Dockerfile` already installs one this way for
   its own build, so there is a worked precedent to copy.
2. Thread `attention_backend` through `InferenceConfig` the way `--dtype` was threaded in
   `8f45ca3f`. `OlmoCoreProvider` already exposes it at
   `src/olmo_eval/inference/providers/olmo_core.py:156-159`; `_OlmoCoreScoringModel` has no
   such field. This alone is not enough — it only helps once the wheel exists.

**Revisit when** any of: a bank needs materially longer outputs than the measured maxima,
a generative CAT run approaches the 1h bound, or a larger checkpoint makes the quadratic
term bite. An L4 is Ada / SM 8.9, so FA2 is the fit — FA3 is Hopper-only and FA4
Blackwell-only.

## This checkpoint never emits eos — answered 2026-08-08

Probe `run_019fe316-0e47` decoded three prompts, one from each generative bank, at 64
tokens each. **Token 0 appeared in none of them and all three ran the full budget:**
ifeval 64/64 in 2.20s, leaderboard_math 64/64 in 1.99s, gpqa 64/64 in 1.67s. `eos_token_id`
is 0 and the checkpoint writes `pad_token_id` as 0 too, which is a separate problem, but
the finding here is simpler: eos is not a stopping mechanism for this model.

Consequences, all live:

- Generation is bounded only by `max_new_tokens` and text-level `stop_sequences`. A bank
  with `stop_sequences: []` — which is `ifeval` and was `gpqa` — pays its full budget on
  every item, so budget figures for those are realised cost, not worst case.
- The decodes are degenerate at this scale. ifeval echoed its own instruction, math looped
  one clause, gpqa invented options (E) through (H) after the real four. Worth knowing
  before grading a 135M base checkpoint generatively.
- Graders that take the *last* match are exposed. `config.yaml` notes gsm8k's extractor
  takes the last number, so a run-on answer can overwrite a correct one.

This is a property of the training corpus rather than the architecture, so it does not
generalise to the next submitter's checkpoint. `RUNNER_REQUEST.md` asks for the sample
output that answers it.

## `free_inference_cache` is still unmeasured

It exists and costs 0.1 ms, but freed 0.0 MB in probe `run_019fe316-0e47` because
`use_cache=False` allocates no cache. That figure describes nothing. Source reading says
memory is a high-water mark rather than a sum — `prepare_inference_cache` resets per call
and `is_reusable` never shrinks the buffer — so the real reason to call it is when leaving
the generate phase, not between items. Re-measure once caching is on.
