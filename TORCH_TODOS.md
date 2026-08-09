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
  missing the flash-attn package or is not supported on this platform`. Measured in probe
  `run_019fe316-0e47`.

  **The cause recorded here was wrong and is corrected 2026-08-08.** This line used to
  blame `pip install ".[hf,s3]"` for not pulling the wheel, which assumed flash-attn had to
  arrive through *our* install. It does not — it belongs to the image, and OLMo-core's own
  `.edullm/Dockerfile` installs it with a pinned release wheel. Two facts rule the old
  explanation out: `olmo-eval-full` declares no `torch` in its base dependencies, `hf` or
  `s3`, so our install cannot move the pin or break the wheel's ABI; and an L4 is Ada /
  SM 8.9, which FA2 supports, so the message's second branch does not apply either.

  The real cause is that `.edullm/Dockerfile` does not exist on OLMo-core's `main` -- it is
  a per-branch directory -- and the wheel line was added on 2026-08-03 in `87df895f`, which
  reached only 4 of the 74 `edullm/*` branches: `flash-attn-for-the-olmo3-configs`,
  `hyper-connections-370m`, `latent-cot-superposition-amy` and `p3-math-split`. Our image
  comes from `edullm/memory-split-135m`, one of the 70 without it. There is no shared base
  to fix once.

**Measured cost of leaving it off:** 64 tokens in ~2.0s on an L4 for the 135M model, over
three prompts (ifeval 2.20s, leaderboard_math 1.99s, gpqa 1.67s). About 32s per
1024-token item, so a 40-item bank is roughly 20 minutes of pure decode against the 1h
`olmo-core-check` bound. Tight but viable, which is why this is pinned rather than fixed.

**Two ways to fix it when it matters, and both are needed:**
1. Put flash-attn in the image. Branch off `edullm/memory-split-135m` -- the branch our
   image is built from -- and add two lines to its `.edullm/Dockerfile`, copied from
   `edullm/p3-math-split`, which has already built them:

   ```dockerfile
   python -c "import torch; assert torch.compiled_with_cxx11_abi(), 'flash-attn wheel requires torch CXX11 ABI'"; \
   python -m pip install --no-cache-dir --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810"; \
   python -c "import flash_attn; assert callable(flash_attn.flash_attn_varlen_func)"
   ```

   Not `pip install flash-attn`: the base carries no nvcc, so PyPI's source distribution
   cannot build there. `--no-deps` stops its unconstrained `torch` requirement moving the
   pin, and the import assertion checks the compiled symbol rather than the package name,
   because a wheel built for the wrong boundary imports and then fails to load.

   **Stay on their image rather than building ours.** This wheel matches
   `edullm/memory-split-135m`'s `torch==2.9.0` exactly and `p3-math-split` has already
   proved it in a real build. `olmo-eval-full/.edullm/Dockerfile` can hold torch itself via
   `INSTALL_OLMO_CORE=1`, but it resolves torch 2.10.0, and flash-attn v2.8.3 ships no cu12
   build for torch 2.10 -- only cu13. Matching it would mean reaching back to v2.8.1 for an
   untested combination, which is a problem created by leaving their image rather than one
   that needed solving.
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
