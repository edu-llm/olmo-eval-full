# TracingLLM on Qwen3-30B-A3B-Thinking — plan

Base model: **Qwen/Qwen3-30B-A3B-Thinking-2507** (30.5B MoE, 3.3B active, 48 layers, thinking-only).

User supplies a **HuggingFace-format checkpoint URI** (staged under `teams/.../runs/`).

## Differences from OLMo smoke

| OLMo 370M smoke | Qwen 30B-A3B thinking |
|-----------------|------------------------|
| OLMo-core → HF on node | **HF pass-through** (no `olmo_core`) |
| `gpu-1xl40s` (~1.7B weights) | **`gpu-8xl40s`** + `device_map=auto` (full weights ~60GB bf16) |
| Layer 6 of 12 | **Layer 24** of 48 (middle third) |
| Plain generate | Thinking model: **`enable_thinking=False`** on `generate()` for evals |
| Probing: raw `encode()` | Same (raw encode avoids chat/thinking template for vectors) |

## Checkpoint layout (required)

Staged directory must look like HF:

```
config.json          # architectures includes Qwen3MoeForCausalLM (or similar)
model.safetensors*   # weights or index + shards
tokenizer.json / tokenizer_config.json
```

Stage with `sb_aws` / `stage_checkpoints.sh` — platform roles cannot read private buckets.

## Three-checkpoint protocol (optional)

Same roles as OLMoE plan (`early`, `chinchilla`, `final`), but **step numbers are manifest-only**
(no OLMo `config.json` schedule inference). Fill `checkpoints_qwen30b-thinking.json` after staging.

## Entry points

| Smoke (1 ckpt) | `generate_steering_vector.py` + `platform-run-qwen30b-thinking-smoke.yaml` |
| Full protocol | `run_tracingllm_qwen30b.py` + `platform-run-qwen30b-thinking.yaml` |

## Audit before submit

```bash
python SteeringVectors/audit_qwen_checkpoint.py --manifest SteeringVectors/checkpoints_qwen30b-thinking.json
python SteeringVectors/generate_steering_vector.py --dry-run --manifest SteeringVectors/checkpoints_qwen30b-thinking.json
```

See `AUDIT_QWEN30B.md` for the full checklist.
