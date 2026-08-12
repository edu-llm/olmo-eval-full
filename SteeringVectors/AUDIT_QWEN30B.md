# Qwen3-30B-A3B-Thinking — pre-submit audit

Run after editing `checkpoints_qwen30b-thinking.json` with your staged URI.

## Checklist

| # | Check | Command / action |
|---|--------|------------------|
| 1 | Placeholder URIs replaced | `grep REPLACE checkpoints_qwen30b-thinking.json` → empty |
| 2 | HF layout staged | `config.json`, sharded `model*.safetensors`, tokenizer files |
| 3 | Manifest dry-run | `python SteeringVectors/audit_qwen_checkpoint.py` |
| 4 | Vector dry-run | `python SteeringVectors/generate_steering_vector.py --dry-run --manifest ...` |
| 5 | S3 config reachable | `python SteeringVectors/audit_qwen_checkpoint.py --check-s3` |
| 6 | Compute | `gpu-8xl40s` + `device_map=auto` (bf16 ~60GB weights) |
| 7 | Platform guard | command includes `--dtype bfloat16` and `EDULLM_LAUNCH_CHECK=waived` (single process + `device_map=auto`) |
| 8 | Thinking disabled for eval | manifest `load.enable_thinking: false` |
| 9 | Layers in range | default smoke layer **24** (of 48); full job uses [20,24,28] |
| 10 | No olmo_core in command | HF-only sync (`--extra hf s3 clients`) |

## Known limitations

- **Probing / vector build** uses raw `tokenizer.encode()` on CSV statements (no chat template). This avoids injecting thinking tokens into activation capture; it is not identical to chat-style inference.
- **Discriminative / toxigen eval** calls `generate()` with `enable_thinking=False` when the model supports it; thinking-only models may still behave differently from the paper's Amber setup.
- **TruthfulQA** remains MC1/MC2, not GPT-judge.
- **Single-node 8×L40S** is the intended shape; one GPU cannot hold full bf16 MoE weights.

## Submit smoke

```bash
edullm submit --dataset none --compute gpu-8xl40s --experiment qwen30b-steering-smoke \
  --spec SteeringVectors/platform-run-qwen30b-thinking-smoke.yaml
```

## Submit full 3-checkpoint protocol

```bash
edullm submit --dataset none --compute gpu-8xl40s --experiment qwen30b-tracingllm \
  --spec SteeringVectors/platform-run-qwen30b-thinking.yaml
```
