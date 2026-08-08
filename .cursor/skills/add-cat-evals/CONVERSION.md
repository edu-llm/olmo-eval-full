# OLMo-core → HF conversion (reference)

The `atlas_arc` CAT runs through vLLM, which only loads **HF-format** weights
(`config.json` + `*.safetensors`). Native OLMo-core sharded checkpoints
(`model_and_optim/` + `.metadata` / `.distcp`) must be converted first. This is
what `scripts/run_cat_diagnostic.sh` does before the diagnostic (unless
`--skip-convert`).

## Canonical command

OLMo-core ships the converter at `src/examples/huggingface/convert_checkpoint_to_hf.py`:

```bash
python convert_checkpoint_to_hf.py \
  -i "${CKPT_DIR}" \
  -o "${CKPT_DIR}-hf" \
  --dtype bfloat16 \
  --skip-validation
```

`-i` must be a checkpoint dir containing both `config.json` (with `model` and
`dataset.tokenizer`) and `model_and_optim/`. `-t <hf-tokenizer-id>` overrides the
tokenizer if it can't be resolved from config.

## In-process equivalent (dense arch)

```python
import json, torch
from olmo_core.config import DType
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from olmo_core.nn.hf.checkpoint import save_hf_model

cfg = json.load(open(f"{ckpt_dir}/config.json"))
model = TransformerConfig.from_dict(cfg["model"]).build(init_device="meta")
model.to_empty(device=torch.device("cpu"))
load_model_and_optim_state(f"{ckpt_dir}/model_and_optim", model)
sd = get_model_state_dict(model, options=StateDictOptions(cpu_offload=True))
save_hf_model(f"{ckpt_dir}-hf", sd, model, dtype=DType.bfloat16, save_overwrite=True)
# then: AutoTokenizer.from_pretrained(tok_id).save_pretrained(f"{ckpt_dir}-hf")
```

## Cost

For a standard **dense** OLMo-2 / OLMo-3 checkpoint:

- **CPU-only**, single-process, no `torch.distributed` init.
- Peak RAM ≈ full model size at bf16 (~2 GB per 1B params); ~2× if `validate=True`.
  Use `--skip-validation` in the hot path and validate out-of-band.
- Disk: ~2 GB in + ~2 GB out per 1B params (bf16); double for fp32.
- Wall-clock: ~30 s for 1B on local NVMe with validation skipped; add S3 transfer
  and cold torch/olmo_core import (~5–15 s) in practice.

Approx RAM by size (bf16, skip-validation): 1B ≈ 4–6 GB · 7B ≈ 16–20 GB · 13B ≈ 28–32 GB.

## Caveats

- **Architecture-gated.** `get_hf_config` raises `NotImplementedError` for
  unsupported blocks. Fused / flash-attention configs force `device=cuda`; MoE /
  dropless variants add reshaping + validation fragility.
- Marked `@beta_feature` in olmo_core and sensitive to the installed `transformers`
  version.
- `olmo_core` is a pinned optional extra in this repo (`ai2-olmo-core==2.4.0`) that
  conflicts with `openhands`, so it is **not** in the default `uv sync`. The
  conversion is expected to run in the **training team's** OLMo-core env, not this
  repo's diagnostic env.
