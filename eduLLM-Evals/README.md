# eduLLM-Evals — InFoBench

This branch contains the InFoBench benchmark and the shared evaluation,
calibration, and computer-adaptive testing (CAT) toolchain used by the eduLLM
evaluation team.

## Repository status

- **Source bank:** 500 scenarios and 2,250 binary criteria in
  `data/InFoBench/`.
- **Calibration cohort:** 52 tutor models from 22 model families, scored by the
  frozen Qwen zero-shot binary judge.
- **Latent structure used by the latest study:** one overall
  `instruction_following` dimension built from InFoBench's five source labels
  (`content`, `format`, `number`, `style`, and `linguistic`).
- **Latest calibration/CAT policy study:** the preregistered 2PL-only follow-up
  completed all Phase-3 calibration panels but failed its CAT validation gates.
- **Latest stop-rule diagnostic:** EAP posterior-SD stopping improved honest
  target attainment from 13.8% to 85.8%, but added roughly 3–4 scenarios and
  therefore failed the preregistered no-length-increase gate (`do_not_adopt`).
- **Release status:** no final fitted bank or deployable CAT policy is approved.
  Phase 4 and final export were correctly not run.

The difficulty and discrimination values in the source rubric files are
synthetic placeholders. Do not treat them as calibrated measurements or use
them to support deployment claims.

## Start here

| Purpose | File |
| --- | --- |
| Benchmark format and five source labels | `data/InFoBench/README.md` |
| Frozen 2PL-only study design | `docs/infobench_2pl_only_v1_plan.md` |
| Machine-readable frozen configuration | `configs/infobench_calibration_cat_2pl_only_v1.json` |
| Terminal result and limitations | `reports/infobench_calibration_cat_2pl_only_v1/EXECUTION_SUMMARY.md` |
| Figure explanations | `reports/infobench_calibration_cat_2pl_only_v1/figures/FIGURE_GUIDE.md` |
| Exact reproduction commands | `reports/infobench_calibration_cat_2pl_only_v1/reproduction_commands.txt` |
| Frozen EAP stop-rule prototype | `configs/infobench_eap_stop_prototype_v1.json` |
| EAP stop-rule result and figures | `reports/infobench_eap_stop_prototype_v1/EAP_STOP_PROTOTYPE_SUMMARY.md` |
| EAP raw metrics and provenance | `runs/calibration/InFoBench_eap_stop_prototype_v1/` |

Earlier InFoBench calibration and numerical studies are retained because the
latest 2PL pipeline imports or hash-verifies parts of that provenance chain.
Files with `v2`, `v3`, or `v4` in their names are therefore not necessarily
obsolete.

## Install and test

```bash
uv sync --frozen --extra irt --extra dev
uv run --frozen --extra irt pytest -q
```

The current 2PL pipeline can be checked more narrowly with:

```bash
uv run --frozen --extra irt pytest -q \
  tests/test_infobench_v2_numerical_followup_v4.py \
  tests/test_nested_scenario_cat_cv_2pl_only_v1.py \
  tests/test_nested_cat_total_uncertainty_2pl_only_v1.py \
  tests/test_finalize_infobench_calibration_cat_2pl_only_v1.py \
  tests/test_render_infobench_2pl_only_v1_terminal.py
```

The optional EAP stopping path and frozen prototype driver can be checked with:

```bash
uv run --frozen --extra irt pytest -q \
  tests/test_mirt.py \
  tests/test_engine_sim.py \
  tests/test_scenario_cat_lib.py \
  tests/test_infobench_eap_stop_prototype.py
```

## Generate tutor responses

The shipped benchmark registry now contains only data present on this branch:

```bash
# Offline prompt preview; no model or GPU is loaded.
tutor-cat generate \
  --benchmarks benchmarks.yaml \
  --only InFoBench \
  --dry-run \
  --limit 3
```

For a real open-weight generation run, install the `gen` dependencies and run
the same registry on a CUDA host. Generated shards are written under
`runs/responses/InFoBench/` and are intentionally not committed.

## Important entry-point note

`config.yaml` is a retained legacy three-skill TutorBench runtime example. Its
data files are not shipped on this branch, and it is **not** the configuration
for the current InFoBench calibration/CAT study. Use the versioned InFoBench
configs and reproduction commands linked above; do not run bare
`tutor-cat validate` or `tutor-cat run` and interpret that legacy config as
an InFoBench result.

The response-generation, judging, and calibration modules remain generic and
can accept externally supplied benchmark banks. `benchmarks.yaml` lists only
what this branch actually ships.
