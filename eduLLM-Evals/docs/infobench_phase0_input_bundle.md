# InFoBench remediation: frozen Phase 0 inputs

Phase 0 preserves the exact data boundary used by the August 4 InFoBench run. It
does not refit item parameters or run CAT.

## Frozen artifact

- Repository archive:
  `runs/calibration/InFoBench_remediation_phase0/infobench_phase0_inputs.tar.gz`
- Archive SHA-256:
  `b033f86f0e206197bfff80c057a5793563989b04ab4062f4e7aa476718c62ca3`
- Tracked provenance:
  `reports/infobench_calibration_remediation/phase0_input_artifact.json`
- Storage status: `tracked_in_repository`

The archive is committed to this repository and is retrievable from a fresh clone.
The `storage` block in the hash-pinned provenance manifest still records the
`local-only://` state that was true when the bundle was built; it is retained as
historical build provenance rather than rewritten after publication. Verify the
repository copy against the SHA-256 above. The freezing script itself does not
upload data.

## Contents and checks

The archive contains the frozen response matrix, judge manifest, normalized Qwen
verdicts, all 52 original InFoBench response shards, an artifact manifest, the
criterion-count reconciliation, and per-file SHA-256 checksums. The builder verifies
the following before producing the archive:

- 52 models and 2,250 criteria form exactly 117,000 matrix cells;
- each normalized verdict agrees with the corresponding matrix cell;
- the matrix, verdicts, judge manifest, and response shards have the same model
  roster;
- every response shard has 500 unique InFoBench scenario rows;
- the Qwen provenance is the frozen `generic-binary`, canonical judge path;
- the known input hashes match; and
- the criterion counts reconcile end to end.

The reconciled criterion flow is:

1. `2,250 - 145 all-fail = 2,105 fitted`;
2. `2,105 - 9 nonpositive-discrimination = 2,096 exported`;
3. therefore 154 source criteria were excluded from the provisional fitted-only
   export.

The 27 Qwen `no_decision` cells remain missing in both the normalized verdict table
and response matrix. They are preserved rather than converted to failures.

## Rebuild

```bash
.venv/bin/python scripts/freeze_infobench_phase0_inputs.py build \
  --response-archive /path/to/full200_results.zip \
  --calibration-archive /path/to/InFoBench.zip \
  --output-dir runs/calibration/InFoBench_remediation_phase0 \
  --provenance-out \
    reports/infobench_calibration_remediation/phase0_input_artifact.json
```

The source archives are pinned by default to the hashes from the original run. The
archive uses sorted members, read-only normalized tar metadata, and a zero gzip
timestamp, so rebuilding from the same inputs and code yields the same archive hash.

## Verify

```bash
.venv/bin/python scripts/freeze_infobench_phase0_inputs.py verify \
  --archive \
    runs/calibration/InFoBench_remediation_phase0/infobench_phase0_inputs.tar.gz \
  --expected-archive-sha256 \
    b033f86f0e206197bfff80c057a5793563989b04ab4062f4e7aa476718c62ca3
```
