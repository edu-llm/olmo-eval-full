# EduLLM–OLMo adaptive feasibility spike

## Verdict

- Adaptive CAT core on OLMo provider contracts: **feasible**
- Production-ready native OLMo integration today: **no**
- Recommended architecture now: **two-mode dispatcher with standard OLMo native execution and an EduLLM adaptive adapter; continue the integrated plugin only after provider-lifecycle, artifact, and discovery gates pass**

The three-scenario CPU-only experiment made tutor and judge calls through OLMo's
inference-provider interface. The first verdict changed the next selected scenario,
EAP and MWLE matched recorded, provenance-bearing legacy-derived regression values,
and malformed judge output stayed `no_decision` without updating ability.

## Gate results

| Gate | Result |
| --- | --- |
| legacy provenance resolves | PASS |
| fixture definition matches recorded oracle | PASS |
| sequential selection uses prior judgments | PASS |
| distinct tutor and judge providers | PASS |
| eap matches recorded legacy regression | PASS |
| mwle matches recorded legacy regression | PASS |
| no decision is not a failure | PASS |
| adaptive sidecars round trip losslessly | PASS |
| evaluation names align | PASS |
| two mode dispatcher is extensible | PASS |
| standard external metrics preserve adaptive metadata | FAIL |
| external runner owns judge lifecycle | FAIL |
| third party mode plugin is auto discovered | FAIL |

## Observed paths

- Strong trajectory: `mid -> hard -> easy`
- Weak trajectory: `mid -> easy -> hard`
- Strong EAP theta: `0.283622568`
- Weak EAP theta: `-0.283622568`
- Strong MWLE theta: `0.499999796`
- Weak MWLE theta: `-0.499999796`

## Scope and limitations

This was an interface-feasibility experiment, not a validation of the full scientific
study. It used one latent dimension, three already-calibrated scenarios, one criterion
per scenario, and mock providers. It did not refit item parameters, run a real Qwen or
tutor model, allocate GPUs, or test multidimensional recovery. Those behaviors remain
covered by the legacy EduLLM studies and must be migrated with their tests.

The spike used deterministic top-1 information selection to make branching easy to
characterize. It did not reproduce or validate the legacy default seeded-random choice
among the top five scenarios.

The regression oracle records its source commit, file paths, and Git blob hashes in
`scripts/internal/edullm_olmo_legacy_regression.json`. Matching it is compatibility
evidence, not independent validation of the legacy method.

## Remaining integration work

The prototype injects the judge lookup manually. OLMo's external runner still needs to
own the auxiliary provider lifecycle/GPU planning, preserve rich adaptive metadata in
its standard result storage, and support discoverable external mode plugins. Until
those gates pass, the safe product design is one two-mode dispatcher with standard
OLMo delegated natively and EduLLM adaptive execution kept behind its own adapter.

The branch audit also found that the legacy live verdict property maps every non-pass
label to zero. The production migration therefore needs the explicit tri-state boundary
demonstrated here so `no_decision` is skipped rather than treated as a failed criterion.
