# P3 — Dedicated results namespace (`olmo-eval-results/*`)

> **NOTHING HERE IS APPLIED YET.** The scripts remain on the interim `smoke` prefix
> until the scoped `olmo-eval-results/*` bucket-policy grant lands. Switching the
> prefix before the grant exists would break writes (the same `AccessDenied` wall
> the smoke test hit). Apply order: settle the execution identity (§3.7) → obtain
> the Principal ARN(s) → apply the bucket policy (Artifact 1) → run the re-verify
> (Artifact 3) → *then* apply the config diff (Artifact 2).

This document persists the authoring-only artifacts for P3 (scoping doc
`CHECKPOINT_LAUNCHER_SCOPING.md` §5, §3.7, §9). No AWS was called to produce it.

---

## 1. Overview

- **Bucket:** `edullm-adaptive-inference-056956104102` (region `us-east-1`) — the single
  team results bucket, defaulted in both scripts (`run_checkpoint_eval.sh:90`,
  `run_checkpoint_batch.sh:138`).
- **Current results prefix:** `S3_PREFIX=smoke` (`run_checkpoint_eval.sh:91`,
  `run_checkpoint_batch.sh:139`), with `S3_GROUP=checkpoints`. Results currently land at
  `s3://edullm-adaptive-inference-056956104102/smoke/checkpoints/step_N/`. This is the IAM
  workaround from §5: the node role `EswManagedInstance` only has `s3:PutObject` on
  `smoke/* | smoke_split/* | full200/*`, so results were parked under `smoke/`.
- **Target:** dedicated `olmo-eval-results/*` namespace (§1.4, §5 option B, §9 P3) with a
  proper scoped bucket-policy grant.
- **Files the proposed diff touches:** `tests/aws/run_checkpoint_eval.sh` and
  `tests/aws/run_checkpoint_batch.sh` (one `S3_PREFIX` default each). Every other prefix —
  tar-staging, `_batch/`, `step_N/`, `_SUCCESS`/`metrics.json` — is derived from
  `$S3_PREFIX`, so it follows automatically.

---

## 2. Scoped S3 bucket-policy JSON (`olmo-eval-results/*`, least-privilege)

The `Principal` is a fill-in placeholder because the exact identity depends on the
unresolved §3.7 execution-identity decision (see §6). This grants write + read + list
scoped to the dedicated namespace only.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "OlmoEvalResultsObjectRW",
      "Effect": "Allow",
      "Principal": {
        "AWS": "<LAUNCHER_OR_SERVICE_PRINCIPAL_ARN>"
      },
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::edullm-adaptive-inference-056956104102/olmo-eval-results/*"
    },
    {
      "Sid": "OlmoEvalResultsList",
      "Effect": "Allow",
      "Principal": {
        "AWS": "<LAUNCHER_OR_SERVICE_PRINCIPAL_ARN>"
      },
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::edullm-adaptive-inference-056956104102",
      "Condition": {
        "StringLike": {
          "s3:prefix": [
            "olmo-eval-results",
            "olmo-eval-results/*"
          ]
        }
      }
    }
  ]
}
```

**Action rationale (tied to what the scripts actually do):**

- `s3:PutObject` — `olmo-eval run --s3-*` uploads `metrics.json`; the driver writes
  `_SUCCESS`, `_batch/{worker_i,summary,todo}.json`, and exfils `eval.log` on failure.
- `s3:GetObject` — laptop/aggregator downloads worker shards for the P2.d merge; fleet
  workers read `_batch/todo.json`.
- `s3:ListBucket` (prefix-conditioned) — the skip-if-done gate and `verify_step` run
  `aws s3 ls --recursive <prefix>` (`run_checkpoint_batch.sh:778,1038`;
  `run_checkpoint_eval.sh:421,645`).
- `s3:DeleteObject` — tar-mode staging cleanup `aws s3 rm "$STAGING_URI"`
  (`run_checkpoint_eval.sh:664`); `STAGING_KEY="${S3_PREFIX}/staging/olmo-eval.tgz"` moves
  under the new prefix too. Matches the §5-B pre-verified `PutObject/GetObject/DeleteObject`
  shape.

**Which principal needs what (both plausibly need access):**

- **Node role `arn:aws:iam::056956104102:role/EswManagedInstance`** (machine identity,
  §3.7.1) — needs **all four** actions. In the fleet (P2.c) path, the *node itself* runs
  `aws s3 ls` (skip/verify), `aws s3 cp` (metrics, `_SUCCESS`, shard manifests, `eval.log`),
  and reads `todo.json`. This is the identity that hit the original `smoke/*`-only
  `AccessDenied`.
- **Launcher/orchestrator principal** (§3.7.2/§3.7.4) — in the K=1 laptop-driven path the
  *operator/launcher* runs the skip-if-done ls, `verify_step`, `write_success`, and
  `write_manifests`; in the fleet path it downloads shards for aggregation. So it needs
  `ListBucket` + `GetObject` + `PutObject` (+ `DeleteObject` for staging cleanup).

If a single service principal ends up running both orchestration and (indirectly) the node
writes, one statement covers it; if node and launcher stay distinct identities, the policy
should list **both** ARNs (or add a second statement) — hence the placeholder.

---

## 3. Proposed config diff (PROPOSAL — NOT applied)

Both scripts thread `--s3-prefix ${S3_PREFIX}` into `olmo-eval run`
(`run_checkpoint_eval.sh:590`, `run_checkpoint_batch.sh:997,1464`), so switching the single
default flips the whole namespace. **Do not apply until the grant lands** — writes would
break on `smoke`→`olmo-eval-results` before the policy exists.

### `tests/aws/run_checkpoint_eval.sh` (lines 87–91)

Current:

```bash
# S3_PREFIX defaults to 'smoke' because the EswManagedInstance role's existing s3:PutObject
# grant covers only smoke/* (+ smoke_split/*, full200/*). Results land under
# smoke/checkpoints/ so auto-upload succeeds with NO IAM change (scoping doc §5 interim).
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-smoke}"
```

Proposed:

```bash
# S3_PREFIX defaults to the dedicated 'olmo-eval-results' namespace (scoping doc §5 option B / P3).
# Requires the scoped bucket-policy grant on olmo-eval-results/* to be applied first; until then
# override S3_PREFIX=smoke to stay on the interim EswManagedInstance smoke/* grant.
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-olmo-eval-results}"
```

### `tests/aws/run_checkpoint_batch.sh` (lines 137–139)

Current:

```bash
# --- S3 result namespace (matches the atom's §5 grant story) ------------------
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-smoke}"            # smoke/* is the covered interim prefix (§5)
```

Proposed:

```bash
# --- S3 result namespace (matches the atom's §5 grant story) ------------------
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-olmo-eval-results}" # dedicated namespace (§5 B / P3); needs the scoped grant first
```

### Sibling prefixes that follow automatically (no separate edit; confirm covered by the grant)

All derive from `$S3_PREFIX`, so they inherit the new value:

- Tar-staging key `STAGING_KEY="${S3_PREFIX}/staging/olmo-eval.tgz"`
  (`run_checkpoint_eval.sh:153`, `run_checkpoint_batch.sh:226`) → `olmo-eval-results/staging/…`
  (needs Put + Delete, covered above).
- Per-step prefix `step_prefix()` and result group (`run_checkpoint_batch.sh:757–758`), and
  the atom's `RESULT_PREFIX` (`run_checkpoint_eval.sh:155`).
- Done-markers `_SUCCESS`/`metrics.json` and `_batch/{worker_i,summary,todo}.json`
  (`run_checkpoint_batch.sh:1159,1330–1331,1436,1802,1912`).
- The fleet node-loop CONFIG header `S3_PREFIX='${S3_PREFIX}'`
  (`run_checkpoint_batch.sh:1337`) is driver-substituted, so it inherits the new value.

### Explicitly NOT to change

- `tests/aws/run_rung1_smoketest.sh` and `tests/aws/rung1_smoketest.sh` also default
  `S3_PREFIX=smoke`, but those are the genuine Rung-1 smoke test and *should* stay on
  `smoke/*`. Out of P3 scope.
- `S3_GROUP` stays `checkpoints`.

### Full-consistency note for the sibling write paths (§1.5 / §5)

When `diagnostics/mcq_cat/runner.py` is wired as the second driver command (P3.5), its
`--s3-out` argument writes via raw `boto3.put_object` and is subject to the identical grant —
it must also be pointed at `s3://edullm-adaptive-inference-056956104102/olmo-eval-results/…`.
Same for `tests/OnNode/checkpoint_infer.py`'s default `checkpoint-infer/` prefix. These live
outside the two scripts and are not part of this diff, but need the same target for a single
clean namespace.

---

## 4. Post-grant one-shot re-verify command

**Run this ONLY after the `olmo-eval-results/*` bucket-policy grant is applied.** It confirms
write (and cleans up) against the new namespace. Uses the same profile/region the scripts
default to.

```bash
# --- run ONLY after the olmo-eval-results/* bucket-policy grant is applied ---
BUCKET=edullm-adaptive-inference-056956104102
KEY=olmo-eval-results/_grant_check/$(date -u +%Y%m%dT%H%M%SZ)-$$.txt

# 1) write a tiny object under the new prefix (this is the real AccessDenied test)
printf 'olmo-eval-results grant OK\n' | \
  aws s3 cp - "s3://${BUCKET}/${KEY}" --profile sbsandbox --region us-east-1

# 2) read + list back to confirm Get/List too
aws s3 cp "s3://${BUCKET}/${KEY}" - --profile sbsandbox --region us-east-1
aws s3 ls "s3://${BUCKET}/olmo-eval-results/_grant_check/" --profile sbsandbox --region us-east-1

# 3) cleanup the probe object (confirms DeleteObject)
aws s3 rm "s3://${BUCKET}/${KEY}" --profile sbsandbox --region us-east-1
```

If step 1 succeeds (no `AccessDenied`), the switch in §3 is safe to apply. Under a
service/instance-role identity, drop `--profile sbsandbox` and let creds resolve ambiently.

---

## 5. Grant-request blurb (paste to the bucket-policy owner)

> **Request: scoped S3 bucket-policy grant for a dedicated eval-results namespace.**
> Please add an additive, prefix-scoped statement to the bucket policy on
> **`edullm-adaptive-inference-056956104102`** (us-east-1) granting **`s3:PutObject`,
> `s3:GetObject`, `s3:DeleteObject`** on
> **`arn:aws:s3:::edullm-adaptive-inference-056956104102/olmo-eval-results/*`**, plus
> **`s3:ListBucket`** on the bucket with a condition limiting `s3:prefix` to
> `olmo-eval-results` / `olmo-eval-results/*`. This moves retroactive OLMo checkpoint-eval
> results out of the interim `smoke/*` workaround into a clean, dedicated namespace (scoping
> doc §5 option B / P3). **The Principal ARN is TBD** pending our execution-identity decision
> (§3.7): it will be either the node role
> `arn:aws:iam::056956104102:role/EswManagedInstance` and/or the orchestrator's
> service/launcher role — we'll supply the exact ARN(s) before you apply. The grant is low
> blast radius (additive, prefix-scoped to our own results bucket) and reversible.
> Ready-to-apply JSON attached (§2 above).

---

## 6. Execution-identity coupling note (§3.7)

The policy `Principal` is a fill-in because P3's grant is gated on the **still-open §3.7
launcher-identity decision (open question #9, due at P2)**: whether the orchestrator
authenticates as a **central service/CI machine identity** (users submit a job, need zero
AWS creds — §3.7.3 option 1, preferred) or a **dedicated assumable launcher role** (§3.7.3
option 2). Two coupling facts drive this:

- **Who writes to `olmo-eval-results/*` differs by identity model.** In the K=1 laptop path
  the *operator* writes `_SUCCESS`/manifests and does the verify; in the P2.c fleet path the
  *node role* (`EswManagedInstance`) does those writes on-box. §3.7.3 explicitly wants the
  batch driver designed for a **single service principal** that "is the single principal that
  writes there," which would collapse this to one ARN. Until that's decided, we can't name
  the definitive Principal — hence the placeholder and the "both may need it" split in §2.
- **Node-role read gap is a related prerequisite (§3.7.1).** Independent of the results-write
  grant, the node role also needs S3 **read** on the checkpoint *source* buckets — a sibling
  role-policy fix, not part of this bucket policy.

**What unblocks finalizing it:** settle §3.7 (service role vs. assumable launcher role) →
obtain the concrete principal ARN(s) → drop them into the policy → apply → run the §4
re-verify → then (and only then) apply the §3 prefix switch.
