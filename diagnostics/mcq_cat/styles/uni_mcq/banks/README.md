# Locally fitted item banks

`irt_item_parameters_combined.csv` for the three self-calibrated MCQ banks, plus the scripts
that produce them. `vendor_bank` reads these through `git_show(source_ref, spec.params_csv)`,
so they are committed rather than generated at vendoring time — a bank whose parameters exist
only on somebody's disk is a bank nobody else can reproduce.

These are the *inputs* to vendoring. The vendored output lives in `calibrated_datasets/`.

## Why these exist at all

Every prior study of these banks fit them in memory, ran a CAT diagnostic, recorded a
correlation, and discarded the parameters. `run_pl_comparison.py` on the `Research` ref never
writes `a`/`b` to disk, and `fit_diagnostics.csv` there records only aggregates — item counts
and the min/max of `a`, not per-item values. So there was nothing to vendor and the fit had to
be re-run.

## Provenance

| Bank | Models | Raw items | Usable | Fit family | Fitter |
| --- | --- | --- | --- | --- | --- |
| `pedagogy` | 78 | 920 | 379 | 1PL | `girth.rasch_mml` |
| `piqa` | 52 | 1838 | 896 | 2PL | `girth.twopl_mml` |
| `socialiqa` | 52 | 1954 | 1012 | 1PL | `girth.rasch_mml` |

Responses come from the 2026-08-01 sweep at
`s3://edullm-adaptive-inference-056956104102/full200/results/Outputs/mcq/`. Fit family per
bank is a measured decision, not a default; see `MCQ_BANK_VENDORING_PLAN.md` §6.2.

`filter_items` and `fit_bank` are transcribed verbatim from `tutor_cat/mcq_irt/matrix.py` and
`Research/scripts/se_sweep_small_pool.py` rather than reimplemented, because the point is to
reproduce that pipeline rather than to write a better one.

## Column shape

`X, a1, d, g, u`, which is what `vendor_bank.load_bank` expects. `d` is the **mirt intercept,
not a difficulty** — the reader recovers `b = -d / a1`. Difficulty is therefore stored as
`d = -a * b`.

**Every position in the raw enumeration gets a row**, including the ones `filter_items`
dropped, which carry `a1 = 0`. This is deliberate. `check_alignment` aborts unless the bank's
maximum index equals the bridge's row count; it compares the max index rather than the row
count specifically so sparse banks are allowed, but a bank whose *final* item was filtered
would still fail, because its max index falls short of the enumeration length. Pedagogy and
piqa are both in that position. A zero-discrimination row keeps the index space complete
without inventing an item: `load_bank` already drops `a1 <= 0` as
`non_positive_discrimination`, the same path the nine existing banks use, and the count
surfaces in the manifest where a reader can see it.

Note for whoever teaches `check_parameter_family` to recognise `1pl`: the Rasch banks have
`a1 = 1` only among their **usable** rows. The zeroed rows carry `a1 = 0` by design, so a
naive "all a1 == 1" test will misclassify them.

## Regenerating

```bash
# 1. Fetch the response matrices (182 CSVs, ~20 MB).
aws s3 cp s3://edullm-adaptive-inference-056956104102/full200/results/Outputs/mcq/ \
    <matrices-dir>/ --recursive

# 2. Gate: this must print 318 / 831 / 969 for the common-52 train slice before the
#    fit below is worth trusting. Those are the counts pl_1_2_3_comparison published.
python reproduce_check.py --matrices <matrices-dir>

# 3. Fit, then emit the mirt-shaped CSVs in place.
python fit_banks.py --matrices <matrices-dir> --out <work-dir>
python emit_mirt_csv.py --fit-dir <work-dir>
```

Needs `girth`, which is not a project dependency — it is only used here, offline, to produce
a committed artifact. `pip install girth` into a scratch environment.

The 2PL fit for piqa takes about four minutes; both Rasch fits take under a second.

## Sanity checks the fit must pass

- `reproduce_check.py` recovers 318 / 831 / 969 on the study's `rng(7)` 40/12 split. The
  published "52 models" means the models common to all three banks, which is why pedagogy's
  published count comes from 52 of its 78.
- Spearman correlation between difficulty and pass rate is **exactly −1** for both Rasch
  banks — under Rasch, difficulty is a strictly monotone function of pass rate, so anything
  else means the fit is wrong. Piqa's 2PL gives −0.768, where discrimination legitimately
  breaks the monotonicity.
- Both 1PL fits return `a = 1.0` for every usable item.
- `b` recovered as `-d / a1` matches the fitted difficulty to floating-point noise.
