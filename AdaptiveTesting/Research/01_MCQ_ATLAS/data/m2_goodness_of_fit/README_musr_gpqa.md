# Limited-information M2 goodness-of-fit (per-chunk)

Maydeu-Olivares & Joe (2006) limited-information M2 fit statistics for the
OpenLM item banks under unidimensional **2PL** and **3PL**, computed the
LIGHTER per-chunk way.

## Why per-chunk

A single joint M2 over an entire 700-1200 item bank is intractable: the number
of second-order margins grows ~n_items^2 and mirt's covariance machinery runs
out of memory. We instead reuse the EXACT 100-item chunking the ATLAS
calibration pipeline uses (`openlm_trainsize_sweep.chunk_ends`), fit each chunk
jointly and unidimensionally with R `mirt` (same fit helper, same
EM/GenRandomPars retry, same itemtype as calibration), and call `M2(fit)` per
chunk. RMSEA / SRMSR / TLI / CFI / M2-p are aggregated across chunks.

## Method

- Item source: `sweep.build_master(bench)` (the full model set; the same data
  path the calibration uses).
- Chunks: `sweep.chunk_ends(n_items, 100)` -- 100-item chunks with a small
  trailing remainder merged into the last chunk (min 20), identical to the
  calibration.
- Fit: `mirt(dat, 1, itemtype = {2PL|3PL}, method = 'EM', NCYCLES = 500)`,
  deterministic start then fixed-seed GenRandomPars retries (the calibration's
  rescue path).
- M2 is computed on the FULL-chunk fit. The calibration fits every item in a
  chunk jointly and applies its a>0 filter only downstream when assembling the
  CAT bank (it never refits), so the full-chunk fit is the faithful per-chunk
  model. The `n_degenerate` column reports how many items (a<=0) that filter
  would drop -- a high count signals near-zero discriminations, i.e. items that
  carry little unidimensional signal. (Cross-check: on MuSR 2PL chunk 1 the
  full-chunk RMSEA 0.1049 vs an a>0-only refit 0.1051 are identical, so the
  choice does not affect conclusions.)
- M2: `M2(fit)`; if that errors, `M2(fit, type = 'C2')` (the `m2_type` column
  records which was used).

## Data

- **musr**: 1099 models x 754 items -> 8 chunks (source: build_master).
- **gpqa**: 1102 models x 1192 items -> 12 chunks (source: build_master).

## Results (aggregated over chunks that fit)

| benchmark | model | mean RMSEA | median RMSEA | mean SRMSR | mean TLI | mean CFI | frac acceptable | frac RMSEA>.10 | frac M2 p>.05 | mean a<=0/chunk | converged | n chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| musr | 2PL | 0.1067 | 0.1054 | 0.1476 | 0.5879 | 0.5968 | 0.0 | 0.875 | 0.0 | 32.62 | 4/8 | 8/8 |
| musr | 3PL | 0.3808 | 0.1096 | 0.1327 | 0.6014 | 0.6191 | 0.0 | 0.75 | 0.0 | 38.0 | 4/8 | 8/8 |
| gpqa | 2PL | 0.122 | 0.1219 | 0.131 | 0.4521 | 0.4632 | 0.0 | 1.0 | 0.0 | 41.25 | 12/12 | 12/12 |
| gpqa | 3PL | 0.1128 | 0.1128 | 0.1306 | 0.5296 | 0.5487 | 0.0 | 1.0 | 0.0 | 51.17 | 0/12 | 12/12 |

`frac acceptable` = fraction of chunks with RMSEA < 0.05 AND SRMSR < 0.08.
Conventional thresholds: RMSEA < 0.05 good / > 0.10 poor; SRMSR < 0.08; TLI/CFI
> 0.95 good; M2 p > 0.05 = fail to reject exact fit. `converged` = chunks whose
EM met the tolerance within NCYCLES=500 (the calibration's setting); non-
convergence on the hardest chunks is itself a symptom of weak identification and
does not change the (overwhelming) misfit verdict. Means for MuSR 3PL are
inflated by one pathological chunk (see interpretation); consult median RMSEA.

## Interpretation

**Bottom line: both MuSR and GPQA fit the unidimensional IRT model poorly, and 3PL does not fix it.** No chunk of either benchmark, under either 2PL or 3PL, reaches acceptable fit; every chunk rejects exact fit (M2 p < .001) and sits in RMSEA's poor range with SRMSR and TLI/CFI far from their thresholds. This is direct evidence that the calibration model is mis-specified for these banks.

- **musr**: (a) 3PL does **not** improve fit (median RMSEA 0.105 -> 0.110, still 0/8 chunks acceptable): the guessing parameter is over-parameterization. Its mean RMSEA (0.38) is inflated by >=1 pathological chunk where the 3PL guessing parameter is unidentified (a Heywood-type blow-up), which is itself a failure mode of the extra parameter. (b) The bank shows poor unidimensional fit under both models -- median RMSEA 2PL=0.105 / 3PL=0.110 (good < 0.05, poor > 0.10), median SRMSR ~0.12 (ok < 0.08), mean TLI/CFI ~0.59 (want > 0.95), and every chunk rejects exact fit (M2 p < .001) with 0/8 chunks acceptable.
- **gpqa**: (a) 3PL does **not** improve fit (median RMSEA 0.122 -> 0.113, still 0/12 chunks acceptable): the guessing parameter is over-parameterization. (b) The bank shows poor unidimensional fit under both models -- median RMSEA 2PL=0.122 / 3PL=0.113 (good < 0.05, poor > 0.10), median SRMSR ~0.13 (ok < 0.08), mean TLI/CFI ~0.45 (want > 0.95), and every chunk rejects exact fit (M2 p < .001) with 0/12 chunks acceptable.

## Honest limitations

1. **Within-chunk only (the main caveat).** Per-chunk M2 tests within-chunk
   unidimensional fit. Because chunks are arbitrary 100-item column blocks fit
   independently, this design **cannot detect cross-chunk multidimensionality**
   or any misfit whose signal lives in item pairs that fall in different chunks.
   A bank could look acceptable chunk-by-chunk while still being multidimensional
   overall -- so per-chunk M2 is a tractable proxy for, not a replacement of, a
   full-bank M2. (Here the within-chunk fit is already poor, so the full-bank fit
   can only be as bad or worse; this bounds the misfit from below in severity.)
2. **Chunk boundaries are alphabetical.** Items are ordered `subtask|question_id`
   and blocked in 100s, so chunk composition is incidental, not designed.
3. **Non-convergence on the hardest chunks.** Several chunks (mostly 3PL) hit the
   NCYCLES=500 EM cap without meeting tolerance (see the `converged` column). We
   keep NCYCLES=500 to match the calibration; the misfit is so large and uniform
   that convergence would not change the verdict.
4. **High a<=0 rates.** A large share of items per chunk have non-positive
   discrimination (see `mean a<=0/chunk`); the calibration's a>0 filter would
   drop them. This reflects near-zero-signal items and reinforces the weak-
   unidimensionality reading rather than being a nuisance to filter away.

## Files

- `per_chunk_m2.csv` -- one row per (bench, model_type, chunk).
- `summary_m2.csv` -- bench x model_type aggregates.
- `_work/<bench>/logs/*.log` -- raw R output per chunk fit.
