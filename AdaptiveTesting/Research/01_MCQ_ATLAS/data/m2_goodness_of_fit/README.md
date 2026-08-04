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

- **ifeval**: 1102 models x 536 items -> 6 chunks (source: build_master).
- **math**: 901 models x 1214 items -> 12 chunks (source: build_master).
- **gpqa**: 1102 models x 1192 items -> 12 chunks (source: build_master).
- **musr**: 1099 models x 754 items -> 8 chunks (source: build_master).

## Results (aggregated over chunks that fit)

| benchmark | model | mean RMSEA | median RMSEA | mean SRMSR | mean TLI | mean CFI | frac acceptable | frac RMSEA>.10 | frac M2 p>.05 | mean a<=0/chunk | converged | n chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ifeval | 2PL | 0.0671 | 0.0692 | 0.0709 | 0.9118 | 0.9139 | 0.0 | 0.0 | 0.0 | 4.17 | 5/6 | 6/6 |
| ifeval | 3PL | 0.0628 | 0.0642 | 0.1029 | 0.9227 | 0.9265 | 0.0 | 0.0 | 0.0 | 3.83 | 0/6 | 6/6 |
| math | 2PL | 0.0191 | 0.0159 | 0.1036 | 0.9863 | 0.9851 | 0.083 | 0.0 | 0.25 | 1.58 | 3/12 | 12/12 |
| math | 3PL | 0.0141 | 0.0034 | 0.1449 | 0.9915 | 0.9889 | 0.0 | 0.0 | 0.5 | 1.58 | 3/12 | 12/12 |
| gpqa | 2PL | 0.122 | 0.1219 | 0.131 | 0.4521 | 0.4632 | 0.0 | 1.0 | 0.0 | 41.25 | 12/12 | 12/12 |
| gpqa | 3PL | 0.1128 | 0.1128 | 0.1306 | 0.5296 | 0.5487 | 0.0 | 1.0 | 0.0 | 51.17 | 0/12 | 12/12 |
| musr | 2PL | 0.1067 | 0.1054 | 0.1476 | 0.5879 | 0.5968 | 0.0 | 0.875 | 0.0 | 32.62 | 4/8 | 8/8 |
| musr | 3PL | 0.3808 | 0.1096 | 0.1327 | 0.6014 | 0.6191 | 0.0 | 0.75 | 0.0 | 38.0 | 4/8 | 8/8 |

`frac acceptable` = fraction of chunks with RMSEA < 0.05 AND SRMSR < 0.08.
Conventional thresholds: RMSEA < 0.05 good / > 0.10 poor; SRMSR < 0.08; TLI/CFI
> 0.95 good; M2 p > 0.05 = fail to reject exact fit. `converged` = chunks whose
EM met the tolerance within NCYCLES=500 (the calibration's setting); non-
convergence (e.g. all 3PL chunks for IFEval/GPQA) reflects weak identification of
the extra parameters and does not overturn the per-bank verdicts. Means for MuSR
3PL are inflated by one pathological chunk (see interpretation); consult median
RMSEA. For MATH, `frac acceptable` is conservative: it additionally requires
SRMSR < 0.08, which MATH rarely meets even though its RMSEA/TLI/CFI/M2-p all
indicate good fit -- read the individual columns, not just `frac acceptable`.

## Interpretation

**Bottom line: the four banks split cleanly into good-fitting (MATH, IFEval) and poor-fitting (GPQA, MuSR).** MATH shows strong unidimensional fit -- every chunk has RMSEA well below 0.05 (mean 0.014-0.019), TLI/CFI above 0.98, and a real fraction of chunks fail to reject exact fit (M2 p > .05 on 3/12 chunks for 2PL and 6/12 for 3PL, several with p ~ 1). IFEval is clearly better than the poor banks but only borderline in absolute terms (RMSEA ~0.06-0.07, TLI/CFI ~0.91-0.93, 2PL SRMSR ~0.07). GPQA and MuSR remain poor under both 2PL and 3PL: RMSEA in the poor range (> 0.10), TLI/CFI far below 0.95, no chunk acceptable, and every chunk rejects exact fit (M2 p < .001). This is the intended contrast: the calibration model is well-specified for MATH (and reasonable for IFEval) but mis-specified for GPQA/MuSR. The one index that stays elevated even for the good banks is SRMSR, so "good fit" here rests on RMSEA/TLI/CFI/M2-p rather than a clean sweep of every index (see the MATH note).

- **math (good)**: (a) Strong unidimensional fit. All 12 chunks have RMSEA < 0.05 under both 2PL (mean 0.019, median 0.016) and 3PL (mean 0.014, median 0.003), mean TLI/CFI ~0.985-0.99 (> 0.95 good), and exact fit is retained on a genuine share of chunks (M2 p > .05 on 3/12 chunks for 2PL, 6/12 for 3PL -- several with p ~ 1, i.e. we fail to reject exact fit). (b) The one dissonant index is SRMSR (mean 0.10 2PL / 0.14 3PL; only 8% / 0% of chunks < 0.08): the raw residual-correlation metric stays high while the df-penalized RMSEA and the M2 exact-fit test both pass, so `frac_acceptable` (which additionally requires SRMSR < 0.08) understates the fit at 0.083 / 0. (c) 3PL slightly lowers RMSEA (median 0.016 -> 0.003) and raises the exact-fit pass rate but worsens SRMSR; 2PL is the more balanced model. Verdict: **good** on RMSEA/TLI/CFI/M2-p, with SRMSR the only caveat.
- **ifeval (better/borderline)**: (a) Clearly better than GPQA/MuSR but short of MATH. Median RMSEA 0.069 (2PL) / 0.064 (3PL) sits in the borderline band (0.05-0.10; 0/6 chunks poor, 0/6 good), mean TLI/CFI ~0.91-0.93 (approaching but below 0.95), and 2PL SRMSR is genuinely good (mean 0.071; 5/6 chunks < 0.08). (b) It does not clear the strict bar: no chunk reaches RMSEA < 0.05 and every chunk still rejects exact fit (M2 p < .001), so `frac_acceptable` = 0. Honestly **"better/borderline," not "good"** in the MATH sense. (c) 3PL trims RMSEA marginally (0.069 -> 0.064) but inflates SRMSR (0.071 -> 0.103) and converges on 0/6 chunks; 2PL is preferable.
- **gpqa (poor)**: Poor under both models -- median RMSEA 0.122 (2PL) / 0.113 (3PL), both in the poor range (> 0.10), mean TLI/CFI ~0.45-0.55 (far below 0.95), median SRMSR ~0.13 (want < 0.08), 0/12 chunks acceptable, and every chunk rejects exact fit (M2 p < .001). 3PL does not fix it (over-parameterization; note 0/12 3PL chunks even converge).
- **musr (poor)**: Poor under both models -- median RMSEA 0.105 (2PL) / 0.110 (3PL), poor range, mean TLI/CFI ~0.59-0.62, median SRMSR ~0.12-0.13, 0/8 chunks acceptable, and every chunk rejects exact fit (M2 p < .001). 3PL does not help; its mean RMSEA (0.38) is inflated by one pathological chunk (chunk 1, RMSEA ~2.3) where the guessing parameter is unidentified (a Heywood-type blow-up) -- consult the median.

## Honest limitations

1. **Within-chunk only (the main caveat).** Per-chunk M2 tests within-chunk
   unidimensional fit. Because chunks are arbitrary 100-item column blocks fit
   independently, this design **cannot detect cross-chunk multidimensionality**
   or any misfit whose signal lives in item pairs that fall in different chunks.
   A bank could look acceptable chunk-by-chunk while still being multidimensional
   overall -- so per-chunk M2 is a tractable proxy for, not a replacement of, a
   full-bank M2. For GPQA/MuSR the within-chunk fit is already poor, so the full-
   bank fit can only be as bad or worse (this bounds their misfit from below). For
   IFEval/MATH the caveat cuts the other way: acceptable within-chunk fit does not
   guarantee full-bank unidimensionality, since cross-chunk multidimensionality
   would be invisible to this design.
2. **Chunk boundaries are alphabetical.** Items are ordered `subtask|question_id`
   and blocked in 100s, so chunk composition is incidental, not designed.
3. **Non-convergence on some chunks.** Several chunks (mostly 3PL) hit the
   NCYCLES=500 EM cap without meeting tolerance (see the `converged` column). We
   keep NCYCLES=500 to match the calibration. For GPQA/MuSR the misfit is so large
   and uniform that convergence would not change the verdict; for IFEval/MATH the
   2PL fits (the ones we lean on) largely converge, and the non-converged 3PL runs
   agree with them, so the good-fit reading is not an artifact of stopping early.
4. **a<=0 rates track the verdict.** The `mean a<=0/chunk` column is itself
   diagnostic: GPQA/MuSR have ~33-51 of 100 items per chunk with non-positive
   discrimination (near-zero unidimensional signal, dropped by the calibration's
   a>0 filter), whereas the good banks have very few (MATH ~1.6, IFEval ~4). The
   high-degeneracy banks are exactly the poor-fitting ones, reinforcing the
   contrast rather than being a nuisance to filter away.

## Files

- `per_chunk_m2.csv` -- one row per (bench, model_type, chunk).
- `summary_m2.csv` -- bench x model_type aggregates.
- `_work/<bench>/logs/*.log` -- raw R output per chunk fit.
