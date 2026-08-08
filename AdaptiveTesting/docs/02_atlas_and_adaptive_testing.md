# ATLAS + the adaptive-testing work already in this repo

Scope: what ATLAS is, the vendored ATLAS repo, the teammate's experiments already
committed under `AdaptiveTesting/`, the standalone inference harness, and every
data schema + id bridge the integration depends on.

Upstream: [Peiyu-Georgia-Li/ATLAS](https://github.com/Peiyu-Georgia-Li/ATLAS)
(ICML 2026 spotlight, [arXiv 2511.04689](https://arxiv.org/abs/2511.04689)).

---

## 1. The ATLAS method

ATLAS replaces a static benchmark score with a **psychometric** one:

1. **Calibrate** a **3PL IRT** model on a big **model × item** binary response
   matrix (rows = LLMs, cols = benchmark questions, cell = correct?). Fit in
   chunks, then link chunks to a common scale via mean–σ transforms.
2. **WLE θ** — full-test ability estimate per model = ground-truth θ.
3. **Adaptive testing (CAT)** — for a new model, start at θ=0 and repeatedly pick
   the item with **maximum Fisher information** at the current θ; re-estimate θ
   after each; **stop when SE(θ) ≤ threshold** (0.1 precise → 0.3 fast). Reaches a
   stable θ with a small subset (headline: full HellaSwag from 41 / 5,608 items).
4. **p-IRT accuracy** — reconstruct full-benchmark accuracy from the observed
   subset + IRT-predicted probabilities for the unseen items.

### 3PL and why we use it

$$P(\text{correct}\mid\theta)=g+(1-g)\,\sigma\!\big(a(\theta-b)\big)$$

- `a` discrimination, `b` difficulty, `g` pseudo-guessing (lower asymptote).
- **We use 3PL, not 2PL**: there are enough calibration models (published banks:
  ARC 4,162 models / GSM8K 4,195 / HellaSwag 3,467 / TruthfulQA 4,635 /
  WinoGrande 4,680) to estimate the guessing parameter stably. `g` matters for
  MCQ where a weak model still gets ~1/n by chance; 2PL forces `g=0` and
  mis-fits easy/low-discrimination items. The teammate's `local_diagnostic_arc`
  experiment already runs both 2PL and 3PL for comparison.

ATLAS stores mirt-style params `(a1, d, g, u)`; convert to CAT form with
`a = a1`, `b = -d/a1`, `c = g`.

---

## 2. What is already in `AdaptiveTesting/`

The teammate has vendored ATLAS and built out calibration + validation. This is
**further along than a green field** — the CAT math is already implemented in
Python.

```
AdaptiveTesting/
├── Inputs/
│   ├── ATLAS/                         # vendored ATLAS repo (R pipeline + published banks)
│   │   ├── scripts/                   # 01_fit_irt.r … 05_compute_actual_acc.r (+ *_custom.r)
│   │   ├── arc/
│   │   │   ├── irt_item_parameters_combined.csv   # PUBLISHED 3PL bank (a1,d,g,u)
│   │   │   ├── atlas_idx_to_question_id.csv        # ★ item-index → ARC question_id bridge
│   │   │   ├── irt_person_scores_WLE_SE_test.csv   # ground-truth θ
│   │   │   ├── actual_accuracy.csv, pirt_accuracy_se_*.csv, pirt_vs_actual_se_*.csv
│   │   ├── arc_0p5_7b/irt_item_parameters_combined.csv  # recalibrated small-model bank
│   │   └── data/gaussian_sampled_arc_response_matrix_{train,test,train_with_scores[_0p5_7b]}.csv
│   ├── Models/models.yaml             # 100-model 0–7B roster (for response-matrix gen)
│   └── Open/LLM-Judge/
│       ├── *.rubric.txt               # Prometheus judge rubrics (open-ended)
│       └── mcq/<benchmark>/<model>.csv# ★ per-model MCQ response CSVs (the response matrix)
├── Experiments/
│   ├── local_diagnostic_arc/          # calibrate 2PL/3PL on OUR matrices, CAT on held-out models
│   ├── atlas_transfer_published/      # PUBLISHED ATLAS 3PL bank → our held-out models
│   ├── atlas_recalibrate_0p5_7b/      # recalibrated bank on 0.5–7B models only
│   └── models_200/                    # size-balanced roster builder
├── Test/Inference/                    # standalone vLLM/HF sweep that PRODUCES the mcq CSVs (see §4)
└── docs/                              # ← these docs
```

### The Python CAT already exists

`Experiments/atlas_transfer_published/atlas_diagnostic_validation.py` is a
complete, dependency-light (numpy) implementation of the ATLAS eval loop:

- `load_atlas_item_bank()` — reads `(a1,d,g)` → `(a,b,c)`.
- `load_atlas_idx_map()` — reads `atlas_idx_to_question_id.csv`.
- `load_our_responses()` — reads `mcq/arc_challenge/*.csv` → per-model 0/1 vector
  aligned to the bank's `question_id`s.
- `prob()`, `eap_se()` — 3PL probability + EAP θ/SE on a fixed θ grid.
- `run_cat()` — **the CAT**: max-Fisher-info selection, EAP re-estimate, stop at
  `SE ≤ SE_STOP` (min 8 / max 200 items).
- `pirt_accuracy()` — observed-subset + predicted-unseen accuracy blend.

```154:171:AdaptiveTesting/Experiments/atlas_transfer_published/atlas_diagnostic_validation.py
def run_cat(resp_all, a, b, c):
    n = len(a); used = np.zeros(n, dtype=bool)
    theta, se, order = 0.0, 1.0, []
    for _ in range(min(MAX_ITEMS, n)):
        p = prob(theta, a, b, c)
        info = (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2
        info[used] = -1.0
        j = int(np.argmax(info)); used[j] = True; order.append(j)
        idx = np.asarray(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        if len(order) >= MIN_ITEMS and se <= SE_STOP:
            break
    return theta, se, order
```

This is effectively the reference implementation to lift into an olmo-eval
result computation. There is also a parallel, more general Python IRT/CAT package
at `eduLLM-Evals/tutor_cat/mcq_irt/` (`matrix.py`, `calibrate.py`, `ability.py`,
`cat.py`, `pipeline.py`, `report.py`) that operates on the same MCQ CSV schema.

### Validation status (important caveat)

The published ATLAS ARC bank was fit on **25-shot Open-LLM-Leaderboard** labels;
our MCQ CSVs are **0-shot log-likelihood**. Held-out transfer over the same 60
models, recomputed from each experiment's own `results/*.csv`:

| Bank | SE stop | r | p-IRT MAE | mean items |
|---|---|---|---|---|
| Published (`arc/`) | 0.2 | **0.830** | 0.147 | 21.1 |
| Published (`arc/`) | 0.3 | **0.738** | 0.172 | 9.9 |
| Recalibrated 0.5–7B (`arc_0p5_7b/`) | 0.2 | 0.589 | 0.147 | 12.5 |
| Recalibrated 0.5–7B (`arc_0p5_7b/`) | 0.3 | 0.585 | 0.150 | 8.3 |

Three things to be clear about, because earlier drafts of this doc got them wrong:

1. **The published bank transfers better than the recalibrated one** (r 0.74–0.83
   vs 0.59). The r≈0.59 figure belongs to `atlas_recalibrate_0p5_7b`, not to the
   published bank.
2. **The recalibration did not address the scoring mismatch.** Its README shows it
   re-fit on `data/gaussian_sampled_arc_response_matrix_train_with_scores_0p5_7b.csv`,
   which is ATLAS's own 25-shot matrix filtered to models with heuristic size in
   [0.5, 7]B. It varied the calibration **population**, not the scoring method, and
   correlation dropped. A true scoring-parity recalibration has not been run.
3. **p-IRT accuracy is not usable as an absolute number yet.** Predicting the
   constant mean accuracy for every model gives MAE 0.084; p-IRT gives 0.147–0.172.
   Predicted accuracy is also range-compressed (sd 0.043–0.054 against an actual sd
   of 0.099). The θ *ranking* is sound; the reconstructed accuracy *level* is not.

The compression has a concrete cause. 7 of the 60 held-out models score **below
four-choice chance** (0.171–0.242), and the bank predicts 0.411–0.521 for exactly
those models. A live run reproduces it: `Qwen/Qwen2.5-0.5B` scores **0.293** on the
full benchmark and p-IRT reconstructs **0.491** (see doc 04 §2e). A 3PL bank cannot represent sub-chance accuracy: the pseudo-guessing
parameter floors every prediction, and this bank's `g` is high (mean 0.264, p90
0.763). Sub-chance results are what 0-shot log-likelihood over choice continuations
produces on weak models, since length and fluency bias pushes them below chance
rather than toward it; 25-shot leaderboard responses rarely go there, so `g` was
never fit against that regime.

Takeaway for planning: **calibrate on responses produced the same way we score at
eval time**, and treat sub-chance behavior as a first-class requirement of whichever
scoring is chosen.

---

## 3. Data schemas (the contracts)

**Published/recalibrated 3PL bank** — `Inputs/ATLAS/**/irt_item_parameters_combined.csv`:

```
"X","a1","d","g","u"
"X1",3.925...,2.856...,0.771...,1        # X<k> = ATLAS 1-based item index k
```

**Item index → question_id** — `Inputs/ATLAS/arc/atlas_idx_to_question_id.csv`:

```
atlas_idx,question_id,question
1,Mercury_SC_410971,"Cities control the amount of pollution ..."
```

`atlas_idx` matches `X<k>`; `question_id` matches ARC's native id and olmo-eval's
`Instance.metadata["id"]`. This CSV was built from leaderboard example order — it
is the artifact that makes ARC alignable today. **Equivalent maps do not yet exist
for the other benchmarks.**

**Effective bank coverage is smaller than the file suggests.** The ARC params CSV
has 839 rows, but `load_bank()` drops items with non-positive discrimination, and
189 of them have `a1 <= 0` (an anti-discriminating fit, where stronger models do
*worse* on the item). That leaves **650 usable items**, against 1,172 questions in
the ARC-Challenge test split, so the CAT sees about 55% of the benchmark. The
bridge itself has 1,172 rows but only 1,170 distinct `question_id`s, so two ids are
duplicated; `ItemBank` keys positions by id, so a duplicate silently resolves to
one position.

**Per-model MCQ response CSV** — `Inputs/Open/LLM-Judge/mcq/<benchmark>/<model>.csv`:

```
question_id,model,benchmark,predicted,gold,result,scoring_method
Mercury_7175875,Qwen/Qwen2.5-0.5B,arc_challenge,C,C,correct,loglikelihood
```

`result ∈ {correct, wrong}` is the 0/1 IRT cell; `question_id` is the join key.
Stacking these files = the model × item response matrix.

**Response matrix (ATLAS native)** — `Inputs/ATLAS/data/gaussian_sampled_arc_response_matrix_*.csv`:
rows = model ids, columns = `X1..Xk` item indices. Used by the R fit scripts and
to recover the ATLAS model roster (to know which models are held-out).

---

## 4. Where the response matrices come from (the harness)

`AdaptiveTesting/Test/Inference/` is a **standalone** vLLM/HF sweep (no `olmo_eval`
import). `run_benchmark.py` loads the 100-model roster (`models_registry.py` +
`Inputs/Models/models.yaml`), normalizes benchmarks (`datasets_registry.py`),
scores MCQ by **log-likelihood** (`mcq_scoring.py`), and writes
`Outputs/mcq/<benchmark>/<slug>.csv` (schema above). It also does open-ended
generation + Prometheus judging (`open_generate.py`, `judge_prometheus.py`). At
scale it runs as an **AWS Batch** array job (`aws/`), not Beaker. The `mcq/` files
under `Inputs/Open/LLM-Judge/` are promoted copies of those outputs.

`question_id` rules (`datasets_registry.py`): ARC/OpenBookQA/SQuAD use native HF
ids; sciq/piqa/boolq/winogrande/hellaswag use `"{prefix}_{index:05d}"` in split
order. **This matters**: only ARC has a ready index→question_id bridge, and
enumeration-based ids are only stable if the HF split order + sample cap are pinned.

The strategic point of the whole integration: this bespoke harness duplicates what
olmo-eval already does (load benchmark, loglik-score MCQ, per-item correctness).
Folding response generation into olmo-eval tasks removes the duplicate harness and
lets the *same* run both score the benchmark and produce the IRT response vector.
