# Research results, data, and figures

This folder collects the experimental evidence for the paper drafted in `Outline.md`,
organized by section. Each subfolder has a `README.md` (the write-up / description), a
`data/` folder (CSV/JSON results), and a `figures/` folder (PNG graphs). Reproducible
scripts for the experiments run for this compilation live in `scripts/`.

## Map to `Outline.md`

| Folder | Outline section(s) | What it contains |
|---|---|---|
| `01_MCQ_ATLAS/` | *MCQ*, *ATLAS Recreation*, *Feasibility*, *Experiments*, *MIRT Discussion (MCQ)* | ATLAS transfer (our responses **and** ATLAS's own prompt-matched responses) & range-restricted recalibration, local 1PL/2PL/3PL diagnostics, train-size / model-count sweeps, an **SE-requirement sweep** (correlation & #items vs SE), OpenLM ATLAS-style 3PL on new benchmarks, and **new** Pedagogy/PIQA/SocialIQA IRT feasibility calibrations. |
| `02_FRQ_Judge/` | *FRQ → LLM-as-judge*, *skills/rubrics* | Judge selection study design, acceptance thresholds, blinded 261-case set, frozen Qwen judge config, skill definitions + Q-matrix banks. |
| `03_FRQ_MIRT/` | *FRQ → MIRT results*, *Limitations* | 82-model 2-skill TutorBench M2PL (existing) **and a new 115-model calibration + CAT** (82 + 33 AWS models), abilities, discriminations, recovery, CAT length, and parameter-SE-vs-N. |
| `04_Inference_Cost/` | *Inference cost* (MCQ feasibility) and *Inference Cost For Tutor Models* (FRQ) | Measured cost & latency per inference vs model size (MCQ and open-ended), plus the per-parameter-range individual-latency table used in the outline. |
| `05_Data_Availability/` | *Availability and Quality of Response Data*, *Parameter Distribution* | Inventory of OpenLM / ATLAS / local response data, model rosters, and parameter distributions, with a discussion of coverage gaps. |

## Headline results (one line each)

- **ATLAS recreation (ARC):** the *published* ATLAS 3PL bank transfers to our model pool at
  Pearson **r=0.83** (SE≤0.2, ~21 items) / **r=0.74** (SE≤0.3, ~10 items). Replaying ATLAS's
  **own** prompt-matched responses (no 0-shot vs 25-shot mismatch) lifts this to **r≈0.91**
  in <10 items — most of the earlier loss was scoring-protocol mismatch, not IRT transfer.
  Restricting calibration to 0.5–7B models drops transfer to **r≈0.59**, quantifying the
  cost of a parameter-range-mismatched, thin calibration pool.
- **SE requirement trade-off:** an SE-sweep shows a well-calibrated bank (ATLAS 3PL) already
  hits r≈0.90 at **~8 items (SE≤0.5)** and plateaus; a thin in-house bank needs a tighter
  SE≤0.3 (~30–45 items) to reach its correlation plateau. #items grows ~5–9× from SE≤0.3 to
  SE≤0.12 for little r gain.
- **New benchmark feasibility (Pedagogy):** fitting our own 2PL on 40 models and running a
  CAT recovers full pedagogy accuracy at **r=0.72** with ~**3%** of items (SE≤0.3); PIQA
  reaches **r=0.94** at ~**1.3%** of items. Tight SE (≤0.15) on a thin, 2PL-only bank needs
  most of the bank (≈87% for pedagogy) — the practical limit that ATLAS's large 3PL pools avoid.
- **FRQ MIRT (new 115-model run):** merging the original 82 models with 33 non-overlapping
  AWS Qwen-judge models gives a 115-model 2-skill TutorBench bank with CAT recovery
  **r=0.97 (correctness) / 0.92 (scaffolding)** and median loading SE **0.34** (vs **0.41** at
  N=82, matched method) — directly demonstrating the outline's "more models shrink parameter SE".
- **Inference cost:** open-ended tutor inference costs ~**$0.20–0.23 per 1,000** on an L4;
  MCQ is ~**20–25× cheaper**; individual (single-stream) latency by parameter range is
  ≈**4 s (0–1B) / 10 s (1–2B) / 19 s (2–5B) / 34 s (5–7B)**.

See `GAPS.md` for what the outline asks for but the data does not (yet) cover.
