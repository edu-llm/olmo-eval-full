# Build notes: CAT for Accelerated Pedagogy Benchmarking

Assembled acmart review draft, built from the staged section drafts in
`AdaptiveTesting/Research/_paper_build/sections/` and bib fragments in
`AdaptiveTesting/Research/_paper_build/refs/`. Nothing outside
`AdaptiveTesting/Research/_paper_build/` was touched, and no git was run.

## acmart options / preamble
```
\documentclass[manuscript,nonacm,review]{acmart}
\settopmatter{printacmref=false}\setcopyright{none}
\renewcommand\footnotetextcopyrightpermission[1]{}
```
Packages: `graphicx, booktabs, amsmath, amssymb, xcolor`; `\graphicspath{{figures/}}`.
Visible AI/review markers: `\aiadd` and `\aibullets` render blue, `\aiflag` (and
literal `\textcolor{red}{...}`) render red. The section files' own
`\providecommand` fallbacks are harmless no-ops because `main.tex` defines the
macros first.

## Section order (flat, as authored)
Abstract is extracted from `00` and placed in `\begin{abstract}...\end{abstract}`
before `\maketitle` (acmart requirement). Body `\input` order:

1. `sections/00_abstract_strategy` — Strategy; supplies `\section{Multiple-Choice Benchmarks}`
2. `sections/10_mcq_atlas_recreation` — `\section{ATLAS Recreation}`
3. `sections/11_mcq_feasibility_data_cost` — `\subsection`-only fragment
4. `sections/12_mcq_experiments` — `\subsection`-only fragment; keeps the red ORACLE placeholder
5. `sections/20_mirt_discussion`
6. `sections/30_recommendations`
7. `sections/40_frq_judge`
8. `sections/41_frq_mirt_results`

Then `\appendix` + `sections/90_appendix_irt`, then
`\bibliographystyle{ACM-Reference-Format}` + `\bibliography{references}`.

Notes:
- `00` was split: the `abstract` environment moved into `main.tex`; the copied
  `00_abstract_strategy.tex` retains only the MCQ "Strategy" content.
- Sections `11` and `12` are authored as `\subsection`-only fragments and nest
  under `10`'s `\section{ATLAS Recreation}`, exactly as their source comments
  request. No heading demotions/promotions were made.
- Grouping headers: "Multiple-Choice Benchmarks" is provided by `00`.
  A "Free-Response Benchmarks" header was intentionally **omitted** — `40` and
  `41` already open with their own `\section{Free-response track: ...}`, so a
  bare grouping section would be empty or force a risky heading demotion. Skipped
  per the brief's "if unsure, skip".

## Bibliography — canonical keys chosen for unified duplicates
`references.bib` = concatenation of all `refs/*.bib`, deduplicated. **45 entries,
45 unique keys, 0 duplicate `@entry` keys.** Every one of the 45 distinct `\cite`
keys used across the included sections resolves to exactly one entry, and there
are **0 unused entries**.

Unified same-paper-different-key collisions (old key -> canonical, `\cite` calls updated):
- ATLAS: `li2026adaptive` -> **`li2025atlas`** (kept the arXiv `@misc` with DOI; updated `\cite` in `10`, `12`).
- Tsutakawa 1990: `tsutakawa1990` -> **`tsutakawa1990effect`** (kept version with DOI + publisher; updated `12`).
- Patton 2013: `patton2013` -> **`patton2013influence`** (kept version with DOI + publisher; updated `12`).
- Rasch 1960: `rasch1960probabilistic` -> **`rasch1960`** (updated `90`).
- Lord 1980: `lord1980applications` -> **`lord1980irt`** (kept `lord1980irt`, added the ISBN; updated `90`).
- Birnbaum 1968: `birnbaum1968some` -> **`birnbaum1968`** (updated `90`).
- Wainer 2000 (extra collision, not in the brief's list, but the same book
  "Computerized Adaptive Testing: A Primer"): `wainer2000` -> **`wainer2000cat`**
  (kept the full author list; updated `\cite` in `00`'s Strategy paragraph).
  Unified because leaving both keys would have printed the same book twice.

Kept one each (already shared under a single key, no `\cite` change needed):
`chalmers2012mirt` (kept the DOI version), `bock1981marginal`, `clark2018arc`,
`fourrier2024openllm`, `lelievre2025pedagogy` (`@misc`), `embretson2000irt`,
`kwon2023vllm`, `rein2024gpqa`, `zhou2023ifeval`, `hendrycks2021math`,
`suzgun2023bbh`, `liang2023helm`, `olmo3`, `llama3herd`.
`beeching2023openllm` (Open LLM Leaderboard v1) and `fourrier2024openllm` (v2)
are deliberately kept as two distinct works.

Open venue question (preserved, not resolved): a `TODO(human)` comment in `20`
notes the ATLAS venue is uncertain (author repo lists ICML 2026; OpenReview shows
a withdrawn ICLR 2026 submission). The canonical entry cites the arXiv preprint
`arXiv:2511.04689`, which is unambiguous.

`refs/oracle_subset.bib` appeared but `12b_oracle_subset.tex` is **not** present,
so the oracle section is not included (per the brief). That bib contributed **no
unique keys** (all duplicates of existing entries), so it did not change
`references.bib`. When the parent splices `12b`, any `\cite{li2026adaptive}` in it
must be retargeted to the canonical `li2025atlas`.

## Figures
`figures/` holds **39 PNGs**. Referenced = 39, present = 39, **0 missing, 0 extra.**
38 were copied from the original absolute source paths listed in each section's
top comment; `irt_icc.png` was copied from the prebuilt `_paper_build/figures/`
(the appendix listed no absolute source for it). `figure*`/`table*` floats are
harmless in single-column `manuscript` mode (they behave as `figure`/`table`).

## Consistency + humanizer sweep
- Pedagogy model count reconciled to the README (`01_MCQ_ATLAS/README.md`:
  "52 models" in-house matrix; feasibility splits it 40 calibration + 12 held-out).
  The abstract's `\aiadd{about 52} calibration models` was changed to
  `\aiadd{about 52 in-house} models` so it agrees with the README and with the
  body (which uses 40 calibration models), instead of implying 52 calibration
  models.
- Fixed `~300` -> `$\sim$300` inside the ORACLE placeholder (a non-breaking space
  was rendering instead of "approximately"). The placeholder is otherwise intact.
- No em-dashes anywhere (the only `---` occurrences are inside LaTeX comments,
  which do not render).
- All existing `\aiadd`/`\aibullets` (blue) and every red flag were preserved;
  none were removed or rewritten. Only merge-level edits (cite keys, abstract
  extraction, the two fixes above) were made, so no new AI-tell prose was
  introduced to humanize.

## Remaining red flags / TODOs (all preserved; none fabricated)
Rendered red in the PDF:
- `main.tex`: `[AUTHOR NAME]`, `[AFFILIATION]` placeholders, and an empty `\email{}`.
- `12` §"Oracle latent-ability subset selection": `[ORACLE SUBSET RESULT PENDING ...]` (left as instructed).
- `40` (five): `[CITATION NEEDED: ... Flow, Selene, and Gemma judges]`;
  `[CHECK: ... "Qwen/Qwen3.5-9B" ...]`;
  `[CHECK: ... three vs four skills ...]`;
  `[CHECK: content reliability reported two ways ...]`;
  `[CHECK: Bridge 642 vs 250 scenarios]`.

Rendered blue author stubs (preserved): `90` appendix `\aibullets{TODO (author): ...}` (two).
Non-rendered LaTeX comment TODO (preserved): `20` ATLAS-venue `TODO(human)`.

## Compile status
**No LaTeX engine was available** (`latexmk`, `pdflatex`, `tectonic`, `xelatex`,
`lualatex` and `bibtex`/`biber` are all absent), so the paper was **structurally
validated, not compiled**. Structural checks (all passed):
- Environments balanced: `document`, `abstract`, `figure`(21), `figure*`(8),
  `table`(25), `table*`(2), `tabular`(27), `itemize`(15), `enumerate`(1),
  `equation`(6), `minipage`(1) — every `\begin` matches its `\end`.
- Braces balanced (1226/1226 after discounting escaped `\{`/`\}`).
- Inline math `$` balanced (1054, even, after discounting 7 escaped currency `\$`).
- `abstract` appears before `\maketitle`; `\appendix` before `\bibliography`;
  `\bibliography` before `\end{document}`; exactly one each of `\begin{document}`,
  `\begin{abstract}`, `\maketitle`.
- 45/45 `\cite` keys resolve; 39/39 figures present; multicolumn table column
  counts spot-checked and consistent.
