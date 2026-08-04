"""Build the MCQ Feasibility results section (continuation after the intro passage).

Run: uv run --with python-docx python build_feasibility_docx.py
Output: Feasibility_Experiments_MCQ.docx (does not touch the other docx files).
"""
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
MCQ = os.path.abspath(os.path.join(HERE, "..", "01_MCQ_ATLAS"))
DATA = os.path.join(MCQ, "data")
FIG01 = os.path.join(MCQ, "figures")
PED = os.path.join(DATA, "pedagogy_feasibility")
COST = os.path.abspath(os.path.join(HERE, "..", "04_Inference_Cost", "graphs"))
OUT = os.path.join(HERE, "Feasibility_Experiments_MCQ.docx")
IMG_W = Inches(5.5)

embedded, missing = [], []
doc = Document()

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)


def para(text, size=11, bold=False, italic=False, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.bold, r.italic, r.font.size = bold, italic, Pt(size)
    return p


def heading(text):
    para(text, size=13, bold=True, space_after=4)


def subhead(text):
    para(text, size=11, bold=True, space_after=2)


def figure(path, caption):
    if not os.path.exists(path):
        missing.append(os.path.basename(path))
        para("[missing figure: %s]" % os.path.basename(path), size=9, italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(path, width=IMG_W)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    r = cap.add_run(caption)
    r.italic, r.font.size = True, Pt(9)
    r.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    embedded.append(os.path.basename(path))


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i].paragraphs[0]
        rr = c.add_run(h)
        rr.bold, rr.font.size = True, Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cp = cells[i].paragraphs[0]
            rr = cp.add_run(str(v))
            rr.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


# ============================================================================
# Part A: Pedagogy Benchmark calibration
# ============================================================================
heading("Part A: Pedagogy Benchmark calibration")
para(
    "Pedagogy Benchmark is a narrow, skill-specific benchmark with no public per-question "
    "response data, so there is no leaderboard matrix to calibrate against. We therefore built "
    "the response matrix from in-house model runs and calibrated an IRT item bank on those. "
    "The question is whether an adaptive test can recover a model's full pedagogy accuracy from "
    "a small handful of items."
)
para(
    "It can, at the rank level. A 2PL bank recovers full pedagogy accuracy at Pearson r=0.716 "
    "(SE\u22640.3) after a mean of 10.3 items, about 3.2% of the 318-item bank. A 1PL/Rasch bank "
    "reaches r=0.890 at the same stopping rule; it just spends more items to get there "
    "(about 44). Ranking is good, but the raw predicted values are miscalibrated in level: true "
    "pedagogy accuracy is squeezed into roughly 0.22 to 0.31, while the p-IRT prediction reads "
    "off a much wider axis, so the raw cloud sits on a line about 6x too steep with a large "
    "positive offset (raw MAE up to about 0.10 for the 1PL bank)."
)
table(
    ["Benchmark (bank)", "SE", "r", "MAE", "Mean items", "% of bank"],
    [
        ["Pedagogy (2PL)", "0.3", "0.716", "0.063", "10.3", "3.2%"],
        ["Pedagogy (1PL)", "0.3", "0.890", "0.101", "43.8", "13.8%"],
        ["PIQA (2PL)", "0.3", "0.937", "0.040", "10.6", "1.3%"],
        ["SocialIQa (2PL)", "0.3", "0.271", "0.093", "8.0", "0.8%"],
    ],
)
para(
    "Because the error is a deterministic slope/offset, not noise, a single linear map "
    "(fit on 40 calibration models, applied to 12 held-out models) removes almost all of it and "
    "leaves r unchanged. Honest held-out MAE drops 76% to 90% across both banks and both SE "
    "targets. The 1PL bank, the worst raw case at MAE about 0.10, ends with the lowest "
    "calibrated MAE, about 0.010. Leave-one-out agrees (0.008 to 0.016), so the gain is real "
    "rather than fit noise."
)
table(
    ["Bank", "SE", "r", "MAE raw", "MAE calibrated", "MAE reduction"],
    [
        ["2PL", "0.3", "0.716", "0.063", "0.015", "76%"],
        ["2PL", "0.15", "0.900", "0.080", "0.012", "85%"],
        ["1PL", "0.3", "0.890", "0.101", "0.010", "90%"],
        ["1PL", "0.15", "0.941", "0.097", "0.010", "90%"],
    ],
)
para(
    "Two public MCQ benchmarks act as controls and show the method is not automatic. PIQA "
    "replicates cleanly (r=0.94 at SE\u22640.3 using about 1.3% of its bank), while SocialIQa does "
    "not (r=0.27 at SE\u22640.3, rising only to 0.43 at SE\u22640.15). The bank recovers a skill "
    "when its items carry signal for that skill and fails when they do not. Pedagogy sits in "
    "between: strong rank recovery from a few percent of items, rescued to low absolute error by "
    "one linear recalibration."
)
figure(os.path.join(PED, "diag_pedagogy_se0.3.png"),
       "Pedagogy 2PL CAT, SE\u22640.3: p-IRT predicted vs actual full accuracy. r=0.716 from ~3% of the bank; good rank, level offset.")
figure(os.path.join(PED, "linear_calibration", "calib_pedagogy_1pl_se0.3.png"),
       "Pedagogy 1PL, SE\u22640.3: raw prediction (left) vs after one linear map (right). MAE 0.101 to 0.010, a 90% cut, r unchanged.")

# ============================================================================
# Part B: ATLAS-subset trade-offs
# ============================================================================
heading("Part B: ATLAS-subset trade-offs")
para(
    "The pedagogy case has to build its own calibration matrix. To show what that costs in "
    "practice we vary the calibration pool on ATLAS-scale data along the three levers named "
    "above: how the models are chosen (diversity), how many there are (count), and how well "
    "their parameter range matches the test set."
)

subhead("Model diversity: choose the pool well and use fewer models")
para(
    "Holding the held-out test set fixed, we compare random model sampling against two smart "
    "selectors: ability-spread (models that cover the accuracy range evenly) and response "
    "diversity (a farthest-point k-center on the model-by-item correctness matrix). Smart "
    "selection reaches random's plateau correlation with about 1.4x to 1.6x fewer models. On "
    "math, response diversity reaches r about 0.89 by N=50 (SE\u22640.3), at or above random's "
    "N=120 plateau of about 0.865, a 1.6x saving. On ifeval, ability spread matches the plateau "
    "at N=80 versus about N=113 for random, a 1.4x saving. The small-N points are genuinely "
    "noisy: 3PL fits on 10 to 30 models against hundreds of items are over-parameterized and the "
    "usable item bank shrinks as constant items drop, so the trend matters more than any single "
    "point."
)
figure(os.path.join(DATA, "model_diversity_selection", "selection_curve_combined.png"),
       "Correlation vs number of calibration models per strategy. Smart selectors hit random's plateau with ~1.4-1.6x fewer models.")

subhead("Count: correlation stabilizes fast, then plateaus")
para(
    "Sweeping only the number of calibration models, correlation is unstable below about 15 "
    "models (r bounces from 0.55 at N=5 to 0.45 at N=10 in the fine single-benchmark sweep), "
    "then settles into a high band from roughly 25 to 40 models (r about 0.83 to 0.94). The "
    "extended multi-benchmark sweep out to nearly 1,000 models is flat: correlation plateaus by "
    "about 100 to 200 models (ifeval about 0.91 to 0.95, math about 0.86 to 0.93 across that "
    "whole range) and adding more calibration models past that point does not help. The "
    "practical pool size is tens, not thousands."
)
figure(os.path.join(FIG01, "trainsize_corr_ext.png"),
       "Correlation vs calibration pool size. Unstable below ~15, stabilizes ~25-40, and the extended sweep plateaus by ~100-200.")

subhead("Parameter-range match: ability spread beats proximity")
para(
    "Three experiments look like they disagree until read under one rule. First, a 3PL bank "
    "calibrated on one parameter band and tested across a gap on another band transfers well "
    "for strongly-linking benchmarks (ifeval is essentially free; math pays a modest penalty of "
    "+0.118 r calibrating small and testing large, +0.032 the other way) and fails for "
    "weakly-linking gpqa. Transfer is asymmetric: small-to-large is the harder direction. "
    "Second, when the test set is fixed to 7B models, a wide 0-5B calibration pool (upward "
    "extrapolation) actually beats a closer 3-6.5B in-range pool at matched N: the in-range pool "
    "is lower by 0.036 r on ifeval and 0.130 r on math (SE\u22640.3). Closer range, worse "
    "recovery. Third, restricting calibration to the 0.5-7B slice (about 95% at 7B, a narrow "
    "ability spread) drops recovery to r about 0.59 against the wide published bank's 0.83, and "
    "neither re-balancing the size mix nor changing the count at matched N moves it (all land at "
    "r about 0.56 to 0.65)."
)
para(
    "The single rule that reconciles them: what matters is the calibration pool's ability "
    "spread, not its proximity to the test set's parameter range. A pool that spans a wide "
    "ability range predicts an out-of-range target better than a narrow pool sitting right next "
    "to it, because a narrow pool is dominated by items that saturate or lose discrimination "
    "outside its band."
)
figure(os.path.join(DATA, "param_extrapolation", "param_extrapolation_combined.png"),
       "Cross-band transfer (SE\u22640.3): ifeval near-free, math a modest penalty, gpqa fails; small-to-large is the harder direction.")
figure(os.path.join(DATA, "calib_range_to_7b", "calib_range_to_7b_r_A_vs_B.png"),
       "Recovering 7B models: wide 0-5B pool (A) beats closer 3-6.5B in-range pool (B). B minus A is negative for ifeval and math.")
figure(os.path.join(DATA, "size_balanced_recal", "size_balance_r_comparison.png"),
       "0.5-7B recalibration: rebalancing the size mix at matched N does not recover r. The limit is the restricted range, not skew or count.")

# ============================================================================
# Part C: Inference cost
# ============================================================================
heading("Part C: Inference cost")
para(
    "The two discussion points from the intro are time and dollars. On time, adaptive testing "
    "wins twice over. CAT reaches its stopping rule after roughly 1% to 15% of the item bank "
    "(the MCQ banks in Part A stop at about 1% to 4%), which is on the order of 7x to 30x fewer "
    "item inferences than scoring the whole bank. On top of that, running the models under "
    "batched vLLM instead of one request at a time is about 14x to 21x faster per request. "
    "Together these turn a per-model evaluation from tens of seconds into roughly 1 to 2 seconds "
    "of warm batched serving, which is what makes checkpoint-time evaluation during training and "
    "fast experimentation practical rather than a separate offline job."
)
table(
    ["Parameter range", "Single-stream (s)", "Batched (s)", "Batch speed-up", "Mean output tokens"],
    [
        ["0-1B", "4.0", "0.25", "~16x", "878"],
        ["1-2B", "10.3", "0.74", "~14x", "648"],
        ["2-5B", "19.7", "1.13", "~17x", "658"],
        ["5-7B", "34.7", "1.64", "~21x", "489"],
    ],
)
figure(os.path.join(COST, "individual_latency_by_param_range.png"),
       "Per-request latency by parameter range: single-stream vs batched vLLM. Batching is ~14-21x faster; larger models cost more per request.")
para(
    "On dollars, the task type dominates. MCQ costs about $0.009 per 1,000 inferences because "
    "the generated answer is only a few tokens, so there is almost no decode. Open-ended tasks "
    "cost about $0.20 to $0.23 per 1,000, roughly 20x to 25x more, and that gap is driven by "
    "output tokens rather than input length (the short, medium, and long input buckets differ "
    "by only about 15%). MCQ and open-ended use different item and model sets, so this is an "
    "order-of-magnitude comparison, not a controlled one."
)
table(
    ["Task", "Cost per 1k inferences"],
    [
        ["MCQ (cdpk_main, 24 models)", "~$0.009"],
        ["Open-ended, short input", "~$0.20"],
        ["Open-ended, medium input", "~$0.21"],
        ["Open-ended, long input", "~$0.23"],
    ],
)
para(
    "One caveat on the latency table, stated plainly: the single-stream column is an L4 "
    "roofline estimate (300 GB/s, 60% efficiency, 2 bytes per parameter), not a direct "
    "measurement, and the batched column is measured under vLLM at temperature 0. The 0-7B cost "
    "basis is a single NVIDIA L4 (g6.xlarge) at $0.8048/hr. An earlier version of this table was "
    "labeled L40S; an L40S at about 864 GB/s would decode roughly 2x to 3x faster, so the table "
    "here should be read as an L4 estimate.",
    space_after=4,
)

doc.save(OUT)
print("SAVED:", OUT)
print("SIZE_BYTES:", os.path.getsize(OUT))
print("EMBEDDED (%d):" % len(embedded), embedded)
print("MISSING (%d):" % len(missing), missing)
