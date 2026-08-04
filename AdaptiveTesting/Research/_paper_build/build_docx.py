#!/usr/bin/env python3
"""Condensed (~10pp) build of CAT_for_Accelerated_Pedagogy_Benchmarking.{md,docx}.

Faithful (condensed) reproduction of the source PDF, plus two integrated
experiments: MCQ ATLAS recreation (01_MCQ_ATLAS) and FRQ TutorBench 115-model
MIRT (03_FRQ_MIRT). Authored (non-PDF, non-number) prose is essentially zero;
the few unavoidable figure/experiment labels are marked blue. 2PL equation is
rendered correctly. python-docx embeds figures; no pandoc needed.
"""

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "docx_assets")
MD_PATH = os.path.join(HERE, "CAT_for_Accelerated_Pedagogy_Benchmarking.md")
DOCX_PATH = os.path.join(HERE, "CAT_for_Accelerated_Pedagogy_Benchmarking.docx")
BLUE = RGBColor(0x1A, 0x56, 0xDB)


def n(t): return {"t": t}
def a(t): return {"t": t, "c": "add"}
def it(t): return {"t": t, "i": True}
def P(*r): return {"k": "p", "runs": list(r)}
def BUL(lvl, *r): return {"k": "bul", "lvl": lvl, "runs": list(r)}
def H(l, t): return {"k": "h%d" % l, "runs": [n(t)]}
def TITLE(t): return {"k": "title", "runs": [n(t)]}
def TABLE(h, rows, cap=None): return {"k": "table", "header": h, "rows": rows, "cap": cap}
def FIG(f, w, cap): return {"k": "fig", "path": os.path.join(ASSETS, f),
                            "rel": "docx_assets/%s" % f, "width": w, "cap": cap}
def EQ(t, md): return {"k": "eq", "t": t, "md": md}


BLOCKS = []
B = BLOCKS.append

B(TITLE("CAT for Accelerated Pedagogy Benchmarking"))

# ---- Abstract
B(H(1, "Abstract"))
B(P(n("Benchmarking and evaluation take an estimated 10\u201315% of total model training "
      "time (OLMo-3, Llama). Fast, granular, trustworthy benchmarking is crucial for "
      "EDU-LLM given the number of novel hypotheses tested. We propose a skill-based "
      "Computerized Adaptive Test (CAT) that reduces the items needed to report a "
      "predicted per-skill accuracy while holding precision close to full-length "
      "benchmarks. Findings split between Multiple Choice (MCQ) and Free Response (FRQ) "
      "benchmarks.")))
B(BUL(1, n("MCQ. We reproduce ATLAS-style adaptive testing on "), a("ARC: "),
      n("using the published ATLAS 3PL bank, CAT recovers full benchmark accuracy at "),
      a("r = 0.83 with under 2% of items (~21 of 1,172 at SE \u2264 0.2); a "
        "range-restricted 0.5\u20137B refit is weaker (r = 0.59).")))
B(BUL(1, n("FRQ. We grade responses with an LLM-as-judge against per-scenario rubrics to "
           "build a criterion-level response matrix, derive a Q-matrix, and fit 2PL/MIRT "
           "so CAT selects scenarios adaptively.")))

# ================================================================= 1. MCQ
B(H(1, "1. MCQ"))
B(H(2, "1.1 Strategy"))
B(P(n("Our work builds on ATLAS, which showed adaptive testing can dramatically reduce "
      "the items required to predict benchmark accuracy. Items calibrated with Item "
      "Response Theory (IRT) also provide information beyond accuracy: models that get "
      "harder questions right and miss easy ones are ranked appropriately under CAT, "
      "which a simple accuracy comparison cannot capture.")))
B(P(n("We show the feasibility of recreating the ATLAS methodology on new benchmarks and "
      "formalize the requirements to use CAT as a true MCQ benchmark replacement, "
      "highlighting the promises and shortcomings of MIRT for MCQ \u2014 a discussion "
      "largely missing from ATLAS.")))

B(H(2, "1.2 ATLAS Recreation"))
B(P(n("There is a gap between ATLAS results and evaluations during model training. "
      "Applying IRT calibration to a benchmark without public response data requires "
      "analyzing the upfront cost to reach a target correlation, and the ability to "
      "discriminate models across parameter ranges without many calibration models was "
      "not explicitly shown. Two feasibility experiments follow: calibration on a "
      "skill-based benchmark lacking public responses, and calibration on subsets of "
      "ATLAS data mimicking real-world model diversity, count, and parameter-range "
      "trade-offs. Inference used L4 GPUs (CPUs parallelized model downloads).")))
B(P(a("Concretely, on ARC we run ATLAS-style CAT two ways on the same 60 held-out models: "
      "the published ATLAS 3PL bank vs. the pipeline refit on only the 0.5\u20137B slice. "
      "r is Pearson between CAT-predicted and true full-benchmark accuracy.")))
B(TABLE(["Condition", "SE stop", "Pearson r", "MAE", "Mean CAT items"],
        [["Published ATLAS 3PL bank", "\u2264 0.2", "0.830", "0.147", "21.1"],
         ["Published ATLAS 3PL bank", "\u2264 0.3", "0.738", "0.172", "9.9"],
         ["Refit on 0.5\u20137B models", "\u2264 0.2", "0.589", "0.147", "12.5"],
         ["Prompt-matched (ATLAS own resp.)", "\u2264 0.2", "0.915", "0.031", "13.0"]],
        [a("Table 1. ATLAS 3PL transfer on held-out ARC models (integrated from "
           "01_MCQ_ATLAS).")]))
B(FIG("atlas_arc_heldout_se0.2.png", 4.2,
      [a("Figure 1. Published ATLAS 3PL bank on 60 held-out ARC models (SE \u2264 0.2): "
         "r = 0.83, ~21 items.")]))
B(FIG("atlas_selfresp_arc_se0.2.png", 4.2,
      [a("Figure 2. Prompt-matched responses: r = 0.91, MAE = 0.031 in ~13 items \u2014 "
         "most earlier degradation was scoring-protocol mismatch, not IRT transfer.")]))

B(H(2, "1.3 Recommendations"))
B(BUL(1, n("Open response resources are useful but limited; choose the parameter range "
           "appropriately.")))
B(BUL(1, n("1PL/2PL matters more when fewer calibration models are available; adjust the "
           "standard error to match accuracy requirements; model diversity is "
           "important.")))
B(BUL(1, n("Inference is a significant upfront cost, but it speeds up on-demand "
           "evaluation and unlocks benchmarking during checkpoints.")))

# ================================================================= 2. FRQ
B(H(1, "2. FRQ"))
B(H(2, "2.1 LLM-as-Judge"))
B(P(n("Prometheus 2 had been used without proving its judgments matched human graders. We "
      "graded three tutor responses for each of 10 scenarios against their criteria "
      "(261 binary response\u2013criterion cases), targeting Macro-F1 \u2265 0.80, "
      "critical-failure sensitivity \u2265 0.9, per-skill F1 \u2265 0.70, repeat agreement "
      "\u2265 0.90, and prompt-flip rate \u2264 0.10. Judges were blinded to human labels "
      "and tutor identity; local judges ran six waves (three identical runs plus "
      "whitespace, header, and politeness prompt variations). Parsing failures were "
      "separated from grading errors, and Qwen's disagreements were reviewed by hand.")))
B(P(n("No judge passed every threshold, and frontier judges (limited to 174 cases each, "
      "barred from their own provider family) performed similarly without clearing every "
      "bar. We froze Qwen/Qwen3.5-9B zero-shot binary judging, one criterion at a time, "
      "evidence required, p_fail \u2265 0.33 = fail \u2014 the strongest, most stable local "
      "candidate, self-hostable with a frozen checkpoint for reproducibility, privacy, and "
      "low marginal cost across the hundreds of thousands of judgments needed.")))

B(H(2, "2.2 Skills and Rubrics"))
B(P(n("MIRT needs a set of skills and a Q-matrix recording the skills each criterion "
      "tests. An LLM proposes each 1 with evidence, an explanation, and a counterfactual; "
      "labels are reassessed by two LLM verifiers and, on a subset, two human reviewers, "
      "keeping positive labels only on unanimous agreement. Benchmarks: TutorBench "
      "(662 scenarios, 6,462 criteria across content, diagnosis, scaffolding), TutorEval "
      "(828 scenarios; conceptual and quantitative), WildBench (1,001 scenarios, 11 "
      "capability tags), EduBench (nine deterministically-verified task types), and Bridge "
      "(642 scenarios, 39 rubric templates, five skills).")))

B(H(2, "2.3 Feasibility"))
B(P(n("For calibration, 82 open models under 7B answer all 662 TutorBench scenarios on a "
      "single L4 GPU, and the frozen judge grades each answer criterion by criterion; a "
      "second run of 52 models overlaps the first by 19. The smallest models fall into "
      "repetitive loops the judge marks unscorable, and long inputs are capped at 32k "
      "tokens (near-complete coverage).")))
B(TABLE(["Parameter Range (B params)", "Avg. Inference Time on L40S GPU (s)"],
        [["0 \u2013 1", "4"], ["1 \u2013 2", "10"], ["2 \u2013 5", "19"], ["5 \u2013 7", "34"]],
        [n("Table 2. Inference cost for tutor models.")]))

B(H(2, "2.4 MIRT Results \u2014 TutorBench"))
B(P(n("Skills: correctness (factually true information and accurate diagnosis) and "
      "scaffolding (framework/hints that advance the student without giving the answer); "
      "presentation (Markdown/LaTeX, second-person, convention) is a third skill in the "
      "3-skill model. Content and diagnosis were collapsed into correctness due to "
      "collinearity. Choice factors: number of skills (1/2/3) and estimator (Gaussian vs. "
      "Batch EAP vs. MWLE). Ideal conditions: 115 models (0.1B\u20137B), 2-skill, Batch "
      "EAP/MWLE, trace (Fisher) selection, presentation reported separately. Metrics use "
      "the out-of-sample correlation R and slope against the full-bank EAP; total SE "
      "combines posterior SE with parameter uncertainty; seed stability \u2248 0.15.")))
B(P(it("Minimum-scenario floor (2-skill, in sample):")))
B(TABLE(["Min scenarios", "Correctness MWLE r", "Scaffolding MWLE r",
         "Corr. MWLE slope", "Scaff. MWLE slope", "Scenario mean"],
        [["0", ".9815", ".9419", "1.006", "1.085", "21.5"],
         ["12", ".9819", ".9457", "1.002", "1.090", "21.9"],
         ["15", ".9822", ".9481", "1.000", "1.087", "22.5"],
         ["20", ".9822", ".9526", "1.007", "1.093", "24.0"]],
        [n("Table 3. Minimum-scenario floor vs. recovery (chose 12).")]))
B(TABLE(["Estimator", "Corr. Slope", "Corr. R", "Scaff. Slope", "Scaff. R"],
        [["Gaussian", ".644", ".921", ".875", ".903"],
         ["Batch EAP", ".755", ".939", ".918", ".913"],
         ["MWLE", ".981", ".945", "1.032", ".902"]],
        [n("Table 4. Estimator comparison (2-skill).")]))
B(TABLE(["Model", "Correctness r", "Scaffolding r", "Presentation r", "Precision",
         "Mean scenario"],
        [["1 skill", ".949", "N/A", "N/A", "115/115", "12.1"],
         ["2 skill", ".945", ".902", "N/A", "115/115", "21.9"],
         ["3 skill", ".969", ".916", ".965", "58/115", "41.2"]],
        [n("Table 5. Skill-count comparison.")]))
B(TABLE(["Skill", "MWLE R", "MWLE Slope", "SE Post", "SE Total"],
        [["Correctness", "0.945", "0.981", "0.334", "0.412"],
         ["Scaffolding", "0.902", "1.032", "0.307", "0.359"]],
        [n("Table 6. Recovery and uncertainty (2-skill).")]))
B(FIG("tb115_recovery_scatter_correctness.png", 3.4,
      [a("Figure 3. 2-skill correctness recovery, 115-model fit (in-sample EAP r = 0.97; "
         "out-of-sample MWLE 0.945 in Table 6).")]))
B(FIG("tb115_se_reduction_curve.png", 5.2,
      [a("Figure 4. Posterior SE falls with items administered, crossing the SE = 0.3 "
         "target for both skills.")]))
B(P(n("With the finalized 2-skill 2PL model calibrated on 115 models (0.1B\u20137B), "
      "trace selection, SE target 0.30, and a minimum-scenario floor of 12, all 115/115 "
      "models reach the target in about 22 scenarios. It recovers correctness at "
      "out-of-sample r = 0.94 and scaffolding at 0.91, with slopes 0.981 and 1.032 \u2014 "
      "a roughly 30x reduction that nearly matches the full-bank grade in distribution and "
      "scale. Accounting for item-parameter uncertainty, total SE runs slightly high "
      "(0.412 correctness, 0.359 scaffolding).")))
B(P(it("Limitations.")))
B(BUL(1, n("Parameter SE estimated on 115 models is added into every skill-estimate SE; "
           "more calibration models would reduce it from ~0.2 on average.")))
B(BUL(1, n("The judge is imperfect and does not always agree with human graders; with "
           "only 115 models the scale compresses and weak models pull toward the mean "
           "(partially mitigated by MWLE).")))
B(BUL(1, n("Parameter-SE references: Tsutakawa & Johnson (1990), Psychometrika 55(2), "
           "371\u2013390; Patton, Cheng, Yuan & Diao (2013), Applied Psychological "
           "Measurement 37(1), 24\u201340.")))

B(H(2, "2.5 MIRT Results \u2014 TutorEval"))
B(P(n("TutorEval uses the \u2018question\u2019 field as scenarios (with textbook excerpts "
      "for open-book items) and splits \u2018key points\u2019 into criteria; 52 "
      "open-source models (0.2B\u20137B) calibrated discrimination/difficulty. CAT stops "
      "at SE < 0.3 with a 15-criteria minimum. Both unidimensional and 2D (conceptual, "
      "quantitative) were tried.")))
B(TABLE(["Estimator", "Conceptual Slope", "Conceptual R", "Quant. Slope", "Quant. R"],
        [["Gaussian", "0.43", "0.8506", "0.75", "0.9572"],
         ["Batch EAP", "0.573", "0.9063", "0.846", "0.9739"],
         ["MWLE", "0.67", "0.9122", "0.936", "0.9721"]],
        [n("Table 7. TutorEval, two-dimensional.")]))
B(TABLE(["Estimator", "Ability Slope", "Ability R"],
        [["Gaussian", "0.545", "0.9051"],
         ["Batch EAP", "0.659", "0.9349"],
         ["MWLE", "0.686", "0.9309"]],
        [n("Table 8. TutorEval, unidimensional.")]))
B(BUL(1, n("The 2D model collapses back to unidimensional (skills correlate 0.98); we "
           "keep unidimensional for this benchmark (both shown for completeness).")))
B(BUL(1, n("Reduction \u2014 unidimensional: ~10.4 scenarios (~80x), ~15 criteria "
           "(>100x from 1,786). 2D: 18.3 (trace) / 27.8 (D-opt) scenarios (~40x), "
           "~34 criteria.")))
B(BUL(1, n("Recovery \u2014 unidimensional MWLE r = 0.931; 2D MWLE r = 0.912 "
           "(conceptual) / 0.972 (quantitative). MWLE has the least score shrinkage.")))
B(BUL(1, n("Limitation: 52 models is relatively minimal; more would strengthen "
           "validity.")))

# ================================================= Appendix: IRT Fundamentals
B(H(1, "Appendix A. IRT Calibration Fundamentals (Unidimensional)"))
B(P(n("IRT calibrates per-item parameters that let CAT distinguish a model's ability. In "
      "the unidimensional case each model has a single latent ability \u03b8, and each "
      "item is a sigmoid in the probability of a correct response given \u03b8. The "
      "difficulty \u03b2\u1d62 sets the boundary \u03b8; the discrimination "
      "\u03b1\u1d62 sets its steepness, giving the 2PL item response function:")))
B(EQ("P(y = 1 | \u03b8, \u03b2\u1d62, \u03b1\u1d62) = "
     "exp[\u03b1\u1d62(\u03b8 \u2212 \u03b2\u1d62)] / "
     "(1 + exp[\u03b1\u1d62(\u03b8 \u2212 \u03b2\u1d62)]) = "
     "1 / (1 + exp[\u2212\u03b1\u1d62(\u03b8 \u2212 \u03b2\u1d62)])",
     r"\[ P(y=1\mid\theta,\beta_i,\alpha_i)=\frac{\exp[\alpha_i(\theta-\beta_i)]}"
     r"{1+\exp[\alpha_i(\theta-\beta_i)]}=\frac{1}{1+\exp[-\alpha_i(\theta-\beta_i)]} \]"))
B(P(n("Calibration (EM) assumes each model's ability is normal with mean 0, discretizes "
      "\u03b8 over a range, and for each model computes the likelihood of its response "
      "pattern at each \u03b8, multiplying by the prior to update the latent distribution. "
      "For each item, R(\u03b8)/N(\u03b8) \u2014 expected fraction correct at each \u03b8 "
      "\u2014 gives the points the sigmoid must fit, and item parameters are updated by "
      "logistic regression; the steps repeat until convergence.")))


# ---------------------------------------------------------------- docx render
def _bg(cell, hexc):
    tcpr = cell._tc.get_or_add_tcPr()
    tcpr.append(tcpr.makeelement(qn("w:shd"), {qn("w:val"): "clear",
                qn("w:color"): "auto", qn("w:fill"): hexc}))


def _runs(p, runs, size=None):
    for r in runs:
        run = p.add_run(r["t"])
        if r.get("c") == "add":
            run.font.color.rgb = BLUE
        if r.get("i"):
            run.italic = True
        if r.get("b"):
            run.bold = True
        if size:
            run.font.size = Pt(size)


def _widths(table, widths):
    table.autofit = False
    table.allow_autofit = False
    table._tbl.tblPr.append(table._tbl.tblPr.makeelement(
        qn("w:tblLayout"), {qn("w:type"): "fixed"}))
    for row in table.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)


def render_docx(blocks):
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    for s in ("Heading 1", "Heading 2", "Heading 3"):
        pass
    usable = 6.5
    for blk in blocks:
        k = blk["k"]
        if k == "title":
            p = doc.add_paragraph(style=doc.styles["Title"])
            _runs(p, blk["runs"])
        elif k in ("h1", "h2", "h3", "h4"):
            p = doc.add_heading(level=int(k[1]))
            _runs(p, blk["runs"])
        elif k == "p":
            _runs(doc.add_paragraph(), blk["runs"])
        elif k == "bul":
            style = "List Bullet" if blk["lvl"] == 1 else "List Bullet %d" % blk["lvl"]
            try:
                p = doc.add_paragraph(style=style)
            except KeyError:
                p = doc.add_paragraph(style="List Bullet")
            _runs(p, blk["runs"])
        elif k == "eq":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(blk["t"])
            run.italic = True
            run.font.size = Pt(12)
        elif k == "table":
            header, rows = blk["header"], blk["rows"]
            ncol = len(header)
            table = doc.add_table(rows=1, cols=ncol)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, h in enumerate(header):
                run = table.rows[0].cells[i].paragraphs[0].add_run(h)
                run.bold = True
                run.font.size = Pt(9)
                _bg(table.rows[0].cells[i], "D9E2F3")
            for row in rows:
                cells = table.add_row().cells
                for i, v in enumerate(row):
                    run = cells[i].paragraphs[0].add_run(str(v))
                    run.font.size = Pt(9)
            w0 = usable * 0.30 if ncol > 2 else usable * 0.5
            rest = (usable - w0) / (ncol - 1) if ncol > 1 else usable
            _widths(table, [w0] + [rest] * (ncol - 1))
            if blk.get("cap"):
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _runs(cap, [dict(r, i=True) for r in blk["cap"]], size=8.5)
            doc.add_paragraph()
        elif k == "fig":
            if not os.path.exists(blk["path"]):
                doc.add_paragraph("[missing figure: %s]" % blk["rel"])
                continue
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(blk["path"], width=Inches(blk["width"]))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _runs(cap, [dict(r, i=True) for r in blk["cap"]], size=8.5)
    doc.save(DOCX_PATH)
    return DOCX_PATH


def _md_runs(runs):
    out = []
    for r in runs:
        t = r["t"]
        if r.get("b"):
            t = "**%s**" % t
        if r.get("i"):
            t = "*%s*" % t
        if r.get("c") == "add":
            t = '<span style="color:#1A56DB">%s</span>' % t
        out.append(t)
    return "".join(out)


def render_md(blocks):
    lines = []
    for blk in blocks:
        k = blk["k"]
        if k == "title":
            lines.append("# %s\n" % _md_runs(blk["runs"]))
        elif k in ("h1", "h2", "h3", "h4"):
            lines.append("%s %s\n" % ("#" * (int(k[1]) + 1), _md_runs(blk["runs"])))
        elif k == "p":
            lines.append("%s\n" % _md_runs(blk["runs"]))
        elif k == "bul":
            lines.append("%s- %s" % ("  " * (blk["lvl"] - 1), _md_runs(blk["runs"])))
        elif k == "eq":
            lines.append("%s\n" % blk["md"])
        elif k == "table":
            lines.append("| " + " | ".join(blk["header"]) + " |")
            lines.append("|" + "|".join(["---"] * len(blk["header"])) + "|")
            for row in blk["rows"]:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("\n*%s*\n" % _md_runs(blk["cap"]) if blk.get("cap") else "")
        elif k == "fig":
            cap = _md_runs(blk["cap"]) if blk.get("cap") else ""
            lines.append("![%s](%s)\n\n*%s*\n" % (cap, blk["rel"], cap))
    with open(MD_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return MD_PATH


if __name__ == "__main__":
    print("wrote markdown:", render_md(BLOCKS))
    print("wrote docx:", render_docx(BLOCKS))
