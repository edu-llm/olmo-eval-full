"""Convert the Bridge dataset into Scenario + Rubric Schema JSONL (5-skill artifact).

Source: https://huggingface.co/datasets/rose-e-wang/bridge  (train/validation/test,
700 tutor-student math remediation conversations total).
Paper: *Bridging the Novice-Expert Gap via Models of Decision-Making: A Case Study on
Remediating Math Mistakes* (Wang et al.).

What Bridge is: each row is a real tutoring conversation that ends on the STUDENT's
turn, and a tutor model must produce the next tutor turn (the remediation). The dataset
ships the original novice tutor reply (`c_r`) and an expert-revised reply (`c_r_`), plus
the expert's annotation of the error type (`e`) and their remediation strategy/intention
(`z_what` / `z_why`). Bridge is math-only: every `lesson_topic` is a TEKS math code
(grades 1-8 plus a few Algebra 2 rows).

HAND-AUTHORING NOTE (deliberate exception to data/README.md's standing rule that a
benchmark "must ship its own rubrics -- no hand-authoring"): Bridge ships NO per-response
criterion checklist. Its native rubric is a 4-dimension conversation-level scale, not a
set of binary per-response criteria. So this ingester attaches hand-authored criteria
drawn from the bank below. The mapping of criteria -> skills (the q-matrix) is likewise
hand-authored, not dataset-derived. Discrimination is left for calibration.

Skills (5). This is Bridge's own q-matrix axis, NOT the engine's default
content/diagnosis/scaffolding tuple. Like InFoBench's 5-skill artifact, Bridge ships as
an on-disk artifact only and is NOT engine-loadable under the default 3-skill
`tutor_cat.SKILLS`. The list order is the fixed q-matrix column order and must not be
reordered in place -- downstream MIRT code indexes skills positionally.

    diagnosis      - correctly identifying the student's specific error and its cause
    strategy       - pedagogical decision-making: choosing an effective remediation move
    math           - mathematical correctness of the tutor's own statements
    communication  - clarity, structure, and audience-appropriateness of the explanation
    affective      - supportive stance: tone, encouragement, preserving student agency

Mapping (Bridge field -> schema field):
    c_id                   -> source_id            (join key back to HuggingFace)
    c_h[-1].text           -> prompt               (student's final turn)
    c_h[:-1]               -> conversation_context  ({role, content}; role student/tutor)
    c_r_ (joined)          -> reference_solution    (expert revised reply = the gold key)
    c_r  (joined)          -> novice_response        (original tutor reply; provenance)
    e                      -> error_type             (drives exclusion + applicability)
    z_what / z_why         -> expert_strategy / expert_intention (provenance)
    lesson_topic           -> lesson_topic, and derives grade_band + topic_domain
    <HF split>             -> native_split            (provenance; `split` is the pipeline role)

EXCLUSIONS (deterministic)
--------------------------
  no_clear_mistake     `e` is not one of the six clean error types -- these are the
                       free-text "the student did not make a mistake" / "end session" /
                       "unresponsive" annotations.
  empty_conversation   `c_h` is empty.
  empty_student_turn   the final student turn has no text.

FLAGGED, NOT EXCLUDED -- `visible_mistake`: false when the final student turn is a bare
acknowledgment ("yes", "no", "done", ...) AND no digit appears anywhere in the
conversation. Those transcripts record only tutor turns plus an assent, so there is no
student position to remediate and the diagnosis criteria (D1/D2, both critical) are
effectively unpassable. They are KEPT so the bank stays complete, and carry the flag so a
calibration run can filter or model them explicitly. Bare acknowledgments WITH a digit
somewhere are `visible_mistake: true` -- the student's answer is recoverable from an
earlier turn or the tutor's restatement, or the assent is itself the wrong answer to a
yes/no question.

DUPLICATE CONVERSATIONS: one scenario is emitted per ROW, so the differing expert
revisions stay as separate items and `source_id` is intentionally NON-unique. Rows sharing
a `c_id` have the same stimulus and the same criterion set, differing only in
`reference_solution`. That is genuine local dependence: group or hold out on `source_id`
at calibration.

APPLICABILITY
-------------
Kept scenarios receive a VARIABLE criterion set assembled from four tiers:

    core                    every scenario                     (16 criteria)
    error-type module       canonical per conversation         (2-3 criteria)
    topic-domain module     keyed on `lesson_topic` keywords   (1-2 criteria)
    grade-band module       keyed on the `lesson_topic` grade  (1 criterion)

CONVERSATION-LEVEL ERROR GATING: Bridge re-annotates the same conversation point by
different experts, who often disagree about `e`. Keying the error module on the row's own
`e` gave byte-identical stimuli mutually contradictory critical criteria. The module is
therefore resolved once per CONVERSATION -- the FIRST annotation in split order wins -- and
every row sharing that `c_id` draws it. A deterministic pre-pass, no API.

Taking the UNION of all annotators' modules was tried and rejected: it kept every expert's
framing but gave disagreement-heavy conversations an extra module, so those scenarios
carried one more `critical` criterion (8 vs 7) than agreed-on ones. Item difficulty would
then partly reflect whether two annotators happened to disagree rather than the tutoring
task. A single canonical module keeps criterion counts comparable across the bank; the
discarded labels are still recorded in `error_types` for provenance.

WHY NO "OPTIONAL" CRITERIA: criteria whose applicability depends on what the CANDIDATE
writes cannot be resolved here -- the response does not exist at ingest time. The repo's
`optional: true` flag is not an N/A mechanism: run_calibration_judging.py drops optional
criteria from the run entirely, and the judge prompt forbids an N/A verdict. So every
criterion below is worded to be answerable pass/fail for ANY response, and M1 states its
empty case explicitly.

Thirteen criteria about content a response need not contain are stated in NEGATIVE form
(fail on misuse, pass on silence) so the empty case has a determinate verdict. That was
originally applied to nineteen; a pilot judging run measured negative-form criteria passing
at 0.866 against 0.646 for positive-form ones, so six were rewritten toward a positive,
specific demand. See the Pilot validation section of data/Bridge/README.md -- the pass rate
and discrimination of every criterion there is measured, not assumed.

Criterion ids use a STABLE suffix per code (see CODE_INDEX): a given suffix always
denotes the same criterion across scenarios. Because each scenario draws a subset of the
bank, its criterion_ids are intentionally non-contiguous.

The records carry NO `difficulty` / `discrimination` / `irt_params`. Those are calibrated
from real judge responses; until that fit exists the bank stays parameter-free rather than
shipping synthetic stand-ins that read like measurements. scripts/assign_irt_params.py can
still append placeholders if some downstream step needs the fields present, but that is a
deliberate opt-in, not part of the build.

    python scripts/ingest_bridge.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "Bridge"

HF_DATASET = "rose-e-wang/bridge"
SOURCE_URL = "https://huggingface.co/datasets/rose-e-wang/bridge"
HF_SPLITS = ["train", "validation", "test"]   # iterated in order; ids number across all
SPLIT = "calibration"   # pipeline-role label (matches TutorBench/InFoBench), not the HF split
USE_CASE = "mistake_remediation"   # unknown to respgen -> falls back to the adaptive_
                                   # explanation system prompt, which preserves multi-turn
SUBJECT = "mathematics"
VERSION = "3.0"   # 3.0 = visible_mistake flag, canonical per-conversation error module,
                  #       8 topic domains, 16 core criteria, negative-form wording

# The fixed q-matrix column order. Bridge's own 5-skill axis (see module docstring).
SKILLS = ["diagnosis", "strategy", "math", "communication", "affective"]

# The six clean error-type labels in `e`. Anything else is free text ("no mistake" /
# "end session" / "unresponsive") and the row is excluded.
CLEAN_ERROR_TYPES = {"guess", "right-idea", "diagnose", "careless", "imprecise", "misinterpret"}

# Bare acknowledgments: a final student turn matching one of these carries no position to
# remediate. Combined with "no digit anywhere" it triggers the no_visible_mistake drop.
ACKNOWLEDGMENTS = {
    "yes", "no", "y", "n", "ok", "okay", "done", "si", "yeah", "yep", "nope",
    "yes maam", "yes mam", "no mam", "no maam", "nhw", "sure", "a little bit",
    "idk", "i dont know", "yes sir", "no sir", "yup", "mhm", "k",
}

# `explicitness` asks whether the tutor was actually TOLD to do this. Bridge scenario
# prompts are student turns ("4 m"), so they never ask for anything -- but the benchmark
# system prompt does. SYSTEM_PROMPTS_BY_BENCHMARK["Bridge"] in tutor_cat/respgen/prompts.py
# reads:
#     "Identify the specific error, then help the student correct it by guiding them
#      toward the right approach rather than simply giving away the answer. Keep a
#      supportive, encouraging tone."
# which restates D1, P1 and A1 almost verbatim. Those three are therefore `explicit`;
# every other criterion is an unstated pedagogical expectation.
#
# NOTE the measurement consequence: every model is instructed to do exactly what D1/P1/A1
# check, so expect them near ceiling and treat their difficulty with suspicion -- D1 and P1
# are both `critical`. Confirm against pilot pass rates before reading anything into them.
EXPLICIT_CODES = {"D1", "P1", "A1"}

# A criterion entry: (code, primary_skill, [skills marked 1], criticality, objectivity,
#                     criterion text, q_rationale)
Criterion = tuple[str, str, list[str], str, str, str, str]

# ---------------------------------------------------------------------------
# TIER 1 -- CORE. Attached to every kept scenario.
# Each latent skill needs enough PURE (single-loading) criteria to be separably
# identified, so math carries M1/M4/M5 and affective carries A1/A3/A4 rather than
# resting on one criterion text apiece.
# ---------------------------------------------------------------------------
CORE_CRITERIA: list[Criterion] = [
    ("D1", "diagnosis", ["diagnosis"], "critical", "objective",
     "The response correctly identifies the specific error the student made.",
     "Locating the student's actual error is a pure diagnosis act."),
    ("D2", "diagnosis", ["diagnosis"], "critical", "objective",
     "The response addresses the student's real mistake and does not invent or correct an "
     "error the student did not make.",
     "Distinguishing the real error from a nonexistent one is a diagnosis act."),
    ("M1", "math", ["math"], "critical", "objective",
     "Every mathematical statement, calculation, and piece of notation in the response is "
     "correct. A response that makes no mathematical statement satisfies this criterion.",
     "Correctness of the tutor's own mathematics is pure content."),
    ("M4", "math", ["math"], "critical", "objective",
     "The response does not treat the student's incorrect answer or statement as if it "
     "were correct.",
     "Falsely endorsing a wrong value is a mathematical-correctness failure, distinct from "
     "the tutor's own arithmetic."),
    ("M5", "math", ["math"], "standard", "objective",
     "The mathematics the response uses is relevant to the problem the student is working "
     "on: it does not solve a different problem or apply an irrelevant procedure.",
     "Relevance of the mathematics to the task at hand is a content judgment separate from "
     "whether the arithmetic is correct."),
    ("M3", "math", ["math", "diagnosis"], "standard", "objective",
     "The response moves the student toward the mathematically correct resolution of their "
     "specific error.",
     "The path offered must be mathematically sound (content) and must actually target the "
     "diagnosed error (diagnosis)."),
    ("P1", "strategy", ["strategy"], "critical", "subjective",
     "The response guides the student toward the correction (e.g., a question, hint, or "
     "prompt to reconsider) rather than simply stating the right answer.",
     "Guiding rather than telling is the core pedagogical-strategy choice; giving the "
     "answer away is a strategy failure."),
    ("P2", "strategy", ["strategy"], "standard", "subjective",
     "The instructional move is appropriate for the situation (e.g., re-teaching for a "
     "misconception versus a brief nudge for a slip).",
     "Selecting a remediation move that fits the situation is a strategy judgment."),
    ("P3", "strategy", ["strategy", "affective"], "standard", "subjective",
     "The response leaves at least one substantive step for the student to complete "
     "themselves rather than carrying the solution through to its conclusion.",
     "Preserving productive struggle is a scaffolding strategy, and sustaining the "
     "student's sense of ownership is an affective goal."),
    ("P4", "strategy", ["strategy", "communication"], "standard", "subjective",
     "The response follows coherently from the preceding conversation and does not "
     "contradict or pointlessly repeat what the tutor already said.",
     "Tracking the dialogue so far is a strategy decision, and expressing that continuity "
     "is a communication quality."),
    ("C1", "communication", ["communication"], "standard", "subjective",
     "Whatever reasoning the response gives is easy to follow. Offering a single step or "
     "question, rather than a chain of reasoning, satisfies this.",
     "Clarity and organization are communication qualities."),
    ("C3", "communication", ["communication", "strategy"], "standard", "subjective",
     "The response stays focused on the error and avoids overwhelming the student with "
     "unnecessary content.",
     "Concise, focused delivery is a communication quality, and managing how much to "
     "reveal is a scaffolding-strategy decision."),
    ("A1", "affective", ["affective"], "standard", "subjective",
     "The response conveys some encouragement to the student. A few words (\"Nice try\", "
     "\"Good start\") suffice -- elaborate warmth is not required.",
     "The presence of positive affect is an affective quality, distinct from the mere "
     "absence of harshness. Bridge's own expert tutors encourage in three or four words, "
     "so length is not evidence of it."),
    ("A2", "affective", ["affective", "communication"], "standard", "subjective",
     "The response treats the mistake as something to work on rather than as a verdict. A "
     "brief \"not quite, let's check\" does this as well as a longer reassurance.",
     "Constructive framing is an affective stance enacted through how the message is "
     "communicated."),
    # PILOT: 0.996 pass, 0.032 discrimination -- effectively a constant. Kept anyway:
    # it is a genuine safety tripwire (a tutor that demeans a child IS a critical
    # failure), not a measurement item. Exclude it from an IRT fit rather than delete
    # the only check on it. See data/Bridge/README.md.
    ("A3", "affective", ["affective"], "critical", "subjective",
     "The response avoids discouraging or judgmental language that could undermine the "
     "student's confidence.",
     "Avoiding harm to the student's confidence is an affective concern; a serious "
     "violation is a critical failure."),
    ("A4", "affective", ["affective"], "standard", "subjective",
     "The response does not imply the student ought to already know this, or that the "
     "mistake is careless or disappointing. Explicit reassurance is not required.",
     "Establishing psychological safety is an affective goal distinct from tone and from "
     "how the mistake is framed. Stated negatively because expert tutors establish it by "
     "simply carrying on helpfully, not by announcing it."),
]

# ---------------------------------------------------------------------------
# TIER 2 -- ERROR-TYPE MODULES. Unioned per conversation (see docstring).
# ---------------------------------------------------------------------------
ERROR_MODULE_OF = {
    "misinterpret": "conceptual",
    "diagnose": "conceptual",
    "guess": "guess",
    "right-idea": "right_idea",
    "careless": "careless",
    "imprecise": "imprecise",
}

ERROR_MODULES: dict[str, list[Criterion]] = {
    # The student's reasoning itself is wrong.
    "conceptual": [
        ("D3", "diagnosis", ["diagnosis", "strategy"], "critical", "subjective",
         "The response addresses the underlying misconception or faulty reasoning behind "
         "the error, not just the surface wrong answer.",
         "Recognizing the misconception is diagnostic; choosing to target it rather than "
         "only the surface answer is a remediation-strategy decision."),
        ("D4", "diagnosis", ["diagnosis", "math"], "standard", "objective",
         "The response does not misstate the cause of the student's error.",
         "Any claim about why the student went wrong must be diagnostically and "
         "mathematically accurate; a response making no such claim does not misstate one."),
        ("D5", "strategy", ["strategy", "communication"], "standard", "subjective",
         "The response ends by asking the student to produce something specific -- an "
         "answer, a next step, or an explanation in their own words.",
         "Handing the next step back to the student is a scaffolding move realized through "
         "how the turn is closed."),
    ],
    # The student produced an answer with no visible reasoning.
    "guess": [
        ("G1", "diagnosis", ["diagnosis", "strategy"], "critical", "subjective",
         "The response does not attribute to the student a specific misconception the "
         "student never expressed.",
         "Withholding an unevidenced diagnosis is a diagnostic discipline enacted through "
         "the choice to probe rather than assert."),
        ("G2", "strategy", ["strategy"], "standard", "subjective",
         "The response gives the student a concrete, accessible starting point they can act "
         "on, given that they showed no working.",
         "Supplying an entry point when the student is stuck is a scaffolding decision."),
    ],
    # The student's approach is sound but the execution or completion is not.
    "right_idea": [
        ("R1", "affective", ["affective", "diagnosis"], "critical", "subjective",
         "The response does not treat the correct part of the student's thinking as wrong, "
         "nor ignore it and restart from scratch.",
         "Preserving what the student got right requires recognizing it (diagnosis) and is "
         "a confidence-preserving affective move."),
        ("R2", "strategy", ["strategy"], "standard", "subjective",
         "The response builds on the student's partial understanding rather than restarting "
         "the explanation from the beginning.",
         "Meeting the student where they already are is a scaffolding-strategy judgment."),
    ],
    # The student knows the method but slipped in execution.
    "careless": [
        ("S1", "strategy", ["strategy", "diagnosis"], "critical", "subjective",
         "The response does not re-teach the whole concept as though the student "
         "misunderstood it.",
         "Reading the error as procedural rather than conceptual is diagnostic, and "
         "calibrating the response size to it is a strategy choice."),
        ("S2", "strategy", ["strategy"], "standard", "subjective",
         "The response prompts the student to locate and fix the slip themselves rather "
         "than simply pointing at it.",
         "Handing back the correction is a scaffolding decision that preserves agency."),
    ],
    # The answer is on the right track but imprecise, unlabeled, or incomplete.
    "imprecise": [
        ("I1", "math", ["math", "diagnosis"], "critical", "objective",
         "The response does not treat the student's answer as entirely wrong when the issue "
         "is a precision gap (such as units, labels, rounding, or completeness).",
         "Judging which part is imprecise rather than wrong is diagnostic, and knowing what "
         "precision the answer requires is mathematical."),
        ("I2", "communication", ["communication", "math"], "standard", "objective",
         "The response does not itself use imprecise mathematical language or notation "
         "(e.g., dropping units or labels where they matter).",
         "Modelling precision is a communication act whose correctness is mathematical."),
    ],
}

# ---------------------------------------------------------------------------
# TIER 3 -- TOPIC-DOMAIN MODULES. Exactly one fires, keyed on `lesson_topic` keywords.
# Checked in order; first match wins.
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("proportional_reasoning",
     ("ratio", "unit rate", "percent", "proportional", "speed")),
    ("data_graphing",
     ("graph", "pictograph", "line plot", "coordinate", "ordered pair", "data",
      "populations and samples", "range and deviation")),
    ("geometry_spatial",
     ("shape", "area", "quadrilateral", "symmetry", "geometric line", "figure",
      "polygon", "reflection", "angle", "perimeter", "volume", "length",
      "unit square", "decompos", "triangle", "circle", "prism", "cylinder", "solid")),
    ("measurement_conversion",
     ("conversion", "converting", "units of measure", "standard measurement",
      "time", "money")),
    ("fractions",
     ("fraction", "mixed number")),
    ("place_value_number",
     ("place value", "rounding", "decimal", "comparing", "expanded notation",
      "counting numbers", "ordering", "prime", "factors and multiples",
      "rational number")),
    ("algebra_expressions",
     ("expression", "order of operations", "function", "slope", "algebra",
      "polynomial", "inverse", "unknown number", "equation", "variable")),
]
DEFAULT_TOPIC_DOMAIN = "operations_arithmetic"

TOPIC_MODULES: dict[str, list[Criterion]] = {
    "geometry_spatial": [
        ("V1", "communication", ["communication", "strategy"], "standard", "subjective",
         "The response refers to, describes, or invites a visual or spatial representation "
         "where doing so would help the student see the geometric relationship.",
         "Choosing to make the spatial idea visible is a scaffolding decision realized "
         "through how the explanation is expressed."),
        ("V2", "math", ["math", "communication"], "standard", "objective",
         "The response names the relevant figure, attribute or unit precisely (e.g. \"the "
         "base\", \"square centimetres\") rather than vague reference like \"this side\".",
         "Correct technical vocabulary is mathematical content conveyed through "
         "communication."),
    ],
    "fractions": [
        ("F1", "math", ["math", "diagnosis"], "standard", "objective",
         "The response treats the fraction or mixed number as a quantity -- referring to its "
         "size, or to what it is a part of -- and not only as two numbers to manipulate.",
         "The whole/part referent is the mathematical crux of this strand and the usual "
         "site of the student's error."),
    ],
    "place_value_number": [
        ("N1", "math", ["math", "communication"], "standard", "objective",
         "The response does not describe or use place value incorrectly (e.g., naming the "
         "wrong place, or treating digits only by position on the page).",
         "Precise place-value language is mathematical content delivered through "
         "communication."),
    ],
    "measurement_conversion": [
        ("U1", "math", ["math", "communication"], "standard", "objective",
         "The response does not confuse or omit the units involved where the unit is what "
         "the problem turns on.",
         "Unit reasoning is the mathematical crux of conversion and measurement problems "
         "and must be made explicit to the student."),
    ],
    "data_graphing": [
        ("B1", "math", ["math", "communication"], "standard", "objective",
         "The response does not misread or misdescribe the features of the data display "
         "(such as the scale, key, axis, or category).",
         "Reading a representation correctly is mathematical content that depends on "
         "directing the student's attention to the display."),
    ],
    "algebra_expressions": [
        ("X1", "math", ["math", "diagnosis"], "standard", "objective",
         "The response draws attention to the structure -- what the unknown stands for, or "
         "which operation applies first -- whether by stating it or by asking about it.",
         "Structural reasoning is the mathematical content of this strand and the usual "
         "site of the student's error."),
    ],
    "proportional_reasoning": [
        ("Z1", "math", ["math", "diagnosis"], "standard", "objective",
         "The response identifies which two quantities the rate or ratio compares.",
         "Identifying the compared quantities is the mathematical crux of proportional "
         "reasoning and the usual site of the student's error."),
    ],
    "operations_arithmetic": [
        ("O1", "math", ["math", "strategy"], "standard", "subjective",
         "The response connects the operation to its meaning or to the problem situation, "
         "rather than reciting the procedure alone.",
         "Grounding a procedure in its meaning is a mathematical-content choice made for "
         "pedagogical reasons."),
    ],
}

# ---------------------------------------------------------------------------
# TIER 4 -- GRADE-BAND MODULE. Exactly one fires, keyed on the lesson_topic grade.
# ---------------------------------------------------------------------------
GRADE_MODULES: dict[str, list[Criterion]] = {
    "1-3": [
        ("Y1", "communication", ["communication"], "standard", "subjective",
         "The language is concrete and everyday, suitable for an early-elementary student, "
         "and avoids abstract or formal notation the student would not yet know.",
         "Pitching vocabulary to an early-elementary audience is a communication quality."),
    ],
    "4-5": [
        ("Y2", "communication", ["communication"], "standard", "subjective",
         "The language is suitable for an upper-elementary student: it may name standard "
         "terms but explains them rather than assuming fluency.",
         "Pitching vocabulary to an upper-elementary audience is a communication quality."),
    ],
    "6-12": [
        ("Y3", "communication", ["communication", "math"], "standard", "subjective",
         "The language suits a secondary student: it is neither over-simplified nor "
         "needlessly technical, and any formal term it uses is used correctly.",
         "Appropriate register at secondary level is a communication quality that depends "
         "on correct use of formal mathematical terms."),
    ],
}

# Master bank + stable id suffix per code. Order fixes the suffix, so appending new
# codes at the end keeps every existing criterion_id stable.
ALL_CRITERIA: list[Criterion] = [
    *CORE_CRITERIA,
    *ERROR_MODULES["conceptual"], *ERROR_MODULES["guess"], *ERROR_MODULES["right_idea"],
    *ERROR_MODULES["careless"], *ERROR_MODULES["imprecise"],
    *TOPIC_MODULES["geometry_spatial"], *TOPIC_MODULES["fractions"],
    *TOPIC_MODULES["place_value_number"], *TOPIC_MODULES["measurement_conversion"],
    *TOPIC_MODULES["data_graphing"], *TOPIC_MODULES["algebra_expressions"],
    *TOPIC_MODULES["proportional_reasoning"], *TOPIC_MODULES["operations_arithmetic"],
    *GRADE_MODULES["1-3"], *GRADE_MODULES["4-5"], *GRADE_MODULES["6-12"],
]
CODE_INDEX = {code: i for i, (code, *_) in enumerate(ALL_CRITERIA, start=1)}
CRITERION_BY_CODE = {c[0]: c for c in ALL_CRITERIA}

# Which tier each code came from -- written onto the rubric as `applicability`.
TIER_OF_CODE: dict[str, str] = {
    **{c[0]: "core" for c in CORE_CRITERIA},
    **{c[0]: f"error_type:{mod}" for mod, cs in ERROR_MODULES.items() for c in cs},
    **{c[0]: f"topic_domain:{mod}" for mod, cs in TOPIC_MODULES.items() for c in cs},
    **{c[0]: f"grade_band:{mod}" for mod, cs in GRADE_MODULES.items() for c in cs},
}

SCENARIO_KEYS = [
    "scenario_id", "source_id", "use_case", "subject", "grade_band", "topic_domain",
    "modality", "prompt", "conversation_context", "reference_solution", "novice_response",
    "criterion_ids", "error_type", "error_types", "error_module", "visible_mistake",
    "expert_strategy", "expert_intention", "lesson_topic", "native_split", "source",
    "split", "version",
]
RUBRIC_KEYS = [
    "criterion_id", "scenario_id", "criterion", "expected_evidence", "scoring_type",
    "score_anchors", "criterion_code", "applicability", "primary_skill", "q_mapping",
    "q_rationale", "criticality", "objectivity", "explicitness", "source", "status",
    "version",
]

# TEKS lesson_topic codes start with the grade ("3.6B.Quadrilaterals") or an Algebra
# course prefix ("A2.7D.Factoring Polynomials").
_GRADE_RE = re.compile(r"^(?:(\d+)|A(\d+))\.")


def scenario_id(index: int) -> str:
    """Zero-based, 4-digit so the ids sort lexicographically."""
    return f"bridge_{index:04d}"


def join_turns(turns: list[dict] | None) -> str:
    """Join a list of message bubbles into one text block (one bubble per line)."""
    return "\n".join((t.get("text") or "").strip() for t in (turns or []) if (t.get("text") or "").strip())


def q_mapping(marked: list[str]) -> dict[str, int]:
    """Multi-hot over the five skills: 1 for every skill this criterion loads on."""
    marked_set = set(marked)
    return {skill: int(skill in marked_set) for skill in SKILLS}


def normalize_ack(text: str) -> str:
    """Lowercase and strip punctuation, for matching against ACKNOWLEDGMENTS."""
    return re.sub(r"[^a-z0-9 ]", "", (text or "").strip().lower()).strip()


def has_visible_mistake(prompt: str, context: list[dict]) -> bool:
    """True unless the student's turn is a bare assent with no numeric content anywhere.

    A bare acknowledgment leaves nothing to remediate only when the transcript also
    carries no figure the student could have gotten wrong; if a digit appears (in an
    earlier student turn, or in the tutor's restatement of the answer) the student's
    position is recoverable and the scenario is kept.
    """
    if normalize_ack(prompt) not in ACKNOWLEDGMENTS:
        return True
    blob = " ".join([t.get("content", "") for t in context] + [prompt])
    return bool(re.search(r"\d", blob))


def grade_band(lesson_topic: str | None) -> str:
    """Map a TEKS lesson_topic code to a grade band. Algebra courses count as secondary."""
    match = _GRADE_RE.match((lesson_topic or "").strip())
    if not match:
        return "unknown"
    if match.group(2) is not None:      # "A2." -- an Algebra course, i.e. secondary
        return "6-12"
    grade = int(match.group(1))
    if grade <= 3:
        return "1-3"
    if grade <= 5:
        return "4-5"
    return "6-12"


def topic_domain(lesson_topic: str | None) -> str:
    """Map a TEKS lesson_topic code to a math strand (first keyword match wins)."""
    text = (lesson_topic or "").lower()
    for domain, keywords in TOPIC_KEYWORDS:
        if any(k in text for k in keywords):
            return domain
    return DEFAULT_TOPIC_DOMAIN


def applicable_codes(error_module: str, domain: str, band: str) -> list[str]:
    """Assemble a scenario's criterion codes: core + one module from each tier.

    Returned in master-bank order so criterion_ids come out sorted.
    """
    codes = {c[0] for c in CORE_CRITERIA}
    codes.update(c[0] for c in ERROR_MODULES[error_module])
    codes.update(c[0] for c in TOPIC_MODULES[domain])
    codes.update(c[0] for c in GRADE_MODULES.get(band, GRADE_MODULES["4-5"]))
    return [code for (code, *_rest) in ALL_CRITERIA if code in codes]


def build() -> tuple[list[dict], list[dict], list[dict]]:
    ds = load_dataset(HF_DATASET)

    dropped: list[dict] = []
    staged: list[dict] = []          # rows surviving exclusion, pre-union
    # Canonical error module per conversation: the FIRST annotation wins (split order).
    # Every repeat of a conversation then draws the same module, so byte-identical stimuli
    # can no longer receive contradictory criteria -- and, unlike taking the union of all
    # annotators' modules, every scenario ends up with the SAME NUMBER of error criteria.
    # Unioning made difficulty depend on whether two experts happened to disagree.
    module_by_cid: dict[str, str] = {}

    # --- Pass 1: exclusions, and collect each conversation's error modules -------------
    for native_split in HF_SPLITS:
        for row in ds[native_split]:
            error_type = (row.get("e") or "").strip()
            c_id = row.get("c_id")

            def drop(reason: str) -> None:
                dropped.append({
                    "source_id": c_id,
                    "native_split": native_split,
                    "error_type": row.get("e"),
                    "reason": reason,
                })

            if error_type not in CLEAN_ERROR_TYPES:
                drop("no_clear_mistake")
                continue
            turns = row.get("c_h") or []
            if not turns:
                drop("empty_conversation")
                continue
            prompt_text = (turns[-1].get("text") or "").strip()
            if not prompt_text:
                drop("empty_student_turn")
                continue

            context = [
                {"role": t.get("user", "student"), "content": (t.get("text") or "").strip()}
                for t in turns[:-1]
            ]
            staged.append({
                "row": row, "native_split": native_split, "error_type": error_type,
                "c_id": c_id, "prompt": prompt_text, "context": context,
                # Flagged rather than dropped: see `visible_mistake` in the docstring.
                "visible_mistake": has_visible_mistake(prompt_text, context),
            })
            module_by_cid.setdefault(c_id, ERROR_MODULE_OF[error_type])

    # --- Pass 2: one scenario per ROW ---------------------------------------------------
    # Every surviving row becomes its own scenario, so the differing expert revisions stay
    # as separate items. `source_id` is therefore NON-unique by design: rows sharing a c_id
    # have the same stimulus and (after the union) the same criterion set, differing only
    # in `reference_solution`. That is real local dependence -- group or hold out on
    # `source_id` at calibration. See the README.
    error_types_by_cid: dict[str, set[str]] = defaultdict(set)
    for item in staged:
        error_types_by_cid[item["c_id"]].add(item["error_type"])

    scenarios: list[dict] = []
    rubrics: list[dict] = []

    for index, item in enumerate(staged):
        row = item["row"]
        sid = scenario_id(index)
        lesson_topic = row.get("lesson_topic")
        band = grade_band(lesson_topic)
        domain = topic_domain(lesson_topic)
        module = module_by_cid[item["c_id"]]

        codes = applicable_codes(module, domain, band)
        criterion_ids = [f"{sid}_c{CODE_INDEX[code]:02d}" for code in codes]

        scenarios.append({
            "scenario_id": sid,
            "source_id": item["c_id"],
            "use_case": USE_CASE,
            "subject": SUBJECT,
            "grade_band": band,
            "topic_domain": domain,
            "modality": "text",
            "prompt": item["prompt"],
            "conversation_context": item["context"],
            "reference_solution": join_turns(row.get("c_r_")),
            "novice_response": join_turns(row.get("c_r")),
            "criterion_ids": criterion_ids,
            "error_type": item["error_type"],
            "error_types": sorted(error_types_by_cid[item["c_id"]]),
            "error_module": module,
            "visible_mistake": item["visible_mistake"],
            "expert_strategy": row.get("z_what"),
            "expert_intention": row.get("z_why"),
            "lesson_topic": lesson_topic,
            "native_split": item["native_split"],
            "source": SOURCE_URL,
            "split": SPLIT,
            "version": VERSION,
        })

        for code in codes:
            _c, primary, marked, criticality, objectivity, text, rationale = (
                CRITERION_BY_CODE[code]
            )
            rubrics.append({
                "criterion_id": f"{sid}_c{CODE_INDEX[code]:02d}",
                "scenario_id": sid,
                "criterion": text,
                "expected_evidence": [],
                "scoring_type": "binary",
                "score_anchors": None,
                "criterion_code": code,
                "applicability": TIER_OF_CODE[code],
                "primary_skill": primary,
                "q_mapping": q_mapping(marked),
                "q_rationale": rationale,
                "criticality": criticality,
                "objectivity": objectivity,
                "explicitness": "explicit" if code in EXPLICIT_CODES else "implicit",
                "source": SOURCE_URL,
                "status": "approved",
                "version": VERSION,
            })

    return scenarios, rubrics, dropped


def validate(scenarios: list[dict], rubrics: list[dict]) -> list[str]:
    errs: list[str] = []

    for name, records, keys in (
        ("scenario", scenarios, SCENARIO_KEYS),
        ("rubric", rubrics, RUBRIC_KEYS),
    ):
        for r in records:
            if list(r.keys()) != keys:
                errs.append(f"{name} {list(r.values())[0]}: key set/order mismatch")
                break

    codes_all = [c[0] for c in ALL_CRITERIA]
    if len(set(codes_all)) != len(codes_all):
        errs.append("duplicate criterion code in the bank")

    sids = [s["scenario_id"] for s in scenarios]
    cids = [r["criterion_id"] for r in rubrics]
    if len(set(sids)) != len(sids):
        errs.append("duplicate scenario_id")
    if len(set(cids)) != len(cids):
        errs.append("duplicate criterion_id")

    declared = {c for s in scenarios for c in s["criterion_ids"]}
    actual = set(cids)
    if declared != actual:
        errs.append(
            f"criterion_ids mismatch: {len(declared - actual)} declared-but-missing, "
            f"{len(actual - declared)} present-but-undeclared"
        )
    orphans = {r["scenario_id"] for r in rubrics} - set(sids)
    if orphans:
        errs.append(f"{len(orphans)} rubrics reference unknown scenarios")

    valid_codes = set(codes_all)
    for s in scenarios:
        if not (s["prompt"] or "").strip():
            errs.append(f"{s['scenario_id']}: empty prompt (student turn)")
        if not s["criterion_ids"]:
            errs.append(f"{s['scenario_id']}: no criteria")
        if s["error_type"] not in CLEAN_ERROR_TYPES:
            errs.append(f"{s['scenario_id']}: error_type {s['error_type']!r} not a clean type")
        if s["topic_domain"] not in TOPIC_MODULES:
            errs.append(f"{s['scenario_id']}: unknown topic_domain {s['topic_domain']!r}")
        if s["grade_band"] not in GRADE_MODULES:
            errs.append(f"{s['scenario_id']}: unknown grade_band {s['grade_band']!r}")
        if s["error_module"] not in ERROR_MODULES:
            errs.append(f"{s['scenario_id']}: bad error_module {s['error_module']!r}")
        elif s["error_module"] not in {ERROR_MODULE_OF[e] for e in s["error_types"]}:
            errs.append(f"{s['scenario_id']}: error_module not implied by any error_type")
        if s["error_type"] not in s["error_types"]:
            errs.append(f"{s['scenario_id']}: canonical error_type missing from error_types")
        if s["visible_mistake"] != has_visible_mistake(s["prompt"], s["conversation_context"]):
            errs.append(f"{s['scenario_id']}: visible_mistake flag disagrees with the text")
        for turn in s["conversation_context"]:
            if turn.get("role") not in {"student", "tutor"}:
                errs.append(f"{s['scenario_id']}: context role {turn.get('role')!r} not student/tutor")
                break

    for r in rubrics:
        if not (r["criterion"] or "").strip():
            errs.append(f"{r['criterion_id']}: empty criterion")
        if r["criterion_code"] not in valid_codes:
            errs.append(f"{r['criterion_id']}: unknown criterion_code {r['criterion_code']!r}")
        q = r["q_mapping"]
        if list(q.keys()) != SKILLS:
            errs.append(f"{r['criterion_id']}: q_mapping keys/order != {SKILLS}")
        if set(q.values()) - {0, 1}:
            errs.append(f"{r['criterion_id']}: q_mapping values must be 0/1")
        if sum(q.values()) < 1:
            errs.append(f"{r['criterion_id']}: q_mapping maps to no skill")
        if r["primary_skill"] not in SKILLS:
            errs.append(f"{r['criterion_id']}: primary_skill {r['primary_skill']!r} not a skill")
        elif q.get(r["primary_skill"]) != 1:
            errs.append(f"{r['criterion_id']}: primary_skill not marked in q_mapping")
        want_expl = "explicit" if r["criterion_code"] in EXPLICIT_CODES else "implicit"
        if r["explicitness"] != want_expl:
            errs.append(f"{r['criterion_id']}: explicitness {r['explicitness']!r} != {want_expl!r}")

    # Every skill needs pure (single-loading) anchors or its dimension is not separably
    # identified; two distinct criterion texts is the minimum this bank commits to.
    pure_codes: dict[str, set[str]] = defaultdict(set)
    for code, _p, marked, *_rest in ALL_CRITERIA:
        if len(marked) == 1:
            pure_codes[marked[0]].add(code)
    for skill in SKILLS:
        if len(pure_codes[skill]) < 2:
            errs.append(f"skill {skill!r} has fewer than 2 pure-loading criteria: "
                        f"{sorted(pure_codes[skill])}")

    # Applicability consistency.
    for s in scenarios:
        expected = applicable_codes(s["error_module"], s["topic_domain"], s["grade_band"])
        expected_ids = [f"{s['scenario_id']}_c{CODE_INDEX[c]:02d}" for c in expected]
        if s["criterion_ids"] != expected_ids:
            errs.append(f"{s['scenario_id']}: criterion_ids do not match its tier modules")

    # source_id is intentionally NON-unique (one scenario per expert annotation), but every
    # row of a conversation must agree on the unioned module set -- that agreement is the
    # whole point of the conversation-level gate.
    by_cid: dict[str, set[str]] = defaultdict(set)
    for s in scenarios:
        by_cid[s["source_id"]].add(s["error_module"])
    for cid, sets in by_cid.items():
        if len(sets) > 1:
            errs.append(f"conversation {cid}: inconsistent error_module across repeats")

    by_sid = {s["scenario_id"]: s for s in scenarios}
    for r in rubrics:
        scen = by_sid.get(r["scenario_id"])
        if scen is None:
            continue
        tier = TIER_OF_CODE[r["criterion_code"]]
        if r["applicability"] != tier:
            errs.append(f"{r['criterion_id']}: applicability {r['applicability']!r} != {tier!r}")
        if tier.startswith("error_type:"):
            if tier.split(":", 1)[1] != scen["error_module"]:
                errs.append(f"{r['criterion_id']}: {tier} != {scen['error_module']}")
        elif tier.startswith("topic_domain:") and tier != f"topic_domain:{scen['topic_domain']}":
            errs.append(f"{r['criterion_id']}: {tier} attached to {scen['topic_domain']!r}")
        elif tier.startswith("grade_band:") and tier != f"grade_band:{scen['grade_band']}":
            errs.append(f"{r['criterion_id']}: {tier} attached to {scen['grade_band']!r}")

    unused = valid_codes - {r["criterion_code"] for r in rubrics}
    if unused:
        errs.append(f"criteria never attached to any scenario: {sorted(unused)}")

    return errs


def write(name: str, records: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json_path = OUT_DIR / f"{name}.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    size = jsonl_path.stat().st_size / 1e6
    print(f"wrote {jsonl_path.relative_to(ROOT)} + .json  ({len(records)} rows, {size:.1f} MB)")


def main() -> None:
    scenarios, rubrics, dropped = build()

    errs = validate(scenarios, rubrics)
    if errs:
        print(f"VALIDATION FAILED ({len(errs)} issues):", file=sys.stderr)
        for e in errs[:20]:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"validation passed: {len(scenarios)} scenarios, {len(rubrics)} criteria "
          f"from a bank of {len(ALL_CRITERIA)}")

    write("scenarios", scenarios)
    write("rubrics", rubrics)
    if dropped:
        write("dropped", dropped)

    loads = Counter(s for r in rubrics for s in SKILLS if r["q_mapping"][s])
    primaries = Counter(r["primary_skill"] for r in rubrics)
    pure = Counter()
    pure_codes: dict[str, set[str]] = defaultdict(set)
    for r in rubrics:
        marked = [k for k, v in r["q_mapping"].items() if v == 1]
        if len(marked) == 1:
            pure[marked[0]] += 1
            pure_codes[marked[0]].add(r["criterion_code"])
    print("\nq_mapping loads / primary / PURE single-loading instances:")
    for skill in SKILLS:
        print(f"  {skill:<14} load={loads.get(skill, 0):>6}  primary={primaries.get(skill, 0):>6}"
              f"  pure={pure.get(skill, 0):>6} via {sorted(pure_codes[skill])}")

    per_scenario = Counter(len(s["criterion_ids"]) for s in scenarios)
    print("\ncriteria per scenario: " +
          ", ".join(f"{n}->{c}" for n, c in sorted(per_scenario.items())))

    print("\ntopic_domain / grade_band distribution:")
    for key in ("topic_domain", "grade_band"):
        dist = Counter(s[key] for s in scenarios)
        print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in dist.most_common()))

    nmod = Counter(s["error_module"] for s in scenarios)
    disagree = sum(1 for s in scenarios if len(s["error_types"]) > 1)
    print("\nerror module (canonical, one per conversation): "
          + ", ".join(f"{k}={v}" for k, v in nmod.most_common()))
    print(f"scenarios whose conversation had disagreeing expert labels: {disagree} "
          f"(all still draw the same module, so criterion counts stay comparable)")

    cid_counts = Counter(s["source_id"] for s in scenarios)
    dup_scenarios = sum(n for n in cid_counts.values() if n > 1)
    print(f"unique conversations (source_id): {len(cid_counts)}   "
          f"scenarios sharing a conversation: {dup_scenarios}")
    no_vis = sum(1 for s in scenarios if not s["visible_mistake"])
    print(f"visible_mistake=false (kept, but the student's error is not in the text): "
          f"{no_vis} scenarios / "
          f"{sum(len(s['criterion_ids']) for s in scenarios if not s['visible_mistake'])} criteria")
    print("\ndropped by reason: " +
          ", ".join(f"{k}={v}" for k, v in Counter(d["reason"] for d in dropped).most_common()))
    print("difficulty/discrimination: absent by design -- to be calibrated from real "
          "judge responses, not synthesized.")


if __name__ == "__main__":
    main()
