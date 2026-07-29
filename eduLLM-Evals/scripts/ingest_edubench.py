"""Convert the EduBench English task splits into Scenario + Rubric Schema JSONL.

Source: https://huggingface.co/datasets/DirectionAI/EduBench  (`en_data/`, nine
JSONL files, one per educational task type, 9,163 records total -- 517 fewer than
Table 2 of the paper reports; the published files are what this script trusts).

Mirrors the InFoBench pipeline shape (scripts/ingest_infobench.py): this emits the
base schema fields plus a **q-matrix**, and leaves `difficulty`/`discrimination`
as explicit nulls for scripts/assign_irt_params.py to fill. Re-running this script
resets them, so always re-run the assign step after a rebuild:

    python scripts/ingest_edubench.py
    python scripts/assign_irt_params.py \
        --input data/EduBench/rubrics.jsonl \
        --skills answering_questions,error_correction,idea_provision,learning_support,mental_health,question_generation,grading,material_generation,personalized_content_creation \
        --log-dir data/EduBench/irt_logs --no-backup

Mapping (EduBench field -> schema field):
    <file stem>            -> use_case + source     (one task type per file)
    information["Subject"] -> subject
    information[<alias>]   -> grade_band            (three key names, see GRADE_KEYS)
    prompt                 -> prompt                (verbatim)
    information["Answer"]  -> reference_solution     (Q&A / EC only)
    Table 7 row            -> criterion + q_mapping (one rubric row per checkmark)

EduBench allocates its 12 rubric metrics per task type, not per item: Table 7 of
the paper is a metric-by-task applicability matrix, so every record in a file gets
the same set of criteria (between 3 for PCC and 8 for IP). The criterion text is
EduBench's own -- title, `Description:` and the 9-10/7-8 score anchors quoted from
Appendix H -- with the paper's generic referents resolved to the current task, and
the two top anchors joined by "or" to dichotomize the 10-point scale at >= 7-8.

`model_predictions` (five model responses per record) is discarded at parse time;
nothing downstream consumes it, and keeping it would multiply the artifact size.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "EduBench"

HF_DATASET = "DirectionAI/EduBench"
SOURCE_URL = "https://huggingface.co/datasets/DirectionAI/EduBench"
DATA_DIR = "en_data"
SPLIT = "calibration"   # pipeline-role label (matches InFoBench/WildBench), not an HF split
VERSION = "1.0"

# Placeholder metadata (EduBench has no native equivalent; matches InFoBench/WildBench).
CRITICALITY = "critical"
OBJECTIVITY = "objective"
EXPLICITNESS = "explicit"

# The subject the paper's rubric leaves implicit ("Did it fully understand...").
MODEL = "the tutoring model"

# One task type per file. `slug` is used for BOTH `use_case` and the q-matrix column
# name so a task never has two names. The list order is the fixed q-matrix column
# order and the scenario concatenation order -- do not reorder in place, downstream
# MIRT code indexes skills positionally.
#
# `core_task` and `display` resolve the generic referents in the Appendix H text:
# "the core task (e.g., solving problems, ...)" and "In error correction scenarios".
#
# `n_records` is the line count of the published `en_data/<stem>.jsonl`; `n_paper` is
# the English count Table 2 of the paper reports for the same task. Every file ships
# fewer records than the paper claims (9,163 against 9,680), so `n_records` is the
# figure validate() enforces -- the artifact is authoritative, the table is not.
TASKS = [
    {
        "stem": "Q&A",
        "slug": "answering_questions",
        "display": "Question Answering",
        "core_task": "answering the student's subject question and producing the correct answer",
        "reference_key": "Answer",
        "n_records": 1285,
        "n_paper": 1328,
    },
    {
        "stem": "EC",
        "slug": "error_correction",
        "display": "Error Correction",
        "core_task": "correcting the student's answer and explaining the error",
        "reference_key": "Corrected Answer",
        "n_records": 1301,
        "n_paper": 1350,
    },
    {
        "stem": "IP",
        "slug": "idea_provision",
        "display": "Idea Provision",
        "core_task": "providing reasoning and hints toward the question without giving the answer directly",
        "reference_key": None,
        "n_records": 1301,
        "n_paper": 1350,
    },
    {
        "stem": "PLS",
        "slug": "learning_support",
        "display": "Personalized Learning Support",
        "core_task": "generating personalized learning content or tasks from the student's profile",
        "reference_key": None,
        "n_records": 448,
        "n_paper": 561,
    },
    {
        "stem": "ES",
        "slug": "mental_health",
        "display": "Emotional Support",
        "core_task": "responding with emotional support to the student's dialogue at the stated anxiety level",
        "reference_key": None,
        "n_records": 1061,
        "n_paper": 1074,
    },
    {
        "stem": "QG",
        "slug": "question_generation",
        "display": "Question Generation",
        "core_task": "generating a question from the subject, education level, knowledge point, and question type",
        "reference_key": None,
        "n_records": 1288,
        "n_paper": 1338,
    },
    {
        "stem": "AG",
        "slug": "grading",
        "display": "Automatic Grading",
        "core_task": "grading the student's answer with a score, scoring details, and personalized feedback",
        "reference_key": None,
        "n_records": 1042,
        "n_paper": 1073,
    },
    {
        "stem": "TMG",
        "slug": "material_generation",
        "display": "Teaching Material Generation",
        "core_task": "generating teaching materials with objectives, key points and difficulties, "
                     "and classroom activity design",
        "reference_key": None,
        "n_records": 1185,
        "n_paper": 1347,
    },
    {
        "stem": "PCC",
        "slug": "personalized_content_creation",
        "display": "Personalized Content Creation",
        "core_task": "creating personalized learning path planning and recommendations from the "
                     "student's profile",
        "reference_key": None,
        "n_records": 252,
        "n_paper": 259,
    },
]
SKILLS = [t["slug"] for t in TASKS]

# The same semantic field carries three different key names across the nine files:
# "Education Level" (Q&A, PLS, ES, QG), "Level" (AG, TMG, PCC), "Difficulty" (EC, IP).
# EC/IP's "Difficulty" holds a grade band ("High School"), not an easy/medium/hard
# rating -- no difficulty field exists anywhere in the source.
GRADE_KEYS = ["Education Level", "Level", "Difficulty"]
SUBJECT_KEY = "Subject"

# EduBench's 12 rubric metrics (Appendix H), in the paper's H.1.1 -> H.3.4 order.
#
# `applies` is the metric's row of Table 7: the set of task types the paper scores it
# in. `desc` is the paper's `Description:` line and `pass_high`/`pass_mid` are its
# 9-10 and 7-8 anchors, quoted verbatim except for the {model}/{core_task}/{display}
# substitutions and a lowercased initial on `pass_mid` so it reads after the "or".
METRICS = [
    {
        "abbr": "IFTC",
        "title": "Instruction Following & Task Completion",
        "applies": {
            "answering_questions", "error_correction", "idea_provision", "learning_support",
            "mental_health", "question_generation", "grading", "material_generation",
            "personalized_content_creation",
        },
        "desc": "Did {model} fully understand and execute the user's instruction? Was the core "
                "task ({core_task}) completed? Is the output formatting correct?",
        "pass_high": "{Model} fully understood and precisely executed all instructions; achieved "
                     "core task with perfect accuracy; output format is fully compliant.",
        "pass_mid": "{model} accurately understood main instructions and correctly completed the "
                    "task; core goals are well achieved; format is mostly correct with only minor "
                    "omissions or deviations.",
    },
    {
        "abbr": "RTC",
        "title": "Role & Tone Consistency",
        "applies": {
            "mental_health", "material_generation", "error_correction", "idea_provision",
            "grading", "answering_questions", "personalized_content_creation",
            "learning_support",
        },
        "desc": "Does the language style, tone, and level of professionalism of {model} match the "
                "assigned role (e.g., teacher, teaching assistant, peer) and the target learner "
                "group (e.g., elementary, college)?",
        "pass_high": "Excellent role-playing (e.g., teacher/TA); language style, professionalism, "
                     "and tone (e.g., encouraging/serious) are perfectly aligned with the assumed "
                     "role and audience.",
        "pass_mid": "role and tone are mostly consistent and appropriate for the scenario, with "
                    "minor deviation in individual expressions.",
    },
    {
        "abbr": "CRSC",
        "title": "Content Relevance & Scope Control",
        "applies": {
            "answering_questions", "idea_provision", "learning_support", "question_generation",
            "grading", "material_generation", "personalized_content_creation",
        },
        "desc": "Is the content produced by {model} tightly aligned with the specified topic, "
                "theme, or question? Is it kept within the specified difficulty level, scenario, "
                "or scope?",
        "pass_high": "Content is highly relevant to the specified topic/theme/question; strictly "
                     "within required difficulty/scope/discipline without redundant or irrelevant "
                     "information.",
        "pass_mid": "overall relevance is high; scope control is good with possibly a small amount "
                    "of slightly off-topic or mildly overreaching information.",
    },
    {
        "abbr": "SEI",
        "title": "Scenario Element Integration",
        "applies": {
            "error_correction", "idea_provision", "learning_support", "mental_health",
            "personalized_content_creation",
        },
        "desc": "Did {model} effectively use scenario-specific information (e.g., previous student "
                "answers, learning preferences, specific teaching goals)? This is especially "
                "important in {display} contexts.",
        "pass_high": "{Model} fully integrated all key scenario elements (e.g., student history, "
                     "learning preferences); output is highly personalized and well-matched to the "
                     "teaching context.",
        "pass_mid": "{model} used major scenario elements effectively; response is targeted, "
                    "possibly overlooks minor details but does not affect overall results.",
    },
    {
        "abbr": "BFA",
        "title": "Basic Factual Accuracy",
        "applies": {
            "answering_questions", "error_correction", "idea_provision", "question_generation",
            "grading", "material_generation", "personalized_content_creation",
        },
        "desc": "Are objective facts such as concept definitions, formulas, dates, terminology, "
                "code syntax, legal clauses correctly presented by {model}?",
        "pass_high": "All stated factual elements (definitions, formulas, dates, terms, syntax, "
                     "etc.) are completely accurate.",
        "pass_mid": "the vast majority of facts are correct; possibly contains very minor, "
                    "non-critical typos or omissions.",
    },
    {
        "abbr": "DKA",
        "title": "Domain Knowledge Accuracy",
        "applies": {
            "idea_provision", "question_generation", "material_generation",
            "error_correction", "grading", "answering_questions",
            "personalized_content_creation",
        },
        "desc": "Is the use of subject matter knowledge (math, programming, law, finance, etc.) by "
                "{model} not only correct but also appropriately specialized and aligned with "
                "domain standards?",
        "pass_high": "Subject matter application is not only accurate but also shows appropriate "
                     "depth and rigor; adheres to industry or academic standards.",
        "pass_mid": "proper use of professional knowledge reflecting a good degree of proficiency; "
                    "minor shortcomings in depth or detail not affecting validity.",
    },
    {
        "abbr": "RPR",
        "title": "Reasoning Process Rigor",
        "applies": {"answering_questions", "error_correction", "idea_provision", "grading"},
        "desc": "For content requiring reasoning (e.g., math steps, code logic, legal arguments, "
                "case analysis), is the logical flow of {model} complete and sound?",
        "pass_high": "Reasoning is complete, clear, and rigorous; all steps are correct; arguments "
                     "are strong and free of logical fallacies.",
        "pass_mid": "reasoning is largely correct and logically coherent with minor issues in "
                    "individual steps or details that do not affect the conclusion.",
    },
    {
        "abbr": "EICP",
        "title": "Error Identification & Correction Precision",
        "applies": {"error_correction", "grading"},
        "desc": "In {display} scenarios, are errors precisely identified by {model} (no missed or "
                "false positives)? Are the corrections correct and optimal?",
        "pass_high": "{Model} precisely identified all errors (no omission or false positives); "
                     "provided completely correct, clear, and optimal correction suggestions.",
        "pass_mid": "{model} correctly located most major errors; suggestions are generally "
                    "accurate and effective with only minor omissions or less-than-perfect advice.",
    },
    {
        "abbr": "CSI",
        "title": "Clarity, Simplicity & Inspiration",
        "applies": {
            "error_correction", "idea_provision", "question_generation", "material_generation",
            "grading", "answering_questions", "personalized_content_creation",
        },
        "desc": "Are the explanations, descriptions, and feedback of {model} clear, concise, and "
                "easy for the target learners to understand? Is the delivery inspiring and "
                "thought-provoking?",
        "pass_high": "Extremely clear and concise explanations; fully accessible for target "
                     "learners; vibrant and engaging delivery that inspires deep thought and "
                     "interest.",
        "pass_mid": "clear and easy to understand; appropriate for learner level; somewhat "
                    "thought-provoking and can trigger reflection.",
    },
    {
        "abbr": "MGP",
        "title": "Motivation, Guidance & Positive Feedback",
        "applies": {"error_correction", "mental_health", "grading"},
        "desc": "Does the interaction provide encouragement and support? Is constructive and "
                "positive language used? In {display}, does {model} guide thinking or just give "
                "away answers?",
        "pass_high": "Strongly supportive and encouraging; consistently uses constructive and "
                     "positive language; offers highly effective heuristic guidance instead of "
                     "simply giving answers.",
        "pass_mid": "generally supportive tone and positive language; provides useful guidance "
                    "though occasionally too direct.",
    },
    {
        "abbr": "PAS",
        "title": "Personalization, Adaptation & Learning Support",
        "applies": {"learning_support", "mental_health", "personalized_content_creation"},
        "desc": "Can {model} provide differentiated content, advice, or feedback based on a "
                "student's level, traits, or needs? Does {model} recommend effective learning "
                "paths or resources?",
        "pass_high": "Highly personalized content/advice/feedback based on student "
                     "level/traits/needs; resource and learning path suggestions are accurate, "
                     "practical, and valuable.",
        "pass_mid": "demonstrates some adaptation to student situation; provides relevant learning "
                    "advice or resources with good utility.",
    },
    {
        "abbr": "HOTS",
        "title": "Higher-Order Thinking & Skill Development",
        "applies": {
            "idea_provision", "learning_support", "question_generation", "material_generation",
        },
        "desc": "Does the interaction or content of {model} help foster students' critical "
                "thinking, creativity, problem-solving, or knowledge transfer skills?",
        "pass_high": "Skillfully designed to promote critical/creative thinking, problem-solving, "
                     "or transfer of knowledge (e.g., through open-ended questions, comparative "
                     "analysis, case study, project-based tasks).",
        "pass_mid": "includes guiding questions or moderately challenging tasks that positively "
                    "support the development of higher-order thinking (e.g., analysis, evaluation, "
                    "application).",
    },
]

Q_RATIONALE = "Table 7 of EduBench paper"

SCENARIO_KEYS = [
    "scenario_id", "use_case", "subject", "grade_band", "modality", "prompt",
    "conversation_context", "reference_solution", "criterion_ids", "source", "split",
    "version",
]
RUBRIC_KEYS = [
    "criterion_id", "scenario_id", "criterion", "expected_evidence", "scoring_type",
    "score_anchors", "q_mapping", "q_rationale", "criticality", "objectivity",
    "explicitness", "source", "status", "version", "difficulty", "discrimination",
]


def metrics_for(slug: str) -> list[dict]:
    """The Table 7 column for one task, in the paper's H.1.1 -> H.3.4 row order."""
    return [m for m in METRICS if slug in m["applies"]]


def build_criterion(metric: dict, task: dict) -> str:
    """Title + Appendix H description + the two passing anchors joined by "or".

    Concatenating the 9-10 and 7-8 anchors is what makes the criterion binary: a
    pass is a response that would have scored at least 7-8 on the paper's scale.
    """
    sub = {
        "model": MODEL,
        "Model": MODEL[0].upper() + MODEL[1:],
        "core_task": task["core_task"],
        "display": task["display"],
    }
    return (
        f"Criteria: {metric['title']} ({metric['abbr']}).\n"
        f"General Description: {metric['desc'].format(**sub)}\n"
        f"Passing Description: {metric['pass_high'].format(**sub)} "
        f"or {metric['pass_mid'].format(**sub)}"
    )


def q_mapping(metric: dict, task: dict, mode: str) -> dict[str, int]:
    """`row`: the metric's full Table 7 row. `onehot`: that row ∩ this task.

    A Table 7 row records where a metric is *scored*, not what a given response
    *exercises*, so `row` makes every IFTC criterion load on all nine task
    dimensions. `onehot` is the between-item multidimensional reading instead.
    """
    if mode == "onehot":
        return {s: int(s == task["slug"]) for s in SKILLS}
    return {s: int(s in metric["applies"]) for s in SKILLS}


def use_os_trust_store() -> None:
    """Verify HTTPS against the operating system trust store instead of certifi.

    Needed where a TLS-inspecting antivirus or corporate proxy re-signs traffic with
    a root the OS trusts but certifi does not ship. Certificates are still verified
    and hostnames still checked; the context simply omits the RFC 5280 strictness
    that such generated roots tend to violate.
    """
    import httpx
    from huggingface_hub import set_client_factory

    try:
        from huggingface_hub.utils._http import hf_request_event_hook
        hooks = {"request": [hf_request_event_hook]}
    except ImportError:
        hooks = {}

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    set_client_factory(
        lambda: httpx.Client(
            verify=context, event_hooks=hooks, follow_redirects=True, timeout=None
        )
    )


def resolve_paths() -> dict[str, str]:
    """Map file stem -> repo path, checked against the repo's actual file listing."""
    available = {
        f for f in list_repo_files(HF_DATASET, repo_type="dataset")
        if f.startswith(f"{DATA_DIR}/") and f.endswith(".jsonl")
    }
    paths, missing = {}, []
    for task in TASKS:
        candidate = f"{DATA_DIR}/{task['stem']}.jsonl"
        if candidate in available:
            paths[task["stem"]] = candidate
        else:
            missing.append(candidate)
    if missing:
        raise SystemExit(
            f"expected files not in {HF_DATASET}: {missing}\n"
            f"available under {DATA_DIR}/: {sorted(available)}"
        )
    return paths


def read_records(path: str) -> list[dict]:
    """Line-by-line json.loads, not datasets.load_dataset.

    The HF dataset-viewer 500s on this repo because `information["Student Profile"]`
    is an object in some records and a bare string in others, which Arrow cannot
    unify into one schema. Reading the raw JSONL sidesteps the type conflict.
    """
    local = hf_hub_download(HF_DATASET, path, repo_type="dataset")
    records = []
    with open(local, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: {exc}") from exc
    return records


def build(mode: str) -> tuple[list[dict], list[dict], dict]:
    paths = resolve_paths()

    scenarios: list[dict] = []
    rubrics: list[dict] = []
    stats: dict = {
        "counts": {},            # stem -> records read
        "info_keys": {},         # stem -> sorted information keys seen
        "grade_key": {},         # stem -> which GRADE_KEYS alias was used
        "grade_values": {},      # stem -> Counter of grade_band values
        "subjects": {},          # stem -> Counter of subject values
        "ref_missing": Counter(),
        "grade_missing": Counter(),
    }

    index = 0
    for task in TASKS:
        stem, slug = task["stem"], task["slug"]
        records = read_records(paths[stem])
        stats["counts"][stem] = len(records)

        allocated = metrics_for(slug)
        info_keys: set[str] = set()
        grade_keys_used: Counter = Counter()
        grade_values: Counter = Counter()
        subjects: Counter = Counter()

        for row in records:
            info = row.get("information")
            if not isinstance(info, dict):
                raise SystemExit(f"{stem} record {index}: `information` is {type(info).__name__}")
            info_keys.update(info.keys())

            grade, grade_key = None, None
            for key in GRADE_KEYS:
                value = info.get(key)
                if isinstance(value, str) and value.strip():
                    grade, grade_key = value.strip(), key
                    break
            if grade is None:
                stats["grade_missing"][stem] += 1
            else:
                grade_keys_used[grade_key] += 1
                grade_values[grade] += 1

            subject = info.get(SUBJECT_KEY)
            subject = subject.strip() if isinstance(subject, str) else None
            if subject:
                subjects[subject] += 1

            reference = ""
            if task["reference_key"]:
                value = info.get(task["reference_key"])
                if isinstance(value, str) and value.strip():
                    reference = value.strip()
                else:
                    stats["ref_missing"][stem] += 1

            sid = f"eb_{index}"
            criterion_ids = [f"{sid}_c{n:02d}" for n in range(1, len(allocated) + 1)]

            scenarios.append({
                "scenario_id": sid,
                "use_case": slug,
                "subject": subject,
                "grade_band": grade,
                "modality": "text",   # zh_data and any image split are out of scope
                "prompt": row.get("prompt") or "",
                "conversation_context": [],   # ES's dialogue is already inside `prompt`
                "reference_solution": reference,
                "criterion_ids": criterion_ids,
                "source": f"Edubench_{stem} JSONL",
                "split": SPLIT,
                "version": VERSION,
            })

            for cid, metric in zip(criterion_ids, allocated):
                rubrics.append({
                    "criterion_id": cid,
                    "scenario_id": sid,
                    "criterion": build_criterion(metric, task),
                    "expected_evidence": [],
                    "scoring_type": "binary",
                    "score_anchors": None,   # binary; the 10-point scale is folded into `criterion`
                    "q_mapping": q_mapping(metric, task, mode),
                    "q_rationale": Q_RATIONALE,
                    # Placeholders (uniform, like InFoBench); feed only the synthetic IRT step.
                    "criticality": CRITICALITY,
                    "objectivity": OBJECTIVITY,
                    "explicitness": EXPLICITNESS,
                    "source": SOURCE_URL,
                    "status": "approved",
                    # Explicit nulls: the bank is declaredly uncalibrated until
                    # assign_irt_params.py overwrites these in place.
                    "version": VERSION,
                    "difficulty": None,
                    "discrimination": None,
                })
            index += 1

        stats["info_keys"][stem] = sorted(info_keys)
        stats["grade_key"][stem] = dict(grade_keys_used)
        stats["grade_values"][stem] = grade_values
        stats["subjects"][stem] = subjects

    return scenarios, rubrics, stats


def validate(scenarios: list[dict], rubrics: list[dict], stats: dict, mode: str) -> list[str]:
    errs: list[str] = []

    for name, records, keys in (
        ("scenario", scenarios, SCENARIO_KEYS),
        ("rubric", rubrics, RUBRIC_KEYS),
    ):
        for r in records:
            if list(r.keys()) != keys:
                errs.append(f"{name} {list(r.values())[0]}: key set/order mismatch")
                break

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

    for s in scenarios:
        if not (s["prompt"] or "").strip():
            errs.append(f"{s['scenario_id']}: empty prompt")
        if not s["criterion_ids"]:
            errs.append(f"{s['scenario_id']}: no criteria")
        if not (s["grade_band"] or "").strip():
            errs.append(f"{s['scenario_id']}: no grade_band from any of {GRADE_KEYS}")
        if not (s["subject"] or "").strip():
            errs.append(f"{s['scenario_id']}: no subject")
        if s["use_case"] not in SKILLS:
            errs.append(f"{s['scenario_id']}: use_case {s['use_case']!r} not a task slug")

    for r in rubrics:
        if not (r["criterion"] or "").strip():
            errs.append(f"{r['criterion_id']}: empty criterion")
        if "<" in r["criterion"] or ">" in r["criterion"]:
            errs.append(f"{r['criterion_id']}: unresolved <> placeholder in criterion")
        if "{" in r["criterion"] or "}" in r["criterion"]:
            errs.append(f"{r['criterion_id']}: unresolved {{}} substitution in criterion")
        q = r["q_mapping"]
        if list(q.keys()) != SKILLS:
            errs.append(f"{r['criterion_id']}: q_mapping keys/order != the 9 task slugs")
        if set(q.values()) - {0, 1}:
            errs.append(f"{r['criterion_id']}: q_mapping values must be 0/1")
        if sum(q.values()) == 0:
            errs.append(f"{r['criterion_id']}: q_mapping is all zeros")
        if r["difficulty"] is not None or r["discrimination"] is not None:
            errs.append(f"{r['criterion_id']}: difficulty/discrimination must ship as null")

    # Per-task criterion count must equal that task's Table 7 column total.
    per_scenario = Counter(r["scenario_id"] for r in rubrics)
    by_task = {s["scenario_id"]: s["use_case"] for s in scenarios}
    seen: dict[str, set] = defaultdict(set)
    for sid, n in per_scenario.items():
        seen[by_task[sid]].add(n)
    for task in TASKS:
        want = len(metrics_for(task["slug"]))
        got = seen.get(task["slug"], set())
        if got != {want}:
            errs.append(f"{task['stem']}: criteria per scenario {sorted(got)} != Table 7 total {want}")

    # Record counts against the published files (not the paper -- see TASKS).
    for task in TASKS:
        got = stats["counts"][task["stem"]]
        if got != task["n_records"]:
            errs.append(
                f"{task['stem']}: read {got} records, en_data/{task['stem']}.jsonl "
                f"held {task['n_records']} at ingest time"
            )

    if stats["ref_missing"]:
        errs.append(f"reference_solution key missing for: {dict(stats['ref_missing'])}")

    # Under 'row', each metric's q_mapping is its own Table 7 applicability row, so distinct
    # count can be less than len(METRICS) if two metrics happen to apply to the same task
    # set -- that's a legitimate coincidence, not a bug. It can never exceed len(METRICS)
    # (mode 'onehot': len(TASKS)), since every row's vector is one of those fixed sets.
    distinct_q = len({json.dumps(r["q_mapping"], sort_keys=True) for r in rubrics})
    max_q = len(METRICS) if mode == "row" else len(TASKS)
    if distinct_q > max_q:
        errs.append(f"{distinct_q} distinct q_mapping values, expected at most {max_q} under "
                    f"{mode!r}")

    return errs


def write(name: str, records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json_path = out_dir / f"{name}.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    size = jsonl_path.stat().st_size / 1e6
    pretty = json_path.stat().st_size / 1e6
    try:
        shown = jsonl_path.relative_to(ROOT)
    except ValueError:
        shown = jsonl_path
    print(f"wrote {shown} ({len(records)} rows, {size:.1f} MB) + .json ({pretty:.1f} MB)")


def report(scenarios: list[dict], rubrics: list[dict], stats: dict, mode: str) -> None:
    print("\nper-file records, grade_band alias, criteria/scenario:")
    total = paper = 0
    for task in TASKS:
        stem = task["stem"]
        got = stats["counts"][stem]
        flag = "" if got == task["n_records"] else f"  <-- file changed, was {task['n_records']}"
        alias = ", ".join(f"{k}={v}" for k, v in stats["grade_key"][stem].items()) or "NONE"
        n_crit = len(metrics_for(task["slug"]))
        abbrs = " ".join(m["abbr"] for m in metrics_for(task["slug"]))
        total += got
        paper += task["n_paper"]
        print(f"  {stem:<4} {got:>5} records (paper: {task['n_paper']:>4})   "
              f"grade_band via {alias:<24} {n_crit} criteria: {abbrs}{flag}")
    print(f"  {'':<4} {total:>5} total    (paper: {paper}, i.e. the artifact ships "
          f"{paper - total} fewer records than Table 2 claims)")

    print("\ninformation keys per file:")
    for task in TASKS:
        print(f"  {task['stem']:<4} {stats['info_keys'][task['stem']]}")

    print("\ngrade_band values per file:")
    for task in TASKS:
        vals = stats["grade_values"][task["stem"]]
        top = ", ".join(f"{k} ({n})" for k, n in vals.most_common(6))
        more = f" ... +{len(vals) - 6} more" if len(vals) > 6 else ""
        print(f"  {task['stem']:<4} {len(vals):>3} distinct: {top}{more}")

    print("\nsubject values per file:")
    for task in TASKS:
        vals = stats["subjects"][task["stem"]]
        top = ", ".join(f"{k} ({n})" for k, n in vals.most_common(5))
        more = f" ... +{len(vals) - 5} more" if len(vals) > 5 else ""
        print(f"  {task['stem']:<4} {len(vals):>3} distinct: {top}{more}")

    if stats["grade_missing"]:
        print(f"\nrecords with no grade_band alias: {dict(stats['grade_missing'])}")
    if stats["ref_missing"]:
        print(f"records missing their reference_solution key: {dict(stats['ref_missing'])}")

    loads = Counter(s for r in rubrics for s in SKILLS if r["q_mapping"][s])
    print(f"\nq_mapping mode {mode!r} -- criteria loading each task dimension:")
    for slug in SKILLS:
        print(f"  {slug:<32} {loads.get(slug, 0):>6}")
    distinct_q = len({json.dumps(r["q_mapping"], sort_keys=True) for r in rubrics})
    print(f"  distinct q_mapping values: {distinct_q}")

    filled = sum(1 for s in scenarios if s["reference_solution"])
    print(f"\nscenarios: {len(scenarios)}   criteria: {len(rubrics)}   "
          f"reference_solution populated: {filled}")
    print("difficulty/discrimination: explicit null until assign_irt_params.py runs.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--q-mapping", choices=["row", "onehot"], default="row",
        help="row: the criterion's full Table 7 row (default). onehot: that row "
             "intersected with the scenario's own task type.",
    )
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    ap.add_argument("--dry-run", action="store_true", help="parse and report, write nothing")
    ap.add_argument(
        "--os-trust-store", action="store_true",
        help="verify HTTPS against the OS trust store rather than certifi; needed "
             "behind a TLS-inspecting antivirus or proxy",
    )
    args = ap.parse_args()

    if args.os_trust_store:
        use_os_trust_store()

    scenarios, rubrics, stats = build(args.q_mapping)
    errs = validate(scenarios, rubrics, stats, args.q_mapping)

    report(scenarios, rubrics, stats, args.q_mapping)

    if errs:
        print(f"\nVALIDATION FAILED ({len(errs)} issues):", file=sys.stderr)
        for e in errs[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errs) > 20:
            print(f"  ... and {len(errs) - 20} more", file=sys.stderr)
        if not args.dry_run:
            raise SystemExit(1)
    else:
        print(f"\nvalidation passed: {len(scenarios)} scenarios, {len(rubrics)} criteria")

    if args.dry_run:
        print("dry run: nothing written")
        return

    write("scenarios", scenarios, args.out)
    write("rubrics", rubrics, args.out)


if __name__ == "__main__":
    main()
