"""Benchmark registry + normalization loaders.

Every loader converts a raw dataset into a list of :class:`common.Question`.
For MCQ the ``prompt`` is the scoring context and ``options`` are the
*continuation strings* to be ranked by log-likelihood (``gold_index`` marks the
correct one). For open-ended the ``prompt`` is the full model input and
``reference`` is the gold answer / key points (may be empty).

Normalized items are cached as JSONL under ``Inputs/{MCQ,Open}/Benchmarks``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import (
    MCQ_BENCH_DIR,
    OPEN_BENCH_DIR,
    BenchType,
    Question,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_hf(path: str, name: str | None = None, split: str = "test", **kw):
    from datasets import load_dataset

    if name:
        return load_dataset(path, name, split=split, **kw)
    return load_dataset(path, split=split, **kw)


def _cap(items: list[Question], max_samples: int | None, seed: int) -> list[Question]:
    if max_samples is None or len(items) <= max_samples:
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = sorted(idx[:max_samples])
    return [items[i] for i in idx]


def _first_col(row: dict, candidates: list[str]) -> Any:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return None


def _qa_prompt(stem: str) -> str:
    return f"Question: {stem.strip()}\nAnswer:"


def _letter_choices(texts: list[str]) -> list[str]:
    return [f"{chr(ord('A') + i)}. {t}" for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# MCQ loaders
# ---------------------------------------------------------------------------


def _mcq_from_choices(stem: str, texts: list[str], gold_index: int, qid: str) -> Question:
    return Question(
        qid=qid,
        prompt=_qa_prompt(stem),
        options=[" " + t.strip() for t in texts],
        gold_index=gold_index,
        meta={"stem": stem, "choices_text": texts, "style": "qa"},
    )


def load_openbookqa(max_samples, seed):
    ds = _load_hf("allenai/openbookqa", "main", split="test")
    out: list[Question] = []
    for i, row in enumerate(ds):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        key = row["answerKey"]
        if key not in labels:
            continue
        gold = labels.index(key)
        qid = row.get("id") or f"obqa_{i:05d}"
        out.append(_mcq_from_choices(row["question_stem"], texts, gold, qid))
    return _cap(out, max_samples, seed)


def load_sciq(max_samples, seed):
    ds = _load_hf("allenai/sciq", split="test")
    rng = random.Random(seed)
    out: list[Question] = []
    for i, row in enumerate(ds):
        correct = row["correct_answer"]
        texts = [correct, row["distractor1"], row["distractor2"], row["distractor3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [texts[j] for j in order]
        gold = order.index(0)
        out.append(_mcq_from_choices(row["question"], shuffled, gold, f"sciq_{i:05d}"))
    return _cap(out, max_samples, seed)


def load_piqa(max_samples, seed):
    # Script-based loading was removed in datasets v5; use the auto-converted
    # parquet branch instead.
    ds = _load_hf("ybisk/piqa", split="validation", revision="refs/convert/parquet")
    out: list[Question] = []
    for i, row in enumerate(ds):
        texts = [row["sol1"], row["sol2"]]
        gold = int(row["label"])
        out.append(_mcq_from_choices(row["goal"], texts, gold, f"piqa_{i:05d}"))
    return _cap(out, max_samples, seed)


def load_mathqa(max_samples, seed):
    ds = _load_hf("allenai/math_qa", split="test", revision="refs/convert/parquet")
    out: list[Question] = []
    for i, row in enumerate(ds):
        texts = _parse_mathqa_options(row["options"])
        if not texts:
            continue
        key = str(row["correct"]).strip().lower()
        gold = ord(key) - ord("a")
        if gold < 0 or gold >= len(texts):
            continue
        out.append(_mcq_from_choices(row["Problem"], texts, gold, f"mathqa_{i:05d}"))
    return _cap(out, max_samples, seed)


def _parse_mathqa_options(raw: str) -> list[str]:
    """Parse ``"a ) 10 , b ) 12 , c ) 14 ..."`` into a list of option texts."""
    import re

    parts = re.split(r"\b([a-e])\s*\)", raw)
    # parts = ['', 'a', ' 10 , ', 'b', ' 12 , ', ...]
    texts: list[str] = []
    it = iter(parts[1:])
    for _label, text in zip(it, it, strict=False):
        texts.append(text.strip().rstrip(",").strip())
    return texts


def load_educationq(max_samples, seed):
    """MMLU-Pro stratified by category up to ``max_samples``."""
    ds = _load_hf("TIGER-Lab/MMLU-Pro", split="test")
    by_cat: dict[str, list[Question]] = {}
    for i, row in enumerate(ds):
        texts = row["options"]
        gold = row.get("answer_index")
        if gold is None:
            key = str(row["answer"]).strip().upper()
            gold = ord(key) - ord("A")
        if gold < 0 or gold >= len(texts):
            continue
        q = _mcq_from_choices(row["question"], texts, gold, f"eduq_{i:05d}")
        q.meta["category"] = row.get("category", "unknown")
        by_cat.setdefault(q.meta["category"], []).append(q)
    return _stratified_sample(by_cat, max_samples, seed)


def _stratified_sample(
    by_group: dict[str, list[Question]], max_samples: int | None, seed: int
) -> list[Question]:
    all_items = [q for items in by_group.values() for q in items]
    if max_samples is None or len(all_items) <= max_samples:
        return all_items
    rng = random.Random(seed)
    groups = sorted(by_group.keys())
    per = max(1, max_samples // len(groups))
    picked: list[Question] = []
    for g in groups:
        items = list(by_group[g])
        rng.shuffle(items)
        picked.extend(items[:per])
    if len(picked) > max_samples:
        rng.shuffle(picked)
        picked = picked[:max_samples]
    return picked


def load_pedagogy(max_samples, seed):
    """AI-for-Education/pedagogy-benchmark: options are ``answer_a`` .. ``answer_g``.

    Unused option columns are present but filled with ``None``.
    """
    ds = _load_hf("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
    letters = "abcdefg"
    out: list[Question] = []
    for i, row in enumerate(ds):
        texts: list[str] = []
        keys: list[str] = []
        for ch in letters:
            val = row.get(f"answer_{ch}")
            if val in (None, "", "None"):
                continue
            texts.append(str(val))
            keys.append(ch)
        stem = _first_col(row, ["question", "Question"])
        answer = _first_col(row, ["correct_answer", "answer", "correct"])
        if stem is None or answer is None or len(texts) < 2:
            continue
        key = str(answer).strip().lower()
        gold = keys.index(key) if key in keys else _resolve_gold(answer, texts)
        if gold is None:
            continue
        qid = row.get("question_id")
        qid = f"pedagogy_{qid}" if qid not in (None, "") else f"pedagogy_{i:05d}"
        out.append(_mcq_from_choices(str(stem), texts, gold, qid))
    return _cap(out, max_samples, seed)


def _resolve_gold(answer: Any, texts: list) -> int | None:
    """Resolve a gold answer that may be a letter, an index, or the option text."""
    s = str(answer).strip()
    if len(s) == 1 and s.upper().isalpha():
        idx = ord(s.upper()) - ord("A")
        return idx if 0 <= idx < len(texts) else None
    if s.isdigit():
        idx = int(s)
        return idx if 0 <= idx < len(texts) else None
    for i, t in enumerate(texts):
        if str(t).strip() == s:
            return i
    return None


# ---------------------------------------------------------------------------
# Open-ended loaders
# ---------------------------------------------------------------------------


def _open(qid: str, prompt: str, reference: str = "", meta: dict | None = None) -> Question:
    return Question(qid=qid, prompt=prompt, reference=reference, meta=meta or {})


def load_squad_v2(max_samples, seed):
    ds = _load_hf("rajpurkar/squad_v2", split="validation")
    out: list[Question] = []
    for i, row in enumerate(ds):
        answers = row["answers"]["text"]
        ref = answers[0] if answers else "[no answer]"
        prompt = (
            f"Context: {row['context']}\n\nQuestion: {row['question']}\n"
            "If the question cannot be answered from the context, reply 'no answer'.\nAnswer:"
        )
        out.append(
            _open(
                row.get("id") or f"squad_{i:05d}",
                prompt,
                ref,
                {"unanswerable": len(answers) == 0, "all_answers": answers},
            )
        )
    return _cap(out, max_samples, seed)


def load_svamp(max_samples, seed):
    ds = _load_hf("ChilleD/SVAMP", split="test")
    out: list[Question] = []
    for i, row in enumerate(ds):
        body = _first_col(row, ["Body", "body"]) or ""
        question = _first_col(row, ["Question", "question"]) or ""
        answer = _first_col(row, ["Answer", "answer"])
        equation = _first_col(row, ["Equation", "equation"]) or ""
        prompt = (
            f"{body} {question}".strip() + "\nSolve the problem and give the final numeric answer."
        )
        out.append(
            _open(
                row.get("ID") or f"svamp_{i:05d}",
                prompt,
                str(answer),
                {"equation": equation},
            )
        )
    return _cap(out, max_samples, seed)


def load_mathdial(max_samples, seed):
    ds = _load_hf("eth-nlped/mathdial", split="test")
    out: list[Question] = []
    for i, row in enumerate(ds):
        conv = _first_col(row, ["conversation", "dialog", "dialogue"]) or ""
        conv_text = _serialize_conversation(conv)
        question = _first_col(row, ["question", "problem"]) or ""
        gt = _first_col(row, ["ground_truth", "answer"]) or ""
        wrong = _first_col(row, ["student_incorrect_solution", "student_solution"]) or ""
        prompt = (
            f"Math problem: {question}\n"
            f"Student's (incorrect) attempt: {wrong}\n"
            f"Tutoring conversation so far:\n{conv_text}\n"
            "As the tutor, write your next response."
        )
        out.append(_open(f"mathdial_{i:05d}", prompt, str(gt), {}))
    return _cap(out, max_samples, seed)


def _serialize_conversation(conv: Any) -> str:
    if isinstance(conv, str):
        return conv
    if isinstance(conv, list):
        lines = []
        for turn in conv:
            if isinstance(turn, dict):
                role = turn.get("role") or turn.get("speaker") or ""
                text = turn.get("content") or turn.get("text") or ""
                lines.append(f"{role}: {text}".strip())
            else:
                lines.append(str(turn))
        return "\n".join(lines)
    return str(conv)


def load_tutoreval(max_samples, seed):
    ds = _load_hf("princeton-nlp/TutorEval", split="train")
    out: list[Question] = []
    for i, row in enumerate(ds):
        question = _first_col(row, ["question", "prompt"]) or ""
        context = _first_col(row, ["chapter", "context", "textbook"]) or ""
        key_points = _first_col(row, ["key_points", "keypoints", "reference", "answer"]) or ""
        if isinstance(key_points, list):
            key_points = "\n".join(f"- {k}" for k in key_points)
        prompt = (f"Textbook context: {context}\n\n" if context else "") + f"Question: {question}"
        out.append(_open(f"tutoreval_{i:05d}", prompt, str(key_points), {}))
    return _cap(out, max_samples, seed)


def load_tutorbench(max_samples, seed):
    ds = _load_hf("tutorbench/tutorbench", split="train")
    out: list[Question] = []
    for i, row in enumerate(ds):
        prompt = _first_col(row, ["PROMPT", "prompt", "question", "conversation", "input"])
        if isinstance(prompt, list):
            prompt = _serialize_conversation(prompt)
        follow_up = _first_col(row, ["FOLLOW_UP_PROMPT"])
        if prompt is not None and follow_up:
            prompt = f"{prompt}\n\nFollow-up: {follow_up}"
        ref = (
            _first_col(
                row,
                ["RUBRICS", "UC1_INITIAL_EXPLANATION", "reference", "answer", "rubric", "solution"],
            )
            or ""
        )
        if prompt is None:
            continue
        out.append(_open(f"tutorbench_{i:05d}", str(prompt), str(ref), {}))
    return _cap(out, max_samples, seed)


def load_edubench(max_samples, seed):
    ds = _load_hf("DirectionAI/EduBench", split="test")
    out: list[Question] = []
    for i, row in enumerate(ds):
        prompt = _first_col(row, ["prompt", "question", "instruction", "input", "query"])
        ref = _first_col(row, ["reference", "answer", "output", "response"]) or ""
        subtask = _first_col(row, ["task", "scenario", "type", "category"]) or "default"
        if prompt is None:
            continue
        out.append(_open(f"edubench_{i:05d}", str(prompt), str(ref), {"subtask": str(subtask)}))
    return _cap(out, max_samples, seed)


def load_socialiqa(max_samples, seed):
    """Social IQa / SocialQA — 3-way MCQ social commonsense."""
    ds = _load_hf("allenai/social_i_qa", split="validation", revision="refs/convert/parquet")
    out: list[Question] = []
    for i, row in enumerate(ds):
        stem = f"{row.get('context', '').strip()}\n\n{row.get('question', '').strip()}"
        texts = [row.get("answerA", ""), row.get("answerB", ""), row.get("answerC", "")]
        if not all(texts) or not stem.strip():
            continue
        label = int(str(row.get("label", "1"))) - 1
        if label < 0 or label >= len(texts):
            continue
        out.append(_mcq_from_choices(stem, texts, label, f"socialiqa_{i:05d}"))
    return _cap(out, max_samples, seed)


def load_bridge(max_samples, seed):
    """Bridge tutoring remediation — next tutor turn (open). Uses validation split."""
    ds = _load_hf("rose-e-wang/bridge", split="validation")
    out: list[Question] = []
    for i, row in enumerate(ds):
        hist = row.get("c_h") or []
        if not hist:
            continue
        last = hist[-1] if isinstance(hist, list) else hist
        student = ""
        if isinstance(last, dict):
            student = str(last.get("text") or last.get("content") or "")
        else:
            student = str(last)
        if not student.strip():
            continue
        ctx = _serialize_conversation(hist[:-1] if isinstance(hist, list) and len(hist) > 1 else [])
        prompt = (
            (f"Conversation so far:\n{ctx}\n\n" if ctx else "")
            + f"Student: {student.strip()}\n\nWrite the tutor's next remediation turn:"
        )
        ref_parts = row.get("c_r_") or row.get("c_r") or []
        if isinstance(ref_parts, list):
            ref = " ".join(
                str(p.get("text") if isinstance(p, dict) else p) for p in ref_parts
            ).strip()
        else:
            ref = str(ref_parts)
        qid = str(row.get("c_id") or f"bridge_{i:05d}")
        out.append(_open(qid, prompt, ref, {"lesson_topic": row.get("lesson_topic")}))
    return _cap(out, max_samples, seed)


def load_biggen(max_samples, seed):
    ds = _load_hf("prometheus-eval/BiGGen-Bench", split="test")
    out: list[Question] = []
    for i, row in enumerate(ds):
        prompt = _first_col(row, ["input", "prompt", "instruction", "question"])
        ref = _first_col(row, ["reference_answer", "reference", "answer"]) or ""
        if prompt is None:
            continue
        out.append(
            _open(
                f"biggen_{i:05d}",
                str(prompt),
                str(ref),
                {"capability": row.get("capability"), "task": row.get("task")},
            )
        )
    return _cap(out, max_samples, seed)


def load_ifeval(max_samples, seed):
    ds = _load_hf("google/IFEval", split="train")
    out: list[Question] = []
    for i, row in enumerate(ds):
        prompt = row.get("prompt")
        if not prompt:
            continue
        key = row.get("key", i)
        out.append(
            _open(
                f"ifeval_{key}",
                str(prompt),
                reference="",
                meta={
                    "instruction_id_list": row.get("instruction_id_list"),
                    "kwargs": row.get("kwargs"),
                },
            )
        )
    return _cap(out, max_samples, seed)


def load_infobench(max_samples, seed):
    ds = _load_hf("kqsong/InFoBench", split="train")
    out: list[Question] = []
    for i, row in enumerate(ds):
        prompt = _first_col(row, ["input", "prompt", "instruction", "query"])
        if prompt is None:
            continue
        ref = _first_col(row, ["output", "reference", "answer"]) or ""
        qid = str(row.get("id") or row.get("instruction_id") or f"infobench_{i:05d}")
        out.append(_open(qid, str(prompt), str(ref), {"category": row.get("category")}))
    return _cap(out, max_samples, seed)


def load_wildbench(max_samples, seed):
    ds = _load_hf("allenai/WildBench", "v2", split="test")
    out: list[Question] = []
    for i, row in enumerate(ds):
        conv = row.get("conversation_input") or row.get("conversation") or row.get("messages")
        prompt = _serialize_conversation(conv) if conv else _first_col(row, ["prompt", "instruction"])
        if not prompt:
            continue
        ref = _first_col(row, ["reference", "answer", "checklist"]) or ""
        qid = str(row.get("session_id") or row.get("id") or f"wildbench_{i:05d}")
        out.append(_open(qid, str(prompt), str(ref), {"primary_tag": row.get("primary_tag")}))
    return _cap(out, max_samples, seed)


# ---------------------------------------------------------------------------
# synthetic loaders (offline smoke tests / CI; no network, no weights)
# ---------------------------------------------------------------------------


def load_synth_mcq(max_samples, seed):
    n = max_samples or 8
    out: list[Question] = []
    for i in range(n):
        out.append(
            _mcq_from_choices(
                f"What is 2 + {i}?",
                [str(2 + i), str(2 + i + 1), str(2 + i + 2), str(2 + i + 3)],
                0,
                f"synth_mcq_{i:03d}",
            )
        )
    return out


def load_synth_open(max_samples, seed):
    n = max_samples or 8
    out: list[Question] = []
    for i in range(n):
        out.append(
            _open(
                f"synth_open_{i:03d}",
                f"Explain educational concept number {i} to a student.",
                reference=f"A correct explanation of concept {i}.",
            )
        )
    return out


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSpec:
    name: str
    type: BenchType
    loader: Callable[[int | None, int], list[Question]]
    rubric_key: str = "default"


BENCHMARKS: dict[str, BenchmarkSpec] = {
    # MCQ
    "openbookqa": BenchmarkSpec("openbookqa", BenchType.MCQ, load_openbookqa),
    "pedagogy": BenchmarkSpec("pedagogy", BenchType.MCQ, load_pedagogy),
    "sciq": BenchmarkSpec("sciq", BenchType.MCQ, load_sciq),
    "piqa": BenchmarkSpec("piqa", BenchType.MCQ, load_piqa),
    "educationq": BenchmarkSpec("educationq", BenchType.MCQ, load_educationq),
    "mathqa": BenchmarkSpec("mathqa", BenchType.MCQ, load_mathqa),
    "socialiqa": BenchmarkSpec("socialiqa", BenchType.MCQ, load_socialiqa),
    # Open-ended
    "tutorbench": BenchmarkSpec("tutorbench", BenchType.OPEN, load_tutorbench, "tutorbench"),
    "tutoreval": BenchmarkSpec("tutoreval", BenchType.OPEN, load_tutoreval, "tutoreval"),
    "edubench": BenchmarkSpec("edubench", BenchType.OPEN, load_edubench, "edubench"),
    "bridge": BenchmarkSpec("bridge", BenchType.OPEN, load_bridge, "default"),
    "biggen": BenchmarkSpec("biggen", BenchType.OPEN, load_biggen, "default"),
    "ifeval": BenchmarkSpec("ifeval", BenchType.OPEN, load_ifeval, "default"),
    "infobench": BenchmarkSpec("infobench", BenchType.OPEN, load_infobench, "default"),
    "wildbench": BenchmarkSpec("wildbench", BenchType.OPEN, load_wildbench, "default"),
    "squad_v2": BenchmarkSpec("squad_v2", BenchType.OPEN, load_squad_v2, "squad_v2"),
    "svamp": BenchmarkSpec("svamp", BenchType.OPEN, load_svamp, "svamp"),
    "mathdial": BenchmarkSpec("mathdial", BenchType.OPEN, load_mathdial, "mathdial"),
    # Synthetic (offline testing only; excluded from the "all"/"mcq"/"open" groups)
    "synth_mcq": BenchmarkSpec("synth_mcq", BenchType.MCQ, load_synth_mcq),
    "synth_open": BenchmarkSpec("synth_open", BenchType.OPEN, load_synth_open),
}

# Named suite for the 200-model CPU response sweep (MCQ correctness + open responses).
CPU_SWEEP_BENCHMARKS = [
    "openbookqa",
    "socialiqa",
    "piqa",
    "pedagogy",
    "bridge",
    "edubench",
    "biggen",
    "ifeval",
    "infobench",
    "tutorbench",
    "tutoreval",
    "wildbench",
]

_SYNTHETIC = {"synth_mcq", "synth_open"}

MCQ_BENCHMARKS = [
    n for n, s in BENCHMARKS.items() if s.type == BenchType.MCQ and n not in _SYNTHETIC
]
OPEN_BENCHMARKS = [
    n for n, s in BENCHMARKS.items() if s.type == BenchType.OPEN and n not in _SYNTHETIC
]
ALL_BENCHMARKS = MCQ_BENCHMARKS + OPEN_BENCHMARKS


def _cache_path(name: str, spec: BenchmarkSpec, max_samples: int | None, seed: int) -> Path:
    base = MCQ_BENCH_DIR if spec.type == BenchType.MCQ else OPEN_BENCH_DIR
    tag = "all" if max_samples is None else str(max_samples)
    return base / f"{name}.n{tag}.s{seed}.jsonl"


def _serialize(q: Question) -> str:
    return json.dumps(
        {
            "qid": q.qid,
            "prompt": q.prompt,
            "options": q.options,
            "gold_index": q.gold_index,
            "reference": q.reference,
            "meta": q.meta,
        },
        ensure_ascii=False,
    )


def _deserialize(line: str) -> Question:
    d = json.loads(line)
    return Question(
        qid=d["qid"],
        prompt=d["prompt"],
        options=d.get("options"),
        gold_index=d.get("gold_index"),
        reference=d.get("reference"),
        meta=d.get("meta", {}),
    )


def load_benchmark(
    name: str,
    max_samples: int | None,
    seed: int = 1234,
    use_cache: bool = True,
) -> list[Question]:
    """Load + normalize (+ cache) a benchmark's questions."""
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Known: {sorted(BENCHMARKS)}")
    spec = BENCHMARKS[name]
    cache = _cache_path(name, spec, max_samples, seed)
    if use_cache and cache.exists():
        with open(cache) as f:
            return [_deserialize(line) for line in f if line.strip()]
    questions = spec.loader(max_samples, seed)
    ensure_dirs(cache.parent)
    with open(cache, "w") as f:
        for q in questions:
            f.write(_serialize(q) + "\n")
    return questions


def _main() -> None:
    ap = argparse.ArgumentParser(description="Download + normalize a benchmark.")
    ap.add_argument("benchmark", nargs="?", help="benchmark name (default: list all)")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    if not args.benchmark:
        print("MCQ:", ", ".join(MCQ_BENCHMARKS))
        print("Open:", ", ".join(OPEN_BENCHMARKS))
        return
    qs = load_benchmark(args.benchmark, args.max_samples, args.seed, use_cache=not args.no_cache)
    print(f"{args.benchmark}: {len(qs)} questions")
    if qs:
        q = qs[0]
        print("first:", json.dumps(_deserialize(_serialize(q)).__dict__, ensure_ascii=False)[:400])


if __name__ == "__main__":
    _main()
