"""Benchmark registry + normalization loaders.

Two item sources, by benchmark type:

  * **MCQ** - loaded from HuggingFace and normalized to :class:`common.Question`,
    where ``prompt`` is the scoring context and ``options`` are the *continuation
    strings* ranked by log-likelihood (``gold_index`` marks the correct one).
    Normalized items are cached as JSONL under ``Inputs/MCQ/Benchmarks``.
  * **FRQ / open-ended** - loaded from the local eduLLM-Evals ``scenarios.jsonl``
    banks (see :mod:`frq_scenarios`), NOT from HuggingFace. Those banks are the
    only source carrying ``use_case`` / ``conversation_context`` / native
    ``system_prompt``, which faithful tutor-prompt construction requires.

Use :func:`load_items` to load either type; :func:`load_benchmark` is the
MCQ-only normalizer (kept for the prefetch/caching path).
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
    BenchType,
    Question,
    ensure_dirs,
)
from frq_scenarios import FRQ_BANKS, Scenario, load_frq_scenarios

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_hf(path: str, name: str | None = None, split: str = "test", **kw):
    from datasets import load_dataset

    if name:
        return load_dataset(path, name, split=split, **kw)
    return load_dataset(path, split=split, **kw)


def _cap(items: list, max_samples: int | None, seed: int) -> list:
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


# ---------------------------------------------------------------------------
# MCQ loaders (PedagogyBench, PIQA, SocialQA)
# ---------------------------------------------------------------------------


def _mcq_from_choices(stem: str, texts: list[str], gold_index: int, qid: str) -> Question:
    return Question(
        qid=qid,
        prompt=_qa_prompt(stem),
        options=[" " + t.strip() for t in texts],
        gold_index=gold_index,
        meta={"stem": stem, "choices_text": texts, "style": "qa"},
    )


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


def load_socialiqa(max_samples, seed):
    """Social IQa / SocialQA - 3-way MCQ social commonsense."""
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
    """Offline FRQ items as Scenarios, so the full tutor-prompt path (system turn,
    conversation context, coalescing) is exercised without network or weights."""
    n = max_samples or 8
    use_cases = ["adaptive_explanation", "feedback", "hint_generation"]
    out: list[Scenario] = []
    for i in range(n):
        uc = use_cases[i % len(use_cases)]
        ctx = (
            [{"role": "student", "content": f"I tried concept {i} and got stuck."}]
            if uc != "adaptive_explanation"
            else []
        )
        out.append(
            Scenario(
                scenario_id=f"synth_open_{i:03d}",
                prompt=f"Explain educational concept number {i} to a student.",
                use_case=uc,
                conversation_context=ctx,
                reference_solution=f"A correct explanation of concept {i}.",
                benchmark="TutorBench",
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
    # MCQ benchmarks carry an HF loader. FRQ benchmarks load from a local
    # scenarios.jsonl bank instead, so `loader` is None for them.
    loader: Callable[[int | None, int], list] | None = None
    rubric_key: str = "default"


def _frq_spec(key: str) -> BenchmarkSpec:
    bank = FRQ_BANKS[key]
    return BenchmarkSpec(key, BenchType.OPEN, None, bank.rubric_key)


BENCHMARKS: dict[str, BenchmarkSpec] = {
    # --- MCQ (HuggingFace; cached to Inputs/MCQ/Benchmarks) ---
    "pedagogy": BenchmarkSpec("pedagogy", BenchType.MCQ, load_pedagogy),
    "piqa": BenchmarkSpec("piqa", BenchType.MCQ, load_piqa),
    "socialiqa": BenchmarkSpec("socialiqa", BenchType.MCQ, load_socialiqa),
    # --- FRQ / open-ended (local eduLLM-Evals scenario banks) ---
    "tutorbench": _frq_spec("tutorbench"),
    "tutoreval": _frq_spec("tutoreval"),
    "bridge": _frq_spec("bridge"),
    "biggen": _frq_spec("biggen"),
    "infobench": _frq_spec("infobench"),
    "wildbench": _frq_spec("wildbench"),
    # --- Synthetic (offline testing only; excluded from the groups below) ---
    "synth_mcq": BenchmarkSpec("synth_mcq", BenchType.MCQ, load_synth_mcq),
    "synth_open": BenchmarkSpec("synth_open", BenchType.OPEN, load_synth_open),
}

_SYNTHETIC = {"synth_mcq", "synth_open"}

MCQ_BENCHMARKS = [
    n for n, s in BENCHMARKS.items() if s.type == BenchType.MCQ and n not in _SYNTHETIC
]
OPEN_BENCHMARKS = [
    n for n, s in BENCHMARKS.items() if s.type == BenchType.OPEN and n not in _SYNTHETIC
]
ALL_BENCHMARKS = MCQ_BENCHMARKS + OPEN_BENCHMARKS

# The full pre-calibration suite (3 MCQ + 6 FRQ).
CPU_SWEEP_BENCHMARKS = list(ALL_BENCHMARKS)


def is_frq(name: str) -> bool:
    return BENCHMARKS[name].type == BenchType.OPEN


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _cache_path(name: str, max_samples: int | None, seed: int) -> Path:
    tag = "all" if max_samples is None else str(max_samples)
    return MCQ_BENCH_DIR / f"{name}.n{tag}.s{seed}.jsonl"


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
    """Load + normalize (+ cache) an MCQ benchmark's questions from HuggingFace."""
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Known: {sorted(BENCHMARKS)}")
    spec = BENCHMARKS[name]
    if spec.loader is None:
        raise ValueError(
            f"'{name}' is an FRQ benchmark loaded from a local scenarios.jsonl bank; "
            f"use load_items() / frq_scenarios.load_frq_scenarios()"
        )
    cache = _cache_path(name, max_samples, seed)
    if use_cache and cache.exists():
        with open(cache, encoding="utf-8") as f:
            return [_deserialize(line) for line in f if line.strip()]
    questions = spec.loader(max_samples, seed)
    # Synthetic FRQ items are Scenarios, not Questions - never cached.
    if name in _SYNTHETIC and spec.type == BenchType.OPEN:
        return questions
    ensure_dirs(cache.parent)
    with open(cache, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(_serialize(q) + "\n")
    return questions


def load_items(
    name: str,
    max_samples: int | None,
    seed: int = 1234,
    use_cache: bool = True,
) -> list:
    """Load a benchmark's items regardless of type.

    MCQ -> list[Question] (HuggingFace, cached). FRQ -> list[Scenario] (local
    eduLLM-Evals bank). Raises frq_scenarios.FRQBankNotFound when an FRQ bank is
    missing, which the driver surfaces as a loud ALERT.
    """
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Known: {sorted(BENCHMARKS)}")
    spec = BENCHMARKS[name]
    if spec.type == BenchType.OPEN and spec.loader is None:
        return load_frq_scenarios(name, max_samples, seed)
    return load_benchmark(name, max_samples, seed, use_cache=use_cache)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Download + normalize a benchmark.")
    ap.add_argument("benchmark", nargs="?", help="benchmark name (default: list all)")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    if not args.benchmark:
        print("MCQ :", ", ".join(MCQ_BENCHMARKS))
        print("FRQ :", ", ".join(OPEN_BENCHMARKS))
        return
    items = load_items(args.benchmark, args.max_samples, args.seed, use_cache=not args.no_cache)
    print(f"{args.benchmark}: {len(items)} items")
    if items:
        print("first:", json.dumps(items[0].__dict__, ensure_ascii=False, default=str)[:400])


if __name__ == "__main__":
    _main()
