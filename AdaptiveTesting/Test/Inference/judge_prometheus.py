"""Prometheus LLM-judge for open-ended responses.

Reads a model's ``responses.jsonl`` and writes ``judged.csv`` with a single
``result`` (pass/fail) column plus ``reasoning`` (feedback) and the raw 1-5
``score``. The judge model is loaded once (resident) and reused across all
(benchmark, model) pairs.
"""

from __future__ import annotations

import json
import re

from common import open_judged_path, open_responses_path
from config import JudgeConfig, WriterConfig
from engine import Engine, GenParams
from models_registry import ModelSpec
from results_writer import CsvResultWriter, existing_ids

JUDGE_FIELDS = [
    "question_id",
    "model",
    "benchmark",
    "result",
    "reasoning",
    "score",
    "exact_match",
    "final_answer_match",
]

ABS_PROMPT = (
    "###Task Description:\n"
    "An instruction (might include an Input inside it), a response to evaluate, "
    "a reference answer that gets a score of 5, and a score rubric representing "
    "an evaluation criteria are given.\n"
    "1. Write detailed feedback assessing the quality of the response strictly "
    "based on the given score rubric, not evaluating in general.\n"
    "2. After writing the feedback, write a score that is an integer between 1 "
    "and 5. You should refer to the score rubric.\n"
    '3. The output format should look as follows: "Feedback: (write a feedback '
    'for criteria) [RESULT] (an integer number between 1 and 5)"\n'
    "4. Please do not generate any other opening, closing, and explanations.\n\n"
    "###The instruction to evaluate:\n{instruction}\n\n"
    "###Response to evaluate:\n{response}\n\n"
    "###Reference Answer (Score 5):\n{reference}\n\n"
    "###Score Rubrics:\n{rubric}\n\n"
    "###Feedback:"
)

_RESULT_RE = re.compile(r"\[RESULT\]\s*([1-5])")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class Judge:
    """Resident Prometheus judge."""

    def __init__(self, cfg: JudgeConfig, backend: str | None = None):
        self.cfg = cfg
        self._rubric_cache: dict[str, str] = {}
        judge_spec = ModelSpec(
            id=cfg.model,
            params_b=7.0,
            apply_chat_template=True,
            trust_remote_code=True,
        )
        self.engine = Engine(judge_spec, backend=backend or cfg.backend)

    def _rubric(self, rubric_key: str) -> str:
        if rubric_key in self._rubric_cache:
            return self._rubric_cache[rubric_key]
        path = self.cfg.rubric_dir / f"{rubric_key}.rubric.txt"
        if not path.exists():
            path = self.cfg.rubric_dir / self.cfg.default_rubric
        text = path.read_text() if path.exists() else _FALLBACK_RUBRIC
        self._rubric_cache[rubric_key] = text
        return text

    def judge_file(
        self,
        benchmark: str,
        model_id: str,
        rubric_key: str,
        writer_cfg: WriterConfig,
    ) -> int:
        """Judge one model's responses for a benchmark; returns rows written."""
        resp_path = open_responses_path(benchmark, model_id)
        if not resp_path.exists():
            return 0
        out_path = open_judged_path(benchmark, model_id)
        done = existing_ids(out_path)

        records = []
        with open(resp_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["question_id"] not in done:
                    records.append(rec)
        if not records:
            return 0

        rubric = self._rubric(rubric_key)
        prompts = [
            ABS_PROMPT.format(
                instruction=r.get("prompt", ""),
                response=r.get("response", ""),
                reference=r.get("reference", "") or "[no reference provided]",
                rubric=rubric,
            )
            for r in records
        ]
        gen = GenParams(temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
        feedbacks = self.engine.generate(prompts, gen)

        written = 0
        with CsvResultWriter(
            out_path,
            JUDGE_FIELDS,
            fsync_every_rows=writer_cfg.fsync_every_rows,
            fsync_every_seconds=writer_cfg.fsync_every_seconds,
        ) as w:
            for rec, fb in zip(records, feedbacks, strict=True):
                score, reasoning = _parse_feedback(fb)
                passed = score is not None and score >= self.cfg.pass_threshold
                result = "pass" if passed else "fail"
                em = _hybrid_exact_match(benchmark, rec)
                fam = _hybrid_final_answer(benchmark, rec)
                w.write_row(
                    {
                        "question_id": rec["question_id"],
                        "model": model_id,
                        "benchmark": benchmark,
                        "result": result,
                        "reasoning": reasoning,
                        "score": score if score is not None else "",
                        "exact_match": "" if em is None else int(em),
                        "final_answer_match": "" if fam is None else int(fam),
                    }
                )
                written += 1
        return written

    def close(self) -> None:
        self.engine.close()


def _parse_feedback(text: str) -> tuple[int | None, str]:
    m = _RESULT_RE.search(text)
    score = int(m.group(1)) if m else None
    reasoning = text.split("[RESULT]")[0].replace("Feedback:", "").strip()
    reasoning = reasoning.replace("\n", " ").strip()
    if not reasoning:
        reasoning = text.strip().replace("\n", " ")
    return score, reasoning


# --------------------------------------------------------------------------
# Hybrid grounding for benchmarks with exact gold answers
# --------------------------------------------------------------------------


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _hybrid_exact_match(benchmark: str, rec: dict) -> bool | None:
    if benchmark != "squad_v2":
        return None
    resp = rec.get("response", "")
    meta = rec.get("meta", {}) or {}
    if meta.get("unanswerable"):
        return "no answer" in resp.lower() or "noanswer" in _normalize(resp)
    golds = meta.get("all_answers") or ([rec["reference"]] if rec.get("reference") else [])
    norm_resp = _normalize(resp)
    return any(_normalize(g) and _normalize(g) in norm_resp for g in golds)


def _hybrid_final_answer(benchmark: str, rec: dict) -> bool | None:
    if benchmark != "svamp":
        return None
    ref = rec.get("reference", "")
    nums = _NUM_RE.findall(rec.get("response", ""))
    ref_nums = _NUM_RE.findall(str(ref))
    if not nums or not ref_nums:
        return False
    try:
        return abs(float(nums[-1]) - float(ref_nums[-1])) < 1e-4
    except ValueError:
        return False


_FALLBACK_RUBRIC = """[Is the response correct, helpful, and appropriate for the instruction?]
Score 1: The response is incorrect, irrelevant, or harmful.
Score 2: The response is largely incorrect or unhelpful.
Score 3: The response is partially correct but has notable gaps.
Score 4: The response is correct and helpful with minor issues.
Score 5: The response is fully correct, helpful, and well-justified."""
