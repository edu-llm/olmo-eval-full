"""Prometheus LLM-judge for FRQ responses.

Reads a model's ``responses.jsonl`` - written in the Model Output schema by
``frq_generate`` - and writes ``judged.csv`` with a single ``result`` (pass/fail)
column plus ``reasoning`` (feedback) and the raw 1-5 ``score``. The judge model is
loaded once (resident) and reused across all (benchmark, model) pairs.

Two schema notes:
  * response rows use the Title-Case Model Output keys (``Scenario``, ``Output``,
    ``Issue``, ...); rows with ``Issue == 1`` are failure cells with no text and
    are skipped rather than judged as empty answers;
  * the reference answer is not stored in the response row (it is judge-only), so
    it is looked up from the FRQ scenario bank by scenario id.
"""

from __future__ import annotations

import json
import re

from common import open_judged_path, open_responses_path
from config import JudgeConfig, WriterConfig
from engine import Engine, GenParams
from frq_scenarios import FRQBankNotFound, load_frq_scenarios
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
            # Explicit: the judge's window is unrelated to the FRQ prompt budget,
            # and pinning it keeps judging behaviour (and its GPU footprint)
            # exactly as before the manifest moved to auto-resolved windows.
            max_model_len=4096,
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
        with open(resp_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from an interrupted append
                if rec.get("Issue", 0) == 1:
                    continue  # failure cell: no text to judge
                sid = rec.get("Scenario")
                if isinstance(sid, str) and sid not in done:
                    records.append(rec)
        if not records:
            return 0

        # References live in the scenario bank, not in the response row.
        refs = _bank_references(benchmark)
        rubric = self._rubric(rubric_key)
        prompts = [
            ABS_PROMPT.format(
                instruction=_instruction_for(r, refs),
                response=r.get("Output", ""),
                reference=refs.get(r["Scenario"], {}).get("reference")
                or "[no reference provided]",
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
                w.write_row(
                    {
                        "question_id": rec["Scenario"],
                        "model": model_id,
                        "benchmark": benchmark,
                        "result": result,
                        "reasoning": reasoning,
                        "score": score if score is not None else "",
                        "exact_match": "",
                        "final_answer_match": "",
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
# Reference / instruction lookup from the FRQ scenario bank
# --------------------------------------------------------------------------

_BANK_CACHE: dict[str, dict[str, dict]] = {}


def _bank_references(benchmark: str) -> dict[str, dict]:
    """scenario_id -> {prompt, reference} for one FRQ benchmark, loaded once.

    The Model Output row deliberately stores no reference (it is judge-only), so
    the bank is the source. Returns {} if the bank is unavailable, in which case
    the judge falls back to the rendered prompt and no reference.
    """
    if benchmark in _BANK_CACHE:
        return _BANK_CACHE[benchmark]
    table: dict[str, dict] = {}
    try:
        for s in load_frq_scenarios(benchmark):
            table[s.scenario_id] = {"prompt": s.prompt, "reference": s.reference_solution}
    except (FRQBankNotFound, KeyError):
        table = {}
    _BANK_CACHE[benchmark] = table
    return table


def _instruction_for(rec: dict, refs: dict[str, dict]) -> str:
    """The instruction shown to the judge: the scenario's own prompt when we have
    the bank (clean, no tutor system prompt or template tokens), else the rendered
    prompt that actually reached the model."""
    entry = refs.get(rec.get("Scenario", ""))
    if entry and entry.get("prompt"):
        return entry["prompt"]
    return rec.get("Rendered Prompt", "")


_FALLBACK_RUBRIC = """[Is the response correct, helpful, and appropriate for the instruction?]
Score 1: The response is incorrect, irrelevant, or harmful.
Score 2: The response is largely incorrect or unhelpful.
Score 3: The response is partially correct but has notable gaps.
Score 4: The response is correct and helpful with minor issues.
Score 5: The response is fully correct, helpful, and well-justified."""
