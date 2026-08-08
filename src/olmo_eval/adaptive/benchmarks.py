"""Config-driven ATLAS benchmark registry.

Single source of truth mapping each ATLAS benchmark to its ``olmo-eval`` base
task, its calibrated 3PL bank subdirectory, and the per-item scoring method the
CAT needs. The offline adaptive tasks, the online adaptive external evals, and
the ``atlas`` suite all read this registry, so the two entry points and the
suite cannot drift on which benchmarks exist or where their banks live.

Scoring methods:

- ``mcq_loglik`` -- argmax over choice continuations vs ``gold_idx``
  (arc/hellaswag/winogrande/csqa/piqa).
- ``generative`` -- generate a completion, run the base task's own answer
  extraction, and exact-match against gold (gsm8k). The CAT still consumes a
  0/1 correctness cell; only how that cell is produced differs from MCQ.
- ``varies`` -- truthfulqa is mixed and has no olmo-eval base task yet, so it is
  intentionally *not* wired (excluded from :func:`cat_benchmarks`).

:func:`cat_benchmarks` is the single driver for offline task, online eval, and
suite registration: it includes every benchmark with a base task and a scoring
method the CAT can derive (``mcq_loglik`` or ``generative``), so gsm8k rides the
same wiring as the MCQ set while truthfulqa stays out until it has a base task.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from olmo_eval.adaptive.bank import resolve_bank_dir

# Per-item correctness methods.
SCORING_MCQ_LOGLIK = "mcq_loglik"
SCORING_GENERATIVE = "generative"
SCORING_VARIES = "varies"

# Only this benchmark honors the legacy $OLMO_EVAL_ATLAS_BANK_DIR override, which
# predates multi-benchmark banks and points at a single (ARC) directory.
_ENV_COMPAT_BENCHMARK = "arc_challenge"


@dataclass(frozen=True)
class AtlasBenchmark:
    """One ATLAS benchmark: base task, bank location, and scoring method.

    Attributes:
        name: Registry key and ``atlas_<name>`` task/eval suffix.
        base_task: olmo-eval task spec providing items/formatting/scoring, or
            ``None`` when no equivalent task exists yet (truthfulqa).
        bank_subdir: Directory under ``AdaptiveTesting/Inputs/ATLAS`` holding the
            benchmark's ``irt_item_parameters_combined.csv`` +
            ``atlas_idx_to_question_id.csv``.
        scoring: One of the ``SCORING_*`` constants.
        online_name: Explicit online external-eval name; defaults to
            ``atlas_<name>``. ARC keeps the historical ``atlas_arc``.
        positional_id: ``True`` when the base task emits a *positional* index as
            ``metadata["id"]`` (winogrande, piqa) rather than a stable native id
            (arc/hellaswag/csqa). The bank join keys on this id, so for
            positional benchmarks the ``atlas_idx_to_question_id.csv`` bridge
            must be generated against the exact split ordering olmo-eval
            enumerates, or the join finds no overlap and the adaptive report is
            skipped (a mismatch degrades to base metrics rather than silently
            misaligning). See docs 04 section 3b.
    """

    name: str
    base_task: str | None
    bank_subdir: str
    scoring: str = SCORING_MCQ_LOGLIK
    online_name: str | None = None
    positional_id: bool = False

    @property
    def offline_task_name(self) -> str:
        return f"atlas_{self.name}"

    @property
    def online_eval_name(self) -> str:
        return self.online_name or f"atlas_{self.name}"

    @property
    def cat_supported(self) -> bool:
        """Whether today's MCQ-loglik CAT can score this benchmark end to end."""
        return self.scoring == SCORING_MCQ_LOGLIK and self.base_task is not None

    @property
    def registered(self) -> bool:
        """Whether this benchmark is wired end to end (offline task + online eval).

        True when it has an olmo-eval base task and a per-item scoring method the
        shared CAT can derive: ``mcq_loglik`` (argmax) or ``generative`` (generate
        + base-task exact match). truthfulqa (``varies`` / no base task) is False.
        """
        return self.base_task is not None and self.scoring in (
            SCORING_MCQ_LOGLIK,
            SCORING_GENERATIVE,
        )


_BENCHMARKS: tuple[AtlasBenchmark, ...] = (
    AtlasBenchmark("arc_challenge", "arc_challenge", "arc", online_name="atlas_arc"),
    AtlasBenchmark("hellaswag", "hellaswag", "hellaswag"),
    AtlasBenchmark("winogrande", "winogrande", "winogrande", positional_id=True),
    AtlasBenchmark("csqa", "csqa", "csqa"),
    AtlasBenchmark("piqa", "piqa", "piqa", positional_id=True),
    # gsm8k is generative: the CAT gets its 0/1 cell from generate + the base
    # task's exact-match, not loglik argmax. Its base task emits a positional
    # index as metadata["id"], so the bank join is positional (zero overlap ->
    # graceful skip, never silent misalignment), same as winogrande/piqa.
    #
    # Parity caveat: the S3-supplied gsm8k bank was calibrated on ATLAS's gsm8k
    # responses under ATLAS's own scoring. olmo-eval's generative exact-match
    # (extract last number, normalize, compare) must match that scoring for the
    # 3PL item params to transfer; a scoring mismatch shifts difficulty/theta.
    # This is a calibration-parity note (mirrors docs/02 section 2 for arc), not
    # a wiring blocker -- the pipeline runs regardless.
    AtlasBenchmark("gsm8k", "gsm8k", "gsm8k", scoring=SCORING_GENERATIVE, positional_id=True),
    # OpenLM (Open LLM Leaderboard v2) banks. Unlike the ATLAS-sourced banks
    # above, these were calibrated on lm-eval-harness responses; the id-bridge
    # joins each bank's item to the leaderboard doc_id (ifeval) / "{subtask}|{doc_id}"
    # (math) rather than a raw split position, so positional_id stays False.
    #
    # ifeval is generative + verifier-scored: the CAT's 0/1 cell comes from the
    # base task's prompt-level *strict* accuracy (IFEvalPromptStrictAccuracy.
    # compute_instance), matching the metric the OpenLM bank was calibrated on
    # (prompt_level_strict_acc).
    AtlasBenchmark("ifeval", "ifeval", "ifeval", scoring=SCORING_GENERATIVE),
    # math is generative + sympy answer-equivalence. Parity caveat (mirrors the
    # gsm8k note): the OpenLM bank was calibrated under lm-eval's math_verify
    # exact_match; olmo-eval's Minerva-style extraction/equivalence is close but
    # not identical, which can shift item difficulty. Wiring runs regardless.
    AtlasBenchmark("math", "leaderboard_math", "math", scoring=SCORING_GENERATIVE),
    # truthfulqa: mixed scoring and no olmo-eval base task yet, so it is the one
    # unregistered benchmark (excluded from cat_benchmarks()).
    AtlasBenchmark("truthfulqa", None, "truthfulqa", scoring=SCORING_VARIES),
)

_BY_NAME: dict[str, AtlasBenchmark] = {b.name: b for b in _BENCHMARKS}


def get_benchmark(name: str) -> AtlasBenchmark:
    """Look up a benchmark by name."""
    try:
        return _BY_NAME[name]
    except KeyError:
        available = ", ".join(_BY_NAME)
        raise KeyError(f"Unknown ATLAS benchmark {name!r}. Available: {available}") from None


def list_benchmarks() -> list[AtlasBenchmark]:
    """All registered ATLAS benchmarks (any scoring method)."""
    return list(_BENCHMARKS)


def mcq_benchmarks() -> list[AtlasBenchmark]:
    """Benchmarks scored by MCQ-loglik argmax (arc/hellaswag/winogrande/csqa/piqa)."""
    return [b for b in _BENCHMARKS if b.cat_supported]


def cat_benchmarks() -> list[AtlasBenchmark]:
    """Benchmarks wired end to end through the CAT (offline task + online eval).

    Includes every benchmark with a base task and a CAT-derivable scoring method
    (``mcq_loglik`` or ``generative``): the MCQ set plus gsm8k. Excludes
    truthfulqa. This is the single source of truth for task/eval/suite
    registration so the three entry points cannot drift.
    """
    return [b for b in _BENCHMARKS if b.registered]


def bank_dir_for(benchmark: AtlasBenchmark, bank_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the calibrated-bank directory for a benchmark.

    Precedence: explicit ``bank_dir`` > ``$OLMO_EVAL_ATLAS_BANK_DIR`` (ARC only,
    for back-compat) > the benchmark's ``bank_subdir``. The caller still passes
    the result to :func:`olmo_eval.adaptive.load_bank`, which raises
    ``FileNotFoundError`` when the CSVs are absent so callers degrade gracefully.
    """
    use_env = benchmark.name == _ENV_COMPAT_BENCHMARK
    return resolve_bank_dir(bank_dir, subdir=benchmark.bank_subdir, use_env=use_env)
