"""Test whether a vendored bank's difficulties belong to the questions beside them.

Run offline by a developer, after vendoring::

    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.check_bridge_alignment \\
        --dataset hellaswag

Every guard vendoring applies is internal, and a bridge that names the right items in
the wrong order passes all of them: the max-index check compares counts, the overlap
floor compares a join rate, and a permutation preserves both. Downstream is no better --
a permuted bank still selects items, still collapses the standard error, and still
reports a theta. Nothing in the pipeline computes a quantity that moves.

So the test has to come from outside it. If a bank's difficulties really describe its
items, then models answering those items should find the high-``b`` ones harder, and the
rank correlation between an item's p-value and its implied ``b = -d / a1`` should be
strongly negative. Under a permutation the two are independent and it sits at zero.
On ARC, whose bridge shipped with the ATLAS release and is the one join here that is
independently attested, this reads -0.857 against the 3,747-model calibration matrix and
-0.319 against the four checkpoints available for the other banks; a bridge regenerated
on the refuted positional assumption reads -0.017, and a one-position shift of the
correct one reads -0.027. The measurement is therefore sharp enough to be worth trusting
and blunt enough to need a null beside it, which ``--permutations`` supplies.

Spearman rather than Pearson, deliberately. ``b`` is a ratio with ``a1`` in the
denominator, so a bank keeps a long tail of well-ordered items with implausible
magnitudes; Pearson reads ARC's verified join at -0.38 where Spearman reads -0.86, which
would make a correct bank look like a doubtful one.

**Two sources of p-value, and they do not attest the same thing.** Read which one a
dataset uses off the output's ``p_value_source`` before quoting its number.

``harvest`` is the strong one: per-question response CSVs from our own inference sweeps,
keyed by something the *item* carries -- a native id, or a split position the harvest
recorded independently -- so a bank whose bridge names the right questions in the wrong
order pairs each difficulty with a stranger's p-value and the correlation collapses.
That is the test the three rebuilt ATLAS bridges were accepted on. :data:`RESPONSE_KEYS`
lists the datasets that have one.

``matrix`` is the weaker one, and it is all the Route B banks have: the calibration's own
response matrix, whose columns are the same positional index as the rows of the parameter
CSV. Both sides therefore come from one enumeration and no independent statement about
*which question* column ``k`` is exists anywhere, so a permutation between bank index and
question is not something this can see. What it does test is everything between the
upstream fit and the committed artifact -- that the parameters were read from the
calibration the spec names, that ``b = -d / a1`` came out with the right sign and
magnitude, that the ``a > 0`` filter kept the items whose difficulty the data supports,
and that the fit describes its own responses at all. A bank that fails it is broken; a
bank that passes it has had its parameters attested and not its join. For those banks the
join rests instead on the bridge's key being self-describing (``<subtask>|<position>``
rather than a bare index) and on vendoring's per-subtask overlap floor.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import random
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..datasets import CONTENT_HASH, content_item_id, get_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [uni_mcq.align] %(levelname)s: %(message)s",
)
log = logging.getLogger("uni_mcq.align")

#: repo_root/diagnostics/mcq_cat/styles/uni_mcq/scripts/check_bridge_alignment.py
REPO_ROOT = Path(__file__).resolve().parents[5]
CALIBRATED_DATASETS = REPO_ROOT / "calibrated_datasets"

#: Where the per-model, per-question response CSVs live on the source ref.
RESPONSES_DIR = "AdaptiveTesting/Inputs/Open/LLM-Judge/mcq"

#: Where the local Route B fits live on the source ref, calibration and matrices together.
_EXPERIMENTS = "AdaptiveTesting/Experiments"

#: Draws used to estimate the null. Two hundred is enough to place the null's standard
#: deviation to about a hundredth, which is the precision the comparison needs: the
#: results being separated are around -0.3 against a null near 0.00 +/- 0.04.
DEFAULT_PERMUTATIONS = 200


@dataclass(frozen=True)
class ResponseKey:
    """How a dataset's response CSVs name the item a row scored.

    Attributes:
        subdir: The directory under :data:`RESPONSES_DIR` holding its models.
        of_instance: Builds the response ``question_id`` for one task instance. Every
            harvest keys on something the instance already carries, but not the same
            thing in each case, and the differences matter -- ARC on its native id,
            HellaSwag on the native ``ind``, WinoGrande on a zero-padded split position.
    """

    subdir: str
    of_instance: Callable[[Any], str]


def _metadata_id(instance: Any) -> str:
    return str((instance.metadata or {}).get("id", "")).strip()


#: The datasets with committed per-question responses, and how those responses are keyed.
#:
#: GSM8K is absent and its absence is the finding rather than a gap: no response matrix,
#: ``actual_accuracy.csv``, item-selection frequency or per-item statistic for it is
#: committed on any ref in this checkout, so this test cannot be run for that bank at
#: all and no amount of care with its bridge will produce one.
RESPONSE_KEYS: dict[str, ResponseKey] = {
    "arc_challenge": ResponseKey(subdir="arc_challenge", of_instance=_metadata_id),
    "hellaswag": ResponseKey(
        subdir="hellaswag", of_instance=lambda inst: f"hellaswag_{_metadata_id(inst)}"
    ),
    "winogrande": ResponseKey(
        subdir="winogrande", of_instance=lambda inst: f"winogrande_{int(_metadata_id(inst)):05d}"
    ),
}

#: The Route B banks, and the response matrices their own calibration was fit from.
#:
#: Train and test are both listed because the split is a property of the fit rather than
#: of the items: the same columns appear in each, over disjoint model rows, and pooling
#: them is simply a larger sample for the same per-item p-value. A matrix row is one
#: model and a column is one bank index, 1-based, matching the ``X{k}`` labels of
#: ``irt_item_parameters_combined.csv``.
#:
#: GPQA's pair is named ``gpqa_response_matrix_*`` where the other four are
#: ``response_matrix_*``; the paths are spelled out per dataset rather than templated so
#: that inconsistency stays a fact about the data instead of a rule to be inferred. BBH
#: is the case that shows why: its fit lives beside math's, musr's and ifeval's under
#: ``openlm_atlas_3pl`` and follows their unprefixed naming, while gpqa's lives in its own
#: experiment directory and does not, so neither convention generalizes from the other.
RESPONSE_MATRICES: dict[str, tuple[str, ...]] = {
    "bbh": (
        f"{_EXPERIMENTS}/openlm_atlas_3pl/bbh/data/response_matrix_train.csv",
        f"{_EXPERIMENTS}/openlm_atlas_3pl/bbh/data/response_matrix_test.csv",
    ),
    "ifeval": (
        f"{_EXPERIMENTS}/openlm_atlas_3pl/ifeval/data/response_matrix_train.csv",
        f"{_EXPERIMENTS}/openlm_atlas_3pl/ifeval/data/response_matrix_test.csv",
    ),
    "leaderboard_math": (
        f"{_EXPERIMENTS}/openlm_atlas_3pl/math/data/response_matrix_train.csv",
        f"{_EXPERIMENTS}/openlm_atlas_3pl/math/data/response_matrix_test.csv",
    ),
    "musr": (
        f"{_EXPERIMENTS}/openlm_atlas_3pl/musr/data/response_matrix_train.csv",
        f"{_EXPERIMENTS}/openlm_atlas_3pl/musr/data/response_matrix_test.csv",
    ),
    "gpqa": (
        f"{_EXPERIMENTS}/openlm_gpqa_atlas_3pl/data/gpqa_response_matrix_train.csv",
        f"{_EXPERIMENTS}/openlm_gpqa_atlas_3pl/data/gpqa_response_matrix_test.csv",
    ),
}

#: Written into the output so a reader knows which of the two tests produced the number.
HARVEST_SOURCE = "harvest"
MATRIX_SOURCE = "matrix"


def git_show(ref: str, path: str) -> str:
    """Return the contents of ``path`` at ``ref`` without checking anything out."""
    completed = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"git show {ref}:{path} failed (exit {completed.returncode}).\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def git_ls(ref: str, path: str) -> list[str]:
    """Return the files under ``path`` at ``ref``."""
    completed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"git ls-tree {ref}:{path} failed.\n{completed.stderr.strip()}")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def ranks(values: Sequence[float]) -> list[float]:
    """Return average ranks, so ties do not manufacture an ordering.

    A bank has real ties on both sides -- four models give a p-value only six values can
    take -- and breaking them by input order would let the arbitrary order of
    ``params.json`` leak into the statistic.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            out[index] = average
        position = end + 1
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Pearson's r, or 0.0 where a side has no variance to correlate."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = (sum(a * a for a in dx) ** 0.5) * (sum(b * b for b in dy) ** 0.5)
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / denominator


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Spearman's rho, as Pearson over average ranks."""
    return pearson(ranks(xs), ranks(ys))


@dataclass(frozen=True)
class BankRow:
    """One vendored item, in the two keys the two p-value sources join on.

    ``item_id`` is what a harvest is reached through, by way of the task's enumeration;
    ``atlas_idx`` is the bank column the calibration's own matrix scored. Carrying both
    off the same ``params.json`` record is what lets one pairing routine serve either
    source without either side reconstructing an order.
    """

    item_id: str
    difficulty: float
    atlas_idx: int | None


def load_bank(dataset: str, bank_dir: Path) -> tuple[list[BankRow], str]:
    """Return the vendored items and the bridge kind the bank was keyed under.

    The kind is read from the bank's own ``manifest.json`` rather than from its spec,
    on the same rule that makes ``fit_family`` authoritative there: the manifest records
    how the artifacts were produced, while the spec records what would be produced now.
    They differ exactly when a bridge has been replaced, which is when this test is worth
    running -- taking the kind from the spec would rebuild the item keys under the new
    scheme, find nothing in a bank keyed under the old one, and report the replacement as
    an empty join rather than scoring the bank in front of it.
    """
    params_path = bank_dir / "params.json"
    if not params_path.is_file():
        raise SystemExit(f"No vendored bank at {params_path}. Vendor the dataset first.")
    records = json.loads(params_path.read_text(encoding="utf-8"))
    bank = [
        BankRow(
            item_id=str(record["item_id"]),
            difficulty=float(record["difficulty"]),
            atlas_idx=_atlas_idx(record),
        )
        for record in records
    ]

    manifest_path = bank_dir / "manifest.json"
    kind = get_spec(dataset).bridge_kind
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kind = str(manifest.get("bridge_kind", kind))
    return bank, kind


def _atlas_idx(record: dict[str, Any]) -> int | None:
    """Return the bank column a vendored record came from, if it recorded one."""
    value = (record.get("metadata") or {}).get("atlas_idx")
    return None if value is None else int(value)


def load_responses(
    dataset: str, source_ref: str, *, only: Sequence[str] = ()
) -> tuple[dict[str, float], list[str]]:
    """Return ``question_id -> p-value`` and the models it was computed over.

    A p-value here is the fraction of models answering the item correctly, which is the
    coarsest possible difficulty estimate and the only one four checkpoints can support.
    It is enough: what is being tested is whether two orderings agree at all, not how
    precisely either is measured.

    ``only`` restricts the roster, and the reason to restrain it is comparability rather
    than speed. How negative a correct join reads depends on how well the p-value ranks
    items, which depends on how many models produced it -- ARC's verified bank reads
    -0.542 over the 63 checkpoints harvested for it and about -0.32 over the four that
    HellaSwag and WinoGrande have. Judging a four-model result against a 63-model
    reference would condemn a correct bank.
    """
    key = RESPONSE_KEYS[dataset]
    paths = git_ls(source_ref, f"{RESPONSES_DIR}/{key.subdir}")
    if only:
        paths = [path for path in paths if Path(path).stem in set(only)]
    if not paths:
        raise SystemExit(
            f"{dataset}: no response CSVs under {source_ref}:{RESPONSES_DIR}/{key.subdir}"
            + (f" matching {list(only)}." if only else ".")
        )

    correct: dict[str, int] = defaultdict(int)
    asked: dict[str, int] = defaultdict(int)
    for path in sorted(paths):
        for row in csv.DictReader(io.StringIO(git_show(source_ref, path))):
            question_id = str(row["question_id"]).strip()
            asked[question_id] += 1
            if str(row["result"]).strip() == "correct":
                correct[question_id] += 1

    p_values = {qid: correct[qid] / asked[qid] for qid in asked}
    models = [Path(path).stem for path in sorted(paths)]
    return p_values, models


def load_matrix_p_values(
    dataset: str, source_ref: str, *, only: Sequence[str] = ()
) -> tuple[dict[int, float], list[str]]:
    """Return ``bank column -> p-value`` from the calibration's own response matrices.

    A matrix row is one model and a column is one bank index, so the p-value is the
    column mean. Train and test are pooled: they hold the same columns over disjoint
    model rows, and which side of the fit's own split a model fell on says nothing about
    how hard an item is.

    A cell that is neither 0 nor 1 -- blank, ``NA`` -- is not counted rather than being
    read as a wrong answer, so an item no model was asked keeps no p-value at all and
    drops out of the pairing instead of arriving as a spurious zero.
    """
    correct: dict[int, int] = defaultdict(int)
    asked: dict[int, int] = defaultdict(int)
    models: list[str] = []
    restrict = set(only)

    for path in RESPONSE_MATRICES[dataset]:
        reader = csv.reader(io.StringIO(git_show(source_ref, path)))
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"{dataset}: response matrix {path} is empty.")
        columns = [int(name) for name in header[1:]]
        for row in reader:
            if not row:
                continue
            if restrict and row[0] not in restrict:
                continue
            models.append(row[0])
            for column, cell in zip(columns, row[1:], strict=True):
                value = cell.strip()
                if value not in {"0", "1"}:
                    continue
                asked[column] += 1
                correct[column] += value == "1"

    if not models:
        raise SystemExit(
            f"{dataset}: no model rows in {len(RESPONSE_MATRICES[dataset])} response "
            f"matrices" + (f" matching {sorted(restrict)}." if restrict else ".")
        )
    return {column: correct[column] / asked[column] for column in asked}, models


def item_response_keys(dataset: str, bridge_kind: str) -> tuple[dict[str, str], int]:
    """Return ``bank item id -> response question_id``, dropping ambiguous ones.

    The two keys are derived from the same enumeration, so building both here is what
    ties a bank row to a response row without either side assuming an order.

    Ambiguity is dropped rather than resolved, and on HellaSwag that is the whole reason
    this function returns a count. Its harvest keys on the native ``ind``, which names
    two different validation rows 433 times over, so a response row carrying one of
    those values could have scored either item. Keeping them would pair some fraction of
    the bank with p-values belonging to other questions -- noise added to precisely the
    measurement meant to detect that.
    """
    from olmo_eval.evals.tasks.common.registry import get_task

    spec = get_spec(dataset)
    response_key = RESPONSE_KEYS[dataset].of_instance

    pairs: list[tuple[str, str]] = []
    for index, instance in enumerate(get_task(spec.task).instances):
        if bridge_kind == CONTENT_HASH:
            item_id = content_item_id(instance.question, tuple(instance.choices or ()))
        else:
            item_id = str((instance.metadata or {}).get("id", index)).strip()
        pairs.append((item_id, response_key(instance)))

    contested = {qid for qid, n in Counter(qid for _, qid in pairs).items() if n > 1}
    keys = {item_id: qid for item_id, qid in pairs if qid not in contested}
    return keys, sum(1 for _, qid in pairs if qid in contested)


def scrambled_null(
    p_values: Sequence[float], difficulties: Sequence[float], *, draws: int, seed: int
) -> tuple[float, float]:
    """Return the mean and standard deviation of Spearman under a random pairing.

    The number that makes a result readable. A correlation of -0.30 means nothing until
    it is known what this many items of this shape produce by chance, and permuting one
    side answers that using the same marginals, the same ties and the same sample size
    as the real measurement.
    """
    rng = random.Random(seed)
    shuffled = list(p_values)
    draws_seen: list[float] = []
    for _ in range(draws):
        rng.shuffle(shuffled)
        draws_seen.append(spearman(shuffled, difficulties))
    mean = sum(draws_seen) / len(draws_seen)
    variance = sum((value - mean) ** 2 for value in draws_seen) / len(draws_seen)
    return mean, variance**0.5


def pair_through_harvest(
    dataset: str,
    bank: Sequence[BankRow],
    bridge_kind: str,
    source_ref: str,
    only: Sequence[str],
) -> tuple[list[tuple[float, float]], list[str]]:
    """Pair each bank item with the p-value our own harvest recorded for its question."""
    p_values, models = load_responses(dataset, source_ref, only=only)
    keys, contested = item_response_keys(dataset, bridge_kind)
    log.info(
        "%d harvested response rows over %d models; %d instances dropped for an "
        "ambiguous response key",
        len(p_values),
        len(models),
        contested,
    )
    pairs = [
        (p_values[keys[row.item_id]], row.difficulty)
        for row in bank
        if row.item_id in keys and keys[row.item_id] in p_values
    ]
    return pairs, models


def pair_through_matrix(
    dataset: str, bank: Sequence[BankRow], source_ref: str, only: Sequence[str]
) -> tuple[list[tuple[float, float]], list[str]]:
    """Pair each bank item with the p-value of the column its parameters were fit from."""
    unindexed = [row.item_id for row in bank if row.atlas_idx is None]
    if unindexed:
        raise SystemExit(
            f"{dataset}: {len(unindexed)} of {len(bank)} vendored items record no "
            f"metadata['atlas_idx'] (for example {unindexed[:3]}), so there is nothing "
            f"to look their column up by. The bank predates that field; re-vendor it."
        )

    p_values, models = load_matrix_p_values(dataset, source_ref, only=only)
    log.info("%d scored bank columns over %d model rows", len(p_values), len(models))
    pairs = [(p_values[row.atlas_idx], row.difficulty) for row in bank if row.atlas_idx in p_values]
    return pairs, models


def check(
    dataset: str,
    *,
    bank_dir: Path,
    source_ref: str,
    draws: int,
    seed: int,
    only: Sequence[str] = (),
) -> int:
    """Run the alignment test for one dataset and report it."""
    if dataset not in RESPONSE_KEYS and dataset not in RESPONSE_MATRICES:
        log.error(
            "%s has neither a per-question harvest nor a calibration response matrix "
            "on any ref, so this test cannot be run for it. That is a fact about the "
            "evidence available, not a passing result: its bridge rests on whatever the "
            "ARC control establishes about the recovery procedure and on nothing "
            "measured about the bank itself.",
            dataset,
        )
        return 2

    bank, bridge_kind = load_bank(dataset, bank_dir)
    source = HARVEST_SOURCE if dataset in RESPONSE_KEYS else MATRIX_SOURCE
    log.info("Bank has %d items keyed as %r; p-values from the %s", len(bank), bridge_kind, source)

    if source == HARVEST_SOURCE:
        pairs, models = pair_through_harvest(dataset, bank, bridge_kind, source_ref, only)
    else:
        pairs, models = pair_through_matrix(dataset, bank, source_ref, only)

    if len(pairs) < 2:
        raise SystemExit(
            f"{dataset}: only {len(pairs)} bank items have a response, which is too "
            f"few to correlate. The bank and the {source} are not keyed to the same items."
        )

    paired_p = [p for p, _ in pairs]
    paired_b = [b for _, b in pairs]
    rho = spearman(paired_p, paired_b)
    r = pearson(paired_p, paired_b)
    null_mean, null_sd = scrambled_null(paired_p, paired_b, draws=draws, seed=seed)

    log.info(
        "%s: Spearman(p, b) = %+.3f over %d items (Pearson %+.3f). Scrambled control "
        "%+.3f +/- %.3f over %d draws.",
        dataset,
        rho,
        len(paired_p),
        r,
        null_mean,
        null_sd,
        draws,
    )
    print(
        json.dumps(
            {
                "dataset": dataset,
                "bank_dir": str(bank_dir),
                "bridge_kind": bridge_kind,
                "p_value_source": source,
                "items_correlated": len(paired_p),
                "models": len(models),
                # Named only for a harvest, whose roster is four checkpoints and is what
                # --models restricts; a matrix carries about a thousand and listing them
                # would bury the result it qualifies.
                "model_names": models if source == HARVEST_SOURCE else None,
                "spearman": round(rho, 4),
                "pearson": round(r, 4),
                "scrambled_mean": round(null_mean, 4),
                "scrambled_sd": round(null_sd, 4),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the alignment check's argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.check_bridge_alignment",
        description="Correlate item p-value against a vendored bank's implied difficulty.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset whose bank to test.")
    parser.add_argument(
        "--bank-dir",
        default=None,
        help=(
            "Directory holding the params.json to test (default: "
            "calibrated_datasets/<dataset>). Point it elsewhere to score a bank vendored "
            "against a different bridge without disturbing the committed one."
        ),
    )
    parser.add_argument(
        "--source-ref",
        default="origin/Research",
        help="Git ref holding the per-question response CSVs (default: origin/Research).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help=f"Draws used to estimate the scrambled null (default: {DEFAULT_PERMUTATIONS}).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the scrambled null.")
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model names to restrict the roster to, matched against a "
            "harvest's CSV stems (Qwen__Qwen2.5-0.5B) or a response matrix's row labels. "
            "Use it to read ARC's verified bank over the same four checkpoints the other "
            "harvested banks have, which is the only like-for-like reference."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the alignment test for one dataset."""
    args = build_parser().parse_args(argv)
    bank_dir = Path(args.bank_dir) if args.bank_dir else CALIBRATED_DATASETS / args.dataset
    only = tuple(name.strip() for name in args.models.split(",")) if args.models else ()
    return check(
        args.dataset,
        bank_dir=bank_dir,
        source_ref=args.source_ref,
        draws=args.permutations,
        seed=args.seed,
        only=only,
    )


if __name__ == "__main__":
    sys.exit(main())
