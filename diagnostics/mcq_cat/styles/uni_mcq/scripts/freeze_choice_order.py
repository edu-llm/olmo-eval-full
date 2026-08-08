"""Freeze a bank's option ordering out of its committed stems, before it is re-vendored.

Run offline by a developer, once per bank that needs one::

    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.freeze_choice_order \\
        --dataset gpqa --ref HEAD

Written for the one migration that cannot be checked any other way. GPQA was vendored
as a generative bank, so each item's stem is the whole user turn ``MCQAChatFormatter``
builds -- the question, a blank line, then ``(A) ...`` through ``(D) ...`` in the order
``GPQATask.process_doc`` shuffled them -- and the record's ``choices`` list is empty
because a chain of thought has nothing to rank. Scoring it as MCQ means putting that
choice list back, and the ordering is the whole risk: the gold letter indexes one
permutation and nothing downstream can tell a correct choice list from a rotated one.
Every guard passes, the CAT converges, the standard error collapses on schedule, and the
theta is noise with a healthy interval printed beside it. That is the same failure the
ATLAS bridges hid, and it went unnoticed there until someone correlated item p-values
against difficulty.

So the ordering is not re-derived, it is *transferred*. The stems already carry the exact
text in the exact order the gold letter indexes, and parsing them is a transformation
with no ordering assumption in it, whereas re-enumerating the source dataset reproduces
the order only if every seed and code path still matches. What this writes is that
transfer, committed, so :func:`~.vendor_bank.check_choice_order` can hold a fresh
enumeration to it item by item and a test can keep holding the shipped bank to it long
after the old stems are gone.

Two properties make the parse trustworthy rather than merely plausible, and both are
enforced here instead of assumed. The decomposition of a stem into a question and an
``n``-option block must be **unique** -- every increasing run of ``(A)``..``(D)`` markers
is enumerated and more than one aborts, which is what would catch an option whose own
text carried a lettered line. And it must **round-trip**: the question and block
reassembled have to be the byte-identical stem. A recorded digest of that stem travels
with each entry so the round-trip stays checkable against whatever is vendored later,
when the stem this was read from no longer exists on disk.

Neither of those says the ordering was *right*, only that it is unchanged, and a mapping
shifted consistently before the stems were written would survive both. That is what
:class:`CrossCheck` is for: where an outside record of the correct answer exists, the
option set it gives and the option it names are frozen beside ours and compared, so the
control attests which option is correct and which four options there are, as well as
which order this bank puts them in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

from ..datasets import get_spec, supported_names

log = logging.getLogger("uni_mcq.freeze")

#: repo_root/diagnostics/mcq_cat/styles/uni_mcq/scripts/freeze_choice_order.py
REPO_ROOT = Path(__file__).resolve().parents[5]

#: Where a frozen ordering is committed, relative to the repo root. Beside the bridges
#: because it answers the same kind of question they do -- which item is this row about --
#: and because ``arc_challenge.control.json`` already establishes the directory as the
#: home for a known-answer control as well as for the joins themselves.
CONTROLS_DIR = "diagnostics/mcq_cat/styles/uni_mcq/bridges"

#: Suffix distinguishing an ordering control from a bridge in the same directory.
CONTROL_SUFFIX = ".choice_order.json"

#: Option labels, in the order a lettered block writes them.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: The separator ``MCQAChatFormatter`` puts between the question and the choice block.
BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class CrossCheck:
    """An outside record of which option is correct, to be frozen beside our own.

    The transfer above establishes that the new choice list is the old one. It cannot
    establish that the old one was right, because both halves of it are this repo's:
    a mapping shifted consistently at some earlier point would be reproduced faithfully
    and every byte-equality test would pass. Answering that needs a source that shares
    no code with the pipeline, and for a Route B bank one exists -- Open LLM Leaderboard
    v2 published its own per-example records, each carrying the question's options and
    the answer, under its own presentation rather than ours.

    Freezing the correct option's *text*, rather than the letter or position naming it,
    is what makes the comparison independent of both orderings. Two orderings of one
    option set agree about the answer or they do not, whatever positions they put it in.

    Two answers are read from each record, because they attest different things and one
    of them does not reach every item.

    ``scored_*`` is the stronger claim and the incomplete one: the options as the harness
    *showed* them and the letter it *scored*, so agreeing with it means our gold names
    the option a real evaluation counted as correct. It runs through lm-evaluation-harness's
    ``leaderboard/gpqa/utils.preprocess``, which deletes bracketed spans, and where an
    option is written entirely inside brackets that leaves nothing to compare.

    ``source_*`` is the weaker claim and the complete one: the benchmark's own
    ``Correct Answer`` and ``Incorrect Answer`` columns, passed through untouched. It
    reaches every item, including the ones preprocessing empties, and it establishes
    which option is correct without establishing that anything scored it.

    Attributes:
        repo: The details dataset on the Hub.
        revision: Pinned, because a details repo is mutable and an unpinned control is
            a control against whatever it says today.
        pattern: Filename template taking ``subtask``, under ``prefix``.
        prefix: Directory inside the repo holding the sample files.
        scored_gold: The document field naming the scored option, as a letter.
        scored_options: The document fields holding the options as shown, in that order.
        source_gold: The document field holding the benchmark's own correct answer.
        source_distractors: The fields holding its incorrect answers.
    """

    repo: str
    revision: str
    prefix: str
    pattern: str
    scored_gold: str = "answer"
    scored_options: tuple[str, ...] = ("choice1", "choice2", "choice3", "choice4")
    source_gold: str = "Correct Answer"
    source_distractors: tuple[str, ...] = (
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    )


#: Datasets with an outside record of the correct option, and where to read it.
#:
#: One entry, and the reason there is only one is the reason gpqa needs a control at all:
#: it is the bank whose modality changed underneath a per-question shuffle. The model is
#: chosen for having evaluated all three subsets in one run at a pinned revision, not for
#: anything about the model -- what is read off it is the benchmark's own documents, which
#: every submission was shown identically.
CROSS_CHECKS: dict[str, CrossCheck] = {
    "gpqa": CrossCheck(
        repo="open-llm-leaderboard/microsoft__Phi-3-mini-4k-instruct-details",
        revision="427d64648094a1aea773aec93fe5e3b5ca069074",
        prefix="microsoft__Phi-3-mini-4k-instruct",
        pattern="samples_leaderboard_gpqa_{subtask}_2024-07-18T11-04-02.101450.jsonl",
    ),
}


def control_path(dataset: str) -> Path:
    """Where ``dataset``'s frozen ordering lives in this checkout."""
    return REPO_ROOT / CONTROLS_DIR / f"{dataset}{CONTROL_SUFFIX}"


def render_block(choices: list[str]) -> str:
    """Return the lettered block for ``choices``, as ``MCQAChatFormatter`` writes it.

    One definition, imported by everything that needs it, because the guard and the
    control have to agree about what a block looks like down to the space after the
    closing bracket. Two spellings that matched today would be free to drift, and the
    drift would surface as a bank nobody can re-vendor rather than as an edit anyone
    could read.
    """
    return "\n".join(f"({LETTERS[index]}) {choice}" for index, choice in enumerate(choices))


def sha256(text: str) -> str:
    """The digest a stem is recorded and later re-checked under."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenOrder:
    """One item's option ordering, as the committed stem stated it.

    Attributes:
        choices: The option texts in block order, which is the order the gold letter
            indexes and the only order it means anything against.
        gold_letter: The letter written beside that block.
        stem_sha256: Digest of the whole stem the two were read out of, so a later
            reader can prove a rebuilt question and this block reassemble it exactly
            rather than merely resembling it.
        leaderboard_gold: The option text an outside evaluation scored as correct, or
            empty where there is no such record or its preprocessing left nothing to
            compare. See :class:`CrossCheck` for why it is the text rather than a letter.
        source_options: The same question's options as the benchmark's own answer
            columns give them, correct one first, or empty where there is no outside
            record. Unordered evidence on purpose: what it attests is which option is
            correct and which four options there are, not what order anything showed
            them in.
    """

    choices: list[str]
    gold_letter: str
    stem_sha256: str
    leaderboard_gold: str = ""
    source_options: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """The committed form."""
        return {
            "choices": list(self.choices),
            "gold_letter": self.gold_letter,
            "stem_sha256": self.stem_sha256,
            "leaderboard_gold": self.leaderboard_gold,
            "source_options": list(self.source_options),
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> FrozenOrder:
        """Read one entry back, without trusting its types."""
        return cls(
            choices=[str(choice) for choice in record["choices"]],
            gold_letter=str(record["gold_letter"]),
            stem_sha256=str(record["stem_sha256"]),
            leaderboard_gold=str(record.get("leaderboard_gold", "")),
            source_options=[str(choice) for choice in record.get("source_options", ())],
        )

    @property
    def gold_index(self) -> int:
        """The position the gold letter names."""
        return LETTERS.index(self.gold_letter)

    @property
    def source_gold(self) -> str:
        """The benchmark's own correct answer, or empty where none was recorded."""
        return self.source_options[0] if self.source_options else ""


def comparable(text: str) -> str:
    """Reduce an option's text to the form two harnesses can be compared in.

    Deliberately lossy, and only in the two ways the harnesses are known to differ.
    lm-evaluation-harness's ``leaderboard/gpqa/utils.preprocess`` deletes bracketed spans
    and collapses doubled spaces where olmo-eval's ``_clean_text`` keeps them, so a
    literal comparison would report a disagreement about the *answer* where the only
    disagreement is about punctuation. Case and surrounding whitespace go for the same
    reason. Nothing else is touched: the numbers, units and words that distinguish one
    option from another all survive, which is what the comparison is about.
    """
    return re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", text)).strip().lower()


def load_cross_check(dataset: str) -> dict[str, tuple[str, list[str]]]:
    """Return ``item id -> (scored answer, source options)`` from an outside record.

    Reads the leaderboard's own per-example files at the pinned revision and keys them
    the way this bank's bridge does, ``<subtask>|<doc_id>``, which the bridge recovery
    established names the document the leaderboard evaluated for all 1,192 of them.

    The scored answer is the option text the harness's own letter names, and is empty
    where its preprocessing emptied that option. The source options are the benchmark's
    four answer columns, correct one first, untouched. See :class:`CrossCheck`.

    Returns an empty mapping for a dataset with no outside record, which is every dataset
    but one; the caller freezes what it has rather than failing.
    """
    check = CROSS_CHECKS.get(dataset)
    if check is None:
        return {}

    # Corporate TLS interception makes certifi's bundle the wrong one, and the failure
    # arrives as a certificate error rather than as anything about the Hub.
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    from huggingface_hub import HfFileSystem

    # Read into memory rather than downloaded, for the reason build_leaderboard_bridge
    # does it: these filenames carry ISO timestamps with colons, and the Hub cache lays
    # them under a snapshot path long enough to defeat a default Windows install.
    handle = HfFileSystem()
    spec = get_spec(dataset)
    outside: dict[str, tuple[str, list[str]]] = {}
    for label, _ in spec.subtasks:
        name = check.pattern.format(subtask=label)
        text = handle.read_text(
            f"datasets/{check.repo}@{check.revision}/{check.prefix}/{name}",
            encoding="utf-8",
        )
        # Split on the newline only. str.splitlines also breaks on U+2028 and the
        # vertical tab, which JSON permits unescaped inside a string and which these
        # question texts contain, so it would cut records in half.
        for line in text.split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            document = record["doc"]
            letter = str(document[check.scored_gold]).strip("()").strip()
            shown = [str(document[name]) for name in check.scored_options]
            source = [str(document[check.source_gold])]
            source += [str(document[name]) for name in check.source_distractors]
            outside[f"{label}|{record['doc_id']}"] = (shown[LETTERS.index(letter)], source)
    log.info("Read %d outside records from %s", len(outside), check.repo)
    return outside


def marker_positions(stem: str, marker: str) -> list[int]:
    """Every index at which ``marker`` occurs in ``stem``, overlaps included."""
    found: list[int] = []
    start = 0
    while (at := stem.find(marker, start)) >= 0:
        found.append(at)
        start = at + 1
    return found


def decompositions(stem: str, options: int) -> list[tuple[str, list[str]]]:
    """Every ``(question, choices)`` split of ``stem`` consistent with a lettered block.

    Enumerated in full rather than found greedily, because the count is the evidence.
    A single decomposition means the block's boundaries are forced by the text and the
    parse is a fact about the stem; two or more mean an option's own text carries a
    lettered line and the split is a guess, which is exactly the case a first-match
    parser would sail through. GPQA's answers are short scientific strings and none of
    them does this, but that is a property of the data rather than of the format, and
    it is worth having measured instead of assumed.
    """
    heads = marker_positions(stem, f"{BLOCK_SEPARATOR}({LETTERS[0]}) ")
    tails = [marker_positions(stem, f"\n({LETTERS[index]}) ") for index in range(1, options)]

    found: list[tuple[str, list[str]]] = []
    for cuts in product(heads, *tails):
        if any(cuts[i] >= cuts[i + 1] for i in range(len(cuts) - 1)):
            continue
        choices = []
        for index, at in enumerate(cuts):
            label = len(BLOCK_SEPARATOR) + 4 if index == 0 else 5
            end = cuts[index + 1] if index + 1 < len(cuts) else len(stem)
            choices.append(stem[at + label : end])
        found.append((stem[: cuts[0]], choices))
    return found


def parse_frozen_stem(item_id: str, stem: str, *, options: int) -> tuple[str, list[str]]:
    """Split a frozen stem into its question and its lettered options, or abort.

    Raises:
        SystemExit: If the stem has no lettered block, if more than one split is
            consistent with it, or if the split does not reassemble the stem exactly.
    """
    splits = decompositions(stem, options)
    if not splits:
        raise SystemExit(
            f"Item {item_id!r} has no {options}-option block of the form "
            f"'{BLOCK_SEPARATOR}(A) ...\\n(B) ...'. Its stem is not a frozen "
            f"MCQAChatFormatter turn, so there is no ordering in it to transfer and "
            f"the choices would have to be re-derived. Nothing was written."
        )
    if len(splits) > 1:
        raise SystemExit(
            f"Item {item_id!r} admits {len(splits)} different splits into a question "
            f"and an {options}-option block, so one of its options contains a lettered "
            f"line of its own and the parse is a guess rather than a reading. A guessed "
            f"ordering is the one thing this file exists not to contain. Nothing was "
            f"written."
        )

    question, choices = splits[0]
    rebuilt = question + BLOCK_SEPARATOR + render_block(choices)
    if rebuilt != stem:
        raise SystemExit(
            f"Item {item_id!r} parsed into a question and {options} options that do not "
            f"reassemble its stem. The block layout is not the one this parser assumes, "
            f"so the options it recovered are not the ones the gold letter indexes. "
            f"Nothing was written."
        )
    return question, choices


def git_show(ref: str, path: str) -> str:
    """Return ``path`` at ``ref`` without checking anything out."""
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


def read_records(dataset: str, ref: str) -> list[dict[str, Any]]:
    """Return the bank's committed item records at ``ref``.

    Read out of git rather than off the working tree, deliberately. This runs
    immediately before a re-vendor overwrites the file it is reading, so naming the
    commit is what makes the result reproducible afterwards -- and re-running it against
    the same ref later has to produce the same control or the transfer was never a
    transfer.
    """
    path = f"calibrated_datasets/{dataset}/items.jsonl"
    return [json.loads(line) for line in git_show(ref, path).splitlines() if line.strip()]


def freeze(dataset: str, ref: str, *, options: int) -> dict[str, Any]:
    """Build ``dataset``'s ordering control from its committed stems at ``ref``."""
    spec = get_spec(dataset)
    records = read_records(dataset, ref)
    if not records:
        raise SystemExit(f"{dataset}: no items at {ref}; there is no ordering to freeze.")

    outside = load_cross_check(dataset)
    frozen: dict[str, dict[str, Any]] = {}
    scored_agreed = source_agreed = 0
    for record in records:
        item_id = str(record["id"])
        stem = str(record["question"])
        gold_letter = str((record.get("metadata") or {}).get("gold_answer", "")).strip()
        _, choices = parse_frozen_stem(item_id, stem, options=options)

        if gold_letter not in LETTERS[:options]:
            raise SystemExit(
                f"Item {item_id!r} records gold_answer {gold_letter!r}, which is not one "
                f"of {list(LETTERS[:options])}. The letter is the only statement this "
                f"bank makes about which option is correct, so a control built without "
                f"it would freeze an ordering and lose the answer. Nothing was written."
            )

        ours = choices[LETTERS.index(gold_letter)]
        leaderboard_gold, source_options = outside.get(item_id, ("", []))
        if leaderboard_gold:
            if comparable(ours) != comparable(leaderboard_gold):
                raise SystemExit(
                    f"Item {item_id!r} answers {ours!r} and the outside evaluation scored "
                    f"{leaderboard_gold!r}. This is the check a byte-equality test cannot "
                    f"do: the two orderings are allowed to differ and the answer is not, "
                    f"so a disagreement here means the gold in this bank names a "
                    f"distractor. Freezing it would preserve the error rather than the "
                    f"ordering. Nothing was written."
                )
            scored_agreed += 1

        if source_options:
            if comparable(ours) != comparable(source_options[0]):
                raise SystemExit(
                    f"Item {item_id!r} answers {ours!r} where the benchmark's own correct "
                    f"answer is {source_options[0]!r}. The gold letter in this bank names "
                    f"a distractor. Nothing was written."
                )
            if sorted(map(comparable, choices)) != sorted(map(comparable, source_options)):
                raise SystemExit(
                    f"Item {item_id!r} carries options {choices!r} where the benchmark's "
                    f"own four are {source_options!r}. The stem was parsed into something "
                    f"that is not this question's option set, so whatever ordering it "
                    f"recovered belongs to another question. Nothing was written."
                )
            source_agreed += 1

        frozen[item_id] = FrozenOrder(
            choices=choices,
            gold_letter=gold_letter,
            stem_sha256=sha256(stem),
            leaderboard_gold=leaderboard_gold,
            source_options=source_options,
        ).as_dict()

    commit = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    log.info("Froze the option ordering of %d %s items read at %s", len(frozen), dataset, ref)
    if outside:
        log.info(
            "Outside agreement about the answer: %d of %d against what was scored, "
            "%d of %d against the benchmark's own columns",
            scored_agreed,
            len(frozen),
            source_agreed,
            len(frozen),
        )

    check = CROSS_CHECKS.get(dataset)
    return {
        "dataset": dataset,
        "options": options,
        "items": len(frozen),
        "cross_checked": scored_agreed,
        "source_checked": source_agreed,
        "cross_check": (
            None
            if check is None
            else {
                "repo": check.repo,
                "revision": check.revision,
                "what": (
                    "Open LLM Leaderboard v2's own per-example records. leaderboard_gold "
                    "is the option its harness scored as correct, which is the stronger "
                    "claim and empties on an option written entirely inside brackets, "
                    "since its preprocessing deletes bracketed spans. source_options is "
                    "the benchmark's own Correct Answer and Incorrect Answer columns "
                    "untouched, correct one first, which reaches every item. Both are "
                    "frozen as text so the comparison is independent of every ordering."
                ),
                "compared_by": "scripts/freeze_choice_order.comparable",
            }
        ),
        "source": {
            "ref": ref,
            "commit": commit,
            "path": f"calibrated_datasets/{dataset}/items.jsonl",
            "modality": spec.modality,
            "layout": (
                "olmo_eval.common.formatters.MCQAChatFormatter, whose user turn is the "
                "question, a blank line, then '(A) choice' lines"
            ),
        },
        "derivation": (
            "Each stem was split into its question and its lettered block by enumerating "
            "every increasing run of (A)..(D) markers and requiring exactly one, then "
            "requiring the two halves to reassemble the byte-identical stem. The letter "
            "is the bank's own metadata['gold_answer']. Nothing was enumerated from the "
            "source dataset and no shuffle was re-applied, which is the point: this is a "
            "transfer of an ordering that already existed rather than a reconstruction of "
            "one."
        ),
        "checked_by": (
            "scripts/vendor_bank.check_choice_order at vendoring time, and "
            "tests/test_gpqa_bank.py against whatever is committed."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "order": frozen,
    }


def load_control(dataset: str) -> dict[str, FrozenOrder]:
    """Return ``item id -> frozen ordering`` for ``dataset``, or abort.

    Raises:
        SystemExit: If the control is missing or holds no orderings.
    """
    path = control_path(dataset)
    if not path.is_file():
        raise SystemExit(
            f"{dataset}: no option-ordering control at {path}. The bank's choices are "
            f"only meaningful in one order and this file is the record of it, so "
            f"vendoring will not write a choice list it cannot check. Build it from the "
            f"last commit that carried the frozen stems:\n"
            f"    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.freeze_choice_order "
            f"--dataset {dataset} --ref <commit>"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    order = payload.get("order") or {}
    if not order:
        raise SystemExit(f"{dataset}: the ordering control at {path} records no items.")
    return {item_id: FrozenOrder.from_dict(record) for item_id, record in order.items()}


def build_parser() -> argparse.ArgumentParser:
    """Build the freezing script's argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.freeze_choice_order",
        description="Transfer a vendored bank's option ordering out of its frozen stems.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset whose stems to read. One of: {', '.join(supported_names())}.",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git ref holding the frozen stems (default: HEAD, before a re-vendor).",
    )
    parser.add_argument(
        "--options",
        type=int,
        default=4,
        help="Options per item the block is expected to carry (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report without writing the control.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Freeze one dataset's option ordering.

    Logging is configured here rather than at import, unlike the other scripts in this
    directory, because this module is also imported by ``vendor_bank`` for its parser and
    its control loader. ``basicConfig`` is a no-op after the first call and the format
    string names its script, so configuring it on import would relabel every line
    vendoring writes as coming from this one.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [uni_mcq.freeze] %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    payload = freeze(args.dataset, args.ref, options=args.options)

    if args.dry_run:
        log.info(
            "[dry-run] %s: %d orderings parsed. Nothing written.",
            args.dataset,
            payload["items"],
        )
        return 0

    path = control_path(args.dataset)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
