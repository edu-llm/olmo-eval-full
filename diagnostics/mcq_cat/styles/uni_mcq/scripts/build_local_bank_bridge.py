"""Build content-hash bridges for the three locally fitted banks.

A bridge answers the question a calibration output does not: column ``X<k>`` holds a
difficulty, and which question is that? These three banks answer it positionally — their
item ids were minted as ``<bank>_<row index>`` by the 2026-08-01 inference sweep — so bank
index *k* names the *k*-th row that sweep enumerated.

`bridges/README.md` explains why a position is not shipped as the key: it is only usable if
the enumeration can be reproduced exactly, which is an assumption rather than a measurement,
and when the other eight banks were measured it was false on two. So the position is resolved
to a content hash here, once, and the hash is what ships. A re-rendered or reordered split
then *misses* rather than silently naming its neighbour's question, and vendoring's overlap
floor turns a miss into an abort.

What makes that resolution safe for these three is that the assumption was measured rather
than assumed, on both boundaries it has to cross:

* bank ids to the upstream rows — every item of all three banks was compared against the
  sweep's own frozen item cache and matched exactly on stem, choices and gold;
* upstream rows to this harness's enumeration — the tasks were enumerated and compared
  position by position against that same cache.

Neither check is reproduced here; this script consumes their result. Re-run
``diagnostics/mcq_cat/styles/uni_mcq/banks/reproduce_check.py`` and the Phase 0b enumeration
check before trusting a rebuilt bridge.

Ids are derived with :func:`datasets.content_item_id` against the *instance's* question and
choices, which is what ``vendor_bank.task_item_key`` calls. Sharing the function is the point:
two implementations that agreed today would be free to drift, and the failure mode would be a
bridge that joins nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from ..datasets import content_item_id
from .vendor_bank import task_registry

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
BRIDGES = HERE.parent / "bridges"

#: Bank name -> the task spec whose enumeration its positions index.
#:
#: The bare spec, deliberately. ``socialiqa:xlarge`` and ``socialiqa:mc_olmo3base`` set
#: ``limit=10000``, which switches the loader to validation+train and then samples it under
#: a fixed seed: 10,000 instances in an order unrelated to the one calibrated. That bridge
#: would build, join, and be wrong, so the variant is named here rather than left to a
#: caller's choice.
TASKS = {"pedagogy": "pedagogy", "piqa": "piqa", "socialiqa": "socialiqa"}

#: Instances each task must yield, from the frozen 2026-08-01 item cache. A task that
#: enumerates a different number has moved, and a bridge built from it would be misaligned.
EXPECTED_INSTANCES = {"pedagogy": 920, "piqa": 1838, "socialiqa": 1954}

#: How much of the stem to carry in the CSV. Purely for a human reading a diff; nothing
#: joins on it.
QUESTION_PREVIEW = 96


def build(bank: str, task_spec: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Return the bridge rows for one bank, and a count of any duplicated ids.

    A duplicate is not repaired here. ``vendor_bank.drop_ambiguous_ids`` removes every
    claimant of a contested id, on the grounds that a bank cannot say which of two rows a
    hash belongs to; recording the count is this script's job, dropping is that one's.
    """
    get_task = task_registry()
    task = get_task(task_spec)
    instances = list(task.instances)

    expected = EXPECTED_INSTANCES[bank]
    if len(instances) != expected:
        raise SystemExit(
            f"{bank}: task {task_spec!r} enumerated {len(instances)} instances but the "
            f"calibrated bank indexes {expected}. Its enumeration has moved, so every "
            f"position after the first difference would name the wrong question. Nothing "
            f"was written."
        )

    rows: list[dict[str, object]] = []
    seen: dict[str, int] = {}
    for index, instance in enumerate(instances):
        item_id = content_item_id(instance.question, tuple(instance.choices or ()))
        seen[item_id] = seen.get(item_id, 0) + 1
        rows.append(
            {
                # X{k} is 1-based; the sweep's ids were 0-based row indices.
                "atlas_idx": index + 1,
                "item_id": item_id,
                "split_index": index,
                "question": " ".join(instance.question.split())[:QUESTION_PREVIEW],
            }
        )

    duplicates = {k: n for k, n in seen.items() if n > 1}
    return rows, duplicates


def write(bank: str, rows: list[dict[str, object]], duplicates: dict[str, int]) -> None:
    csv_path = BRIDGES / f"{bank}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["atlas_idx", "item_id", "split_index", "question"])
        writer.writeheader()
        writer.writerows(rows)

    contested = sum(duplicates.values())
    sidecar = {
        "dataset": bank,
        "bridge_kind": "content_hash",
        "rows": len(rows),
        "ordering": {
            "source": "AdaptiveTesting inference sweep, full200",
            "responses": (
                f"s3://edullm-adaptive-inference-056956104102/full200/results/Outputs/mcq/{bank}/"
            ),
            "item_cache": (
                "s3://edullm-adaptive-inference-056956104102/full200/mcq_cache/"
                f"{bank}.n2000.s1234.jsonl"
            ),
            "cache_written": "2026-08-01T15:23:51+00:00",
            "loader": (
                "AdaptiveTesting/Test/Inference/datasets_registry.py::load_"
                f"{bank} on ref Research@6998270f"
            ),
            "task_spec": TASKS[bank],
        },
        "item_id": {
            "scheme": "content_hash",
            "derivation": (
                "sha256 of the olmo_eval instance's question and choices, each "
                "NUL-terminated, truncated to the first 16 hex digits; see "
                "datasets.content_item_id"
            ),
            "task": TASKS[bank],
        },
        "matching": {
            "recipe": (
                "bank index k names the k-th row the calibration loader enumerated, "
                "resolved here to the content hash of the k-th task instance"
            ),
            "verified": (
                "every bank item matched the frozen 2026-08-01 item cache exactly on stem, "
                "choices and gold; the task enumeration was then compared against that same "
                "cache position by position"
            ),
            "contested_ids": len(duplicates),
            "rows_claiming_a_contested_id": contested,
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (BRIDGES / f"{bank}.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    note = f", {contested} rows share {len(duplicates)} contested ids" if duplicates else ""
    log.info("%s: %d bridge rows%s -> %s", bank, len(rows), note, csv_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(TASKS), action="append", dest="datasets")
    args = parser.parse_args()

    for bank in args.datasets or sorted(TASKS):
        rows, duplicates = build(bank, TASKS[bank])
        write(bank, rows, duplicates)


if __name__ == "__main__":
    main()
