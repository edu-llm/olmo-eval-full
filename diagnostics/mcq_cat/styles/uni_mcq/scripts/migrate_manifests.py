"""Add the scoring-convention block to banks vendored before it existed.

Run offline by a developer::

    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.migrate_manifests --dry-run
    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.migrate_manifests

Six banks were vendored before any manifest recorded what convention it was to be
scored under, and :func:`~diagnostics.mcq_cat.styles.uni_mcq.convention.check_runtime_convention`
now refuses to run one that does not. Re-vendoring would produce the block, and it would
also rewrite ``params.json`` and ``items.jsonl`` from a fresh enumeration of six
HuggingFace datasets -- four of which join by position in that enumeration, so a dataset
revision that inserted or dropped a single row would rekey every item after it while
every guard still passed. Nothing about recording a convention needs the parameters
touched, so this rewrites ``manifest.json`` and only ``manifest.json``.

That leaves this script owing the reader the evidence a re-vendor would have supplied,
so it checks rather than asserts. Every bank's recorded ``sha256`` is recomputed from
the bytes on disk and every recorded item count is recounted from both artifacts, and a
disagreement aborts that dataset instead of being written over -- a manifest that
already misdescribes its own bank is not one to add a fresh claim to.

The block it writes is built from ``datasets.py`` and the style's ``config.yaml`` by the
same code the vendoring script calls, so a migrated manifest and a re-vendored one agree
except in ``recorded_by``. That field is the honest part: a block written by vendoring
records the config that was in force when the artifacts were produced, while one written
here records the config as it stands today, because the config in force back then was
never written down. Re-running is idempotent and preserves whatever ``recorded_by`` a
block already carries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .. import convention
from ..datasets import get_spec, supported_names
from ..resolve import ITEMS_NAME, MANIFEST_NAME, PARAMS_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [uni_mcq.migrate] %(levelname)s: %(message)s",
)
log = logging.getLogger("uni_mcq.migrate")

#: repo_root/diagnostics/mcq_cat/styles/uni_mcq/scripts/migrate_manifests.py
REPO_ROOT = Path(__file__).resolve().parents[5]
CALIBRATED_DATASETS = REPO_ROOT / "calibrated_datasets"

#: Stamped into a block this script had to reconstruct.
MIGRATION_RECORDED_BY = "migrate_manifests"

#: The block is written before this key, so a migrated manifest and a freshly vendored
#: one have the same shape as well as the same content.
INSERT_BEFORE = "notes"


def sha256(text: str) -> str:
    """The digest ``vendor_bank`` records, over the same encoding it wrote."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_artifacts(dataset: str, root: Path, manifest: dict[str, Any]) -> None:
    """Abort unless the manifest still describes the bytes beside it.

    The migration's whole claim is that nothing about the bank moved, and the manifest
    is the only place that could say otherwise. Both halves matter and they fail
    differently: a changed digest means the parameters or the item text were edited
    after vendoring, which invalidates the recorded provenance outright, while a count
    that no longer matches means the two artifacts have drifted apart and the CAT would
    silently run over their intersection.
    """
    for name in (PARAMS_NAME, ITEMS_NAME):
        recorded = (manifest.get("sha256") or {}).get(name)
        actual = sha256((root / name).read_text(encoding="utf-8"))
        if recorded is None:
            raise SystemExit(
                f"{dataset}: {MANIFEST_NAME} records no sha256 for {name}, so there is "
                f"nothing to verify the bank against before adding a claim about how it "
                f"must be scored. Re-vendor it instead. Nothing was written."
            )
        if recorded != actual:
            raise SystemExit(
                f"{dataset}: {name} hashes to {actual} but {MANIFEST_NAME} records "
                f"{recorded}. The artifact changed after it was vendored, so its "
                f"recorded provenance -- the drop accounting, the source commit, the "
                f"item count -- describes a different bank. Re-vendor it rather than "
                f"recording a scoring convention over the top. Nothing was written."
            )

    params = json.loads((root / PARAMS_NAME).read_text(encoding="utf-8"))
    items = [line for line in (root / ITEMS_NAME).read_text(encoding="utf-8").splitlines() if line]
    recorded_items = manifest.get("items")
    if not (len(params) == len(items) == recorded_items):
        raise SystemExit(
            f"{dataset}: {MANIFEST_NAME} records {recorded_items} items, "
            f"{PARAMS_NAME} holds {len(params)} and {ITEMS_NAME} holds {len(items)}. "
            f"The three must agree. Nothing was written."
        )
    log.info("%s: %d items, both artifacts hash as recorded", dataset, len(params))


def migrated(manifest: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
    """Return ``manifest`` with ``block`` inserted at the position vendoring writes it."""
    updated: dict[str, Any] = {}
    for key, value in manifest.items():
        if key == convention.CONVENTION_KEY:
            continue
        if key == INSERT_BEFORE:
            updated[convention.CONVENTION_KEY] = block
        updated[key] = value
    updated.setdefault(convention.CONVENTION_KEY, block)
    return updated


def recorded_by(manifest: dict[str, Any]) -> str:
    """Keep an existing attribution, so re-running does not relabel a vendored block."""
    existing = manifest.get(convention.CONVENTION_KEY)
    if isinstance(existing, dict):
        return str(existing.get(convention.RECORDED_BY_KEY) or MIGRATION_RECORDED_BY)
    return MIGRATION_RECORDED_BY


def migrate(dataset: str, config: dict[str, Any], *, root: Path, dry_run: bool) -> bool:
    """Add the block to one bank, returning whether its manifest changed."""
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    check_artifacts(dataset, root, manifest)

    spec = get_spec(dataset)
    block = convention.manifest_block(spec, config, recorded_by=recorded_by(manifest))
    updated = migrated(manifest, block)
    if updated == manifest:
        log.info("%s: already up to date", dataset)
        return False

    payload = json.dumps(updated, indent=2) + "\n"
    if dry_run:
        log.info(
            "[dry-run] %s: would record %s. Nothing written.",
            dataset,
            json.dumps(block[convention.RUNTIME_KEY], sort_keys=True),
        )
        return True

    manifest_path.write_text(payload, encoding="utf-8")
    log.info("%s: recorded the scoring convention in %s", dataset, manifest_path)
    return True


def build_parser() -> argparse.ArgumentParser:
    """Build the migration script's argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.migrate_manifests",
        description="Record the scoring convention in already-vendored bank manifests.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset to migrate; repeatable. Defaults to every vendored bank. One of: "
            f"{', '.join(supported_names())}."
        ),
    )
    parser.add_argument(
        "--calibrated-datasets",
        default=None,
        help=f"Override the bank root (default: {CALIBRATED_DATASETS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every check and report what would change without writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Migrate the requested banks, or every vendored one."""
    args = build_parser().parse_args(argv)
    banks = Path(args.calibrated_datasets) if args.calibrated_datasets else CALIBRATED_DATASETS
    config = convention.load_config()

    requested = args.dataset or supported_names()
    changed = 0
    for dataset in requested:
        root = banks / dataset
        if not (root / MANIFEST_NAME).is_file():
            if args.dataset:
                log.error("%s has no vendored bank at %s", dataset, root)
                return 2
            continue
        changed += migrate(dataset, config, root=root, dry_run=args.dry_run)

    log.info("%d manifest(s) %s", changed, "would change" if args.dry_run else "updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
