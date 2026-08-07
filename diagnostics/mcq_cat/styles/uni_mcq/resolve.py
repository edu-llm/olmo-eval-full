"""Dataset resolution: the allowlist check and the three-branch ladder.

This is the style's only guard against reporting an ability score computed from the
wrong source, so every failure path raises rather than warning. A warning that scrolls
past in a GPU log is exactly how a run ends up publishing a confident wrong number.

The ladder, in order:

1. **Not on the allowlist** -> raise, listing what is ready and, when the name is a
   known exclusion, why it is excluded.
2. **Carrying a recorded blocker** -> raise with it.
3. **On the allowlist and the bank is vendored** -> proceed. This is the only branch
   implemented on this branch.
4. **Bank absent but a graded response matrix is cached** -> raise. Calibrating from a
   matrix needs R with ``mirt`` plus the chunked-fit and mean-sigma-linking port, none
   of which exists here.
5. **Neither** -> raise, pointing at the vendoring script.

**The blocker is checked before the disk, and the order is load-bearing.** The obvious
reading of a blocker is "this could not be vendored", and under that reading testing it
after the artifacts would be harmless, because there would be none. It also has to cover
"this was vendored and the join was afterwards shown wrong", which is the state the three
ATLAS banks were in until their bridges were rebuilt from Open LLM Leaderboard v1 example
order. Those banks were on disk the whole time and loaded cleanly, so a ladder that asked
the filesystem first would have answered yes and run them -- the one outcome this module
exists to prevent, since well-formed artifacts make the failure surface as a confident
ability score rather than as an error. The ordering stays this way now that they are
repaired, because the situation it guards against is the one that recurs.

Resolution runs *before* the checkpoint is downloaded, so an unsupported request costs
nothing. Discovering it after staging a multi-gigabyte checkpoint and booting a GPU
costs real money.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .datasets import DatasetSpec, get_spec, ready_names

log = logging.getLogger("mcq_cat.uni_mcq.resolve")

#: repo_root/diagnostics/mcq_cat/styles/uni_mcq/resolve.py
REPO_ROOT = Path(__file__).resolve().parents[4]

#: Where vendored banks live, one directory per dataset.
CALIBRATED_DATASETS = REPO_ROOT / "calibrated_datasets"

#: Where a graded response matrix would live if branch 3 were implemented. Nothing
#: writes here yet; the ladder only reads it to give a specific error message.
RESPONSE_MATRICES = REPO_ROOT / "response_matrices"

PARAMS_NAME = "params.json"
ITEMS_NAME = "items.jsonl"
MANIFEST_NAME = "manifest.json"


class DatasetNotAvailable(RuntimeError):
    """A dataset cannot be run, with a message explaining exactly why."""


@dataclass(frozen=True, slots=True)
class ResolvedBank:
    """A vendored bank located on disk and ready to load."""

    spec: DatasetSpec
    root: Path
    params_path: Path
    items_path: Path
    manifest: dict

    @property
    def fit_family(self) -> str:
        """The family the parameters were estimated under, from the manifest.

        Authoritative. The manifest is the record of how the bank was produced, so it
        outranks any value supplied at run time.
        """
        return str(self.manifest.get("fit_family", self.spec.fit_family))

    def provenance(self) -> dict:
        """The manifest fields worth carrying into the report."""
        keys = (
            "dataset",
            "task",
            "route",
            "modality",
            "answer_type",
            "scoring_convention",
            "fit_family",
            "source_ref",
            "source_commit",
            "bank_dir",
            "upstream_bank_rows",
            "bridge_rows",
            "items",
            "dropped",
            "ungradable_instances",
            "positional_ids",
            "generated_at",
        )
        return {key: self.manifest[key] for key in keys if key in self.manifest}


def bank_root(dataset: str) -> Path:
    """Return the directory a vendored bank for ``dataset`` would occupy."""
    return CALIBRATED_DATASETS / dataset


def _matrix_candidates(dataset: str) -> list[Path]:
    """Paths that would hold a graded response matrix for ``dataset``."""
    root = RESPONSE_MATRICES / dataset
    return [root / "response_matrix.csv", root / "response_matrix.npy"]


def resolve(dataset: str) -> ResolvedBank:
    """Resolve ``dataset`` to a vendored bank, or raise explaining why not.

    Args:
        dataset: The name the user supplied.

    Returns:
        The located :class:`ResolvedBank`.

    Raises:
        DatasetNotAvailable: For every branch of the ladder except a present bank.
    """
    if not dataset or not dataset.strip():
        raise DatasetNotAvailable(
            "No dataset was given. This style has no default: a dataset is always the "
            "user's to choose, because silently measuring something unasked-for is worse "
            "than failing. Pass --benchmark NAME, or set BENCHMARK=NAME when going "
            f"through tests/aws/run_checkpoint_diag.sh. Ready datasets: "
            f"{', '.join(ready_names())}."
        )

    try:
        spec = get_spec(dataset)
    except KeyError as exc:
        raise DatasetNotAvailable(exc.args[0]) from None

    if spec.blocked is not None:
        raise DatasetNotAvailable(
            f"Dataset {spec.name!r} has a calibrated bank upstream but is blocked: "
            f"{spec.blocked}. Ready datasets: {', '.join(ready_names())}."
        )

    root = bank_root(spec.name)
    params_path = root / PARAMS_NAME
    items_path = root / ITEMS_NAME
    manifest_path = root / MANIFEST_NAME

    if params_path.is_file() and items_path.is_file():
        manifest: dict = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            log.warning(
                "%s has no %s; provenance will be missing from the report.",
                root,
                MANIFEST_NAME,
            )
        resolved = ResolvedBank(
            spec=spec,
            root=root,
            params_path=params_path,
            items_path=items_path,
            manifest=manifest,
        )
        log.info(
            "Resolved %s: %d items, Route %s, %s fit, from %s",
            spec.name,
            manifest.get("items", "?"),
            spec.route,
            resolved.fit_family,
            manifest.get("source_commit", "unknown commit")[:12],
        )
        return resolved

    existing_matrix = next((p for p in _matrix_candidates(spec.name) if p.is_file()), None)
    if existing_matrix is not None:
        raise DatasetNotAvailable(
            f"Dataset {spec.name!r} has no vendored bank at {root}, but a graded "
            f"response matrix is cached at {existing_matrix}. Calibrating a bank from a "
            f"response matrix is not implemented on this branch: it needs R with the "
            f"mirt package plus a port of the chunked 3PL fit "
            f"(Inputs/ATLAS/scripts/01_fit_irt.r) and mean-sigma chunk linking "
            f"(02_link_chunks_custom.r). Only the bank-exists branch is wired here."
        )

    raise DatasetNotAvailable(
        f"Dataset {spec.name!r} is supported but has no vendored bank at {root}, and no "
        f"cached response matrix under {RESPONSE_MATRICES / spec.name}. Vendor it first:\n"
        f"    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.vendor_bank "
        f"--dataset {spec.name} --source-ref origin/Research\n"
        f"Ready datasets: {', '.join(ready_names())}."
    )


def check_fit_family(resolved: ResolvedBank, requested: str | None) -> str:
    """Return the bank's fit family, rejecting a conflicting request.

    The family describes how the parameters were *estimated*. Scoring a 3PL-fit bank
    as though it were 2PL is not an approximation: ``a`` and ``b`` were estimated
    jointly with ``g``, and zeroing ``g`` adds roughly 1.2 logits of bias at a true
    theta of -2, flattering weak models specifically. So a mismatch is an error, not a
    preference.
    """
    actual = resolved.fit_family
    if requested is not None and requested != actual:
        raise DatasetNotAvailable(
            f"Requested fit family {requested!r} but the {resolved.spec.name} bank was "
            f"calibrated as {actual!r} (per {resolved.root / MANIFEST_NAME}). The family "
            f"is a property of how the parameters were estimated, not a runtime choice. "
            f"To use a {requested!r} bank, vendor a real {requested!r} fit with "
            f"--fit-family {requested}."
        )
    return actual
