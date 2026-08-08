"""Recover a finished CAT session's ability trajectory from its report, and plot it.

Run offline by a developer, after a run has landed::

    python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.replay_trajectory \\
        results/arc_challenge/cat_report.json --out-dir cat_trajectories

``cat_report.json`` records the *final* theta and standard error and nothing in
between. ``run_cat`` re-estimates after every response and assigns over the previous
value (``common/cat_loop.py:68``), and ``CATState`` has no field for a history, so the
two figures a reader actually wants -- how the standard error fell, and where the
ability estimate settled -- are not in the artifact that was written.

They are, however, exactly recomputable from it. EAP is memoryless: ``eap_theta_se``
(``../irt.py:62-81``) is a pure function of ``(resp, a, b, c)`` on a fixed 81-node grid,
and ``estimate_ability`` conditions on the whole response pattern each time rather than
updating an accumulator (``../style.py:488-489``). The report serializes ``responses``
in administration order (``base.py:165-174``), and ``params.json`` holds the ``(a, b,
c)`` those responses were scored against. So running the estimator over prefixes of a
sequence that already happened reproduces every intermediate estimate. Nothing is
re-simulated and nothing is approximated -- item *selection* is not replayed, only the
estimator, on the items the session actually administered.

That claim is worth exactly as much as its check, which is why
:func:`verify_replay` is the load-bearing function here and not the plotting.
The replayed final point is compared against the report's own ``ability.theta`` and
``ability.standard_error``; a run that disagrees is not plotted at all, because a
trajectory whose endpoint does not land on the recorded estimate is a picture of some
other session. See :data:`DEFAULT_TOLERANCE` for why the bar is set where it is.

**It also refits MWLE, which costs nothing and is the point of doing this offline.**
``--ability-estimator batch_eap+mwle`` changes what a *future* run publishes; every
report already on disk was written under EAP. But MWLE is a pure function of the same
``(responses, a, b, c)`` the trajectory is built from, so this script can say what it
would have reported for a finished run without the checkpoint, the GPU or the network.
:func:`refit_mwle` does that for every report replayed and :func:`print_estimators`
prints the three thetas together. Unlike the trajectory it is one number per run and not
a series: MWLE is applied once, over everything administered, and is not a running
estimate. A run that *was* launched under MWLE additionally has its recorded estimate
checked against the refit, which is :func:`verify_replay`'s guarantee extended to the
estimator that produced its headline number.

**The CSV is the artifact; the figures are a view of it.** ``cat_eval_tutorbench.py``
builds ``se_trace`` and ``theta_trace``, draws the published SE reduction curve from
them, and then drops both columns before writing its per-model CSV -- so the figure
shipped beside that CSV cannot be regenerated from it. This script inverts that: the
trajectory is written to CSV unconditionally, every plotted series is read back out of
those same rows, and the figures are optional. That also follows the two-stage shape
``hf-converter-patch`` uses throughout, where ``aggregate_results.py`` consolidates to
one CSV and ``plot_curves.py`` plots only what the CSV holds.

**Matplotlib is optional and is usually absent.** It sits in the ``analysis`` extra
(``pyproject.toml:70-74``) that nothing in the runtime image or the checked-out
virtualenv syncs, so it is imported lazily inside :func:`pyplot` and its absence is a
printed note rather than an error. The tabular output does not depend on it.

**Scope.** One run in, one trajectory out, repeated over as many reports as are passed
so a later checkpoint sweep can overlay them without a rewrite. Aggregating across
checkpoints is a separate step (``CAT_METRICS.md`` phase 6) and is deliberately not
here: a cross-checkpoint figure has to intersect step grids and refuse mixed banks, and
neither is a decision this script has the inputs to make.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ....common.irt_params import load_irt_params
from .. import resolve as resolve_mod
from ..irt import ESTIMATOR_BATCH_EAP, MwleResult, eap_theta_se, mwle_theta_se

log = logging.getLogger("uni_mcq.replay")

#: What the runner names a report (``runner.py:112``), and what a directory walk looks for.
REPORT_NAME = "cat_report.json"

#: The estimate before any item has been administered. ``estimate_ability`` returns
#: exactly this on an empty response list (``../style.py:491-492``), so the trajectory
#: starts here and the fall from the prior is visible on the standard-error figure
#: rather than being cropped off its left edge.
#:
#: Taken from that short-circuit and not from the estimator, because the two disagree.
#: ``eap_theta_se`` over no responses returns the standard-normal prior as discretized
#: on 81 nodes truncated to [-4, 4], whose standard deviation is 0.9996 rather than 1.
#: The gap is invisible on a figure and four orders too large for the endpoint check
#: below, so a run that administered nothing would be rejected under the other reading.
PRIOR_THETA = 0.0
PRIOR_SE = 1.0

#: How far the replayed final point may sit from the recorded one before the
#: reconstruction is declared wrong.
#:
#: The two should agree to the last bit, not merely closely. ``eap_theta_se`` is pure,
#: the replay hands it the same responses in the same order with the same ``(a, b, c)``
#: rows read from the same ``params.json`` through the same loader, and a NumPy
#: reduction over an identically shaped and identically strided array is deterministic.
#: So the honest default is "exact", and every unit of slack above that is room a real
#: defect can hide in.
#:
#: It is not zero for one reason: the run and the replay need not have happened on the
#: same NumPy build or the same CPU, and pairwise-summation and SIMD reduction order can
#: differ across either. That moves an O(1) posterior mean by a few ULP -- around 1e-16,
#: and under 1e-13 even after the ``exp``/``log`` in the likelihood.
#:
#: 1e-9 is the figure ``CAT_METRICS.md`` phase 1 commits to. It clears that noise floor
#: by four orders while staying far below anything a genuine defect produces: one item's
#: parameters swapped for another's, or one response dropped from a prefix, moves theta
#: by 1e-2 or more on any bank whose discriminations are not degenerate, and a
#: ``params.json`` rewritten at reduced precision surfaces around 1e-8. A tolerance of
#: 1e-6 -- which still reads "tight" -- would swallow that last case in silence, and it
#: is the one most likely to actually occur, because re-vendoring is routine.
DEFAULT_TOLERANCE = 1e-9

#: One row per trajectory point, tidy: repeated identifiers rather than a nested shape,
#: so several reports concatenate into one frame a sweep can group.
#:
#: ``item_index`` and not ``step``: ``plot_curves.py`` and every consolidated CSV in
#: ``hf-converter-patch`` use ``step`` for the *training* step, and phase 6 joins these
#: rows against exactly that. Two different integers under one name in one join is the
#: kind of collision nobody notices until a figure is wrong.
#:
#: ``se_threshold``, ``n_cross`` and ``n_items_administered`` are per-run constants
#: repeated on every row. Denormalized deliberately: the threshold is what the
#: standard-error figure's reference line is drawn at, and ``n_cross`` beside
#: ``n_items_administered`` is the whole point of computing it, so a reader holding one
#: row can see both without a second file.
#: ``theta_mwle`` and ``mwle_ok`` join the run-level constants for the same reason
#: ``n_cross`` is one: the number a reader wants beside the trajectory's endpoint is what
#: the *other* estimator made of the same responses, and a second file to join against
#: would mean nobody looks. There is no per-step MWLE column because there is no per-step
#: MWLE -- it is a single refit over the whole administered set, not a running estimate.
CSV_COLUMNS = (
    "report",
    "benchmark",
    "checkpoint",
    "item_index",
    "item_id",
    "correct",
    "theta",
    "se",
    "se_threshold",
    "n_cross",
    "n_items_administered",
    "theta_mwle",
    "mwle_ok",
)

#: Stamped into both figure titles. ``CAT_METRICS.md`` records that the published
#: ``se_reduction_curve.png`` is a mean over 115 models drawn only where at least five
#: are still administering, which is a different figure that would otherwise carry an
#: indistinguishable caption. Both are legitimate; reading one as the other is not.
SINGLE_RUN_NOTE = "single run, one checkpoint (not a cross-model mean)"

SE_FIGURE_SUFFIX = "se_curve.png"
THETA_FIGURE_SUFFIX = "theta_trajectory.png"


class ReplayError(RuntimeError):
    """A report could not be replayed, or the replay disagreed with the report."""


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """The ability estimate after ``item_index`` items had been administered.

    ``item_id`` and ``correct`` are ``None`` at index 0, which is the prior and not an
    item. Kept as one record per point rather than as two parallel arrays for the reason
    ``CAT_METRICS.md`` phase 4 gives: parallel arrays cannot guarantee that the response
    at position ``k`` is the response the estimate at position ``k`` conditioned on.
    """

    item_index: int
    item_id: str | None
    correct: bool | None
    theta: float
    se: float


@dataclass(frozen=True, slots=True)
class ReplayedRun:
    """One report, replayed and verified against its own recorded final estimate."""

    path: Path
    benchmark: str
    checkpoint: str
    params_path: Path
    bank_size: int
    se_threshold: float | None
    min_items: int | None
    stop_reason: str
    bank_hash_note: str
    trajectory: tuple[TrajectoryPoint, ...]
    recorded_theta: float
    recorded_se: float
    theta_delta: float
    se_delta: float
    reported_estimator: str
    mwle: MwleResult
    mwle_delta: float | None

    @property
    def theta_batch(self) -> float:
        """The batch EAP estimate, which is the verified endpoint of the trajectory."""
        return self.final.theta

    @property
    def n_items(self) -> int:
        """Items administered, which is the trajectory less its prior point."""
        return len(self.trajectory) - 1

    @property
    def n_cross(self) -> int | None:
        """Items administered when the standard error first reached the target."""
        return first_crossing(self.trajectory, self.se_threshold)

    @property
    def final(self) -> TrajectoryPoint:
        """The replayed endpoint, which verification has pinned to the recorded one."""
        return self.trajectory[-1]


# -- reading reports -------------------------------------------------------------


def discover_reports(paths: Sequence[str | Path], *, name: str = REPORT_NAME) -> list[Path]:
    """Expand each argument into report files: a file is itself, a directory is walked.

    Sorted, so a sweep's CSV row order and figure order do not depend on the order the
    filesystem happened to hand back.
    """
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            walked = sorted(path.rglob(name))
            if not walked:
                raise ReplayError(f"No {name} found anywhere under {path}.")
            found.extend(walked)
        elif path.is_file():
            found.append(path)
        else:
            raise ReplayError(f"No such report or directory: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_report(path: Path) -> dict[str, Any]:
    """Read one ``cat_report.json``, rejecting anything that is not one.

    The shape is checked before any of it is used, because the failure this guards is a
    plausible one -- pointing the script at an ``accuracy_consolidated.csv``-era metrics
    file, or at a report from a style with a different estimator -- and the alternative
    is a ``KeyError`` from three frames down.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"{path}: not readable as JSON ({exc}).") from exc

    if not isinstance(payload, dict):
        raise ReplayError(f"{path}: expected a JSON object, found {type(payload).__name__}.")

    missing = [key for key in ("benchmark", "ability", "responses") if key not in payload]
    if missing:
        raise ReplayError(
            f"{path}: missing {', '.join(missing)}. This does not look like a "
            f"{REPORT_NAME} written by diagnostics.mcq_cat.runner."
        )
    return payload


def resolve_params_path(benchmark: str, override: Path | None) -> Path:
    """Locate the ``params.json`` the run was scored against.

    Resolution goes through :mod:`..resolve` so a replay reads the bank by the same
    ladder the run did. ``override`` exists for the two cases the ladder correctly
    refuses: a bank whose dataset has since been blocked, and a bank vendored somewhere
    other than ``calibrated_datasets/`` for comparison. It is the same escape hatch
    ``check_bridge_alignment.py`` offers, under the same flag name.
    """
    if override is not None:
        params_path = override if override.is_file() else override / resolve_mod.PARAMS_NAME
        if not params_path.is_file():
            raise ReplayError(f"No {resolve_mod.PARAMS_NAME} at {override}.")
        return params_path

    try:
        return resolve_mod.resolve(benchmark).params_path
    except resolve_mod.DatasetNotAvailable as exc:
        raise ReplayError(
            f"Cannot resolve the bank for benchmark {benchmark!r}: {exc}\n"
            f"Pass --bank-dir to replay against a bank the resolver will not hand out, "
            f"which is what a since-blocked dataset needs."
        ) from None


def bank_digest(params_path: Path) -> str:
    """Return the digest of ``params.json`` in the form the manifest records.

    Hashed over the newline-normalized UTF-8 *text*, not over the file's bytes, because
    that is what ``vendor_bank._sha256`` hashes (``vendor_bank.py:1381-1382``) while
    ``Path.write_text`` translated the newlines on the way to disk. On a Windows
    checkout the two differ, and hashing bytes would report every committed bank as
    tampered with.
    """
    return hashlib.sha256(params_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def check_bank_digest(params_path: Path, provenance: dict[str, Any]) -> str:
    """Verify the bank against the digest the report pinned, and describe the outcome.

    The replay is exact only if ``params.json`` still holds the parameters the run was
    scored against, and a re-vendoring is a routine event that changes them without
    changing anything a report currently carries. ``bank_provenance`` gains ``sha256``
    in ``CAT_METRICS.md`` phase 2; until then the key is absent and this warns, which is
    the honest state -- an unpinned replay is not a checked one.
    """
    recorded = provenance.get("sha256")
    if isinstance(recorded, dict):
        recorded = recorded.get(resolve_mod.PARAMS_NAME)
    if not recorded:
        log.warning(
            "%s carries no bank_provenance.sha256, so the replay cannot prove it read "
            "the bank the run read. A re-vendoring since the run would move every "
            "estimate below without saying so.",
            params_path,
        )
        return "unpinned"

    actual = bank_digest(params_path)
    if actual != recorded:
        raise ReplayError(
            f"Bank mismatch for {params_path}: the report pins params.json at "
            f"{str(recorded)[:12]} and the file on disk hashes to {actual[:12]}. The "
            f"bank has been re-vendored since the run, so replaying against it would "
            f"produce a trajectory for parameters the session never saw."
        )
    return f"pinned {actual[:12]}"


# -- the replay ------------------------------------------------------------------


def response_sequence(report: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Return the administered item ids in order and their 0/1 outcomes.

    Two internal agreements are checked here rather than assumed, because both are free
    and both are silent when broken. ``metadata.selected_item_ids`` is written from the
    same list as ``responses`` (``../style.py:584``), so a disagreement means one of the
    two was rebuilt somewhere; and ``num_items_administered`` restates the length, so a
    disagreement means the report was assembled from two different states.
    """
    responses = report["responses"]
    if not isinstance(responses, list):
        raise ReplayError(f"responses is {type(responses).__name__}, expected a list.")

    ids = [str(response["item_id"]) for response in responses]
    outcomes = np.asarray([1.0 if response["correct"] else 0.0 for response in responses])

    recorded_n = report.get("num_items_administered")
    if recorded_n is not None and int(recorded_n) != len(ids):
        raise ReplayError(
            f"num_items_administered is {recorded_n} but {len(ids)} responses are "
            f"recorded. The report does not describe one session."
        )

    selected = (report.get("metadata") or {}).get("selected_item_ids")
    if selected is not None and [str(item) for item in selected] != ids:
        raise ReplayError(
            "metadata.selected_item_ids disagrees with the order of responses[]. The "
            "replay conditions on response order, so it cannot proceed against a report "
            "that records two of them."
        )
    return ids, outcomes


def item_arrays(ids: Sequence[str], params_path: Path) -> tuple[np.ndarray, ...]:
    """Return ``(a, b, c)`` for ``ids``, in administration order.

    Loaded through the same ``load_irt_params`` the style uses, so the floats are the
    ones the run estimated from rather than a second parse of the same file that might
    round differently. Built in response order, so ``a[:k]`` is the prefix the estimate
    at step ``k`` conditioned on -- alignment is positional and there is nowhere for an
    index to be reordered between here and the estimator.
    """
    bank = load_irt_params(params_path)
    if bank.dimensions != 1:
        raise ReplayError(
            f"{params_path} holds a {bank.dimensions}-dimensional bank. This replay runs "
            f"the unidimensional EAP estimator and would silently mean nothing on a MIRT "
            f"bank."
        )

    missing = sorted({item_id for item_id in ids if item_id not in bank.params})
    if missing:
        raise ReplayError(
            f"{len(missing)} administered item(s) are absent from {params_path}, for "
            f"example {missing[:3]}. The report and the bank describe different item "
            f"sets, so no alignment between them is trustworthy."
        )

    rows = [bank.params[item_id] for item_id in ids]
    return (
        np.asarray([float(row.discrimination) for row in rows], dtype=float),
        np.asarray([float(row.difficulty) for row in rows], dtype=float),
        np.asarray([float(row.guessing) for row in rows], dtype=float),
    )


def replay(
    ids: Sequence[str], outcomes: np.ndarray, arrays: tuple[np.ndarray, ...]
) -> list[TrajectoryPoint]:
    """Re-run the estimator over every prefix of an administered sequence.

    The whole method, and it is four lines: for k = 1..n, EAP over the first k responses
    and their aligned parameters. Prefixed by the step-0 prior point.
    """
    a, b, c = arrays
    points = [
        TrajectoryPoint(item_index=0, item_id=None, correct=None, theta=PRIOR_THETA, se=PRIOR_SE)
    ]
    for k in range(1, len(ids) + 1):
        theta, se = eap_theta_se(outcomes[:k], a[:k], b[:k], c[:k])
        points.append(
            TrajectoryPoint(
                item_index=k,
                item_id=ids[k - 1],
                correct=bool(outcomes[k - 1]),
                theta=theta,
                se=se,
            )
        )
    return points


def recorded_eap(path: Path, report: dict[str, Any]) -> tuple[float, float]:
    """The report's own batch EAP endpoint, which is what a replay can reproduce.

    Ordinarily that is ``ability``, because ``ability`` is the EAP estimate. A run
    launched with ``--ability-estimator batch_eap+mwle`` publishes Warm's estimate there
    instead, and Warm's estimate is not what running EAP over prefixes converges to -- so
    checking the trajectory against it would reject every MWLE run as a bad
    reconstruction. Those runs record the EAP endpoint separately, under
    ``metadata.theta_batch``, and that is the number this returns for them.

    Preferring the metadata pair whenever it exists rather than only when the estimator
    says MWLE is deliberate: under ``batch_eap`` the two are the same float by
    construction, so the choice cannot change a default run's outcome, and reading one
    field instead of two removes the case where a report's estimator label and its
    numbers disagree.

    Raises:
        ReplayError: If neither pair is two floats.
    """
    metadata = report.get("metadata") or {}
    if metadata.get("theta_batch") is not None and metadata.get("se_batch") is not None:
        try:
            return float(metadata["theta_batch"]), float(metadata["se_batch"])
        except (TypeError, ValueError) as exc:
            raise ReplayError(f"{path}: metadata.theta_batch/se_batch are not two floats.") from exc
    ability = report["ability"]
    try:
        return float(ability["theta"]), float(ability["standard_error"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayError(f"{path}: ability.theta/standard_error are not two floats.") from exc


def verify_replay(
    path: Path,
    trajectory: Sequence[TrajectoryPoint],
    report: dict[str, Any],
    *,
    tolerance: float,
) -> tuple[float, float, float, float]:
    """Check the replayed endpoint against the report's own final EAP estimate.

    This is the property the rest of the script rests on. If it does not hold, the
    trajectory belongs to some other set of parameters or some other response order, and
    every point on the figures would be wrong in a way no reader could detect -- a
    standard-error curve falls smoothly whether or not it describes the session named in
    its title.

    Returns:
        ``(recorded_theta, recorded_se, theta_delta, se_delta)``.

    Raises:
        ReplayError: On any disagreement beyond ``tolerance``.
    """
    recorded_theta, recorded_se = recorded_eap(path, report)
    final = trajectory[-1]
    theta_delta = abs(final.theta - recorded_theta)
    se_delta = abs(final.se - recorded_se)

    if theta_delta > tolerance or se_delta > tolerance:
        raise ReplayError(
            f"{path}: the replay does not reproduce the recorded final estimate, so the "
            f"reconstruction is wrong and no figure was emitted for this run.\n"
            f"    theta: recorded {recorded_theta!r}, replayed {final.theta!r} "
            f"(delta {theta_delta:.3e})\n"
            f"    se:    recorded {recorded_se!r}, replayed {final.se!r} "
            f"(delta {se_delta:.3e})\n"
            f"    tolerance {tolerance:.3e} over {len(trajectory) - 1} administered "
            f"item(s).\n"
            f"    A gap this size is not floating point. The usual causes are a bank "
            f"re-vendored since the run, a report whose responses were reordered, or an "
            f"ability estimated by something other than this style's EAP."
        )
    return recorded_theta, recorded_se, theta_delta, se_delta


def refit_mwle(
    outcomes: np.ndarray, arrays: tuple[np.ndarray, ...], theta_batch: float
) -> MwleResult:
    """Re-fit theta by MWLE over the whole administered set, offline.

    The same call the style makes at report time, over the same inputs, so a report
    written before ``--ability-estimator`` existed can still be told what MWLE would have
    said about it. Free: no model, no GPU, no network -- the responses are in the report
    and the parameters are in ``params.json``.

    Unlike the trajectory this is a single number and not a series. MWLE is applied once,
    at the end, to everything administered; there is no running MWLE to plot.
    """
    a, b, c = arrays
    if outcomes.size == 0:
        return MwleResult(theta_batch, float("inf"), False, "nothing was administered")
    return mwle_theta_se(outcomes, a, b, c, theta0=theta_batch)


def check_recorded_mwle(
    path: Path, report: dict[str, Any], mwle: MwleResult, *, tolerance: float
) -> float | None:
    """Compare the offline MWLE refit against one the run itself recorded, if it did.

    Only a run launched under ``batch_eap+mwle`` carries ``metadata.theta_mwle``, and for
    those this is the same endpoint check :func:`verify_replay` performs on the EAP path,
    on the estimator that actually produced their headline number. A report whose MWLE
    was a fallback records the EAP value under that key and ``mwle_ok: false``, so it is
    skipped -- there is no MWLE estimate in it to disagree with.

    Returns:
        The absolute disagreement, or ``None`` when there was nothing to compare.

    Raises:
        ReplayError: On a disagreement beyond ``tolerance``.
    """
    metadata = report.get("metadata") or {}
    if not metadata.get("mwle_ok") or metadata.get("theta_mwle") is None:
        return None
    recorded = float(metadata["theta_mwle"])
    if not mwle.converged:
        raise ReplayError(
            f"{path}: the report records a converged MWLE theta of {recorded!r}, but "
            f"re-solving it here did not converge ({mwle.note}). The bank or the "
            f"responses are not the ones that produced it."
        )
    delta = abs(mwle.theta - recorded)
    if delta > tolerance:
        raise ReplayError(
            f"{path}: the MWLE refit does not reproduce the recorded one.\n"
            f"    theta_mwle: recorded {recorded!r}, refitted {mwle.theta!r} "
            f"(delta {delta:.3e}), tolerance {tolerance:.3e}."
        )
    return delta


def first_crossing(trajectory: Sequence[TrajectoryPoint], threshold: float | None) -> int | None:
    """Items administered when the standard error first reached ``threshold``.

    Not the same number as the test length, and the gap is the point of computing it. A
    session stops on ``step >= min_items and se <= se_threshold`` (``../style.py:519-528``),
    so a run whose posterior collapsed at item 6 under a floor of 24 crossed eighteen
    items before it stopped, and nothing in the report says so -- ``stop_reason`` reads
    ``precision_reached`` either way. ``CAT_METRICS.md`` expects that distance to be
    largest on ``bbh``, whose floor is 24 against a median discrimination of 3.99.

    The prior point is skipped rather than tested. It is not an administered item, and a
    threshold at or above 1.0 would otherwise report a crossing at zero items.
    """
    if threshold is None:
        return None
    for point in trajectory:
        if point.item_index >= 1 and point.se <= threshold:
            return point.item_index
    return None


def replay_report(
    path: Path, *, bank_dir: Path | None, tolerance: float, se_threshold: float | None
) -> ReplayedRun:
    """Replay and verify one report end to end."""
    report = load_report(path)
    benchmark = str(report["benchmark"])
    metadata = report.get("metadata") or {}
    settings = metadata.get("cat_settings") or {}

    params_path = resolve_params_path(benchmark, bank_dir)
    hash_note = check_bank_digest(params_path, metadata.get("bank_provenance") or {})

    ids, outcomes = response_sequence(report)
    arrays = item_arrays(ids, params_path)
    trajectory = replay(ids, outcomes, arrays)
    recorded_theta, recorded_se, theta_delta, se_delta = verify_replay(
        path, trajectory, report, tolerance=tolerance
    )

    # After verification, not before: the refit is seeded at the trajectory's endpoint,
    # and seeding it from an endpoint that has not been shown to be the run's own would
    # be reporting an MWLE number for a session this replay does not describe.
    mwle = refit_mwle(outcomes, arrays, trajectory[-1].theta)
    mwle_delta = check_recorded_mwle(path, report, mwle, tolerance=tolerance)

    threshold = se_threshold if se_threshold is not None else settings.get("se_threshold")
    return ReplayedRun(
        path=path,
        benchmark=benchmark,
        checkpoint=str((report.get("run") or {}).get("checkpoint") or ""),
        params_path=params_path,
        bank_size=int(metadata.get("bank_size") or 0),
        se_threshold=None if threshold is None else float(threshold),
        min_items=None if settings.get("min_items") is None else int(settings["min_items"]),
        stop_reason=str(metadata.get("stop_reason") or "unrecorded"),
        bank_hash_note=hash_note,
        trajectory=tuple(trajectory),
        recorded_theta=recorded_theta,
        recorded_se=recorded_se,
        theta_delta=theta_delta,
        se_delta=se_delta,
        reported_estimator=str(metadata.get("ability_estimator_reported") or ESTIMATOR_BATCH_EAP),
        mwle=mwle,
        mwle_delta=mwle_delta,
    )


# -- the CSV ---------------------------------------------------------------------


def rows_for(run: ReplayedRun) -> list[dict[str, Any]]:
    """Render one replayed run as tidy rows, one per trajectory point.

    ``theta`` and ``se`` go out as Python floats, whose ``str`` is the shortest
    representation that round-trips. The figures are drawn from these rows, so rounding
    here would mean the figure and the verified trajectory were not quite the same
    series.
    """
    n_cross = run.n_cross
    return [
        {
            "report": run.path.as_posix(),
            "benchmark": run.benchmark,
            "checkpoint": run.checkpoint,
            "item_index": point.item_index,
            "item_id": "" if point.item_id is None else point.item_id,
            "correct": "" if point.correct is None else int(point.correct),
            "theta": point.theta,
            "se": point.se,
            "se_threshold": "" if run.se_threshold is None else run.se_threshold,
            "n_cross": "" if n_cross is None else n_cross,
            "n_items_administered": run.n_items,
            "theta_mwle": run.mwle.theta if run.mwle.converged else "",
            "mwle_ok": int(run.mwle.converged),
        }
        for point in run.trajectory
    ]


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write the tidy trajectory table and return where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


# -- the figures -----------------------------------------------------------------


def pyplot() -> Any | None:
    """Return ``matplotlib.pyplot`` on the Agg backend, or ``None`` if it is not installed.

    Imported here and not at module scope because matplotlib is in the optional
    ``analysis`` extra that neither the GPU image nor the development virtualenv syncs,
    and the tabular half of this script has no use for it. Selecting Agg before pyplot
    is imported is the ordering ``plot_curves.py:15-18`` uses, and is what keeps a
    headless box from reaching for a display.
    """
    try:
        import matplotlib
    except ImportError:
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_slug(run: ReplayedRun, taken: set[str]) -> str:
    """A filename stem for one run's figures, unique within this invocation."""
    parts = [run.benchmark or "unknown", Path(run.checkpoint.rstrip("/")).name]
    base = "_".join(
        "".join(character if character.isalnum() else "_" for character in part).strip("_")
        for part in parts
        if part
    )
    slug = base or "run"
    suffix = 2
    while slug in taken:
        slug = f"{base}_{suffix}"
        suffix += 1
    taken.add(slug)
    return slug


def figure_titles(run: ReplayedRun) -> tuple[str, str]:
    """Titles for the standard-error and ability figures.

    Both name the run and both say it is one. The item count sits in the title in
    ``plot_curves.py:113``'s format, and :data:`SINGLE_RUN_NOTE` is what keeps this
    figure from being read as the published cross-model curve it resembles.
    """
    where = f"{run.benchmark}  ·  {run.checkpoint or 'unrecorded checkpoint'}"
    scale = f"{run.n_items:,} of {run.bank_size:,} items"
    return (
        f"CAT standard error vs items administered — {SINGLE_RUN_NOTE}\n{where}  ·  {scale}",
        f"CAT ability trajectory — {SINGLE_RUN_NOTE}\n{where}  ·  {scale}",
    )


def render_figures(
    run: ReplayedRun,
    rows: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    taken: set[str] | None = None,
) -> list[Path]:
    """Draw the standard-error and ability figures for one run, if matplotlib is here.

    Every series comes out of ``rows`` -- the same records that were written to the CSV
    -- so nothing can appear on a figure that the table does not carry. Only the titles
    read from ``run``, and they carry provenance rather than data.

    ``taken`` is threaded by the caller across a sweep so two checkpoints of one dataset
    do not write over each other's figures.
    """
    plt = pyplot()
    if plt is None:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = figure_slug(run, set() if taken is None else taken)
    se_title, theta_title = figure_titles(run)

    index = [int(row["item_index"]) for row in rows]
    theta = [float(row["theta"]) for row in rows]
    se = [float(row["se"]) for row in rows]
    threshold = run.se_threshold
    n_cross = run.n_cross

    written: list[Path] = []

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(index, se, "-o", color="#2E6F9E", markersize=4, linewidth=1.8, zorder=3)
    if threshold is not None:
        axes.axhline(
            threshold,
            color="#999999",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label=f"se_threshold ({threshold:g})",
        )
    if n_cross is not None:
        axes.plot(
            [n_cross],
            [se[n_cross]],
            "o",
            color="#C2562A",
            markersize=9,
            zorder=4,
            label=f"first crossing (item {n_cross})",
        )
    # The gap between the crossing and the stop is the most informative thing on this
    # figure whenever the min_items floor bound, and it is invisible in the report.
    if n_cross is not None and n_cross < run.n_items:
        axes.axvline(
            run.n_items,
            color="#C2562A",
            linestyle=":",
            linewidth=1.2,
            zorder=2,
            label=f"stopped at {run.n_items} (floor {run.min_items})",
        )
    axes.set_title(se_title, fontsize=10)
    axes.set_xlabel("items administered")
    axes.set_ylabel("posterior SE (logits)")
    axes.set_ylim(bottom=0.0)
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(fontsize=8, loc="best")
    figure.tight_layout()
    se_path = out_dir / f"{slug}_{SE_FIGURE_SUFFIX}"
    figure.savefig(se_path, dpi=150)
    plt.close(figure)
    written.append(se_path)

    figure, axes = plt.subplots(figsize=(8, 5))
    # theta +/- 1 SE, and the band is defensible because the SE is the posterior
    # standard deviation on the same grid the mean came from -- the two are one
    # summary of one distribution, not an estimate paired with an asymptotic
    # approximation of its variance. It is a +/-1 SD envelope and not a 68% interval:
    # the grid posterior is skewed early, when a handful of responses leaves it close
    # to the prior. Drawn anyway, because without it a two-logit swing at item 3 reads
    # as instability rather than as a wide prior doing what a prior does.
    axes.fill_between(
        index,
        [t - s for t, s in zip(theta, se, strict=True)],
        [t + s for t, s in zip(theta, se, strict=True)],
        color="#2E6F9E",
        alpha=0.18,
        zorder=1,
        label="theta ± 1 posterior SD",
    )
    axes.plot(index, theta, "-", color="#2E6F9E", linewidth=1.8, zorder=3, label="theta")
    for label, marker, color in (("correct", "o", "#2E7D52"), ("incorrect", "x", "#C2562A")):
        want = label == "correct"
        picked = [row for row in rows if row["correct"] != "" and bool(row["correct"]) is want]
        if picked:
            axes.plot(
                [int(row["item_index"]) for row in picked],
                [float(row["theta"]) for row in picked],
                marker,
                color=color,
                markersize=5,
                linestyle="none",
                zorder=4,
                label=label,
            )
    axes.set_title(theta_title, fontsize=10)
    axes.set_xlabel("items administered")
    axes.set_ylabel("posterior mean ability (logits)")
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(fontsize=8, loc="best")
    figure.tight_layout()
    theta_path = out_dir / f"{slug}_{THETA_FIGURE_SUFFIX}"
    figure.savefig(theta_path, dpi=150)
    plt.close(figure)
    written.append(theta_path)

    return written


# -- reporting to the terminal ---------------------------------------------------


def print_estimators(runs: Sequence[ReplayedRun]) -> None:
    """Print the three thetas side by side, and how much shrinkage MWLE removed.

    ``theta_online`` is the last sequential estimate and ``theta_batch`` the refit over
    the same responses. On this style they are the same number -- ``estimate_ability``
    already conditions on the whole administered set -- and the column is here so that
    stays visible rather than assumed. The number worth reading is the last one: how far
    dropping the prior moved theta away from zero, which is the compression the prior was
    adding to every reported ability.
    """
    if not runs:
        return
    header = (
        f"{'benchmark':<18} {'reported':>14} {'theta_online':>13} {'theta_batch':>12} "
        f"{'theta_mwle':>11} {'shrinkage removed':>18}"
    )
    print()
    print(header)
    print("-" * len(header))
    for run in runs:
        batch = run.theta_batch
        if run.mwle.converged:
            mwle = f"{run.mwle.theta:.6f}"
            removed = f"{abs(run.mwle.theta) - abs(batch):+.6f}"
        else:
            mwle = "-"
            removed = "did not converge"
        print(
            f"{run.benchmark[:18]:<18} {run.reported_estimator:>14} "
            f"{run.recorded_theta:>13.6f} {batch:>12.6f} {mwle:>11} {removed:>18}"
        )
    print(
        "  theta_online is the run's own recorded EAP endpoint; theta_batch is this "
        "replay's refit of it.\n"
        "  'shrinkage removed' is |theta_mwle| - |theta_batch|: positive means MWLE "
        "placed the checkpoint\n"
        "  further from the prior mean than EAP did, which is the inward pull the prior "
        "was contributing."
    )
    for run in runs:
        if not run.mwle.converged:
            print(f"  {run.benchmark}: MWLE did not converge -- {run.mwle.note}.")


def print_summary(
    runs: Sequence[ReplayedRun],
    failures: Sequence[tuple[Path, str]],
    *,
    tolerance: float,
    figures_drawn: int,
) -> None:
    """Print the consistency check, in the shape ``plot_curves.py:137-145`` uses.

    A figure separated from its provenance is a figure nobody can defend, so the things
    a reader would otherwise have to take on trust -- which bank, whether it was pinned,
    how exactly the replay landed -- are said in words here as well.
    """
    header = (
        f"{'benchmark':<18} {'items':>6} {'n_cross':>8} {'theta':>8} {'SE':>7} "
        f"{'max delta':>10}  bank / stop"
    )
    print()
    print(header)
    print("-" * len(header))
    for run in runs:
        crossing = "-" if run.n_cross is None else str(run.n_cross)
        print(
            f"{run.benchmark[:18]:<18} {run.n_items:>6} {crossing:>8} "
            f"{run.final.theta:>8.3f} {run.final.se:>7.3f} "
            f"{max(run.theta_delta, run.se_delta):>10.2e}  "
            f"{run.bank_hash_note} / {run.stop_reason}"
        )

    print()
    if runs:
        worst = max(max(run.theta_delta, run.se_delta) for run in runs)
        print(
            f"Every replayed run reproduced its own recorded final theta and standard "
            f"error; the largest disagreement over {len(runs)} run(s) was {worst:.2e}, "
            f"against a tolerance of {tolerance:.0e}."
        )
    print_estimators(runs)
    for run in runs:
        if run.n_cross is not None and run.n_cross < run.n_items:
            print(
                f"  {run.benchmark}: SE reached {run.se_threshold:g} at item "
                f"{run.n_cross} and the session ran {run.n_items - run.n_cross} further "
                f"item(s) to the min_items floor of {run.min_items}. Test length is "
                f"that floor, not the precision the run needed."
            )
        elif run.n_cross is None and run.se_threshold is not None:
            print(
                f"  {run.benchmark}: SE never reached {run.se_threshold:g}; the run "
                f"stopped on {run.stop_reason} with SE {run.final.se:.3f}."
            )

    if figures_drawn:
        print(f"\nWrote {figures_drawn} figure(s).")
    else:
        print(
            "\nNo figures: matplotlib is not installed. It is in the optional 'analysis' "
            "extra, which neither the runtime image nor this virtualenv syncs. The CSV "
            "above is the artifact; install the extra to render it, or plot it yourself."
        )

    if failures:
        print(f"\n{len(failures)} report(s) FAILED and were not plotted:")
        for path, message in failures:
            print(f"\n  {path}:\n    " + message.replace("\n", "\n    "))


# -- the CLI ---------------------------------------------------------------------


def run(
    reports: Iterable[Path],
    *,
    out_dir: Path,
    csv_path: Path,
    bank_dir: Path | None,
    tolerance: float,
    se_threshold: float | None,
    figures: bool,
) -> int:
    """Replay every report, write the CSV, and draw what can be drawn."""
    replayed: list[tuple[ReplayedRun, list[dict[str, Any]]]] = []
    failures: list[tuple[Path, str]] = []

    for path in reports:
        try:
            one = replay_report(
                path, bank_dir=bank_dir, tolerance=tolerance, se_threshold=se_threshold
            )
        except ReplayError as exc:
            # One bad report does not stop the sweep. A run that cannot be verified is
            # dropped from the CSV and from the figures and named at the end, so a
            # fifty-report pass reports all of its failures at once instead of one per
            # invocation -- but the exit status is still non-zero, because a partial
            # result that looks complete is the failure mode this whole script exists
            # to avoid.
            failures.append((path, str(exc)))
            continue
        replayed.append((one, rows_for(one)))

    runs = [one for one, _ in replayed]
    rows = [row for _, run_rows in replayed for row in run_rows]

    figures_drawn = 0
    if rows:
        write_csv(rows, csv_path)
        print(f"{len(rows)} trajectory point(s) over {len(runs)} run(s) -> {csv_path}")
        if figures:
            taken: set[str] = set()
            for one, run_rows in replayed:
                for written in render_figures(one, run_rows, out_dir, taken=taken):
                    figures_drawn += 1
                    print(f"  {written}")

    print_summary(runs, failures, tolerance=tolerance, figures_drawn=figures_drawn)
    return 1 if failures or not runs else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the replay script's argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.replay_trajectory",
        description=(
            "Replay a finished CAT session's theta and standard-error trajectory from "
            "its cat_report.json, verify it against the recorded final estimate, and "
            "write it as CSV (plus figures where matplotlib is installed)."
        ),
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help=f"One or more {REPORT_NAME} paths, or directories to walk for them.",
    )
    parser.add_argument(
        "--out-dir",
        default="cat_trajectories",
        help="Where the CSV and figures are written (default: cat_trajectories).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV path (default: <out-dir>/cat_trajectory.csv).",
    )
    parser.add_argument(
        "--bank-dir",
        default=None,
        help=(
            "Directory holding the params.json to replay against, or the file itself "
            "(default: whatever resolve() hands out for the report's benchmark). Needed "
            "for a dataset that has since been blocked, since the resolver refuses those."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            f"How far the replayed final estimate may sit from the recorded one before "
            f"the run is rejected (default: {DEFAULT_TOLERANCE:g}). Raising this hides "
            f"exactly the bugs it exists to catch."
        ),
    )
    parser.add_argument(
        "--se-threshold",
        type=float,
        default=None,
        help=(
            "Override the threshold n_cross is measured against (default: the run's own "
            "metadata.cat_settings.se_threshold)."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write the CSV only, even where matplotlib is available.",
    )
    parser.add_argument(
        "--report-name",
        default=REPORT_NAME,
        help=f"Filename to look for when walking a directory (default: {REPORT_NAME}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Replay every report named on the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [uni_mcq.replay] %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args(argv)

    bank_dir = Path(args.bank_dir) if args.bank_dir else None
    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / "cat_trajectory.csv"

    try:
        reports = discover_reports(args.reports, name=args.report_name)
    except ReplayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bank_dir is not None and len(reports) > 1:
        log.warning(
            "--bank-dir applies to all %d reports. Every one of them will be replayed "
            "against %s regardless of the benchmark it names.",
            len(reports),
            bank_dir,
        )

    return run(
        reports,
        out_dir=out_dir,
        csv_path=csv_path,
        bank_dir=bank_dir,
        tolerance=args.tolerance,
        se_threshold=args.se_threshold,
        figures=not args.no_figures,
    )


if __name__ == "__main__":
    sys.exit(main())
