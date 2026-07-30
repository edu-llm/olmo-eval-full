#!/usr/bin/env python3
"""Download per-question Open LLM Leaderboard responses into OpenLM/.

Pulls sample-level outputs from each model's gated ``*-details`` dataset on
Hugging Face for models in a parameter range (default 0.2–8B), then writes:

    OpenLM/
      ifeval/<org>__<model>.csv
      bbh/<org>__<model>.csv
      math/<org>__<model>.csv
      gpqa/<org>__<model>.csv
      musr/<org>__<model>.csv
      mmlu_pro/<org>__<model>.csv

Each CSV has one row per question with the model response and either a
correct/wrong result (MCQ / exact-match tasks) or a grade (IFEval instruction
following).

Requires a Hugging Face token with access to gated details datasets::

    export HF_TOKEN=hf_...
    # or: huggingface-cli login

Details repos use ``gated=auto``; on first encounter the script requests access
for the authenticated user.

Examples::

    uv run python AdaptiveTesting/Inputs/OpenLM/download_openlm_responses.py --limit-models 3
    uv run python AdaptiveTesting/Inputs/OpenLM/download_openlm_responses.py --benchmarks ifeval,math
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.utils import get_token

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install datasets: uv add datasets") from exc

LOG = logging.getLogger("openlm_download")

CONTENTS_DATASET = "open-llm-leaderboard/contents"
DETAILS_ORG = "open-llm-leaderboard"

# Top-level leaderboard benchmarks → sample filename prefix under leaderboard_*
BENCHMARKS: dict[str, str] = {
    "ifeval": "leaderboard_ifeval",
    "bbh": "leaderboard_bbh_",
    "math": "leaderboard_math_",
    "gpqa": "leaderboard_gpqa_",
    "musr": "leaderboard_musr_",
    "mmlu_pro": "leaderboard_mmlu_pro",
}

CSV_FIELDS = [
    "question_id",
    "model",
    "benchmark",
    "subtask",
    "question",
    "response",
    "predicted",
    "gold",
    "result",
    "grade",
    "scoring_method",
]

SAMPLE_RE = re.compile(
    r"samples_(leaderboard_[a-z0-9_]+)_(\d{4}-\d{2}-\d{2}T[\d.-]+)\.jsonl$"
)


@dataclass(frozen=True)
class ModelRow:
    fullname: str
    params_b: float
    eval_name: str
    average: float | None

    @property
    def details_repo(self) -> str:
        return f"{DETAILS_ORG}/{self.fullname.replace('/', '__')}-details"

    @property
    def slug(self) -> str:
        return self.fullname.replace("/", "__")


def _truthy_metric(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value > 0)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "correct", "yes", "1"}:
            return True
        if low in {"false", "incorrect", "wrong", "no", "0"}:
            return False
    return None


def _flatten_response(value: Any) -> str:
    """Best-effort extract of generated text / choice scores from lm-eval samples."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return ""
        # Common shapes: ["text"], [["text"]], [[[lp0], [lp1], ...]]
        if len(value) == 1:
            return _flatten_response(value[0])
        if all(isinstance(x, (int, float)) for x in value):
            return json.dumps(value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _doc_question(doc: Any) -> str:
    if not isinstance(doc, dict):
        return ""
    for key in (
        "prompt",
        "question",
        "query",
        "problem",
        "input",
        "goal",
        "passage",
        "text",
    ):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # Fall back to a compact JSON of the doc (may be hashed on some repos).
    try:
        return json.dumps(doc, ensure_ascii=False)
    except TypeError:
        return str(doc)


def _doc_gold(doc: Any, target: Any) -> str:
    if target is not None and target != "":
        if isinstance(target, (list, dict)):
            return json.dumps(target, ensure_ascii=False)
        return str(target)
    if isinstance(doc, dict):
        for key in ("answer", "target", "gold", "label", "correct_answer", "answerKey"):
            if key in doc and doc[key] not in (None, ""):
                val = doc[key]
                if isinstance(val, (list, dict)):
                    return json.dumps(val, ensure_ascii=False)
                return str(val)
    return ""


def _pick_metric(sample: dict[str, Any]) -> tuple[str | None, Any]:
    """Return (scoring_method, raw_metric_value) preferring leaderboard metrics."""
    preferred = [
        ("prompt_level_strict_acc", "ifeval_prompt_strict"),
        ("exact_match", "exact_match"),
        ("acc_norm", "acc_norm"),
        ("acc", "acc"),
        ("exact_match,none", "exact_match"),
        ("acc_norm,none", "acc_norm"),
        ("acc,none", "acc"),
    ]
    for key, method in preferred:
        if key in sample and sample[key] is not None:
            return method, sample[key]
    # Any *acc / exact_match-like leftover key
    for key, value in sample.items():
        if key in {"doc", "doc_id", "resps", "filtered_resps", "target", "arguments"}:
            continue
        if value is None:
            continue
        if "strict_acc" in key or key.endswith("exact_match") or key in {"acc", "acc_norm"}:
            return key, value
    return None, None


def _grade_and_result(
    benchmark: str, sample: dict[str, Any]
) -> tuple[str, str, str]:
    """Return (result, grade, scoring_method).

    ``result`` is correct/wrong/partial/unknown.
    ``grade`` is a numeric string when partial credit / IFEval applies.
    """
    method, metric = _pick_metric(sample)

    # IFEval: expose prompt-level correct/wrong plus mean instruction grade.
    if benchmark == "ifeval":
        prompt = sample.get("prompt_level_strict_acc")
        inst = sample.get("inst_level_strict_acc")
        grade = ""
        if isinstance(inst, list) and inst:
            grade = f"{sum(float(x) for x in inst) / len(inst):.6f}"
        elif isinstance(prompt, (int, float, bool)):
            grade = f"{float(prompt):.6f}"
        truth = _truthy_metric(prompt)
        if truth is True:
            result = "correct"
        elif truth is False:
            result = "wrong"
        else:
            result = "unknown"
        return result, grade, method or "ifeval_prompt_strict"

    truth = _truthy_metric(metric)
    grade = ""
    if isinstance(metric, (int, float)) and not isinstance(metric, bool):
        # Non-binary scores (rare) — keep as grade.
        if metric not in (0, 1, 0.0, 1.0):
            grade = f"{float(metric):.6f}"
            result = "partial" if 0 < float(metric) < 1 else ("correct" if float(metric) >= 1 else "wrong")
            return result, grade, method or "metric"
    if truth is True:
        return "correct", grade or "1", method or "metric"
    if truth is False:
        return "wrong", grade or "0", method or "metric"
    return "unknown", grade, method or ""


def _mcq_choice_texts(sample: dict[str, Any]) -> list[str]:
    """Choice strings from lm-eval ``arguments`` (gen_args_i.arg_1)."""
    args = sample.get("arguments")
    if not isinstance(args, dict):
        return []
    choices: list[str] = []
    i = 0
    while f"gen_args_{i}" in args:
        entry = args[f"gen_args_{i}"]
        if isinstance(entry, dict) and "arg_1" in entry:
            choices.append(str(entry["arg_1"]))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            choices.append(str(entry[1]))
        else:
            choices.append("")
        i += 1
    return choices


def _mcq_logprobs(filtered_resps: Any) -> list[float] | None:
    """Parse MCQ filtered_resps: [[logprob, is_greedy], ...] → logprobs."""
    if not isinstance(filtered_resps, list) or not filtered_resps:
        return None
    logprobs: list[float] = []
    for item in filtered_resps:
        # Common shapes: [lp, greedy] or [[lp, greedy]]
        cur = item[0] if isinstance(item, list) and item and isinstance(item[0], list) else item
        if not isinstance(cur, (list, tuple)) or not cur:
            return None
        try:
            logprobs.append(float(cur[0]))
        except (TypeError, ValueError):
            return None
    return logprobs


def _predicted_from_sample(sample: dict[str, Any]) -> tuple[str, str]:
    """Return (response, predicted_answer).

    Generative tasks: both are the decoded text.
    MCQ loglikelihood tasks: ``response`` keeps raw score pairs; ``predicted``
    is the argmax choice from ``arguments``.
    """
    filt = sample.get("filtered_resps")
    choices = _mcq_choice_texts(sample)
    logprobs = _mcq_logprobs(filt)
    if choices and logprobs and len(choices) == len(logprobs):
        best_i = max(range(len(logprobs)), key=lambda i: logprobs[i])
        predicted = choices[best_i]
        # Compact scoreboard for the response column.
        board = [{"choice": c, "logprob": lp} for c, lp in zip(choices, logprobs)]
        return json.dumps(board, ensure_ascii=False), predicted

    text = _flatten_response(filt if filt is not None else sample.get("resps"))
    return text, text


def sample_to_row(
    sample: dict[str, Any],
    *,
    model: str,
    benchmark: str,
    subtask: str,
) -> dict[str, str]:
    doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else sample.get("doc")
    doc_id = sample.get("doc_id")
    if doc_id is None and isinstance(doc, dict):
        doc_id = doc.get("id") or doc.get("key") or doc.get("idx")
    question_id = str(doc_id) if doc_id is not None else ""
    if not question_id and isinstance(doc, dict) and "prompt" in doc:
        question_id = str(abs(hash(doc["prompt"])) % (10**12))

    response, predicted = _predicted_from_sample(sample)
    gold = _doc_gold(doc, sample.get("target"))
    result, grade, method = _grade_and_result(benchmark, sample)

    return {
        "question_id": question_id,
        "model": model,
        "benchmark": benchmark,
        "subtask": subtask,
        "question": _doc_question(doc),
        "response": response,
        "predicted": predicted,
        "gold": gold,
        "result": result,
        "grade": grade,
        "scoring_method": method,
    }


def latest_sample_files(files: list[str]) -> dict[str, str]:
    """Map full task name -> repo path for the newest timestamp per task."""
    best: dict[str, tuple[str, str]] = {}
    for path in files:
        name = path.rsplit("/", 1)[-1]
        match = SAMPLE_RE.search(name)
        if not match:
            continue
        task, stamp = match.group(1), match.group(2)
        prev = best.get(task)
        if prev is None or stamp > prev[0]:
            best[task] = (stamp, path)
    return {task: path for task, (_stamp, path) in best.items()}


def benchmark_for_task(task: str) -> str | None:
    for bench, prefix in BENCHMARKS.items():
        if prefix.endswith("_"):
            if task.startswith(prefix) or task == prefix.rstrip("_"):
                return bench
        elif task == prefix or task.startswith(prefix + "_"):
            return bench
    return None


def subtask_name(task: str, benchmark: str) -> str:
    prefix = BENCHMARKS[benchmark]
    if task == prefix or task == prefix.rstrip("_"):
        return benchmark
    if prefix.endswith("_") and task.startswith(prefix):
        return task[len(prefix) :]
    if task.startswith(prefix + "_"):
        return task[len(prefix) + 1 :]
    if task.startswith("leaderboard_"):
        return task[len("leaderboard_") :]
    return task


def load_leaderboard_models(
    min_params: float,
    max_params: float,
    *,
    include_flagged: bool = False,
) -> list[ModelRow]:
    LOG.info("Loading %s …", CONTENTS_DATASET)
    ds = load_dataset(CONTENTS_DATASET, split="train")
    rows: list[ModelRow] = []
    for rec in ds:
        params = rec.get("#Params (B)")
        if params is None:
            continue
        try:
            params_f = float(params)
        except (TypeError, ValueError):
            continue
        if not (min_params <= params_f <= max_params):
            continue
        if not include_flagged and rec.get("Flagged"):
            continue
        fullname = rec.get("fullname")
        if not fullname or not isinstance(fullname, str):
            continue
        avg = rec.get("Average ⬆️")
        try:
            avg_f = float(avg) if avg is not None else None
        except (TypeError, ValueError):
            avg_f = None
        rows.append(
            ModelRow(
                fullname=fullname,
                params_b=params_f,
                eval_name=str(rec.get("eval_name") or ""),
                average=avg_f,
            )
        )

    # One details repo per model id; keep the highest-average eval.
    by_name: dict[str, ModelRow] = {}
    for row in rows:
        prev = by_name.get(row.fullname)
        if prev is None:
            by_name[row.fullname] = row
            continue
        prev_avg = prev.average if prev.average is not None else float("-inf")
        cur_avg = row.average if row.average is not None else float("-inf")
        if cur_avg > prev_avg:
            by_name[row.fullname] = row

    selected = sorted(by_name.values(), key=lambda r: (r.params_b, r.fullname))
    LOG.info(
        "Selected %d unique models in [%.2f, %.2f] B (from %d content rows)",
        len(selected),
        min_params,
        max_params,
        len(rows),
    )
    return selected


def request_gated_access(api: HfApi, repo_id: str) -> None:
    """Best-effort access request for gated=auto details datasets."""
    token = get_token()
    if not token:
        return
    try:
        # huggingface_hub >= 0.24
        ask = getattr(api, "ask_access", None) or getattr(api, "request_access", None)
        if callable(ask):
            ask(repo_id=repo_id, repo_type="dataset")
            return
    except Exception as exc:  # noqa: BLE001
        LOG.debug("ask_access failed for %s: %s", repo_id, exc)
    # Fallback HTTP endpoint used by the Hub UI.
    try:
        import httpx

        url = f"https://huggingface.co/api/datasets/{repo_id}/user-access-request/ask"
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"fields": {}},
            timeout=30.0,
        )
        LOG.debug("access request %s -> %s %s", repo_id, resp.status_code, resp.text[:200])
    except Exception as exc:  # noqa: BLE001
        LOG.debug("HTTP access request failed for %s: %s", repo_id, exc)


def download_sample(repo_id: str, filename: str, token: str | None) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    tmp.replace(path)


def process_model(
    model: ModelRow,
    *,
    out_root: Path,
    benchmarks: set[str],
    token: str | None,
    skip_existing: bool,
    api: HfApi,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": model.fullname,
        "params_b": model.params_b,
        "status": "ok",
        "benchmarks": {},
        "error": "",
    }

    needed = []
    for bench in benchmarks:
        dest = out_root / bench / f"{model.slug}.csv"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            summary["benchmarks"][bench] = {"skipped": True, "path": str(dest)}
        else:
            needed.append(bench)
    if not needed:
        summary["status"] = "skipped"
        return summary

    repo_id = model.details_repo
    try:
        files = list_repo_files(repo_id, repo_type="dataset", token=token)
    except RepositoryNotFoundError:
        summary["status"] = "missing_details"
        summary["error"] = f"No details dataset: {repo_id}"
        return summary
    except GatedRepoError:
        request_gated_access(api, repo_id)
        time.sleep(1.0)
        try:
            files = list_repo_files(repo_id, repo_type="dataset", token=token)
        except Exception as exc:  # noqa: BLE001
            summary["status"] = "gated"
            summary["error"] = str(exc)
            return summary
    except HfHubHTTPError as exc:
        summary["status"] = "http_error"
        summary["error"] = str(exc)
        return summary

    task_files = latest_sample_files(files)
    if not task_files:
        summary["status"] = "no_samples"
        summary["error"] = "No samples_*.jsonl files found"
        return summary

    # Accumulate rows per top-level benchmark.
    by_bench: dict[str, list[dict[str, str]]] = {b: [] for b in needed}
    for task, rel_path in sorted(task_files.items()):
        bench = benchmark_for_task(task)
        if bench is None or bench not in by_bench:
            continue
        sub = subtask_name(task, bench)
        try:
            local = download_sample(repo_id, rel_path, token)
        except GatedRepoError:
            request_gated_access(api, repo_id)
            time.sleep(1.0)
            try:
                local = download_sample(repo_id, rel_path, token)
            except Exception as exc:  # noqa: BLE001
                summary["status"] = "gated"
                summary["error"] = str(exc)
                return summary
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s: failed to download %s (%s)", model.fullname, rel_path, exc)
            continue

        for sample in iter_jsonl(local):
            by_bench[bench].append(
                sample_to_row(
                    sample,
                    model=model.fullname,
                    benchmark=bench,
                    subtask=sub,
                )
            )

    for bench, rows in by_bench.items():
        dest = out_root / bench / f"{model.slug}.csv"
        if not rows:
            summary["benchmarks"][bench] = {"rows": 0, "path": str(dest), "missing": True}
            continue
        write_csv(dest, rows)
        summary["benchmarks"][bench] = {"rows": len(rows), "path": str(dest)}

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=here,
        help="Root OpenLM folder (benchmark subdirs are created here)",
    )
    p.add_argument("--min-params", type=float, default=0.2, help="Min params in billions")
    p.add_argument("--max-params", type=float, default=8.0, help="Max params in billions")
    p.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(BENCHMARKS),
        help=f"Comma-separated subset of: {','.join(BENCHMARKS)}",
    )
    p.add_argument("--limit-models", type=int, default=0, help="Only process first N models (0=all)")
    p.add_argument("--offset", type=int, default=0, help="Skip the first N models after filtering")
    p.add_argument("--workers", type=int, default=4, help="Parallel model downloads")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--include-flagged", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="List models only; do not download")
    p.add_argument("--models-list", type=Path, default=None, help="Write selected models CSV here")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    token = get_token()
    if not token and not args.dry_run:
        LOG.error(
            "No Hugging Face token found. Details datasets are gated. "
            "Set HF_TOKEN or run `huggingface-cli login`."
        )
        return 2

    benches = {b.strip() for b in args.benchmarks.split(",") if b.strip()}
    unknown = benches - set(BENCHMARKS)
    if unknown:
        LOG.error("Unknown benchmarks: %s", ", ".join(sorted(unknown)))
        return 2

    models = load_leaderboard_models(
        args.min_params,
        args.max_params,
        include_flagged=args.include_flagged,
    )
    if args.offset:
        models = models[args.offset :]
    if args.limit_models and args.limit_models > 0:
        models = models[: args.limit_models]

    list_path = args.models_list or (args.out_dir / "models_selected.csv")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fullname", "params_b", "average", "details_repo"])
        w.writeheader()
        for m in models:
            w.writerow(
                {
                    "fullname": m.fullname,
                    "params_b": m.params_b,
                    "average": m.average if m.average is not None else "",
                    "details_repo": m.details_repo,
                }
            )
    LOG.info("Wrote model list -> %s (%d models)", list_path, len(models))

    if args.dry_run:
        for m in models[:20]:
            LOG.info("  %.3fB  %s", m.params_b, m.fullname)
        if len(models) > 20:
            LOG.info("  … and %d more", len(models) - 20)
        return 0

    api = HfApi(token=token)
    summaries: list[dict[str, Any]] = []
    workers = max(1, args.workers)

    def _job(m: ModelRow) -> dict[str, Any]:
        return process_model(
            m,
            out_root=args.out_dir,
            benchmarks=benches,
            token=token,
            skip_existing=args.skip_existing,
            api=api,
        )

    LOG.info("Downloading details for %d models with %d workers…", len(models), workers)
    if workers == 1:
        for i, m in enumerate(models, 1):
            LOG.info("[%d/%d] %s", i, len(models), m.fullname)
            summaries.append(_job(m))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_job, m): m for m in models}
            done = 0
            for fut in as_completed(futs):
                done += 1
                m = futs[fut]
                try:
                    summary = fut.result()
                except Exception as exc:  # noqa: BLE001
                    summary = {
                        "model": m.fullname,
                        "params_b": m.params_b,
                        "status": "exception",
                        "benchmarks": {},
                        "error": str(exc),
                    }
                summaries.append(summary)
                LOG.info(
                    "[%d/%d] %s -> %s",
                    done,
                    len(models),
                    m.fullname,
                    summary.get("status"),
                )

    status_path = args.out_dir / "download_status.csv"
    with status_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["model", "params_b", "status", "error", "benchmark_rows"],
        )
        w.writeheader()
        for s in sorted(summaries, key=lambda x: x.get("model") or ""):
            bench_rows = {
                b: (info.get("rows") if isinstance(info, dict) else info)
                for b, info in (s.get("benchmarks") or {}).items()
            }
            w.writerow(
                {
                    "model": s.get("model"),
                    "params_b": s.get("params_b"),
                    "status": s.get("status"),
                    "error": s.get("error"),
                    "benchmark_rows": json.dumps(bench_rows),
                }
            )
    LOG.info("Wrote status -> %s", status_path)

    counts: dict[str, int] = {}
    for s in summaries:
        counts[s.get("status") or "unknown"] = counts.get(s.get("status") or "unknown", 0) + 1
    LOG.info("Done. Status counts: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
