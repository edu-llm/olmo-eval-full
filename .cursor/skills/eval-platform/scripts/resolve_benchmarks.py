#!/usr/bin/env python3
"""Turn a benchmark request into a resolved list, a cost estimate and a verdict.

Reads ``benchmarks.json`` and answers one question: given a group name or an
explicit list, what runs, what will it cost, and is any of it impossible?

Two callers share this, which is why it is a file rather than a shell heredoc.
``run_eval_sweep.sh`` uses it to price a local sweep; ``submit_eval_run.sh`` uses
it to refuse a platform submission before it costs a queue slot. A second
implementation would be a second answer to "is this benchmark real", and the
whole point of the registry is that there is one.

Refusing here is what makes the check worth anything. olmo-eval would also
reject an unknown task, but only after a checkpoint had been fetched and a model
loaded, so a name checked at submission time saves the whole run.

    resolve_benchmarks.py --registry benchmarks.json --group smoke
    resolve_benchmarks.py --registry benchmarks.json --benchmarks "csqa piqa" --limit 2

Exit 0 with a JSON document on stdout, or exit 2 with the reason on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class Refused(Exception):
    """A request that cannot run, with the sentences that say why."""

    def __init__(self, *lines: str) -> None:
        super().__init__(lines[0] if lines else "refused")
        self.lines = lines


def group_members(groups: dict[str, Any], name: str, seen: tuple[str, ...] = ()) -> list[str]:
    """A group's benchmark list, following ``like`` inheritance.

    ``seen`` guards a cycle in the registry: a malformed file should produce a
    message rather than recurse until the stack gives out.
    """
    if name in seen:
        raise Refused(f"group {name!r} inherits from itself: {' -> '.join((*seen, name))}")
    spec = groups[name]
    if isinstance(spec, list):
        return list(spec)
    if "benchmarks" in spec:
        return list(spec["benchmarks"])
    parent = spec.get("like")
    if parent not in groups:
        raise Refused(f"group {name!r} inherits unknown group {parent!r}")
    return group_members(groups, parent, (*seen, name))


def group_limit(groups: dict[str, Any], name: str) -> int | None:
    spec = groups[name]
    if not isinstance(spec, dict):
        return None
    limit = spec.get("limit")
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise Refused(f"group {name!r} has an invalid limit {limit!r}; want a positive integer")
    return limit


def resolve(
    registry: dict[str, Any],
    *,
    group: str | None,
    benchmarks: str | None,
    cli_limit: int | None,
    allow_any_task: bool,
) -> dict[str, Any]:
    known: dict[str, Any] = registry["benchmarks"]
    groups: dict[str, Any] = registry.get("groups", {})
    aliases: dict[str, str] = registry.get("aliases", {})
    unsupported: dict[str, str] = registry.get("unsupported", {})
    limit_unsafe: dict[str, str] = registry.get("limit_unsafe", {})

    if benchmarks and group:
        lines = ["pass --benchmarks or --group, not both: a group already names its benchmarks."]
        if group == "smoke":
            lines += [
                "",
                "For a smoke test over your own list, drop --group and cap the instances:",
                f'  --benchmarks "{benchmarks}" --limit 2',
                "That is exactly what --group smoke does to the default set.",
            ]
        raise Refused(*lines)

    requested: list[str]
    from_group: str | None = None
    if benchmarks:
        requested = [name for name in benchmarks.replace(",", " ").split() if name]
        source = "explicit --benchmarks"
    else:
        from_group = group or "default"
        if from_group not in groups:
            raise Refused(
                f"unknown group {from_group!r}",
                "known groups: " + " ".join(sorted(groups)),
                "See BENCHMARKS.md for what each group covers.",
            )
        requested = group_members(groups, from_group)
        source = f"group {from_group!r}" + ("" if group else " (registry default)")
        spec = groups[from_group]
        if isinstance(spec, dict) and spec.get("description"):
            source += f" -- {spec['description']}"

    if not requested:
        raise Refused("nothing to run: the resolved benchmark list is empty")

    # Names the registry positively knows cannot produce a score. Refused even
    # under --allow-any-task: that flag bypasses *this* registry, not
    # olmo-eval's, so it cannot make a task that does not exist run. Letting
    # these through would fail after the checkpoint was fetched and the model
    # loaded, taking every valid benchmark in the request down with it.
    blocked = [name for name in requested if name in unsupported]
    if blocked:
        lines = ["cannot run: " + " ".join(blocked)]
        lines += [f"  {name}: {unsupported[name]}" for name in blocked]
        lines += [
            "",
            "--allow-any-task does not help here; it skips this registry, not olmo-eval's.",
        ]
        runnable = [name for name in requested if name not in unsupported]
        if runnable:
            lines += [
                "The rest of the request is runnable:",
                f'  --benchmarks "{" ".join(runnable)}"',
            ]
        lines += ["See BENCHMARKS.md for what ships today."]
        raise Refused(*lines)

    unknown = [name for name in requested if name not in known]
    if unknown and not allow_any_task:
        lines = ["unknown benchmark(s): " + " ".join(unknown)]
        for name in unknown:
            target = aliases.get(name)
            if target:
                lines.append(f"  {name!r} is a dataset name; the olmo-eval task is {target!r}")
        lines += [
            "known: " + " ".join(sorted(known)),
            "known groups: " + " ".join(sorted(groups)),
            "See BENCHMARKS.md for what each one scores.",
            "If olmo-eval registers one of these and this registry simply omits it, "
            "--allow-any-task will run it without a cost estimate. That flag cannot "
            "run a task olmo-eval does not define.",
        ]
        raise Refused(*lines)

    # An explicit --limit beats the group's, so a group's cap is a default rather
    # than a cage.
    effective_limit = (
        cli_limit
        if cli_limit is not None
        else (group_limit(groups, from_group) if from_group else None)
    )
    if effective_limit is not None and effective_limit < 1:
        raise Refused(f"--limit must be positive, got {effective_limit}")

    instances = prompts = 0
    for name in requested:
        entry = known.get(name)
        if entry is None:
            continue
        # With a limit in force the estimate is the limited count, not the split
        # size; reporting 17k instances for a 10-instance smoke run would be
        # worse than useless.
        count = entry["instances"]
        if effective_limit is not None:
            count = min(count, effective_limit)
        instances += count
        prompts += count * entry["choices"]

    return {
        "benchmarks": requested,
        "source": source,
        "group": from_group,
        "effective_limit": effective_limit,
        "instances": instances,
        "prompts": prompts,
        "unknown": unknown,
        "limit_unsafe": (
            {name: limit_unsafe[name] for name in requested if name in limit_unsafe}
            if effective_limit is not None
            else {}
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="Path to benchmarks.json")
    parser.add_argument("--group", default=None)
    parser.add_argument("--benchmarks", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-any-task", action="store_true")
    parser.add_argument(
        "--field",
        default=None,
        help="Print one field as plain text instead of the whole JSON document. "
        "For shell callers that want a single value without a JSON parser.",
    )
    args = parser.parse_args(argv)

    try:
        registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read registry {args.registry}: {error}", file=sys.stderr)
        return 2

    try:
        resolved = resolve(
            registry,
            group=args.group,
            benchmarks=args.benchmarks,
            cli_limit=args.limit,
            allow_any_task=args.allow_any_task,
        )
    except Refused as refusal:
        for line in refusal.lines:
            print(line, file=sys.stderr)
        return 2
    except KeyError as error:
        print(f"registry is missing a required section: {error}", file=sys.stderr)
        return 2

    if args.field:
        value = resolved.get(args.field)
        if isinstance(value, list):
            print(" ".join(str(item) for item in value))
        elif value is None:
            print("")
        else:
            print(value)
        return 0

    print(json.dumps(resolved, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
