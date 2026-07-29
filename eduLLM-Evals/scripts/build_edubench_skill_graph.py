"""Build the EduBench skill co-occurrence graph from the committed rubric bank.

    reads   data/EduBench/rubrics.jsonl, data/EduBench/scenarios.jsonl
    writes  data/EduBench/skill_graph.json
            plots/edubench_graphs/edubench_skill_graph.png

Nodes are the nine EduBench task-type skills (the `q_mapping` axis). An undirected
edge joins two skills when some criterion loads both, weighted by how many criteria
do. Node weight is the q-matrix column total: how many criteria load that skill at
all. It is a node attribute, not a self-loop -- a criterion cannot contain the same
skill twice, so the co-occurrence matrix has no meaningful diagonal.

Weight is counted over EduBench's 12 rubric metrics, not over the 55,231 rubric
rows. A row is one metric instantiated for one scenario, so row multiplicity would
measure how often a task type appears in the corpus rather than how the skills are
wired together. The ingester specializes each metric's criterion *text* per task
type; the invariant this script enforces is that the specialization left
`q_mapping` alone, so a per-metric weight is well defined.

validate() therefore builds the graph two independent ways and requires them to agree:

    metric   -- rows grouped by the abbreviation in the criterion title
    Table 7  -- METRICS[*]["applies"] in ingest_edubench, ignoring the artifact

A third route -- the distinct `q_mapping` vectors, ignoring the criterion text -- used to
be an equally valid cross-check, back when metric and q_mapping were a bijection (every
metric's Table 7 row was unique). That is a coincidence of the *original* Table 7, not a
structural guarantee: nothing requires two metrics to apply to different task sets. After
a manual Table 7 edit gave BFA, DKA and CSI an identical applicability row, deduping by
q_mapping started under-counting relative to deduping by metric (3 metrics collapse to 1
vector), so that route is no longer checked here -- weight is and remains "how many
metrics", not "how many distinct applicability patterns".

Note that a `cNN` slot is NOT a metric: slots are positional within a scenario, so
c04 names five different metrics depending on task type. Only (task type, slot) is
metric-stable. Everything here keys on the metric abbreviation.

ANALYSIS ONLY for the bank: reads `q_mapping` and `criterion`, writes neither, and
runs against the uncalibrated artifact -- `difficulty`/`discrimination` are never
touched and scripts/assign_irt_params.py does not need to have run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from ingest_edubench import METRICS, SKILLS, TASKS
except ModuleNotFoundError as exc:  # huggingface_hub is a top-level import there
    sys.exit(
        f"cannot import scripts/ingest_edubench.py ({exc}).\n"
        "It imports huggingface_hub at module level; run this with an interpreter that "
        "has it installed, e.g. llm-from-scratch/Scripts/python.exe"
    )

CANONICAL_SKILLS = tuple(SKILLS)  # captured before any set_axis() override, for is_canonical

RUBRICS = ROOT / "data" / "EduBench" / "rubrics.jsonl"
SCENARIOS = ROOT / "data" / "EduBench" / "scenarios.jsonl"
OUT_JSON = ROOT / "data" / "EduBench" / "skill_graph.json"
OUT_PLOT = ROOT / "plots" / "edubench_graphs" / "edubench_skill_graph.png"
OUT_QMATRIX = ROOT / "data" / "EduBench" / "table7_q_matrix.csv"
OUT_GROUPED_QMATRIX = ROOT / "data" / "EduBench" / "table7_q_matrix_grouped.csv"

LABEL = {t["slug"]: t["stem"] for t in TASKS}          # HF filename stem, e.g. "AG", "TMG"
DISPLAY = {t["slug"]: t["display"] for t in TASKS}
ABBRS = [m["abbr"] for m in METRICS]                    # paper's H.1.1 -> H.3.4 order
MORDER = {a: i for i, a in enumerate(ABBRS)}
TITLES = {m["abbr"]: m["title"] for m in METRICS}
APPLIES = {m["abbr"]: set(m["applies"]) for m in METRICS}

# `slug` was chosen to mirror Table 7's own column header text (see data/EduBench/README.md,
# "Skill axis": "slugged from the Table 7 column headers"), so its initials ARE the paper's
# acronym -- e.g. slug "grading" -> "G", the header literally being "Grading". This differs
# from LABEL (the HF file stem) for 5 of 9 tasks, because the stems were named after the
# *display* name instead: AG = "Automatic Grading", TMG = "Teaching Material Generation",
# Q&A = "Question Answering", PLS = "Personalized Learning Support", ES = "Emotional
# Support" -- none of which is the phrase Table 7 actually uses as that column's header.
#
# material_generation is kept as the override "TMG" rather than the mechanically-derived
# "MG", to match LABEL/DISPLAY and every other reference to this task elsewhere.
ACRONYM = {s: "".join(w[0].upper() for w in s.split("_")) for s in SKILLS}
ACRONYM["material_generation"] = "TMG"
assert len(set(ACRONYM.values())) == len(SKILLS), f"acronym collision in {ACRONYM}"

# A user-specified 3-skill distillation of the 9 task types, by request -- not derived from
# the paper. Every task appears in exactly one group.
GROUPS = {
    "student-oriented": ["answering_questions", "idea_provision", "learning_support",
                         "mental_health"],
    "checking-student-work": ["error_correction", "grading"],
    "generation": ["question_generation", "material_generation",
                   "personalized_content_creation"],
}
assert sorted(s for g in GROUPS.values() for s in g) == sorted(SKILLS), (
    "GROUPS must cover every skill exactly once")


def set_axis(skills: list[str], label: dict[str, str], display: dict[str, str]) -> None:
    """Override the skill axis for a non-canonical bank (e.g. a merged one whose
    scenarios/rubrics were built by scripts/build_edubench_merged.py).

    Only SKILLS/LABEL/DISPLAY are swapped -- everything that stays tied to the ORIGINAL
    Table 7 (APPLIES, ABBRS, TITLES, ACRONYM, GROUPS) is deliberately left alone, since the
    functions that read those (the live-source cross-check, --qmatrix-out,
    --grouped-qmatrix-out) are skipped for any non-canonical bank; see main()'s
    `is_canonical`. Never called on the canonical path, so canonical behavior is provably
    unaffected -- the whole module's default state is exactly what it was at import time.
    """
    global SKILLS, LABEL, DISPLAY
    SKILLS = list(skills)
    LABEL = dict(label)
    DISPLAY = dict(display)


# "Criteria: Instruction Following & Task Completion (IFTC)." -- the criterion's first
# line is the only place the metric identity is recorded; criterion_id holds a
# positional slot, not a metric.
TITLE_RE = re.compile(r"^Criteria: (?P<title>.+?) \((?P<abbr>[A-Z]+)\)\.$")


def metric_of(criterion: str) -> str:
    """Return the metric abbreviation a criterion instantiates."""
    head = criterion.split("\n", 1)[0]
    m = TITLE_RE.match(head)
    if not m:
        raise ValueError(f"criterion does not open with a parseable title: {head!r}")
    abbr, title = m.group("abbr"), m.group("title")
    if abbr not in TITLES:
        raise ValueError(f"unknown metric abbreviation {abbr!r} in {head!r}")
    if TITLES[abbr] != title:
        raise ValueError(f"{abbr} title is {title!r}, expected {TITLES[abbr]!r}")
    return abbr


def vec(q_mapping: dict[str, int]) -> tuple[int, ...]:
    """Return a `q_mapping` as a tuple over SKILLS, validating its key set."""
    if list(q_mapping) != SKILLS:
        raise ValueError(f"q_mapping axis is {list(q_mapping)}, expected {SKILLS}")
    bad = {k: v for k, v in q_mapping.items() if v not in (0, 1)}
    if bad:
        raise ValueError(f"q_mapping values must be 0/1, got {bad}")
    return tuple(q_mapping[s] for s in SKILLS)


def skills_of(v: tuple[int, ...]) -> list[str]:
    return [s for s, on in zip(SKILLS, v) if on]


class Bank:
    """The rubric bank reduced to what the graph needs."""

    def __init__(self) -> None:
        self.rows = 0
        self.qm: dict[str, set[tuple[int, ...]]] = defaultdict(set)   # metric -> q_mappings
        self.texts: dict[str, set[str]] = defaultdict(set)            # metric -> criterion texts
        self.tasks: dict[str, set[str]] = defaultdict(set)            # metric -> task types
        self.per_metric_rows: dict[str, int] = defaultdict(int)
        self.all_qm: set[tuple[int, ...]] = set()
        self.slot: dict[str, set[str]] = defaultdict(set)             # cNN -> metrics
        self.task_slot: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.n_criteria: dict[str, set[int]] = defaultdict(set)       # task -> criterion_ids counts
        self.scenarios = 0


def load(rubrics: Path, scenarios: Path) -> Bank:
    b = Bank()
    use_case: dict[str, str] = {}
    with scenarios.open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            use_case[s["scenario_id"]] = s["use_case"]
            b.n_criteria[s["use_case"]].add(len(s["criterion_ids"]))
            b.scenarios += 1

    with rubrics.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            b.rows += 1
            abbr = metric_of(r["criterion"])
            v = vec(r["q_mapping"])
            b.qm[abbr].add(v)
            b.texts[abbr].add(r["criterion"])
            b.per_metric_rows[abbr] += 1
            b.all_qm.add(v)
            sid, slot = r["criterion_id"].rsplit("_c", 1)
            uc = use_case[sid]
            b.tasks[abbr].add(uc)
            b.slot[f"c{slot}"].add(abbr)
            b.task_slot[(uc, f"c{slot}")].add(abbr)
    return b


def cooc(vectors) -> tuple[dict[frozenset[str], int], dict[str, int]]:
    """Edge weights and node weights for a collection of q_mapping vectors."""
    edge: dict[frozenset[str], int] = defaultdict(int)
    node: dict[str, int] = dict.fromkeys(SKILLS, 0)
    for v in vectors:
        on = skills_of(v)
        for s in on:
            node[s] += 1
        for a, c in itertools.combinations(on, 2):
            edge[frozenset((a, c))] += 1
    return dict(edge), node


def active(excluded: frozenset[str] = frozenset()) -> list[str]:
    """The metrics in play, in the paper's row order."""
    return [a for a in ABBRS if a not in excluded]


def qvec(abbr: str) -> tuple[int, ...]:
    """A metric's Table 7 row as a q_mapping vector, per ingest_edubench."""
    return tuple(int(s in APPLIES[abbr]) for s in SKILLS)


def cooc_from_table(excluded: frozenset[str] = frozenset()):
    """Same, straight from Table 7 in ingest_edubench -- no artifact involved."""
    return cooc(qvec(a) for a in active(excluded))


def bank_applies(b: Bank) -> dict[str, set[str]]:
    """Each metric's applicability set as THIS bank actually recorded it (from its own
    q_mapping), not the live-imported Table 7 in ingest_edubench -- which may have moved
    on since this particular rubrics.jsonl was built (e.g. a historical/backup bank)."""
    return {a: set(skills_of(next(iter(v)))) for a, v in b.qm.items() if len(v) == 1}


def contributors(b: Bank, pair_or_skill, excluded: frozenset[str] = frozenset()) -> list[str]:
    """Metrics responsible for an edge or a node, per the bank's own recorded applicability,
    in the paper's row order."""
    want = pair_or_skill if isinstance(pair_or_skill, (set, frozenset)) else {pair_or_skill}
    applies = bank_applies(b)
    return [a for a in active(excluded) if a in applies and want <= applies[a]]


def build_graph(b: Bank, edge: dict[frozenset[str], int], node: dict[str, int],
                excluded: frozenset[str] = frozenset()) -> nx.Graph:
    g = nx.Graph()
    for s in SKILLS:
        g.add_node(s, label=LABEL[s], display=DISPLAY[s], weight=node[s],
                   metrics=contributors(b, s, excluded))
    for pair, w in edge.items():
        a, c = sorted(pair, key=SKILLS.index)
        g.add_edge(a, c, weight=w, metrics=contributors(b, pair, excluded))
    return g


def validate(b: Bank, g: nx.Graph, excluded: frozenset[str] = frozenset(),
            check_against_source: bool = True) -> list[str]:
    """check_against_source gates every comparison against the live-imported
    ingest_edubench.APPLIES. Only meaningful when the bank being validated is the
    canonical one that module currently describes -- for a historical/backup bank built
    under an older Table 7, the source has since moved on, so disagreement there is
    expected, not an error. Graph-internal checks (shape, symmetry, weight==len(metrics))
    always run, since those must hold for ANY bank."""
    err: list[str] = []
    kept = {a: v for a, v in b.qm.items() if a not in excluded}
    e_metric, n_metric = cooc(next(iter(v)) for v in kept.values())

    # The invariant: specializing the criterion text per task type left q_mapping alone.
    forked = {a: len(v) for a, v in b.qm.items() if len(v) != 1}
    if forked:
        err.append(f"q_mapping forked within a metric (must be 1 each): {forked}")
    if set(b.qm) != set(ABBRS):
        err.append(f"metrics in bank {sorted(b.qm)} != METRICS {ABBRS}")

    if check_against_source:
        e_table, n_table = cooc_from_table(excluded)
        for a, vs in b.qm.items():
            want = tuple(int(s in APPLIES[a]) for s in SKILLS)
            if len(vs) == 1 and next(iter(vs)) != want:
                err.append(f"{a} q_mapping {skills_of(next(iter(vs)))} != Table 7 "
                           f"{sorted(APPLIES[a])}")
        # The two routes must agree, scoped to the active metric set. (A third route --
        # the distinct q_mapping vectors -- is deliberately not compared here; see module
        # docstring. It would under-count once two metrics share an applicability row,
        # which is a legitimate state, not an error.)
        if (e_metric, n_metric) != (e_table, n_table):
            err.append("graph from artifact != graph from Table 7 in ingest_edubench")

    # A bare slot is not a metric; only (task type, slot) is stable.
    unstable = {s: sorted(v) for s, v in b.task_slot.items() if len(v) != 1}
    if unstable:
        err.append(f"(task, slot) is not metric-stable: {unstable}")
    if all(len(v) == 1 for v in b.slot.values()):
        err.append("every bare slot maps to one metric -- slot numbering is no longer positional")

    # Graph shape.
    if g.number_of_nodes() != len(SKILLS):
        err.append(f"{g.number_of_nodes()} nodes, expected {len(SKILLS)}")
    if list(nx.selfloop_edges(g)):
        err.append(f"self-loops emitted: {list(nx.selfloop_edges(g))}")
    for u, v, d in g.edges(data=True):
        if d["weight"] != len(d["metrics"]):
            err.append(f"edge {u}-{v} weight {d['weight']} != {len(d['metrics'])} metrics")
    for s, d in g.nodes(data=True):
        if d["weight"] != len(d["metrics"]):
            err.append(f"node {s} weight {d['weight']} != {len(d['metrics'])} metrics")

    # Node weight == criteria per scenario for that task, read off scenarios.jsonl. Only
    # holds unscoped -- excluding a metric drops it from every scenario's count too.
    if not excluded:
        for s in SKILLS:
            seen = b.n_criteria.get(s, set())
            if seen != {g.nodes[s]["weight"]}:
                err.append(f"{s}: scenarios carry {sorted(seen)} criteria, node weight "
                           f"{g.nodes[s]['weight']}")
    if b.rows != sum(b.per_metric_rows.values()):
        err.append("per-metric row counts do not sum to the file's row count")
    return err


def write_qmatrix_csv(path: Path) -> Path:
    """The paper's Table 7 (12 metrics x 9 task types) as CSV, verbatim -- not the skill
    graph. Columns use ACRONYM (derived from Table 7's own header text), not LABEL (the HF
    file stem), since the two disagree for 5 of 9 tasks."""
    lines = ["metric," + ",".join(ACRONYM[s] for s in SKILLS)]
    for a in ABBRS:
        lines.append(a + "," + ",".join(str(int(s in APPLIES[a])) for s in SKILLS))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def grouped_vote(group: list[str], abbr: str) -> tuple[int, int, int]:
    """(checked, count, threshold) for one (group, metric) cell under strict majority --
    checked iff MORE than half the group's task types carry that metric in Table 7. For
    even group sizes, exactly half is not enough (2 of 4 fails; needs 3)."""
    n = len(group)
    threshold = n // 2 + 1
    count = sum(1 for s in group if s in APPLIES[abbr])
    return int(count >= threshold), count, n


def write_grouped_qmatrix_csv(path: Path) -> Path:
    """The 12x9 Table 7 grid distilled to 12x3 via GROUPS, by strict-majority vote per
    (group, metric) cell: a group gets the checkmark only if MORE than half of its member
    task types already have it in the original grid. This is a requested regrouping of Table 7,
    not something EduBench's paper defines."""
    lines = ["metric," + ",".join(GROUPS)]
    for a in ABBRS:
        cells = [str(grouped_vote(members, a)[0]) for members in GROUPS.values()]
        lines.append(a + "," + ",".join(cells))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def matrix(g: nx.Graph) -> str:
    """The NxN co-occurrence grid, empty diagonal, node weight in a trailing column.
    Column width adapts to the longest label -- 5, same as the original fixed width, for
    every canonical label (all <=3 chars); wider only when a merged label like "QG+TMG"
    would otherwise run into its neighbor."""
    w = max(5, max(len(LABEL[s]) for s in SKILLS) + 1)
    lines = [" " * w + "".join(f"{LABEL[s]:>{w}}" for s in SKILLS) + "  |     n"]
    lines.append(" " * w + (f"{'-' * (w - 2):>{w}}" * len(SKILLS)) + "  |  ----")
    for a in SKILLS:
        cells = "".join(
            f"{'-':>{w}}" if a == c else f"{g[a][c]['weight']:>{w}}" if g.has_edge(a, c)
            else f"{'.':>{w}}"
            for c in SKILLS
        )
        lines.append(f"{LABEL[a]:>{w}}{cells}  |  {g.nodes[a]['weight']:>4}")
    return "\n".join(lines)


def summarize(g: nx.Graph) -> dict:
    w = [d["weight"] for *_, d in g.edges(data=True)]
    n = g.number_of_nodes()
    return {
        "n_nodes": n,
        "n_edges": g.number_of_edges(),
        "n_possible_edges": n * (n - 1) // 2,
        "complete": g.number_of_edges() == n * (n - 1) // 2,
        "weight_min": min(w),
        "weight_max": max(w),
        "weight_sum": sum(w),
    }


def to_json(b: Bank, g: nx.Graph, mode: str, excluded: frozenset[str] = frozenset()) -> dict:
    return {
        "source": "data/EduBench/rubrics.jsonl",
        "built_by": "scripts/build_edubench_skill_graph.py",
        "directed": False,
        "q_mapping_mode": mode,
        "excluded_metrics": sorted(excluded, key=MORDER.__getitem__),
        "n_scenarios": b.scenarios,
        "n_rubric_rows": b.rows,
        "n_metrics": len(b.qm) - len(excluded),
        "n_criterion_texts": sum(len(v) for a, v in b.texts.items() if a not in excluded),
        "group_key": "metric",
        "skills": SKILLS,
        "summary": summarize(g),
        "nodes": [
            {"id": s, "label": LABEL[s], "display": DISPLAY[s],
             "weight": g.nodes[s]["weight"], "metrics": g.nodes[s]["metrics"]}
            for s in SKILLS
        ],
        "edges": [
            {"source": a, "target": c, "weight": g[a][c]["weight"],
             "metrics": g[a][c]["metrics"]}
            for a, c in itertools.combinations(SKILLS, 2)
            if g.has_edge(a, c)
        ],
    }


def plot(g: nx.Graph, out_path: Path, excluded: frozenset[str] = frozenset()) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    wmax = max(d["weight"] for *_, d in g.edges(data=True))

    fig, (ax_g, ax_m) = plt.subplots(1, 2, figsize=(16.5, 7.6))

    pos = nx.circular_layout(g)
    for u, v, d in sorted(g.edges(data=True), key=lambda e: e[2]["weight"]):
        frac = d["weight"] / wmax
        nx.draw_networkx_edges(
            g, pos, edgelist=[(u, v)], ax=ax_g, width=0.8 + 4.2 * frac,
            edge_color=[cmap(frac)], alpha=0.35 + 0.55 * frac,
        )
    nx.draw_networkx_nodes(
        g, pos, ax=ax_g, node_color="white", edgecolors="#333333", linewidths=1.4,
        node_size=[420 + 190 * g.nodes[s]["weight"] for s in g],
    )
    nx.draw_networkx_labels(g, pos, ax=ax_g, font_size=10, font_weight="bold",
                            labels={s: LABEL[s] for s in g})
    n_active = len(ABBRS) - len(excluded)
    excl_note = f" -- excludes {', '.join(sorted(excluded, key=MORDER.__getitem__))}" if excluded else ""
    ax_g.set_title(f"skill co-occurrence over {n_active} of EduBench's 12 rubric metrics{excl_note}\n"
                   "edge width and colour = criteria loading both skills; "
                   "node size = criteria per scenario", fontsize=10)
    # Margin scales with the longest label so a merged label like "Q&A+EC+IP+QG+AG+TMG"
    # doesn't run off the axes -- 0.12 unchanged whenever every label is <=3 chars (every
    # canonical stem is), same as the original fixed value.
    longest_label = max(len(LABEL[s]) for s in g)
    ax_g.margins(0.12 + max(0, longest_label - 3) * 0.018)
    ax_g.axis("off")

    m = np.full((len(SKILLS), len(SKILLS)), np.nan)
    for i, a in enumerate(SKILLS):
        for j, c in enumerate(SKILLS):
            if i != j and g.has_edge(a, c):
                m[i, j] = g[a][c]["weight"]
    im = ax_m.imshow(m, cmap=cmap, vmin=0, vmax=wmax)
    ax_m.set_xticks(range(len(SKILLS)), [LABEL[s] for s in SKILLS], rotation=45, ha="right")
    ax_m.set_yticks(range(len(SKILLS)),
                    [f"{LABEL[s]} (n={g.nodes[s]['weight']})" for s in SKILLS])
    for i in range(len(SKILLS)):
        for j in range(len(SKILLS)):
            if not np.isnan(m[i, j]):
                ax_m.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=10,
                          color="white" if m[i, j] < 0.6 * wmax else "black")
    ax_m.set_title("edge weight matrix (diagonal empty -- no self-loops;\n"
                   "n = q-matrix column total for that skill)", fontsize=10)
    fig.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04, label="criteria loading both skills")

    fig.text(0.5, 0.015, "  ".join(f"{LABEL[s]} = {DISPLAY[s]}" for s in SKILLS),
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def rel(p: Path) -> str:
    """p relative to ROOT for display, or the resolved absolute path if p (e.g. a
    user-supplied --out) doesn't live under ROOT at all."""
    resolved = Path(p).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rubrics", type=Path, default=RUBRICS)
    p.add_argument("--scenarios", type=Path, default=SCENARIOS)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--plot", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--qmatrix-out", type=Path, default=OUT_QMATRIX,
                   help="also save the raw Table 7 grid as CSV, columns labeled with the "
                        "paper's own acronyms (see ACRONYM), not the HF file stem")
    p.add_argument("--no-qmatrix-out", action="store_true")
    p.add_argument("--grouped-qmatrix-out", type=Path, default=OUT_GROUPED_QMATRIX,
                   help="also save the 12x3 GROUPS-distilled q-matrix (strict-majority "
                        "vote per group, see grouped_vote())")
    p.add_argument("--no-grouped-qmatrix-out", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="build and validate, write nothing")
    p.add_argument("--skills-meta", type=Path, default=None,
                   help="JSON {slug: {label, display}} for a non-canonical (e.g. merged) "
                        "skill axis -- see scripts/build_edubench_merged.py. Swaps SKILLS/"
                        "LABEL/DISPLAY to match the bank; --qmatrix-out/--grouped-qmatrix-out "
                        "are refused, since Table 7/GROUPS don't generalize to this axis.")
    p.add_argument("--exclude", default="", metavar="ABBR[,ABBR...]",
                   help="drop one or more metrics (e.g. IFTC) before building the graph. "
                        "Table-7-derived checks that assume the full 12-metric set are "
                        "skipped; graph-shape checks still run. Defaults to a "
                        "'_excl_<abbr>' output path so this never overwrites the full graph.")
    args = p.parse_args()
    excluded = frozenset(a.strip().upper() for a in args.exclude.split(",") if a.strip())
    unknown = excluded - set(ABBRS)
    if unknown:
        sys.exit(f"--exclude: unknown metric abbreviation(s) {sorted(unknown)}, "
                 f"expected a subset of {ABBRS}")

    suffix = f"_excl_{'_'.join(sorted(excluded, key=MORDER.__getitem__))}" if excluded else ""
    out_path = args.out or OUT_JSON.with_stem(OUT_JSON.stem + suffix)
    plot_path = args.plot or OUT_PLOT.with_stem(OUT_PLOT.stem + suffix)

    is_canonical = (args.rubrics.resolve() == RUBRICS.resolve()
                   and args.scenarios.resolve() == SCENARIOS.resolve())
    if not is_canonical:
        print(f"note: --rubrics/--scenarios point away from the canonical bank "
              f"({RUBRICS.relative_to(ROOT)}) -- skipping the live Table-7-source-code "
              f"cross-check, since ingest_edubench.py may have changed since this bank "
              f"was built. Graph-internal checks (shape, weight==len(metrics)) still run.")

    if args.skills_meta:
        meta = json.loads(args.skills_meta.read_text(encoding="utf-8"))
        with args.rubrics.open(encoding="utf-8") as fh:
            bank_skills = list(json.loads(fh.readline())["q_mapping"])
        label = {**LABEL, **{s: m["label"] for s, m in meta.items()}}
        display = {**DISPLAY, **{s: m["display"] for s, m in meta.items()}}
        missing = [s for s in bank_skills if s not in label or s not in display]
        if missing:
            sys.exit(f"--skills-meta {args.skills_meta}: no label/display for {missing} "
                     f"(neither in the sidecar nor a canonical singleton slug)")
        set_axis(bank_skills, label, display)
        print(f"skill axis overridden from {rel(args.skills_meta)}: "
              f"{len(bank_skills)} skills -- {[LABEL[s] for s in bank_skills]}")

    b = load(args.rubrics, args.scenarios)
    mode = "onehot" if all(sum(v) == 1 for v in b.all_qm) else "row"
    print(f"{b.scenarios} scenarios, {b.rows} rubric rows, {len(b.qm)} metrics, "
          f"{sum(len(v) for v in b.texts.values())} distinct criterion texts, "
          f"{len(b.all_qm)} distinct q_mapping ({mode})")
    if excluded:
        print(f"excluding: {sorted(excluded, key=MORDER.__getitem__)} "
              f"-> {len(ABBRS) - len(excluded)} of {len(ABBRS)} metrics active")

    if mode == "onehot":
        return _onehot_exit()

    kept = {a: v for a, v in b.qm.items() if a not in excluded}
    edge, node = cooc(next(iter(v)) for v in kept.values())
    g = build_graph(b, edge, node, excluded)

    print("\nq_mapping is constant within every metric (text specialization did not touch it):")
    print(f"  {'metric':<6}{'rows':>8}{'tasks':>7}{'q_mapping':>11}{'texts':>7}")
    for a in ABBRS:
        mark = "  (excluded)" if a in excluded else ""
        print(f"  {a:<6}{b.per_metric_rows[a]:>8}{len(b.tasks[a]):>7}"
              f"{len(b.qm[a]):>11}{len(b.texts[a]):>7}{mark}")

    print("\n" + matrix(g))
    s = summarize(g)
    print(f"\n{s['n_edges']}/{s['n_possible_edges']} possible edges"
          f"{' -- complete graph' if s['complete'] else ''}; "
          f"weights {s['weight_min']}-{s['weight_max']}, sum {s['weight_sum']}")

    if s["n_edges"]:
        thin = [(u, v) for u, v, d in g.edges(data=True) if d["weight"] == s["weight_min"]]
        applies = bank_applies(b)
        universal = [a for a in active(excluded) if applies.get(a) == set(SKILLS)]
        print(f"metrics allocated to all {len(SKILLS)} tasks: {universal or 'none'} "
              f"-- enough on their own to make the graph complete")
        print(f"{len(thin)} edges at weight {s['weight_min']}:")
        for u, v in sorted(thin, key=lambda e: (SKILLS.index(e[0]), SKILLS.index(e[1]))):
            print(f"  {LABEL[u]:>4} - {LABEL[v]:<4} {g[u][v]['metrics']}")
        isolated = [LABEL[n] for n in g if g.degree(n) == 0]
        if isolated:
            print(f"isolated nodes: {isolated}")
        print(f"connected: {nx.is_connected(g)} ({nx.number_connected_components(g)} component(s))")

    err = validate(b, g, excluded, check_against_source=is_canonical)
    print(f"\nvalidate: {'PASS' if not err else f'{len(err)} ERROR(S)'}")
    for e in err:
        print(f"  {e}")
    if err:
        return 1

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_json(b, g, mode, excluded), indent=2, ensure_ascii=False)
                        + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {rel(out_path)}")
    if not args.no_plot:
        print(f"wrote {rel(plot(g, plot_path, excluded))}")

    want_qmatrix = not args.no_qmatrix_out
    want_grouped = not args.no_grouped_qmatrix_out
    if (want_qmatrix or want_grouped) and not is_canonical:
        print(f"\nnote: --qmatrix-out/--grouped-qmatrix-out dump Table 7 and its GROUPS "
              f"distillation VERBATIM from ingest_edubench.py -- both are defined over the "
              f"original 9-skill axis and don't generalize to this one "
              f"({len(SKILLS)} skills). Skipping both.")
        want_qmatrix = want_grouped = False
    if want_qmatrix:
        print(f"wrote {rel(write_qmatrix_csv(args.qmatrix_out))}")
    if want_grouped:
        print(f"\n{'metric':<6}" + "".join(f"{g:>24}" for g in GROUPS))
        for a in ABBRS:
            row = f"{a:<6}"
            for members in GROUPS.values():
                checked, count, n = grouped_vote(members, a)
                row += f"{f'{checked} ({count}/{n}, need >={n // 2 + 1})':>24}"
            print(row)
        print(f"wrote {rel(write_grouped_qmatrix_csv(args.grouped_qmatrix_out))}")
    return 0


def _onehot_exit() -> int:
    print("\nThis bank has a one-hot q_mapping: every criterion loads exactly one skill, so\n"
          "no criterion ever contains two skills and the graph has no edges. A co-occurrence\n"
          "graph only exists under `ingest_edubench.py --q-mapping row`, where a criterion's\n"
          "q_mapping is its full Table 7 row. Refusing to write an edgeless artifact.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
