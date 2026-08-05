"""Exercise the real ContainmentScorer and PopQA.process_doc source.

olmo_eval is not installed in this environment, so rather than reimplementing
the logic (which would test nothing), pull the actual source segments out with
ast and run them against minimal stand-ins for Instance/LMOutput/Scorer.
"""

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO as repo

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def extract(path: Path, *names: str) -> str:
    """Source for the named top-level defs, decorators included.

    ast.get_source_segment() starts at the `class`/`def` keyword and so drops
    any decorator above it, which silently turns a @dataclass into a plain
    class. Slice from the first decorator line instead.
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    wanted = set(names)
    missing = wanted - {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    if missing:
        raise SystemExit(f"could not find in {path.name}: {sorted(missing)}")
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            out.append("\n".join(lines[start - 1 : node.end_lineno]))
    return "\n\n".join(out)


# --- stand-ins ---------------------------------------------------------------
@dataclass
class Instance:
    question: str = "Q"
    gold_answer: Any = None
    metadata: dict = field(default_factory=dict)
    choices: Any = None


@dataclass
class LMOutput:
    text: str = ""
    extracted_answer: Any = None


class Scorer:
    pass


def _output(text: str) -> LMOutput:
    return LMOutput(text=text, extracted_answer=text)


# --- ContainmentScorer -------------------------------------------------------
scorers_src = extract(
    repo / "src/olmo_eval/common/scorers/base.py",
    "_squad_normalize_answer",
    "ContainmentScorer",
    "SQuADExactMatchScorer",
)
ns: dict[str, Any] = {
    "dataclass": dataclass, "Scorer": Scorer, "Instance": Instance, "LMOutput": LMOutput,
}
exec(compile(scorers_src, "<scorers>", "exec"), ns)
Containment = ns["ContainmentScorer"]
SquadEM = ns["SQuADExactMatchScorer"]

print("ContainmentScorer (real source):")
c, em = Containment(), SquadEM()

i_paris = Instance(gold_answer="Paris")
check("bare answer matches", c.score(i_paris, _output("Paris")) == 1.0)
check("answer inside a sentence matches",
      c.score(i_paris, _output("The capital of France is Paris.")) == 1.0)
check("...where exact match says 0",
      em.score(i_paris, _output("The capital of France is Paris.")) == 0.0)
check("wrong answer scores 0", c.score(i_paris, _output("It is Lyon")) == 0.0)
check("empty generation scores 0", c.score(i_paris, _output("")) == 0.0)

i_pol = Instance(gold_answer="politician",
                 metadata={"all_answers": ["politician", "pol"]})
check("paper's substring rule: 'pol' matches inside 'policy'",
      c.score(i_pol, _output("He works in policy")) == 1.0,
      "documented false positive, kept for comparability")
check("real occupation still matches", c.score(i_pol, _output("He is a politician")) == 1.0)
check("unrelated answer still scores 0", c.score(i_pol, _output("He is a chemist")) == 0.0)

i_ny = Instance(gold_answer="New York")
check("multi-token answer matches contiguously",
      c.score(i_ny, _output("She lives in New York today")) == 1.0)
check("scrambled tokens do not match", c.score(i_ny, _output("New Jersey and York")) == 0.0)

i_hedge = Instance(gold_answer="Paris")
check("hedging beats containment (why EM sits beside it)",
      c.score(i_hedge, _output("Paris, London, or Rome")) == 1.0
      and em.score(i_hedge, _output("Paris, London, or Rome")) == 0.0)

check("containment is never stricter than exact match",
      all(c.score(i, o) == 1.0
          for i, o in [(Instance(gold_answer="Tokyo"), _output(t))
                       for t in ["Tokyo", "tokyo", "  Tokyo  "]]
          if em.score(i, o) == 1.0))
check("no extracted answer scores 0",
      c.score(i_paris, LMOutput(text="Paris", extracted_answer=None)) == 0.0)
check("no gold and no references scores 0",
      c.score(Instance(gold_answer=None, metadata={}), _output("Paris")) == 0.0)
check("name is 'containment'", c.name == "containment")

# --- PopQA.process_doc -------------------------------------------------------
print()
print("PopQA.process_doc (real source), against real dataset rows:")
popqa_src = extract(
    repo / "src/olmo_eval/evals/tasks/popqa.py",
    "_format_query", "_parse_answers", "_as_int", "PopQA",
)
# Drop the class's decorator and base so it can be built without the framework.
popqa_src = popqa_src.replace('@register("popqa")\n', "").replace(
    "class PopQA(Task):", "class PopQA:"
)
ns2: dict[str, Any] = {"json": json, "Instance": Instance, "Any": Any, "Iterator": Any,
                       "DataSource": lambda **k: None, "Split": type("S", (), {"TEST": "test"}),
                       "SamplingParams": lambda **k: None, "AccuracyMetric": lambda **k: None,
                       "ContainmentScorer": Containment, "SQuADExactMatchScorer": SquadEM,
                       "POPQA_FIXED_FEWSHOT": [], "LMRequest": None, "RequestType": None,
                       "LMOutput": LMOutput, "Task": object}
exec(compile(popqa_src, "<popqa>", "exec"), ns2)
PopQA = ns2["PopQA"]
proc = PopQA.process_doc

# Verbatim row 0 from akariasai/PopQA test
row = {
    "id": 4222362, "subj": "George Rankin", "prop": "occupation", "obj": "politician",
    "s_pop": 142, "o_pop": 25692,
    "question": "What is George Rankin's occupation?",
    "possible_answers": '["politician", "political leader", "political figure", "polit.", "pol"]',
}
inst = proc(None, row, 0)
check("instance produced from a real row", inst is not None)
check("question uses the paper's bare Q:/A: template",
      inst.question == "Q: What is George Rankin's occupation?\nA:", inst.question)
check("possible_answers JSON string is parsed to a list",
      inst.metadata["all_answers"] == ["politician", "political leader", "political figure",
                                       "polit.", "pol"],
      str(inst.metadata["all_answers"]))
check("gold_answer is the first reference", inst.gold_answer == "politician")
check("native id comes from the dataset id", inst.metadata["id"] == 4222362)

attrs = inst.metadata.get("instance_attributes")
check("instance_attributes present", isinstance(attrs, dict), str(attrs))
check("subject popularity carried over", attrs.get("s_pop") == 142, str(attrs))
check("object popularity carried over", attrs.get("o_pop") == 25692, str(attrs))
check("relation carried over", attrs.get("prop") == "occupation", str(attrs))
check("popularity is an int, not the raw string", isinstance(attrs.get("s_pop"), int))
check("no model-produced field leaked into attributes",
      not ({"model_output", "score", "prediction"} & set(attrs)), str(sorted(attrs)))

check("row with no answers is skipped",
      proc(None, {"question": "Q?", "possible_answers": "[]"}, 1) is None)
check("row with no question is skipped",
      proc(None, {"question": "", "possible_answers": '["x"]'}, 2) is None)
check("malformed answer JSON degrades to the raw string rather than crashing",
      proc(None, {"question": "Q?", "possible_answers": "not json"}, 3)
      .metadata["all_answers"] == ["not json"])
check("already-parsed list is accepted",
      proc(None, {"question": "Q?", "possible_answers": ["a", "b"]}, 4)
      .metadata["all_answers"] == ["a", "b"])
check("missing popularity omits the key instead of writing null",
      "s_pop" not in (proc(None, {"question": "Q?", "possible_answers": '["x"]'}, 5)
                      .metadata["instance_attributes"]))

# --- few-shot set vs the paper and the dataset ------------------------------
print()
print("Few-shot demonstrations:")
ns3: dict[str, Any] = {}
exec((repo / "src/olmo_eval/evals/tasks/constants/popqa.py").read_text(encoding="utf-8"), ns3)
shots = ns3["POPQA_FIXED_FEWSHOT"]

check("15 demonstrations, matching the paper's shot count for open models",
      len(shots) == 15, str(len(shots)))
check("task requests 15", "num_fewshot = 15" in
      (repo / "src/olmo_eval/evals/tasks/popqa.py").read_text(encoding="utf-8"))
check("every demo has a question and at least one answer",
      all(d.get("question") and d.get("answer") for d in shots))

# Relation templates read off real dataset rows. A demo not matching one of
# these would teach a phrasing the scored questions never use.
REAL_TEMPLATES = [
    "'s occupation?",                 # What is [subj]'s occupation?
    "Who is the author of ",
    "Who was the director of ",
    "Who was the screenwriter for ",
    "Who was the producer of ",
    "What genre is ",
    "Who is the father of ",
    "What sport does ",
]
for d in shots:
    q = d["question"]
    check(f"demo matches a real template: {q[:46]}",
          any(t in q for t in REAL_TEMPLATES), q)

covered = {t for t in REAL_TEMPLATES if any(t in d["question"] for d in shots)}
check("demos span at least 8 distinct relation templates", len(covered) >= 8, str(len(covered)))
check("answers are bare entities, not sentences",
      all(len(a.split()) <= 4 for d in shots for a in d["answer"]),
      str([a for d in shots for a in d["answer"] if len(a.split()) > 4]))

# The prompt the model actually sees
prompt_shots = "\n\n".join(
    f"Q: {d['question']}\nA: {d['answer'][0]}" for d in shots[:2]
)
print()
print("  first two demos as rendered:")
for line in prompt_shots.splitlines():
    print(f"    {line}")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("POPQA IMPLEMENTATION CORRECT")
