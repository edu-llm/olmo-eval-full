"""Exercise the real WindowedContainmentScorer and TriviaQA.process_doc source.

Checks the task mirrors Co-LMLM (arXiv:2607.07707) where it claims to, using
verbatim rows from mandarjoshi/trivia_qa rc.nocontext validation.
"""

import ast
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
    any decorator above it. For @dataclass classes that silently yields a plain
    class -- methods still work, but generated __init__ does not -- so slice
    from the first decorator line instead.
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    wanted = set(names)
    found = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    missing = wanted - found
    if missing:
        raise SystemExit(f"could not find in {path.name}: {sorted(missing)}")
    chunks = []
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.ClassDef)) or n.name not in wanted:
            continue
        start = min([d.lineno for d in n.decorator_list] + [n.lineno])
        chunks.append("\n".join(lines[start - 1 : n.end_lineno]))
    return "\n\n".join(chunks)


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


def _out(text: str) -> LMOutput:
    return LMOutput(text=text, extracted_answer=text)


# --- scorer ------------------------------------------------------------------
scorers_src = extract(
    repo / "src/olmo_eval/common/scorers/base.py",
    "_squad_normalize_answer", "WindowedContainmentScorer", "SQuADExactMatchScorer",
    "ContainmentScorer",
)
ns: dict[str, Any] = {"dataclass": dataclass, "Scorer": Scorer,
                      "Instance": Instance, "LMOutput": LMOutput}
exec(compile(scorers_src, "<scorers>", "exec"), ns)
Windowed = ns["WindowedContainmentScorer"]
SquadEM = ns["SQuADExactMatchScorer"]
Containment = ns["ContainmentScorer"]

print("WindowedContainmentScorer -- Co-LMLM's rule:")
w, em = Windowed(), SquadEM()
inst = Instance(gold_answer="Sinclair Lewis",
                metadata={"all_answers": ["Sinclair Lewis", "Harry Sinclair Lewis"]})

check("bare answer matches", w.score(inst, _out("Sinclair Lewis")) == 1.0)
check("answer inside a sentence matches",
      w.score(inst, _out("Sinclair Lewis, the novelist.")) == 1.0)
check("an alias matches", w.score(inst, _out("Harry Sinclair Lewis")) == 1.0)
check("wrong answer scores 0", w.score(inst, _out("Upton Sinclair")) == 0.0)
check("case-insensitive", w.score(inst, _out("sinclair lewis")) == 1.0)
check("no extracted answer scores 0",
      w.score(inst, LMOutput(text="x", extracted_answer=None)) == 0.0)
check("empty generation scores 0", w.score(inst, _out("")) == 0.0)

print()
print("  the 100-character window:")
pad = "I think the answer to this trivia question is probably going to be " * 2
late = pad + "Sinclair Lewis"
check("answer beyond 100 chars does NOT count", w.score(inst, _out(late)) == 0.0,
      f"answer starts at char {late.index('Sinclair')}")
check("...but ContainmentScorer (no window) still finds it",
      Containment.__call__ is not None and Containment().score(inst, _out(late)) == 1.0,
      "confirms the window is the only difference here")
check("answer just inside the window counts",
      w.score(inst, _out("x" * 80 + " Sinclair Lewis")) == 1.0)
check("window is configurable", Windowed(window_chars=10_000).score(inst, _out(late)) == 1.0)

print()
print("  normalization is lowercase ONLY (unlike the SQuAD family):")
beatles = Instance(gold_answer="the Beatles", metadata={"all_answers": ["the Beatles"]})
check("article is NOT stripped, so 'Beatles' misses",
      w.score(beatles, _out("Beatles")) == 0.0,
      "SQuAD-normalized scorers would match here; the paper's rule does not")
check("...and the SQuAD scorer does match it, confirming the difference is real",
      Containment().score(beatles, _out("Beatles")) == 1.0)
check("exact string still matches", w.score(beatles, _out("the Beatles")) == 1.0)

print()
print("  empty reference must not match everything:")
blank = Instance(gold_answer="x", metadata={"all_answers": ["", "  "]})
check("blank aliases are ignored", w.score(blank, _out("anything at all")) == 0.0)

check("name is 'windowed_containment'", w.name == "windowed_containment")
check("default window is 100 chars", Windowed().window_chars == 100)

# --- task --------------------------------------------------------------------
print()
print("TriviaQA.process_doc against real rc.nocontext rows:")
tq_src = extract(
    repo / "src/olmo_eval/evals/tasks/triviaqa.py",
    "_format_query", "_answer_aliases", "TriviaQA",
)
tq_src = tq_src.replace('@register("triviaqa")\n', "").replace(
    "class TriviaQA(Task):", "class TriviaQA:"
)
ns2: dict[str, Any] = {
    "Instance": Instance, "Any": Any, "Iterator": Any, "LMOutput": LMOutput,
    "DataSource": lambda **k: k, "Split": type("S", (), {"VALIDATION": "validation"}),
    "SamplingParams": lambda **k: k, "AccuracyMetric": lambda **k: k,
    "WindowedContainmentScorer": Windowed, "SQuADExactMatchScorer": SquadEM,
    "LMRequest": lambda **k: k, "RequestType": type("R", (), {"COMPLETION": "completion"}),
    "Task": object, "_ANSWER_STUB": "The answer is",
}
exec(compile(tq_src, "<triviaqa>", "exec"), ns2)
TQ = ns2["TriviaQA"]

row = {
    "question": "Which American-born Sinclair won the Nobel Prize for Literature in 1930?",
    "question_id": "tc_33",
    "answer": {
        "value": "Sinclair Lewis",
        "aliases": ["Harry Sinclair Lewis", "Sinclair Lewis", "Lewis, Sinclair"],
        "normalized_value": "sinclair lewis",
        "normalized_aliases": ["harry sinclair lewis", "sinclair lewis"],
    },
}
i = TQ.process_doc(None, row, 0)
check("instance produced", i is not None)
check("prompt is question + answer stub, zero-shot",
      i.question == "Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nThe answer is",
      repr(i.question))
check("no Question:/Answer: wrapper", "Question:" not in i.question and "Answer:" not in i.question)
check("references are value + aliases",
      i.metadata["all_answers"] == ["Sinclair Lewis", "Harry Sinclair Lewis", "Lewis, Sinclair"],
      str(i.metadata["all_answers"]))
check("duplicate alias de-duplicated", i.metadata["all_answers"].count("Sinclair Lewis") == 1)
check("normalized_* fields are NOT used",
      not any(a.islower() and " " in a for a in i.metadata["all_answers"]),
      str(i.metadata["all_answers"]))
check("gold_answer is the raw value", i.gold_answer == "Sinclair Lewis")
check("native id from question_id", i.metadata["id"] == "tc_33")

check("row with no answer dict is skipped",
      TQ.process_doc(None, {"question": "Q?", "answer": None}, 1) is None)
check("row with empty value and aliases is skipped",
      TQ.process_doc(None, {"question": "Q?", "answer": {"value": "", "aliases": []}}, 2) is None)
check("row with no question is skipped",
      TQ.process_doc(None, {"question": "", "answer": {"value": "x"}}, 3) is None)
check("aliases-only row still works",
      TQ.process_doc(None, {"question": "Q?", "answer": {"aliases": ["a"]}}, 4)
      .metadata["all_answers"] == ["a"])

print()
print("  task configuration mirrors the paper:")
check("zero-shot", TQ.num_fewshot == 0)
check("max_tokens is 32", TQ.sampling_params["max_tokens"] == 32)
check("greedy decoding", TQ.sampling_params["temperature"] == 0.0)
check("no stop sequences, as their pipeline sets none",
      "stop_sequences" not in TQ.sampling_params)
check("config is rc.nocontext", TQ.data_source["subset"] == "rc.nocontext")
check("split is validation", TQ.data_source["split"] == "validation")
check("dataset is mandarjoshi/trivia_qa", TQ.data_source["path"] == "mandarjoshi/trivia_qa")

req = TQ.format_request(None, i)
check("request is a bare completion of the stubbed prompt",
      req["prompt"] == i.question and req["request_type"] == "completion", str(req))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("TRIVIAQA MIRRORS CO-LMLM")
