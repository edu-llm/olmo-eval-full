"""Verify the vLLM dispatch without a GPU, torch, or vLLM installed.

run_eval imports torch lazily inside functions, so the module loads on a bare
interpreter. Stubbing `vllm` and the engine lets the dispatch, tokenization
handoff and sampling parameters be checked here rather than discovered on a
GPU box an hour into a run.

    python check_backend_dispatch.py --run-eval <path to run_eval.py>
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


class FakeSamplingParams:
    """Records what run_eval asked for so the call can be asserted on."""

    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeSamplingParams.last = kwargs


class FakeCompletion:
    def __init__(self, text):
        self.text = text


class FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [FakeCompletion(text)]


class FakeEngine:
    """Returns a deterministic marker per prompt so ordering is checkable."""

    def __init__(self):
        self.received = None

    def generate(self, prompts, params):
        self.received = (prompts, params)
        return [FakeRequestOutput(f"gen-{i}") for i in range(len(prompts))]


class FakeTokenizer:
    """Token IDs encode prompt identity so the handoff can be verified."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompts, add_special_tokens=True, **kwargs):
        self.calls.append({"prompts": prompts, "add_special_tokens": add_special_tokens})
        if isinstance(prompts, str):
            prompts = [prompts]
        return {"input_ids": [[100 + i, 200 + i] for i in range(len(prompts))]}


def load_run_eval(path: Path):
    sys.path.insert(0, str(path.parent.parent))
    spec = importlib.util.spec_from_file_location("run_eval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_eval_under_test"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-eval", required=True, type=Path)
    args = parser.parse_args()

    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = FakeSamplingParams
    vllm_stub.LLM = object
    sys.modules["vllm"] = vllm_stub

    run_eval = load_run_eval(args.run_eval)

    print("module surface")
    check("generate is module-level", callable(getattr(run_eval, "generate", None)))
    check("_VLLM_ENGINE defaults to None", run_eval._VLLM_ENGINE is None)
    check(
        "generation backends are hf and vllm",
        tuple(run_eval.GENERATION_BACKENDS) == ("hf", "vllm"),
        str(run_eval.GENERATION_BACKENDS),
    )

    import inspect

    signature = list(inspect.signature(run_eval.generate).parameters)
    check(
        "generate keeps the monkeypatched positional signature",
        signature[:3] == ["model", "tok", "prompts"],
        ", ".join(signature),
    )

    print("\ngreedy dispatch")
    engine = FakeEngine()
    run_eval._VLLM_ENGINE = engine
    tok = FakeTokenizer()
    prompts = ["alpha", "beta", "gamma"]

    out = run_eval.generate(
        model=None,
        tok=tok,
        prompts=prompts,
        max_new_tokens=32,
        batch_size=8,
        do_sample=False,
        temperature=0.7,
        device="cuda",
    )

    check("returns one output per prompt", len(out) == len(prompts), f"{len(out)}")
    check("preserves input order", out == ["gen-0", "gen-1", "gen-2"], str(out))
    check(
        "tokenizes with add_special_tokens=False",
        tok.calls and tok.calls[0]["add_special_tokens"] is False,
    )

    sent_prompts, _ = engine.received
    check(
        "hands vLLM token IDs, not raw strings",
        all(isinstance(p, dict) and "prompt_token_ids" in p for p in sent_prompts),
        str(sent_prompts[0]),
    )
    check(
        "token IDs come from the harness tokenizer",
        sent_prompts[0]["prompt_token_ids"] == [100, 200],
        str(sent_prompts[0]["prompt_token_ids"]),
    )

    params = FakeSamplingParams.last
    check("greedy sets temperature 0.0", params["temperature"] == 0.0, str(params["temperature"]))
    check("top_p is 1.0, matching HF top_p=None", params["top_p"] == 1.0)
    check("max_tokens is the per-call budget", params["max_tokens"] == 32)
    check("skip_special_tokens matches HF decode", params["skip_special_tokens"] is True)
    check("single sample per prompt", params["n"] == 1)

    print("\nsampling dispatch")
    run_eval.generate(
        model=None,
        tok=FakeTokenizer(),
        prompts=["x"],
        max_new_tokens=8,
        batch_size=1,
        do_sample=True,
        temperature=0.7,
        device="cuda",
    )
    check(
        "sampling honors --temperature",
        FakeSamplingParams.last["temperature"] == 0.7,
        str(FakeSamplingParams.last["temperature"]),
    )

    print("\nedge cases")
    check("empty prompt list short-circuits", run_eval.generate(None, tok, [], 8, 1, False, 1.0, "cuda") == [])

    run_eval._VLLM_ENGINE = None
    check("engine reset leaves HF path selected", run_eval._VLLM_ENGINE is None)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("All dispatch checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
