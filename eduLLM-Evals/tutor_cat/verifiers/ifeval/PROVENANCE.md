# Vendored: `instruction_following_eval` (IFEval verifier)

- **Upstream:** https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **Files:** `instructions.py`, `instructions_util.py`, `instructions_registry.py`
- **License:** Apache License 2.0 (per-file headers preserved verbatim).
- **Retrieved from:** `raw.githubusercontent.com/google-research/google-research/master/...`

## Local modifications

The only edits are to make the three files importable as a self-contained package
(they originally imported each other as `from instruction_following_eval import ...`):

- `instructions.py`: `from instruction_following_eval import instructions_util`
  → `from . import instructions_util`
- `instructions_registry.py`: `from instruction_following_eval import instructions`
  → `from . import instructions`

Checker logic is otherwise **unmodified**.

## Runtime dependencies

`langdetect`, `immutabledict`, `absl-py`, `nltk` (with the `punkt` tokenizer data).
`nltk` was already present; the others were added. `punkt`/`punkt_tab` must be
downloaded once (`nltk.download("punkt")`).

## Why vendored (not pip-installed)

`instruction_following_eval` is not published as a standalone PyPI package; the
canonical source is the google-research monorepo. Vendoring pins the exact checker
behaviour so IFEval scoring is reproducible.
