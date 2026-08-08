"""Vendored copy of Google's `instruction_following_eval` verifier (IFEval).

Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval
License: Apache-2.0 (headers preserved in each file). See PROVENANCE.md.

Only intra-package imports were changed (absolute `instruction_following_eval` ->
relative `.`); the checker logic is unmodified. Public surface used by the engine:

    from tutor_cat.verifiers.ifeval import instructions_registry
    cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    checker = cls(instruction_id)
    checker.build_description(**kwargs)      # seeds internal state from kwargs
    passed: bool = checker.check_following(response)
"""

from . import instructions, instructions_registry, instructions_util  # noqa: F401
