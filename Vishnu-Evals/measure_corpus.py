"""Size the two evaluation passes from the corpus itself.

Pass 1 re-prefills from position 0 for every 256-token chunk, so its cost grows
with the square of target length. Pass 2 decodes sequentially, so its cost grows
linearly but cannot be parallelized across time. Knowing the target-length
distribution is what decides which dominates.
"""

import json
import math
import sys
from pathlib import Path

CHUNK = 256
CHARS_PER_TOKEN = 3.5  # rough for formal-proof text; replace with a real tokenizer if needed


def main(corpus):
    eval_dir = Path(corpus) / "eval"
    print(f"{'family':<11} {'rows':>6} {'prompt_tok':>11} {'target_tok':>11} "
          f"{'chunks':>7} {'p1_prefill':>11} {'p2_decode':>10}")
    print("-" * 76)

    totals = {"rows": 0, "p1": 0, "p2": 0}
    for path in sorted(eval_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        prompt_toks, target_toks, p1_tokens, p2_tokens = [], [], 0, 0

        for row in rows:
            target_chars = len(row.get("target", ""))
            text_chars = len(row.get("text", ""))
            target_tok = max(1, int(target_chars / CHARS_PER_TOKEN))
            prompt_tok = max(1, int((text_chars - target_chars) / CHARS_PER_TOKEN))
            prompt_toks.append(prompt_tok)
            target_toks.append(target_tok)

            # pass 1: each chunk re-processes tokens 0..score_end
            n_chunks = math.ceil(target_tok / CHUNK)
            for k in range(1, n_chunks + 1):
                p1_tokens += prompt_tok + min(k * CHUNK, target_tok)
            # pass 2: one sequential decode step per generated token
            p2_tokens += target_tok

        n = len(rows)
        totals["rows"] += n
        totals["p1"] += p1_tokens
        totals["p2"] += p2_tokens
        print(f"{path.stem:<11} {n:>6} {sum(prompt_toks)//n:>11,} {sum(target_toks)//n:>11,} "
              f"{math.ceil((sum(target_toks)/n)/CHUNK):>7} "
              f"{p1_tokens/1e6:>10.1f}M {p2_tokens/1e6:>9.2f}M")

    print("-" * 76)
    print(f"{'TOTAL':<11} {totals['rows']:>6} {'':>11} {'':>11} {'':>7} "
          f"{totals['p1']/1e6:>10.1f}M {totals['p2']/1e6:>9.2f}M")
    print()
    print("prompt_tok / target_tok are per-row means; columns are estimates from")
    print(f"character counts at ~{CHARS_PER_TOKEN} chars/token, facts_present only.")
    print()
    print("p1_prefill = tokens pushed through parallel prefill (GPU-efficient)")
    print("p2_decode  = sequential decode steps (one forward each, batch-limited)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "corpus-v3/corpus-v3")
