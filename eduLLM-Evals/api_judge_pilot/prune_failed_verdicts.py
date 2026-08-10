#!/usr/bin/env python
"""Drop error + unscorable verdict records so a resumed grading run re-grades only those.

Keeps clean verdicts (status ok with a parsed pass/fail) and auto_fail records (empty/missing
tutor responses -- not re-gradable). Removes status==error and unscorable (status==ok with a
reason, i.e. unparseable/truncated JSON) so `regrade_benchmark.py` re-grades exactly those
(model, criterion_id) cells on the next resume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

f = Path(sys.argv[1])
rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
kept, dropped = [], []
for r in rows:
    status = r.get("status")
    reason = r.get("reason") or ""
    if status == "error" or (status == "ok" and reason):
        dropped.append(r)
    else:
        kept.append(r)

f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
n_err = sum(1 for r in dropped if r.get("status") == "error")
n_uns = sum(1 for r in dropped if r.get("status") == "ok")
print(f"kept={len(kept)} dropped={len(dropped)} (errors={n_err} unscorable={n_uns}) -> {f}")
