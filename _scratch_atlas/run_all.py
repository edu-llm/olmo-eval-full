"""Driver: run the whole analysis in one interpreter and tee each report to disk."""

from __future__ import annotations

import contextlib
import io
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("01_load_merge.py", "report_pools.txt"),
    ("02_classify.py", "report_classify.txt"),
    ("03_report_flags.py", "report_flags.txt"),
    ("05_shortlist.py", "report_shortlist.txt"),
    ("06_gap.py", "report_gap.txt"),
]

for script, outfile in STEPS:
    buf = io.StringIO()
    argv = sys.argv[:]
    sys.argv = [script, "40"]
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    finally:
        sys.argv = argv
    text = buf.getvalue()
    with open(os.path.join(HERE, outfile), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[ok] {script:<22} -> {outfile}  ({len(text.splitlines())} lines)")
