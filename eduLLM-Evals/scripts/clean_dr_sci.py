"""Clean the Dr. SCI parquet files (fixes #2, #4, #6a, #7 from the data audit).

Idempotent & reversible: on first run each canonical parquet is backed up to
`*.raw.parquet`; the script always READS from the .raw backup and WRITES the
cleaned result to the canonical filename, so it can be re-run safely.

Fixes applied:
  #1  MathJax "{eq}...{/eq}" custom delimiters (WebInstruct-Verified leftovers) are
      swapped for standard "$...$" across question / ground_truth / reference_answer /
      prompt.
  #8  U+FFFD ("replacement char") corruption is context-recovered where confident
      (backslash-before-LaTeX-command, possessive apostrophes, inter-word dashes,
      "= �17" minus signs, and the self-described "�ã" square-root). Any row whose
      question or ground_truth still contains U+FFFD afterwards is DROPPED.
  #2  Thousands-separator commas in verifiable ground_truth/reference_answer are
      removed (e.g. "64,000" -> "64000"). A smart matcher removes ONLY grouping
      commas (digit , exactly-3-digits , boundary), so multi-value list answers
      ("1,221,000, 932,960" -> "1221000, 932960") keep their ", " separators and
      prose commas are untouched. For every row whose GT changed, " Do not output
      commas." is appended to the prompt instruction. (Verifiable only: open-ended
      is rubric-graded, not exact-match.)
  #4  Rows whose ground_truth is a placeholder ("None"/"NULL") or an empty
      \\boxed{} (answer not inside the box) are DROPPED.
  #6a Rubric criterion descriptions using the short tier form ("Essential:",
      "Important:", "Optional:", "Pitfall:") are rewritten to the canonical
      "<Tier> Criteria:" form. (The 22 positive-tier/negative-weight criteria are
      intentionally left untouched.)
  #7  The two "blood sphere radius of 200 m" microscopic-object-in-metres unit
      errors are DROPPED (their reference answers were computed from the wrong
      scale and are unsalvageable without recomputation).
"""
import json
import os
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
DR_SCI = os.path.normpath(os.path.join(HERE, "..", "data", "dr-sci"))

FILES = {
    "verifiable": "Dr_SCI_verifiable.parquet",
    "open-ended": "Dr_SCI_open-ended.parquet",
}

# Smart thousands-separator matcher: a comma that is (a) preceded by a digit and
# (b) followed by EXACTLY three digits and then a non-digit boundary. This removes
# only grouping commas ("1,221,000" -> "1221000") while preserving list separators
# ("1,221,000, 932,960" keeps the ", " between values) and prose commas.
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
EMPTY_BOXED = re.compile(r"\\boxed\{\s*\}")
# #7 unit-scale error: microscopic "blood" object measured in metres.
BLOOD_UNIT = re.compile(r"200\s*m(eters)?\b")

# #8 U+FFFD ("replacement char") recovery. Rules applied in order; any row still
# containing U+FFFD afterwards is undecipherable and gets dropped.
FFFD = "\ufffd"
POSSESSIVE = re.compile(r"([A-Za-z])\ufffd+(?=(?:s|t|re|ll|ve|d|m)\b)")   # world�s -> world's
DASH_BETWEEN = re.compile(r"(?<=[A-Za-z])\ufffd(?=[a-z])")               # Ion�dipole -> Ion-dipole
LATEX_BS = re.compile(r"\ufffd[ ]?([A-Za-z{])")                          # �frac / � left -> \frac / \left
MINUS_FFFD = re.compile(r"(?<=[\s=(+\-*/])\ufffd(?=\d)")                 # = �17 -> = -17


def fix_eq(text):
    """#1: swap MathJax {eq}...{/eq} custom delimiters for standard $...$."""
    if text is None:
        return text
    return text.replace("{eq}", "$").replace("{/eq}", "$")


def recover_fffd(text):
    if text is None or FFFD not in text:
        return text
    text = POSSESSIVE.sub(r"\1'", text)
    text = text.replace("\ufffd\u00e3", "\u221a")  # "�ã" -> √
    text = DASH_BETWEEN.sub("-", text)
    text = LATEX_BS.sub(lambda m: "\\" + m.group(1), text)
    text = MINUS_FFFD.sub("-", text)
    return text


def clean_text(text):
    return recover_fffd(fix_eq(text))
SHORT_TIER = re.compile(r"^(\s*)(Essential|Important|Optional|Pitfall):")
PLACEHOLDERS = {"None", "NULL"}
COMMA_NOTE = " Do not output commas."


def fix_prompt_comma(prompt):
    out = []
    for msg in prompt:
        c = msg["content"]
        if COMMA_NOTE.strip() not in c:
            if "$ANSWER is your answer." in c:
                c = c.replace("$ANSWER is your answer.",
                              "$ANSWER is your answer." + COMMA_NOTE, 1)
            else:
                c = c + COMMA_NOTE
        out.append({"content": c, "role": msg["role"]})
    return out


def is_blood_unit_error(question):
    return bool(question and "blood" in question.lower() and BLOOD_UNIT.search(question))


def clean_file(kind, fname, log):
    canonical = os.path.join(DR_SCI, fname)
    raw = canonical.replace(".parquet", ".raw.parquet")
    if not os.path.exists(raw):
        print(f"[{kind}] backing up raw -> {os.path.basename(raw)}")
        shutil.copy2(canonical, raw)
    src = pq.ParquetFile(raw)
    schema = src.schema_arrow

    st = {"rows_in": 0, "rows_out": 0, "dropped_placeholder": 0, "dropped_empty_boxed": 0,
          "dropped_unit_error": 0, "eq_fixed_rows": 0, "fffd_recovered_rows": 0,
          "dropped_fffd": 0, "gt_comma_fixed": 0, "prompt_comma_added": 0,
          "rubric_prefix_fixed": 0}

    tmp = canonical + ".tmp"
    writer = pq.ParquetWriter(tmp, schema)
    gidx = 0
    for g in range(src.metadata.num_row_groups):
        rows = src.read_row_group(g).to_pylist()
        kept = []
        for r in rows:
            st["rows_in"] += 1
            rm = r["reward_model"]
            ei = r["extra_info"]
            gt = rm.get("ground_truth")

            # --- #4 drop placeholders / empty boxed ---
            if gt is not None and gt.strip() in PLACEHOLDERS:
                st["dropped_placeholder"] += 1
                gidx += 1
                continue
            if gt is not None and EMPTY_BOXED.search(gt):
                st["dropped_empty_boxed"] += 1
                gidx += 1
                continue

            # --- #7 drop microscopic-blood-in-metres unit errors ---
            if is_blood_unit_error(ei.get("question")):
                st["dropped_unit_error"] += 1
                gidx += 1
                continue

            # --- #1 {eq} delimiter swap + #8 U+FFFD recovery (all text fields) ---
            q0, gt0, ra0 = ei.get("question"), gt, ei.get("reference_answer")
            had_fffd = FFFD in (q0 or "") or FFFD in (gt0 or "") or FFFD in (ra0 or "")
            if "{eq}" in (q0 or "") or "{/eq}" in (q0 or ""):
                st["eq_fixed_rows"] += 1
            q1, gt1, ra1 = clean_text(q0), clean_text(gt0), clean_text(ra0)
            if q1 != q0:
                ei["question"] = q1
            if gt1 != gt0:
                rm["ground_truth"] = gt1
                gt = gt1
            if ra1 != ra0:
                ei["reference_answer"] = ra1
            new_prompt, pchanged = [], False
            for msg in r["prompt"]:
                c1 = clean_text(msg["content"])
                if c1 != msg["content"]:
                    pchanged = True
                new_prompt.append({"content": c1, "role": msg["role"]})
            if pchanged:
                r["prompt"] = new_prompt
            if FFFD in (q1 or "") or FFFD in (gt1 or ""):
                st["dropped_fffd"] += 1   # undecipherable residue -> drop row
                gidx += 1
                continue
            if had_fffd:
                st["fffd_recovered_rows"] += 1

            # --- #2 thousands-comma removal (verifiable exact-match only; smart parser
            #        keeps list separators intact, so multi-value answers are safe) ---
            if kind == "verifiable" and gt is not None and "," in gt:
                new_gt = THOUSANDS.sub("", gt)
                if new_gt != gt:
                    rm["ground_truth"] = new_gt
                    st["gt_comma_fixed"] += 1
                    ra = ei.get("reference_answer")
                    if ra is not None and "," in ra:
                        ei["reference_answer"] = new_gt if ra == gt else THOUSANDS.sub("", ra)
                    r["prompt"] = fix_prompt_comma(r["prompt"])
                    st["prompt_comma_added"] += 1

            # --- #6a rubric short-tier -> canonical (open-ended only) ---
            rubric = rm.get("rubric")
            if rubric:
                for c in rubric:
                    d = c.get("description")
                    if d and SHORT_TIER.match(d):
                        c["description"] = SHORT_TIER.sub(r"\1\2 Criteria:", d, count=1)
                        st["rubric_prefix_fixed"] += 1

            kept.append(r)
            st["rows_out"] += 1
            gidx += 1
        if kept:
            writer.write_table(pa.Table.from_pylist(kept, schema=schema))
        print(f"[{kind}] rg{g}: in={st['rows_in']} out={st['rows_out']}")
    writer.close()
    os.replace(tmp, canonical)
    log[kind] = st
    print(f"[{kind}] DONE -> {fname}")


if __name__ == "__main__":
    log = {}
    for kind, fname in FILES.items():
        clean_file(kind, fname, log)
    print("=" * 70)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    with open(os.path.join(DR_SCI, "clean_dr_sci_changelog.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
