#!/usr/bin/env python3
"""Per-parameter-range inference latency for open-ended (FRQ) tutor inference.

Reads the measured full200 open-ended responses (per-item Prompt/Output tokens and
batched wall-clock Latency) and reports, per parameter band:
  * measured mean prompt / output tokens,
  * measured amortized (batched) per-request latency,
  * an estimated INDIVIDUAL (single-stream) latency from a memory-bandwidth roofline.

The roofline (decode-bound) individual latency is
  t_indiv = prompt_tokens / PREFILL_TPS + output_tokens / decode_tps
  decode_tps = MEM_BW * EFF / (2 * params_bytes)   [2 bytes/param, fp16/bf16]
Defaults model an L4 (the GPU the full200 inference actually ran on). Outputs a CSV and a
grouped bar chart into the inference-cost section.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OPEN_GLOB = str(REPO / "AdaptiveTesting/Outputs/full200_results/Outputs/open/*/*.responses.jsonl")
YAML = REPO / "AdaptiveTesting/Inputs/Models/models_200.yaml"

MEM_BW = 300e9   # L4 memory bandwidth (bytes/s)
EFF = 0.60       # achievable fraction of peak bandwidth
PREFILL_TPS = 3000.0  # prefill throughput (tokens/s), prefill is compute-bound & fast

# Both the outline's coarse bands and a finer split.
BANDS = [(0, 1.0, "0-1B"), (1.0, 2.0, "1-2B"), (2.0, 3.0, "2-3B"),
         (3.0, 5.0, "3-5B"), (5.0, 7.1, "5-7B")]
OUTLINE_BANDS = [(0, 1.0, "0-1B"), (1.0, 2.0, "1-2B"), (2.0, 5.0, "2-5B"), (5.0, 7.1, "5-7B")]


def load_params():
    params = {}
    for ln in open(YAML):
        mid = re.search(r"id:\s*([^,}\s]+)", ln)
        mb = re.search(r"params_b:\s*([0-9.]+)", ln)
        if mid and mb:
            params[mid.group(1).lower()] = float(mb.group(1))
    return params


SIZE_RE = re.compile(r"(\d+(?:[._]\d+)?)\s*([bBmM])(?![a-zA-Z])")


def parse_params(mid, yaml_params):
    if mid.lower() in yaml_params:
        return yaml_params[mid.lower()]
    for num, unit in SIZE_RE.findall(mid.split("/")[-1]):
        v = float(num.replace("_", "."))
        return v / 1000 if unit in "mM" else v
    return None


def band_of(pb, bands):
    for lo, hi, n in bands:
        if pb is not None and lo <= pb < hi:
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    yaml_params = load_params()

    def collect(bands):
        agg = {n: {"pt": [], "ot": [], "lat": [], "pb": []} for _, _, n in bands}
        for fp in glob.glob(OPEN_GLOB):
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                pt = r.get("Prompt Tokens")
                ot = r.get("Output Tokens")
                lat = r.get("Latency (s)")
                m = r.get("Model")
                if not (isinstance(pt, (int, float)) and isinstance(ot, (int, float)) and m):
                    continue
                pb = parse_params(m, yaml_params)
                b = band_of(pb, bands)
                if b is None:
                    continue
                d = agg[b]
                d["pt"].append(pt)
                d["ot"].append(ot)
                d["pb"].append(pb)
                if isinstance(lat, (int, float)):
                    d["lat"].append(lat)
        return agg

    rows = []
    for bands, tag in [(BANDS, "fine"), (OUTLINE_BANDS, "outline")]:
        agg = collect(bands)
        for lo, hi, n in bands:
            d = agg[n]
            if not d["pt"]:
                continue
            mP = statistics.mean(d["pt"])
            mO = statistics.mean(d["ot"])
            mB = statistics.mean(d["pb"])
            dec = MEM_BW * EFF / (2 * mB * 1e9)
            indiv = mP / PREFILL_TPS + mO / dec
            amort = statistics.mean(d["lat"]) if d["lat"] else float("nan")
            rows.append({
                "band_set": tag, "band": n, "n_inferences": len(d["pt"]),
                "mean_params_b": round(mB, 3), "mean_prompt_tokens": round(mP, 1),
                "mean_output_tokens": round(mO, 1), "decode_tps_est": round(dec, 1),
                "individual_latency_s": round(indiv, 2),
                "amortized_batched_latency_s": round(amort, 3),
                "batch_speedup_x": round(indiv / amort, 1) if amort == amort and amort > 0 else None,
            })

    import csv
    csv_path = args.out_dir / "individual_latency_by_param_range.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")
    for r in rows:
        print(r)

    import os
    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fine = [r for r in rows if r["band_set"] == "fine"]
    labels = [r["band"] for r in fine]
    indiv = [r["individual_latency_s"] for r in fine]
    amort = [r["amortized_batched_latency_s"] for r in fine]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar([i - 0.2 for i in x], indiv, width=0.4, label="single-stream (roofline estimate)",
           color="#c44e52")
    ax.bar([i + 0.2 for i in x], amort, width=0.4, label="batched vLLM (measured)",
           color="#4c72b0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("latency per open-ended inference (s)")
    ax.set_xlabel("parameter range (B)")
    ax.set_title("Batching cuts single-stream latency 14-21x on an L4 GPU")
    for i, v in enumerate(indiv):
        ax.annotate(f"{v:.0f}s", (i - 0.2, v), ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "individual_latency_by_param_range.png", dpi=130)
    print(f"wrote {args.out_dir / 'individual_latency_by_param_range.png'}")


if __name__ == "__main__":
    main()
