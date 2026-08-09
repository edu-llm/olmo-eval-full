"""Emit the fitted banks in the mirt-shaped CSV `vendor_bank.load_bank` reads.

`load_bank` expects `X, a1, d, g, u`, where `X` is an `X{k}` positional label, `a1` is
discrimination, and `d` is the mirt INTERCEPT rather than a difficulty — the reader recovers
`b = -d / a1`. Difficulty is therefore stored as `d = -a * b`.

Every position in the raw enumeration gets a row, including the ones `filter_items` dropped,
which carry `a1 = 0`. `check_alignment` aborts unless the bank's maximum index equals the
bridge's row count; it compares the max index rather than the row count specifically so that
sparse banks are allowed, but a bank whose *final* item was filtered would still fail because
its max index falls short of the enumeration length — which is the case for pedagogy and
piqa. A zero-discrimination row keeps the index space complete without inventing an item:
`load_bank` already drops `a1 <= 0` as `non_positive_discrimination`, the same path the nine
existing banks use, and the count surfaces in the manifest.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from reproduce_check import EXPECTED_RAW

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-dir", required=True, type=Path, help="dir of *_item_params.csv")
    ap.add_argument(
        "--banks-dir",
        type=Path,
        default=HERE,
        help="destination; each bank gets <banks-dir>/<bank>/irt_item_parameters_combined.csv",
    )
    args = ap.parse_args()

    failures = []
    for bank, n_raw in EXPECTED_RAW.items():
        fitted = pd.read_csv(args.fit_dir / f"{bank}_item_params.csv").set_index("position")
        dest = args.banks_dir / bank
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "irt_item_parameters_combined.csv"

        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
            w.writerow(["X", "a1", "d", "g", "u"])
            for pos in range(1, n_raw + 1):
                if pos in fitted.index:
                    r = fitted.loc[pos]
                    a, b = float(r["a"]), float(r["b"])
                    w.writerow([f"X{pos}", a, -a * b, float(r["c"]), 1])
                else:
                    w.writerow([f"X{pos}", 0, 0, 0, 1])

        back = pd.read_csv(path)
        idx = back["X"].str.lstrip("X").astype(int)
        usable = back[back["a1"] > 0].copy()
        usable["pos"] = idx[back["a1"] > 0]
        recovered = (-usable["d"] / usable["a1"]).to_numpy()
        merged = usable.assign(b_recovered=recovered).set_index("pos").join(fitted[["b"]])
        err = (merged["b_recovered"] - merged["b"]).abs().max()

        guard = "PASS" if int(idx.max()) == n_raw else "FAIL"
        if guard == "FAIL":
            failures.append(f"{bank}: max index {int(idx.max())} != bridge rows {n_raw}")
        print(
            f"  {bank:10} rows={len(back):5} max_index={int(idx.max()):5} "
            f"usable={int((back['a1'] > 0).sum()):5} zeroed={int((back['a1'] <= 0).sum()):5}  "
            f"alignment={guard}  b round-trip err={err:.1e}"
        )

    if failures:
        raise SystemExit("\nALIGNMENT WOULD FAIL:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
