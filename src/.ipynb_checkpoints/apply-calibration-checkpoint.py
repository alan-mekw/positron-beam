#!/usr/bin/env python3
"""Apply a calibration.json (from calibrate.py) to spectrum CSV files.

Recomputes the `energy` column from the `channel` column:
    energy = c0 + c1*ch + c2*ch^2 + ...

By default each CSV is saved as a new file, <name>_calibrated.csv, and the
original is left alone. Use --in-place to overwrite the originals.

Usage:
    python apply_calibration.py a.csv b.csv
    python apply_calibration.py a.csv b.csv -c my_calibration.json
    python apply_calibration.py a.csv --in-place
    python apply_calibration.py                 # prompts for the files
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_calibration(path):
    cal = json.loads(Path(path).read_text())
    return np.asarray(cal["coefficients_low_to_high"], dtype=float)


def apply(csv_path, coeffs, in_place=False):
    df = pd.read_csv(csv_path)
    if "channel" not in df.columns:
        raise ValueError(f"{csv_path}: no 'channel' column")
    df["energy"] = np.polynomial.polynomial.polyval(df["channel"].to_numpy(float), coeffs)
    out = csv_path if in_place else csv_path.with_name(csv_path.stem + "_calibrated.csv")
    df.to_csv(out, index=False)
    return out


def prompt_files():
    print("Enter CSV files, one per line. Blank line to finish.")
    files = []
    while True:
        s = input(f"  file {len(files) + 1}: ").strip().strip('"').strip("'")
        if not s:
            if files:
                return files
            print("  need at least one file")
            continue
        files.append(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="spectrum CSV files (prompted if omitted)")
    ap.add_argument("-c", "--calibration", default="calibration.json",
                    help="calibration file from calibrate.py (default: calibration.json)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the input CSVs instead of writing *_calibrated.csv")
    args = ap.parse_args()

    if not Path(args.calibration).is_file():
        raise SystemExit(f"Calibration file not found: {args.calibration}")
    coeffs = load_calibration(args.calibration)
    print("energy (keV) = " + " + ".join(f"({c:.6g})*ch^{i}" for i, c in enumerate(coeffs)))

    for name in args.files or prompt_files():
        path = Path(name).expanduser()
        if not path.is_file():
            print(f"  not found: {path}")
            continue
        try:
            out = apply(path, coeffs, args.in_place)
            print(f"  {path.name} -> {out.name}")
        except Exception as exc:
            print(f"  {path.name}: failed ({exc})")


if __name__ == "__main__":
    main()
