#!/usr/bin/env python3
"""Convert an HPGe .Spc spectrum file into a CSV with channel, counts, energy.

Uses becquerel to hold the spectrum + machine energy calibration, and pandas to
build/write the table.

File layout (as found in the example file, 128-byte records):
    record 0-21   header (22 records)
        record 2   : name(16s) date(12s) time(10s) live time(10s) real time(10s)
        record 5   : machine calibration, 3 float32 at byte offset 20:
                     E[keV] = c0 + c1*ch + c2*ch^2
    record 22-277 spectrum, 256 records x 32 channels, uint32 little endian
                  (= 8192 channels)

Usage:
    python spc_to_csv.py INPUT_FILE [-o OUTPUT.csv]
"""

import argparse
import struct
from datetime import datetime
from pathlib import Path

import becquerel as bq
import numpy as np
import pandas as pd

RECORD = 128
N_HEADER_RECORDS = 22
CAL_RECORD = 5
CAL_OFFSET = 20  # bytes into record 5 (after 5 unused float32 slots)


def _record(raw: bytes, i: int) -> bytes:
    return raw[i * RECORD : (i + 1) * RECORD]


def read_spc(path) -> bq.Spectrum:
    """Read the .Spc file into a calibrated becquerel Spectrum."""
    raw = Path(path).read_bytes()
    if len(raw) % RECORD or len(raw) <= N_HEADER_RECORDS * RECORD:
        raise ValueError(f"{path}: size {len(raw)} is not a valid .Spc file")

    # --- spectrum: uint32 counts after the header -------------------------
    counts = np.frombuffer(raw[N_HEADER_RECORDS * RECORD :], dtype="<u4").astype(float)

    # --- acquisition info (record 2) --------------------------------------
    rec2 = _record(raw, 2)
    date_s = rec2[16:28].decode("ascii", "ignore")  # e.g. "30-Oct-261  "
    time_s = rec2[28:38].decode("ascii", "ignore").strip()  # "14:34:45"
    livetime = float(rec2[38:48].decode("ascii", "ignore"))
    realtime = float(rec2[48:58].decode("ascii", "ignore"))
    start_time = None
    try:
        # the 10th character is a century flag (1 -> 20xx), drop it
        start_time = datetime.strptime(f"{date_s[:9]} {time_s}", "%d-%b-%y %H:%M:%S")
    except ValueError:
        pass

    # --- machine energy calibration (record 5) ----------------------------
    c0, c1, c2 = struct.unpack_from("<3f", _record(raw, CAL_RECORD), CAL_OFFSET)

    spec = bq.Spectrum(
        counts=counts,
        livetime=livetime,
        realtime=realtime,
        start_time=start_time,
    )
    spec.apply_calibration(bq.Calibration("p[0] + p[1]*x + p[2]*x**2", [c0, c1, c2]))
    return spec


def spectrum_to_dataframe(spec: bq.Spectrum) -> pd.DataFrame:
    """Channel / counts / energy (keV, bin centre) table."""
    return pd.DataFrame(
        {
            "channel": np.arange(len(spec.counts_vals)),
            "count": spec.counts_vals.astype(np.int64),
            "energy": spec.bin_centers_kev,
        }
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help=".Spc file from the HPGe MCA")
    ap.add_argument("-o", "--output", help="output CSV (default: <input>.csv)")
    args = ap.parse_args()

    spec = read_spc(args.input)
    df = spectrum_to_dataframe(spec)
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".csv")
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} channels to {out}")
    print(f"live time {spec.livetime:.0f} s, real time {spec.realtime:.0f} s, "
          f"total counts {df['count'].sum():,}")


if __name__ == "__main__":
    main()
