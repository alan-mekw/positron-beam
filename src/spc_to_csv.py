#!/usr/bin/env python3
"""Convert an HPGe .Spc spectrum file into CSV files.

Writes two files:
    <input>.csv         channel, count, energy  (the spectrum)
    <input>_stats.csv   acquisition info and summary statistics (quantity, value, unit)

Optionally (--dump-header) also writes <input>_header.csv, a raw dump of every
header record (text + float32 interpretation) for finding fields not decoded here.

Uses becquerel to hold the spectrum + machine energy calibration, and pandas to
build/write the tables.

File layout (as found in the example file, 128-byte records):
    record 0-21   header (22 records)
        record 2   : name(16s) date(12s) time(10s) live time(10s) real time(10s)
        record 5   : machine calibration, 3 float32 at byte offset 20:
                     E[keV] = c0 + c1*ch + c2*ch^2
    record 22-277 spectrum, 256 records x 32 channels, uint32 little endian
                  (= 8192 channels)

Usage:
    python spc_to_csv.py INPUT_FILE [-o OUTPUT.csv] [--stats-output STATS.csv]
                         [--no-stats] [--dump-header]
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


def _read_raw(path) -> bytes:
    raw = Path(path).read_bytes()
    if len(raw) % RECORD or len(raw) <= N_HEADER_RECORDS * RECORD:
        raise ValueError(f"{path}: size {len(raw)} is not a valid .Spc file")
    return raw


def parse_header(raw: bytes) -> dict:
    """Decode the header fields we know about (record 2 and record 5)."""
    rec2 = _record(raw, 2)
    name = rec2[0:16].decode("ascii", "ignore").strip("\x00 ")
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
    c0, c1, c2 = struct.unpack_from("<3f", _record(raw, CAL_RECORD), CAL_OFFSET)
    return {
        "name": name,
        "date_raw": date_s.strip(),
        "time_raw": time_s,
        "start_time": start_time,
        "livetime": livetime,
        "realtime": realtime,
        "cal": (c0, c1, c2),
    }


def read_spc(path) -> bq.Spectrum:
    """Read the .Spc file into a calibrated becquerel Spectrum."""
    raw = _read_raw(path)
    counts = np.frombuffer(raw[N_HEADER_RECORDS * RECORD :], dtype="<u4").astype(float)
    h = parse_header(raw)

    spec = bq.Spectrum(
        counts=counts,
        livetime=h["livetime"],
        realtime=h["realtime"],
        start_time=h["start_time"],
    )
    spec.apply_calibration(bq.Calibration("p[0] + p[1]*x + p[2]*x**2", list(h["cal"])))
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


def stats_to_dataframe(path, spec: bq.Spectrum) -> pd.DataFrame:
    """Acquisition info and summary statistics as a quantity/value/unit table."""
    h = parse_header(_read_raw(path))
    counts = spec.counts_vals
    energy = spec.bin_centers_kev
    live, real = h["livetime"], h["realtime"]
    total = float(counts.sum())
    dead_s = real - live
    ipk = int(np.argmax(counts))

    rows = [
        ("file", Path(path).name, ""),
        ("spectrum_name", h["name"], ""),
        ("start_date_raw", h["date_raw"], ""),
        ("start_time_raw", h["time_raw"], ""),
        ("start_datetime", h["start_time"].isoformat() if h["start_time"] else "", ""),
        ("real_time", real, "s"),
        ("live_time", live, "s"),
        ("dead_time", dead_s, "s"),
        ("dead_time_fraction", dead_s / real * 100 if real else np.nan, "%"),
        ("n_channels", len(counts), ""),
        ("total_counts", int(total), "counts"),
        ("count_rate_live", total / live if live else np.nan, "counts/s"),
        ("count_rate_real", total / real if real else np.nan, "counts/s"),
        ("max_count", int(counts[ipk]), "counts"),
        ("max_count_channel", ipk, "channel"),
        ("max_count_energy", float(energy[ipk]), "keV"),
        ("energy_min", float(energy[0]), "keV"),
        ("energy_max", float(energy[-1]), "keV"),
        ("cal_c0", h["cal"][0], "keV"),
        ("cal_c1", h["cal"][1], "keV/ch"),
        ("cal_c2", h["cal"][2], "keV/ch^2"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "unit"])


def dump_header(path) -> pd.DataFrame:
    """Raw dump of the header: per record, printable text and the first float32/uint32 values.

    Useful for locating fields that are not decoded above.
    """
    raw = _read_raw(path)
    rows = []
    for i in range(N_HEADER_RECORDS):
        rec = _record(raw, i)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in rec)
        f32 = struct.unpack("<32f", rec)
        u32 = struct.unpack("<32I", rec)
        rows.append({
            "record": i,
            "text": text,
            "float32": " ".join(f"{v:.6g}" for v in f32),
            "uint32": " ".join(str(v) for v in u32),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help=".Spc file from the HPGe MCA")
    ap.add_argument("-o", "--output", help="output spectrum CSV (default: <input>.csv)")
    ap.add_argument("--stats-output",
                    help="output statistics CSV (default: <input>_stats.csv)")
    ap.add_argument("--no-stats", action="store_true", help="skip the statistics CSV")
    ap.add_argument("--dump-header", action="store_true",
                    help="also write <input>_header.csv with the raw header records")
    args = ap.parse_args()

    spec = read_spc(args.input)
    df = spectrum_to_dataframe(spec)
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".csv")
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} channels to {out}")
    print(f"live time {spec.livetime:.0f} s, real time {spec.realtime:.0f} s, "
          f"total counts {df['count'].sum():,}")

    if not args.no_stats:
        stats = stats_to_dataframe(args.input, spec)
        sp = (Path(args.stats_output) if args.stats_output
              else out.with_name(out.stem + "_stats.csv"))
        stats.to_csv(sp, index=False)
        print(f"Wrote statistics to {sp}")

    if args.dump_header:
        hp = out.with_name(out.stem + "_header.csv")
        dump_header(args.input).to_csv(hp, index=False)
        print(f"Wrote raw header dump to {hp}")


if __name__ == "__main__":
    main()
