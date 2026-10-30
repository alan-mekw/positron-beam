#!/usr/bin/env python3
"""
Read an ORTEC MAESTRO binary .Spc file.

Usage as a library:
    from read_spc import read_spc, spc_to_dataframe
    spec = read_spc("na.Spc")
    spec["counts"], spec["energy_keV"], spec["live_time_s"], ...
    df = spc_to_dataframe(spec)          # channel / energy_keV / counts

Usage from the command line:
    python read_spc.py na.Spc            # writes na.csv next to the input
    python read_spc.py na.Spc out.csv

File layout (all little-endian, 128-byte records), worked out from an example
file and checked against it (Na-22 peaks land at 511 and 1274.5 keV):

  record 0  : int16 table. [27]=total records, [30]=first data record (1-based),
              [31]=number of data records, [32]=number of channels.
              float32 @ byte 90 = real time (s), float32 @ byte 94 = live time (s)
  record 2  : text: name(16) date(DD-Mon-YY) century-flag time(HH:MM:SS) live real
  records 3,4,6,7 : free text (record 7 is the original file path)
  record 5  : float32 x6 @ byte 20: energy cal (a0, a1, a2) then FWHM cal in
              channels (b0, b1, b2); E = a0 + a1*ch + a2*ch**2
  data      : int32 counts per channel, starting at the first data record
"""
import csv
import re
import struct
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

REC = 128
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]


def _text(rec: bytes) -> str:
    return rec.split(b"\x00")[0].decode("latin-1").strip()


def _parse_header_text(rec: bytes):
    """Return (name, start_time or None, live_int, real_int) from record 2."""
    s = rec.split(b"\x00")[0].decode("latin-1")
    name = s[:16].strip()
    m = re.search(r"(\d{1,2})-([A-Za-z]{3})-(\d{2})(\d)\s+(\d{2}):(\d{2}):(\d{2})", s)
    start = None
    if m:
        day, mon, yy, cent, hh, mm, ss = m.groups()
        year = (2000 if cent == "1" else 1900) + int(yy)
        try:
            start = datetime(year, MONTHS.index(mon.lower()) + 1, int(day),
                             int(hh), int(mm), int(ss))
        except ValueError:
            pass
    nums = re.findall(r"\d+(?:\.\d+)?", s[m.end():]) if m else []
    live_txt = float(nums[0]) if len(nums) > 0 else None
    real_txt = float(nums[1]) if len(nums) > 1 else None
    return name, start, live_txt, real_txt


def read_spc(path) -> dict:
    """Parse an ORTEC MAESTRO .Spc file into a dict ready for analysis."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8 * REC:
        raise ValueError(f"{path.name}: too short to be an ORTEC .Spc file")

    h = struct.unpack_from("<64h", raw, 0)
    total_records, first_data_rec, n_data_recs, n_channels = h[27], h[30], h[31], h[32]

    # Sanity checks so a different/variant file fails loudly rather than silently
    if total_records * REC != len(raw):
        warnings.warn(f"header says {total_records} records but file has {len(raw) / REC:g}")
    if n_channels <= 0 or n_data_recs * REC != n_channels * 4:
        raise ValueError(
            f"unexpected layout: {n_channels} channels in {n_data_recs} data records "
            "(expected 4 bytes per channel)")
    data_start = (first_data_rec - 1) * REC
    data_end = data_start + n_channels * 4
    if data_end > len(raw):
        raise ValueError("channel data runs past the end of the file")

    counts = np.frombuffer(raw, dtype="<i4", count=n_channels, offset=data_start).astype(np.int64)

    real_time = struct.unpack_from("<f", raw, 90)[0]
    live_time = struct.unpack_from("<f", raw, 94)[0]

    name, start, live_txt, real_txt = _parse_header_text(raw[2 * REC:3 * REC])
    # The text header carries whole seconds; make sure it agrees with the floats
    if live_txt is not None and abs(live_txt - live_time) > 1:
        warnings.warn(f"live time mismatch: float {live_time:.2f} vs text {live_txt}")
    if real_txt is not None and abs(real_txt - real_time) > 1:
        warnings.warn(f"real time mismatch: float {real_time:.2f} vs text {real_txt}")

    cal = struct.unpack_from("<6f", raw, 5 * REC + 20)
    energy_cal, fwhm_cal = cal[:3], cal[3:]

    channels = np.arange(n_channels)
    if energy_cal[1] != 0:
        energy = energy_cal[0] + energy_cal[1] * channels + energy_cal[2] * channels**2
        calibrated = True
    else:
        energy = channels.astype(float)
        calibrated = False

    text_fields = {i: _text(raw[i * REC:(i + 1) * REC]) for i in (3, 4, 6, 7)}
    text_fields = {k: v for k, v in text_fields.items() if v}

    return {
        "file": path.name,
        "name": name,
        "source_path": text_fields.get(7, ""),
        "start_time": start,
        "live_time_s": float(live_time),
        "real_time_s": float(real_time),
        "dead_time_pct": 100 * (1 - live_time / real_time) if real_time else float("nan"),
        "n_channels": int(n_channels),
        "total_counts": int(counts.sum()),
        "count_rate_cps": float(counts.sum() / live_time) if live_time else float("nan"),
        "calibrated": calibrated,
        "energy_cal": tuple(float(x) for x in energy_cal),      # keV: a0, a1, a2
        "fwhm_cal_channels": tuple(float(x) for x in fwhm_cal),  # inferred, see docstring
        "text_fields": text_fields,                              # {record index: text}
        "channels": channels,
        "energy_keV": energy,
        "counts": counts,
    }


def spc_to_dataframe(spec: dict):
    """Channel table as a pandas DataFrame (pandas imported only when needed)."""
    import pandas as pd
    return pd.DataFrame({
        "channel": spec["channels"],
        "energy_keV": spec["energy_keV"],
        "counts": spec["counts"],
    })


def write_csv(spec: dict, out_path) -> None:
    """CSV with '# key=value' metadata lines, then channel,energy_keV,counts.
    Read back with pd.read_csv(path, comment='#')."""
    meta = {
        "file": spec["file"],
        "start_time": spec["start_time"].isoformat() if spec["start_time"] else "",
        "live_time_s": f"{spec['live_time_s']:.3f}",
        "real_time_s": f"{spec['real_time_s']:.3f}",
        "energy_cal_a0_a1_a2": " ".join(f"{x:.9g}" for x in spec["energy_cal"]),
        "calibrated": spec["calibrated"],
    }
    with open(out_path, "w", newline="") as f:
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        w = csv.writer(f)
        w.writerow(["channel", "energy_keV", "counts"])
        for ch, e, c in zip(spec["channels"], spec["energy_keV"], spec["counts"]):
            w.writerow([int(ch), f"{e:.4f}", int(c)])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".csv")
    s = read_spc(src)
    write_csv(s, dst)
    print(f"{s['file']}: {s['n_channels']} channels, {s['total_counts']:,} counts, "
          f"live {s['live_time_s']:.1f} s, real {s['real_time_s']:.1f} s "
          f"({s['dead_time_pct']:.1f}% dead) -> {dst}")
