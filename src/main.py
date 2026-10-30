"""Convert an ORTEC MAESTRO .Spc file to a CSV and a Python dict, using becquerel + pandas.

    python spc_to_csv.py na.Spc            # writes na.csv
or
    from spc_to_csv import load_spc
    spec, data, df = load_spc("na.Spc")
"""
import struct
import sys
from datetime import datetime
from pathlib import Path

import becquerel as bq
import numpy as np
import pandas as pd

REC = 128  # file is made of 128-byte records


def load_spc(path):
    raw = Path(path).read_bytes()

    # header: record 0 is an int16 table, record 5 holds the calibration
    h = struct.unpack_from("<64h", raw, 0)
    first_data_rec, n_channels = h[30], h[32]
    real_time, live_time = struct.unpack_from("<2f", raw, 90)
    a0, a1, a2 = struct.unpack_from("<3f", raw, 5 * REC + 20)

    # start time from the text header (record 2), e.g. "30-Oct-261  14:34:45"
    txt = raw[2 * REC:3 * REC].split(b"\0")[0].decode("latin-1")
    start = datetime.strptime(txt[16:25] + " " + txt[28:36], "%d-%b-%y %H:%M:%S")

    # channel data: little-endian int32 counts
    counts = np.frombuffer(raw, "<i4", n_channels, (first_data_rec - 1) * REC)

    # becquerel Spectrum (calibrated in keV)
    edges = np.arange(n_channels + 1) - 0.5
    spec = bq.Spectrum(
        counts=counts,
        bin_edges_kev=a0 + a1 * edges + a2 * edges**2,
        livetime=live_time,
        realtime=real_time,
        start_time=start,
    )

    # pandas table
    df = pd.DataFrame({
        "channel": np.arange(n_channels),
        "energy_keV": spec.bin_centers_kev,
        "counts": counts,
    })

    # plain dict for analysis
    data = {
        "file": Path(path).name,
        "start_time": start,
        "live_time_s": float(live_time),
        "real_time_s": float(real_time),
        "energy_cal": (a0, a1, a2),
        "channel": df["channel"].to_numpy(),
        "energy_keV": df["energy_keV"].to_numpy(),
        "counts": counts,
    }
    return spec, data, df


if __name__ == "__main__":
    src = Path(sys.argv[1])
    spec, data, df = load_spc(src)
    out = src.with_suffix(".csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} channels, {data['counts'].sum():,} counts, "
          f"live {data['live_time_s']:.1f} s, real {data['real_time_s']:.1f} s")
