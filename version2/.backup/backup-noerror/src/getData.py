"""Import every calibration source found in data/calibration-data.

A calibration source is a pair of files sharing the same name:
    <name>.Spc  -> binary spectrum (counts per channel) + approximate calibration
    <name>.csv  -> columns 'peak,energy' with the accepted peak energies (keV)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Layout of the .Spc binary file (found by inspecting the example file)
SPC_CAL_OFFSET = 0x294     # 3 float32: a0 + a1*ch + a2*ch^2  (keV)
SPC_DATA_OFFSET = 0xB00    # int32 counts, one per channel
SPC_N_CHANNELS = 8192

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration-data"


def read_spc(path):
    """Return (counts, approx_cal) from a .Spc file.

    approx_cal = (a0, a1, a2) so that energy ~ a0 + a1*channel + a2*channel**2
    """
    raw = Path(path).read_bytes()
    approx_cal = np.frombuffer(raw, dtype="<f4", count=3, offset=SPC_CAL_OFFSET)
    counts = np.frombuffer(raw, dtype="<i4", count=SPC_N_CHANNELS, offset=SPC_DATA_OFFSET)
    return counts.astype(float), approx_cal.astype(float)


def read_peaks(path):
    """Return the known peak energies (keV) from a csv file as a numpy array."""
    return pd.read_csv(path)["energy"].to_numpy(dtype=float)


def load_calibration_data(folder=CALIBRATION_DIR):
    """Load every .Spc file in `folder` together with its matching .csv.

    Returns a list of dicts: name, counts, approx_cal, energies.
    """
    folder = Path(folder)
    sources = []
    for spc in sorted(p for p in folder.iterdir() if p.suffix.lower() == ".spc"):
        csv = next((p for p in folder.iterdir()
                    if p.stem == spc.stem and p.suffix.lower() == ".csv"), None)
        if csv is None:
            print(f"Warning: no csv with peak energies for {spc.name}, skipping")
            continue
        counts, approx_cal = read_spc(spc)
        sources.append({"name": spc.stem, "counts": counts,
                        "approx_cal": approx_cal, "energies": read_peaks(csv)})
    if not sources:
        raise FileNotFoundError(f"No .Spc/.csv pairs found in {folder}")
    return sources
