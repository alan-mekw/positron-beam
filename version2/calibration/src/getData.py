"""Import every calibration source found in data/calibration-data.

A calibration source is a pair of files sharing the same name:
    <name>.Spc  -> binary spectrum (counts per channel) + approximate calibration
    <name>.csv  -> columns 'peak,energy' with the accepted peak energies (keV);
                   optional column 'energy_err' with their 1-sigma uncertainties
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
    """Return (energies, energy_errs) in keV from a csv file as numpy arrays.

    The optional column 'energy_err' holds the 1-sigma uncertainty on each accepted
    energy; if it is missing the errors are taken as 0.
    """
    df = pd.read_csv(path)
    energies = df["energy"].to_numpy(dtype=float)
    if "energy_err" in df.columns:
        errs = df["energy_err"].fillna(0.0).to_numpy(dtype=float)
    else:
        errs = np.zeros_like(energies)
    return energies, errs


def load_calibration_data(folder=CALIBRATION_DIR):
    """Load every .Spc file in `folder` together with its matching .csv.

    Returns a list of dicts: name, counts, approx_cal, energies, energy_errs.
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
        energies, energy_errs = read_peaks(csv)
        sources.append({"name": spc.stem, "counts": counts, "approx_cal": approx_cal,
                        "energies": energies, "energy_errs": energy_errs})
    if not sources:
        raise FileNotFoundError(f"No .Spc/.csv pairs found in {folder}")
    return sources
