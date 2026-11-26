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

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration-data"
PEAKS_DIR = CALIBRATION_DIR / "IDAHO-LABS"

# header calibration is identical in every file and ~100 channels off at 1 MeV; use one fitted to the Ba133 peaks (a0, a1, a2)
APPROX_CAL = np.array([30.74, 0.21195, 0.0])

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
    # accept either 'energy_err' or 'error' as the uncertainty column
    err_col = next((c for c in ("energy_err", "error") if c in df.columns), None)
    if err_col is not None:
        errs = df[err_col].fillna(0.0).to_numpy(dtype=float)
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
        isotope = spc.stem.split("-")[0]          # e.g. "Ba133-Pb-1655-NoCoin" -> "Ba133"
        csv = PEAKS_DIR / f"{isotope}-ID.csv"
        if not csv.exists():
            print(f"Warning: no csv with peak energies for {spc.name}, skipping")
            continue
        counts, approx_cal = read_spc(spc)        
        approx_cal = APPROX_CAL
        energies, energy_errs = read_peaks(csv)
        sources.append({"name": isotope, "counts": counts, "approx_cal": approx_cal,
                        "energies": energies, "energy_errs": energy_errs})
    if not sources:
        raise FileNotFoundError(f"No .Spc/.csv pairs found in {folder}")
    return sources
