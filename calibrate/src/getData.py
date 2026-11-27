"""Import every calibration source found in data/calibration-data.

A calibration source is a spectrum plus the peak list for its isotope:
    <name>.Spc  -> binary spectrum (counts per channel) + approximate calibration
    <isotope>-ID.csv (in IDAHO-LABS) -> columns 'peak,energy' with the accepted peak
                   energies (keV), plus a column with their 1-sigma uncertainties (keV)
"""
import re
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

# accepted headers for the uncertainty column of the peak csv files
ERROR_COLUMNS = ("energy_err", "error", "err", "unc", "uncertainty",
                 "energy_error", "energy_unc", "energy_uncertainty")


def _normalise_header(header):
    """'Energy Error (keV)' -> 'energy_error'."""
    header = re.sub(r"\(\s*kev\s*\)", "", str(header).lower()).strip()
    return re.sub(r"[\s\-]+", "_", header)


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

    The file needs an 'energy' column and a column holding the 1-sigma uncertainty
    on each energy (any name in ERROR_COLUMNS). Headers are matched ignoring case,
    surrounding spaces and a '(keV)' suffix. A missing or invalid uncertainty raises
    an error rather than being silently treated as 0.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [_normalise_header(c) for c in df.columns]
    name = Path(path).name

    err_cols = [c for c in ERROR_COLUMNS if c in df.columns]
    if "energy" not in df.columns or len(err_cols) != 1:
        raise ValueError(
            f"{name}: need an 'energy' column and exactly one uncertainty column "
            f"(one of {', '.join(ERROR_COLUMNS)}); found columns {list(df.columns)}")

    energies = df["energy"].to_numpy(dtype=float)
    errs = pd.to_numeric(df[err_cols[0]], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(errs) | (errs < 0)
    if bad.any():
        raise ValueError(f"{name}: missing or invalid uncertainty for energies "
                         f"{energies[bad].tolist()}")
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
