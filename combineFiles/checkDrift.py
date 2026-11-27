"""Check Co-60 spectra for gain drift against the calibration.

Usage: python3 checkDrift.py runs/ [--csv Co60-ID.csv] [--cal calibration.json]

The reference peaks (energies) come from the Co-60 ID csv and are converted to
channels using the calibration file. Each run's measured peak position is then
compared with the channel the calibration expects for that energy. If --csv / --cal are not given, they are
looked for in the runs folder, then the current folder, then next to this script.
"""
import argparse
import json
import csv
import struct
import sys
from pathlib import Path
import numpy as np

SEARCH = 120   # search +/- this many channels around each expected peak
HALF = 20      # half-width (channels) used to measure the peak
LIMIT = 2.0    # shift (channels) considered noticeable


def find_file(name, given, folder):
    """Use the path given on the command line, else look in a few places."""
    if given:
        p = Path(given)
        if not p.is_file():
            sys.exit(f"File not found: {p}")
        return p
    for d in (folder, Path.cwd(), Path(__file__).resolve().parent):
        if (d / name).is_file():
            return d / name
    sys.exit(f"Could not find {name}; pass it with --csv / --cal.")


def load_energies(path):
    with open(path, newline="") as fh:
        return [float(row["energy"]) for row in csv.DictReader(fh)]


def energy_to_channel(energy, coeffs, channels):
    """Invert the calibration polynomial (coefficients low->high order)."""
    poly = np.polynomial.Polynomial(coeffs)
    roots = (poly - energy).roots()
    real = roots[np.abs(roots.imag) < 1e-9].real
    real = real[(real >= 0) & (real < channels)]
    if len(real) == 0:
        sys.exit(f"Energy {energy} keV is outside the calibrated channel range.")
    return float(real[0])


def centroid(counts, guess):
    lo = max(guess - SEARCH, 0)
    smooth = np.convolve(counts[lo:guess + SEARCH + 1], np.ones(7) / 7, "same")
    top = lo + int(smooth.argmax())
    y = counts[top - HALF:top + HALF + 1].astype(float)
    x = np.arange(top - HALF, top + HALF + 1)
    base = np.linspace(y[:5].mean(), y[-5:].mean(), len(y))  # background
    net = y - base
    area = net.sum()
    if area < 100 or area < 5 * np.sqrt(base.sum() + 1):
        return None, None  # too weak to measure
    mean = (x * net).sum() / area
    error = np.sqrt(((x - mean) ** 2 * (y + base)).sum()) / area
    return mean, error


def read_spc(path):
    data = path.read_bytes()
    record, _, channels = struct.unpack_from("<3h", data, 60)
    return np.frombuffer(data, "<u4", channels, (record - 1) * 128), channels


ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--csv", help="Co-60 peak ID csv (default: Co60-ID.csv)")
ap.add_argument("--cal", help="calibration json (default: calibration.json)")
args = ap.parse_args()

folder = Path(args.folder)
files = sorted(folder.glob("run*.[sS]pc"))
if not files:
    sys.exit("No run*.Spc files found.")

csv_path = find_file("Co60-ID.csv", args.csv, folder)
cal_path = find_file("calibration.json", args.cal, folder)
energies = load_energies(csv_path)
coeffs = json.loads(cal_path.read_text())["coefficients_low_to_high"]
gain = np.polynomial.Polynomial(coeffs).deriv()  # keV per channel

_, n_channels = read_spc(files[0])
EXPECTED = [energy_to_channel(e, coeffs, n_channels) for e in energies]
PEAKS = [int(round(c)) for c in EXPECTED]  # integer channels for the peak search

print(f"Reference peaks from {csv_path.name} using {cal_path.name}:")
for e, c in zip(energies, EXPECTED):
    print(f"  {e:.3f} keV -> expected channel {c:.1f}")
print("\nEach column: measured centroid (shift from expected, channels / keV)\n")

drifted = []
for f in files:
    counts, _ = read_spc(f)
    line = f"{f.name:22}"
    for p, expected in zip(PEAKS, EXPECTED):
        mean, error = centroid(counts, p)
        if mean is None:
            line += f" | {expected:.0f}: not found"
            drifted.append(f.name)
            continue
        shift = mean - expected
        kev = shift * gain(expected)
        # must exceed the limit AND be larger than counting noise
        flag = abs(shift) > max(LIMIT, 3 * error)
        if flag:
            drifted.append(f.name)
        line += f" | {mean:8.1f} ({shift:+6.1f} ch, {kev:+5.1f} keV){' *' if flag else '  '}"
    print(line)

bad = sorted(set(drifted))
print(f"\nCheck these runs: {', '.join(bad)}" if bad else "\nNo noticeable drift found.")
