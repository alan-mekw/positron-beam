#!/usr/bin/env python3
"""
Positron yield from Co-60 gammas hitting W / Pb targets (HPGe + NaI coincidence).

Reads ORTEC-style .Spc files from the HPGe, applies the HPGe energy calibration
(calibration.json: coefficients low->high, E = c0 + c1*channel), fits the
511 keV annihilation peak and the 1173/1332 keV Co-60 peaks, and reports

    * net peak areas and rates (using live time),
    * ratio of 511 keV counts to Co-60 photopeak counts (a relative
      "positron-detection efficiency" that needs no absolute calibration),
    * optionally an absolute number if you supply HPGe photopeak efficiencies.

Usage:
    python positron_efficiency.py Co60-W-163112.Spc Co60-Pb-max.Spc \
        --cal calibration.json [--no-recal] [--plot] \
        [--eff-ge511 X --eff-ge1173 Y --eff-ge1332 Z]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

E_511, E_1173, E_1332 = 510.999, 1173.228, 1332.492
HEADER_BYTES = 0xB00      # 2816-byte header, then the spectrum
N_CHANNELS = 8192


# ----------------------------------------------------------------- I/O -----
def read_spc(path):
    raw = Path(path).read_bytes()
    counts = np.frombuffer(raw[HEADER_BYTES:HEADER_BYTES + 4 * N_CHANNELS],
                           dtype="<i4").astype(float)
    # header text block: "<name> <date> <time> <live> <real>"
    text = raw[0x100:0x140].decode("ascii", "ignore").replace("\x00", " ")
    toks = text.split()
    try:
        live, real = float(toks[-2]), float(toks[-1])
    except (IndexError, ValueError):
        live = real = np.nan
    return counts, live, real


def load_cal(path):
    c = json.load(open(path))["coefficients_low_to_high"]
    return np.asarray(c, float)


# --------------------------------------------------------- peak fitting ----
def gauss_lin(x, A, mu, s, b0, b1):
    return A * np.exp(-0.5 * ((x - mu) / s) ** 2) + b0 + b1 * (x - mu)


def fit_peak(E, counts, e_nominal, search=30.0, half_width=30.0):
    """Gaussian + linear background fit. Returns dict with area (counts)."""
    sm = np.convolve(counts, np.ones(9) / 9, mode="same")
    win = (E > e_nominal - search) & (E < e_nominal + search)
    i0 = np.argmax(np.where(win, sm, -np.inf))
    c0 = E[i0]
    w = (E > c0 - half_width) & (E < c0 + half_width)
    x, y = E[w], counts[w]
    p0 = [max(y.max() - np.median(y), 1), c0, 1.2, np.median(y), 0.0]
    p, cov = curve_fit(gauss_lin, x, y, p0=p0,
                       sigma=np.sqrt(np.maximum(y, 1)), absolute_sigma=True,
                       maxfev=20000)
    A, mu, s = p[0], p[1], abs(p[2])
    de = np.mean(np.diff(E))                       # keV per channel
    area = A * s * np.sqrt(2 * np.pi) / de
    rel = np.sqrt(cov[0, 0] / A**2 + cov[2, 2] / s**2 + 2 * cov[0, 2] / (A * s))
    return dict(mu=mu, fwhm=2.3548 * s, fwhm_err=2.3548 * np.sqrt(cov[2, 2]), area=area, err=area * rel, popt=p,
                window=(c0 - half_width, c0 + half_width))


def analyse(path, cal, recal=True):
    counts, live, real = read_spc(path)
    ch = np.arange(counts.size)
    E = cal[0] + cal[1] * ch
    nominal = {"511": E_511, "1173": E_1173, "1332": E_1332}

    fits = {k: fit_peak(E, counts, v, search=15 if k == "511" else 30)
            for k, v in nominal.items()}

    shift_info = {k: fits[k]["mu"] - nominal[k] for k in nominal}
    if recal:
        # per-spectrum linear correction E_true = a + b*E_cal from the 3 lines
        meas = np.array([fits[k]["mu"] for k in nominal])
        true = np.array(list(nominal.values()))
        b, a = np.polyfit(meas, true, 1)
        E = a + b * E
        fits = {k: fit_peak(E, counts, v, search=10, half_width=25 if k == "511" else 30)
                for k, v in nominal.items()}
    return dict(name=Path(path).name, counts=counts, E=E, live=live, real=real,
                fits=fits, shift_before=shift_info)


# ----------------------------------------------------------- reporting -----
def ratio(a, da, b, db):
    r = a / b
    return r, r * np.sqrt((da / a) ** 2 + (db / b) ** 2)


def report(res, eff):
    f = res["fits"]
    print(f"\n=== {res['name']} ===")
    print(f"live time {res['live']:.0f} s, real time {res['real']:.0f} s, "
          f"dead time {100 * (1 - res['live'] / res['real']):.2f} %")
    print("calibration offset of fitted centroid before recal (keV): "
          + ", ".join(f"{k}: {v:+.2f}" for k, v in res["shift_before"].items()))
    for k in ("511", "1173", "1332"):
        p = f[k]
        print(f"  {k:>5} keV: centroid {p['mu']:8.2f}  FWHM {p['fwhm']:.2f}  "
              f"area {p['area']:8.0f} +- {p['err']:.0f}  "
              f"rate {p['area'] / res['live'] * 1e3:.3f} mHz")
    a, da = f["511"]["area"], f["511"]["err"]
    co = f["1173"]["area"] + f["1332"]["area"]
    dco = np.hypot(f["1173"]["err"], f["1332"]["err"])
    print("  Relative positron yield (511 area / Co-60 photopeak area):")
    for lab, (b, db) in {"1173": (f["1173"]["area"], f["1173"]["err"]),
                         "1332": (f["1332"]["area"], f["1332"]["err"]),
                         "1173+1332": (co, dco)}.items():
        r, dr = ratio(a, da, b, db)
        print(f"    N511 / N{lab:<9} = {r:.4f} +- {dr:.4f}")

    if all(v is not None for v in eff):
        e511, e1173, e1332 = eff
        # photons that reached the HPGe photopeak -> photons incident on HPGe
        n_inc_co = f["1173"]["area"] / e1173 + f["1332"]["area"] / e1332
        n_511 = a / e511
        r = n_511 / n_inc_co
        dr = r * np.hypot(da / a, np.hypot(f["1173"]["err"] / e1173, f["1332"]["err"] / e1332) / n_inc_co)
        print(f"  Efficiency-corrected: (511 incident on HPGe) / (Co-60 gammas incident on HPGe)"
              f" = {r:.4f} +- {dr:.4f}")
    return a, da


def plot(results, outfile="spectra_fits.png"):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(results), 1, figsize=(10, 4 * len(results)))
    axs = np.atleast_1d(axs)
    for ax, r in zip(axs, results):
        ax.semilogy(r["E"], r["counts"] + 0.5, lw=0.6, color="k")
        for k, p in r["fits"].items():
            lo, hi = p["window"]
            x = np.linspace(lo, hi, 400)
            ax.plot(x, gauss_lin(x, *p["popt"]), color="r", lw=1)
        ax.set_xlim(0, 1500); ax.set_xlabel("Energy (keV)"); ax.set_ylabel("Counts")
        ax.set_title(r["name"])
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"\nSaved {outfile}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spectra", nargs="+")
    ap.add_argument("--cal", default="calibration.json")
    ap.add_argument("--no-recal", action="store_true",
                    help="use calibration.json as-is (no per-spectrum gain correction)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-file", default="spectra_fits.png")
    ap.add_argument("--eff-ge511", type=float)
    ap.add_argument("--eff-ge1173", type=float)
    ap.add_argument("--eff-ge1332", type=float)
    a = ap.parse_args()

    cal = load_cal(a.cal)
    eff = (a.eff_ge511, a.eff_ge1173, a.eff_ge1332)
    results = [analyse(p, cal, recal=not a.no_recal) for p in a.spectra]
    areas = [report(r, eff) for r in results]
    if len(results) == 2:
        (a1, d1), (a2, d2) = areas
        r, dr = ratio(a1, d1, a2, d2)
        print(f"\n511 area, {results[0]['name']} / {results[1]['name']} = {r:.3f} +- {dr:.3f}"
              f"  (raw; not live-time or Co-60-normalised)")
    if a.plot:
        plot(results, a.plot_file)


if __name__ == "__main__":
    sys.exit(main())
