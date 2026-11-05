"""
Energy resolution (and optional efficiency) from your spectra + calibration.json only.

Run:   python run_analysis.py                 -> lists the spectra found and asks which to use
       python run_analysis.py Co60 Eu-152     -> skips the prompt
       python run_analysis.py --all           -> uses every spectrum found
Options: --sig N   minimum peak significance (area / area uncertainty), default 10

A "source" is any <name>.csv spectrum (columns: channel,count) with a matching <name>.json
(livetime_s, dead_time_pct, start_time) in DATA_DIR.

What it does, with no nuclear data needed:
  1. finds peaks automatically in each spectrum and fits Gaussian + linear background
  2. converts centroids/FWHM to keV with calibration.json
  3. E_resolution.resolution(): FWHM^2 = a + b E over all peaks found

Efficiency needs the source activity and gamma intensities, which cannot come from the spectrum.
It only runs if sources.csv and lines.csv exist AND contain rows for a chosen source (formats in
Efficiency.py); detected peaks are matched to your lines.csv energies. Otherwise it prints each
peak's count rate (area / live time) instead.

Outputs: calibration_points.csv, resolution.png (and efficiency_table.csv / efficiency.png if run)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as scpo
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import E_resolution as er
import Efficiency as ef

# ------------------------------------------------------------------ config --
SPECTRUM_DIR = "."                    # where <source>.csv live
DATA_DIR = "."                        # where <source>.json live (the modules default to "data")
CAL_JSON = "calibration.json"
FIT_HALF_WIDTH_CH = 40                # peak fit window half-width, channels (increase for very wide peaks)
MIN_SIG = 10.0                        # keep peaks with area/area_unc above this
SMOOTH_SIGMA_CH = 3.0                 # smoothing used only for finding candidates
DETECT_NSIGMA = 5.0                   # candidate threshold in std devs of the smoothed background
MATCH_TOL_KEV = 1.0                   # detected peak <-> lines.csv tolerance (efficiency only)
NON_SPECTRA = {"calibration_points", "efficiency_table", "lines", "sources", "_lines_used", "_sources_used"}


# ------------------------------------------------------------- peak finding --
def gauss_bg(x, area, mu, sigma, b0, b1):
    """Counts per channel: Gaussian of total `area` + linear background."""
    return area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * (x - mu)


def find_candidates(counts):
    """Channels of statistically significant peaks (smoothed excess over a running-median background)."""
    c = counts.astype(float)
    nz = np.flatnonzero(c > 0)
    if nz.size == 0:
        return []
    lo = nz[0] + 3 * FIT_HALF_WIDTH_CH // 4                       # ignore the hardware threshold edge
    sm = gaussian_filter1d(c, SMOOTH_SIGMA_CH)
    bkg = median_filter(c, size=201, mode="nearest")
    std = np.sqrt(np.maximum(bkg, 1.0) / (2 * np.sqrt(np.pi) * SMOOTH_SIGMA_CH))
    excess = sm - bkg
    idx, _ = find_peaks(excess, height=DETECT_NSIGMA * std, distance=5)
    return [int(i) for i in idx if i >= lo]


def fit_peak(ch, counts, ch0, hw=FIT_HALF_WIDTH_CH):
    """Fit one peak near channel ch0. Returns dict or None."""
    sel = (ch >= ch0 - hw) & (ch <= ch0 + hw)
    x, y = ch[sel].astype(float), counts[sel].astype(float)
    if len(x) < 12:
        return None
    sig = np.sqrt(np.maximum(y, 1.0))
    p0 = [max(y.sum() - np.median(y) * len(y), 1.0), float(ch0), 2.0, float(np.median(y)), 0.0]
    try:
        popt, cov = scpo.curve_fit(gauss_bg, x, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=20000)
    except Exception:
        return None
    if not np.all(np.isfinite(cov)):
        return None
    perr = np.sqrt(np.diag(cov))
    k = 2 * np.sqrt(2 * np.log(2))
    fwhm = k * abs(popt[2])
    ok = (abs(popt[1] - ch0) < 5 and 2 < fwhm < hw and popt[0] > 0 and perr[0] > 0
          and abs(popt[1] - ch0) < hw / 2)
    if not ok:
        return None
    chi2r = np.sum(((y - gauss_bg(x, *popt)) / sig) ** 2) / (len(x) - len(popt))
    return dict(centroid_ch=popt[1], centroid_unc=perr[1], fwhm_ch=fwhm, fwhm_ch_unc=k * perr[2],
                area=popt[0], area_unc=perr[0], fit_chi2r=chi2r)


# ----------------------------------------------------------- source choice --
def parse_args():
    args, sig = sys.argv[1:], MIN_SIG
    if "--sig" in args:
        i = args.index("--sig")
        sig = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    return args, sig


def find_sources():
    return [p.stem for p in sorted(Path(SPECTRUM_DIR).glob("*.csv"))
            if p.stem not in NON_SPECTRA and (Path(DATA_DIR) / f"{p.stem}.json").exists()]


def choose_sources(available, args):
    if "--all" in args:
        return available
    if args:
        bad = [a for a in args if a not in available]
        if bad:
            raise SystemExit(f"Unknown source(s) {bad}. Available: {available}")
        return args
    print("Spectra found (.csv + .json):")
    for i, s in enumerate(available, 1):
        print(f"  {i}. {s}")
    ans = input("Which to include? numbers or names, comma-separated (blank = all): ").strip()
    if not ans:
        return available
    chosen = []
    for tok in [t.strip() for t in ans.split(",") if t.strip()]:
        s = available[int(tok) - 1] if tok.isdigit() and 1 <= int(tok) <= len(available) else tok
        if s not in available:
            raise SystemExit(f"'{tok}' is not one of {available}")
        chosen.append(s)
    return chosen


# --------------------------------------------------------------------- main --
def main():
    args, min_sig = parse_args()
    available = find_sources()
    if not available:
        raise SystemExit(f"No <source>.csv + <source>.json pairs found in {SPECTRUM_DIR}")
    chosen = choose_sources(available, args)
    coeffs = er.load_calibration(CAL_JSON)
    poly = np.polynomial.polynomial.polyval

    # 1. detect + fit peaks ---------------------------------------------------
    rows = []
    for s in chosen:
        spec = pd.read_csv(Path(SPECTRUM_DIR) / f"{s}.csv")
        ch, counts = spec["channel"].to_numpy(), spec["count"].to_numpy()
        cands = find_candidates(counts)
        kept = 0
        for c0 in cands:
            r = fit_peak(ch, counts, c0)
            if r is None or r["area"] / r["area_unc"] < min_sig:
                continue
            rows.append(dict(source=s, energy_kev=float(poly(r["centroid_ch"], coeffs)), **r))
            kept += 1
        print(f"{s}: {len(cands)} candidate peaks, {kept} kept (significance >= {min_sig:g})")
    if not rows:
        raise SystemExit("No peaks survived the fit/significance cuts. Try --sig with a lower value.")
    df = pd.DataFrame(rows).sort_values(["source", "energy_kev"])
    # drop duplicates (two candidates fitting the same peak)
    df = df[~(df.duplicated(["source"], keep=False) &
              (df.groupby("source")["energy_kev"].diff().abs() < 0.5))].reset_index(drop=True)
    df[["source", "centroid_ch", "centroid_unc", "energy_kev", "fwhm_ch", "fwhm_ch_unc", "area", "area_unc"]
       ].to_csv("calibration_points.csv", index=False)

    pts = ef.load_points("calibration_points.csv", DATA_DIR)
    pts = pts.merge(df[["source", "centroid_ch", "fit_chi2r"]], on=["source", "centroid_ch"], how="left")

    print("\n=== Peaks used ===")
    show = pts[["source", "energy_kev", "centroid_ch", "fwhm_ch", "area", "area_unc", "fit_chi2r"]].copy()
    show["rate_cps"] = pts["area"] / pts["livetime_s"]
    print(show.round(3).to_string(index=False))

    # 2. resolution -----------------------------------------------------------
    if len(pts) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, fw, fw_er = er.resolution(pts, coeffs, plot=True)
        plt.savefig("resolution.png", dpi=150)
        print("\n=== Energy resolution (FWHM^2 = a + b E) ===")
        for s, E, f, e in zip(pts["source"], pts["energy_kev"], fw, fw_er):
            print(f"  {s:10s} {E:8.2f} keV: FWHM = {f:.3f} +/- {e:.3f} keV  ({100 * f / E:.3f} %)")
        for k, v in res.items():
            print(f"  {k}: {v}")
        if len(pts) <= 2:
            print("  (2 peaks -> 0 degrees of freedom; reduced chi2 is not meaningful)")
    else:
        print("\nNeed >= 2 peaks for a resolution fit.")

    # 3. efficiency: only with user-supplied activity + intensities -----------
    have = Path("sources.csv").exists() and Path("lines.csv").exists()
    if have:
        sr, ln = pd.read_csv("sources.csv"), pd.read_csv("lines.csv")
        ok = [s for s in chosen if s in set(sr["source"]) and s in set(ln["source"])]
    else:
        ok = []
    if ok:
        sr[sr["source"].isin(ok)].to_csv("_sources_used.csv", index=False)
        ln[ln["source"].isin(ok)].to_csv("_lines_used.csv", index=False)
        try:
            tab = ef.efficiency_table(pts[pts["source"].isin(ok)], "_sources_used.csv", "_lines_used.csv",
                                      tol_kev=MATCH_TOL_KEV)
            print("\n=== Full-energy peak efficiency (from your sources.csv / lines.csv) ===")
            print(tab.round(6).to_string(index=False))
            tab.to_csv("efficiency_table.csv", index=False)
            order = int(min(2, len(tab) - 1))
            if order >= 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ef.efficiency_fit(tab, order=order, plot=True)
                plt.savefig("efficiency.png", dpi=150)
        except ValueError as exc:
            print("\nEfficiency skipped:", exc)
    else:
        print("\nEfficiency not computed: it needs the source activity (sources.csv) and gamma intensities "
              "(lines.csv) for the chosen sources. Peak count rates are listed above.")
    plt.show()


if __name__ == "__main__":
    main()
