"""
Energy resolution from the files your pipeline already writes.
 
Reads:
  calibration_points.csv  (from calibrate.py, after the edit: source, centroid_ch, centroid_unc, energy_kev,
                           fwhm_ch, fwhm_ch_unc, area, area_unc)
  calibration.json        (from calibrate.py: coefficients_low_to_high)
  data/<source>.json      (from spc_to_csv.py: livetime_s, realtime_s, dead_time_pct, start_time)
  sources.csv             (YOU write: source, A0_Bq, A0_rel_unc, cal_date, half_life_days)
  lines.csv               (YOU write: source, energy_kev, I_gamma, I_gamma_unc)
 
'source' is the spectrum file name without extension (e.g. Eu-152), so it matches data/Eu-152.json.
Only peaks listed in lines.csv are used for efficiency, so you choose which lines (avoid summing-prone ones).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.optimize as scpo
import matplotlib.pyplot as plt
 
W_KEV = 0.00296            # keV per e-h pair in Ge (Knoll)
P = np.polynomial.polynomial
 
 
def linear(x, m, c):
    return m * x + c
 
 
def red_chi2(y, yfit, sig, npar):
    return np.sum(((y - yfit) / sig) ** 2) / (len(y) - npar)
 
 
# ---------------------------------------------------------------- loading --
def load_calibration(path="calibration.json"):
    return np.asarray(json.loads(Path(path).read_text())["coefficients_low_to_high"], float)
 
 
def load_points(points_csv="calibration_points.csv", data_dir="data", max_dead_pct=10.0):
    """calibration_points.csv + live time / dead time / start time from data/<source>.json.
    Peaks from runs with dead time above max_dead_pct are dropped (and reported)."""
    pts = pd.read_csv(points_csv)
    need = {"source", "centroid_ch", "energy_kev", "fwhm_ch", "fwhm_ch_unc", "area", "area_unc"}
    if need - set(pts.columns):
        raise ValueError(f"{points_csv} is missing {sorted(need - set(pts.columns))}: rerun calibrate.py after the edit")
    info = {s: json.loads((Path(data_dir) / f"{s}.json").read_text()) for s in pts["source"].unique()}
    for key in ["livetime_s", "dead_time_pct"]:
        pts[key] = pts["source"].map(lambda s: info[s][key])
    pts["start_time"] = pd.to_datetime(pts["source"].map(lambda s: info[s]["start_time"]), errors="coerce")
    bad = pts["dead_time_pct"] > max_dead_pct
    if bad.any():
        print(f"Dropping runs with dead time > {max_dead_pct}%:\n",
              pts[bad].groupby("source")["dead_time_pct"].first().round(1).to_string())
    return pts[~bad].reset_index(drop=True)
 
 
# ------------------------------------------------------------- RESOLUTION --
def resolution(pts, coeffs, plot=True):
    """FWHM^2 = a + b E.  Returns (results dict, FWHM keV, FWHM unc keV)."""
    slope = P.polyval(pts["centroid_ch"], P.polyder(coeffs))        # keV per channel at each peak
    fw = pts["fwhm_ch"].to_numpy() * slope
    fw_er = pts["fwhm_ch_unc"].to_numpy() * slope
    E = pts["energy_kev"].to_numpy()
    y, y_er = fw ** 2, 2 * fw * fw_er
    popt, cov = scpo.curve_fit(linear, E, y, p0=np.polyfit(E, y, 1), sigma=y_er, absolute_sigma=True)
    b, a = popt
    b_er, a_er = np.sqrt(np.diag(cov))
    out = dict(a=a, a_err=a_er, b=b, b_err=b_er, noise_fwhm_keV=np.sqrt(a) if a > 0 else np.nan,
               fano=b / (8 * np.log(2) * W_KEV), fano_err=b_er / (8 * np.log(2) * W_KEV),
               red_chi2=red_chi2(y, linear(E, *popt), y_er, 2))
    if plot:
        xs = np.linspace(E.min() * 0.8, E.max() * 1.05, 300)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].errorbar(E, y, yerr=y_er, fmt="o"); ax[0].plot(xs, linear(xs, *popt), "r")
        ax[0].set_xlabel("Energy (keV)"); ax[0].set_ylabel(r"FWHM$^2$ (keV$^2$)")
        ax[1].errorbar(E, 100 * fw / E, yerr=100 * fw_er / E, fmt="o")
        ax[1].plot(xs, 100 * np.sqrt(np.maximum(a + b * xs, 0)) / xs, "r")
        ax[1].set_xlabel("Energy (keV)"); ax[1].set_ylabel("FWHM / E (%)"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
        plt.tight_layout()
    return out, fw, fw_er
