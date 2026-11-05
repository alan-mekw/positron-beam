
"""
Efficiency from the files your pipeline already writes.
 
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


# -------------------------------------------------------------- EFFICIENCY --
def efficiency_table(pts, sources_csv="sources.csv", lines_csv="lines.csv", tol_kev=1.0):
    """eps = area / (live time * A(t) * I_gamma). Matches each fitted peak to a line in lines.csv
    (same source, energy within tol_kev); unmatched peaks are skipped."""
    src = pd.read_csv(sources_csv).set_index("source")
    src["cal_date"] = pd.to_datetime(src["cal_date"])
    lines = pd.read_csv(lines_csv)
    rows = []
    for _, r in pts.iterrows():
        L = lines[lines["source"] == r["source"]]
        if L.empty:
            continue
        j = (L["energy_kev"] - r["energy_kev"]).abs().idxmin()
        if abs(L.loc[j, "energy_kev"] - r["energy_kev"]) > tol_kev:
            continue
        s = src.loc[r["source"]]
        if pd.isna(r["start_time"]):
            raise ValueError(f"{r['source']}: no start_time in its .json, cannot decay-correct")
        days = (r["start_time"] - s["cal_date"]).total_seconds() / 86400
        A = s["A0_Bq"] * np.exp(-np.log(2) * days / s["half_life_days"])
        eps = r["area"] / (r["livetime_s"] * A * L.loc[j, "I_gamma"])
        rel = np.sqrt((r["area_unc"] / r["area"]) ** 2 + s["A0_rel_unc"] ** 2
                      + (L.loc[j, "I_gamma_unc"] / L.loc[j, "I_gamma"]) ** 2)
        rows.append(dict(source=r["source"], energy_kev=L.loc[j, "energy_kev"], eps=eps, eps_err=eps * rel, A_Bq=A))
    if not rows:
        raise ValueError("no peaks matched lines.csv: check source names and energies")
    return pd.DataFrame(rows)
 
 
def efficiency_fit(tab, order=2, plot=True):
    """ln(eps) = polynomial in ln(E). Returns (params, cov, reduced chi2, eps_at(E) -> value, error)."""
    lnE, ln_eps, sig = np.log(tab["energy_kev"]), np.log(tab["eps"]), tab["eps_err"] / tab["eps"]
    f = lambda x, *p: sum(pi * x ** i for i, pi in enumerate(p))
    popt, cov = scpo.curve_fit(f, lnE, ln_eps, p0=np.polyfit(lnE, ln_eps, order)[::-1], sigma=sig, absolute_sigma=True)
    rc = red_chi2(ln_eps.to_numpy(), f(lnE.to_numpy(), *popt), sig.to_numpy(), len(popt))
 
    def eps_at(Ek):
        V = np.vander(np.log(np.atleast_1d(Ek)), len(popt), increasing=True)
        val = np.exp(V @ popt)
        return val, val * np.sqrt(np.einsum("ij,jk,ik->i", V, cov, V))
    if plot:
        E = tab["energy_kev"].to_numpy(); xs = np.linspace(E.min() * 0.9, E.max() * 1.05, 300); v, e = eps_at(xs)
        plt.figure(figsize=(5, 4)); plt.errorbar(E, tab["eps"], yerr=tab["eps_err"], fmt="o")
        plt.plot(xs, v, "r"); plt.fill_between(xs, v - e, v + e, color="r", alpha=0.2)
        plt.xscale("log"); plt.yscale("log"); plt.xlabel("Energy (keV)"); plt.ylabel("Full-energy peak efficiency"); plt.tight_layout()
    return popt, cov, rc, eps_at
