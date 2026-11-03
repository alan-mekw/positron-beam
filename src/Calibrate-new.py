#!/usr/bin/env python3
"""Interactive energy calibration from one or more source spectra.

1. Prompts for spectrum files (CSV from spc_to_csv.py, or .Spc), one at a time.
2. Fits the peaks in each (Gaussian + linear background, via becquerel).
3. Prompts for the expected energy of each fitted peak (Enter to skip it).
4. Fits energy vs centroid channel and plots the calibration curve.

Needs spc_to_csv.py and plot_spectrum.py in the same folder.

Outputs (written next to where you run the script):
    calibration_points.csv   every peak used: source, centroid, energy
    calibration.json         polynomial coefficients (energy = sum c[i] * ch**i)
    calibration.png          calibration curve + residuals
    <name>_calibrated.csv    (optional) your spectra with a recalibrated energy column
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_spectrum import fit_peak, find_peak_guesses, load

HALF_WIDTH = 60  # default fit window half-width in channels (asked at start-up)


# ---------------------------------------------------------------- prompts --
def ask(prompt, default=None):
    s = input(prompt).strip()
    return s if s else default


def ask_float(prompt, default=None):
    while True:
        s = ask(prompt)
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            print("  please enter a number (or press Enter to skip)")


def ask_files():
    files = []
    print("Enter spectrum files (CSV or .Spc), one per line. Blank line to finish.")
    while True:
        s = ask(f"  file {len(files) + 1}: ")
        if not s:
            if files:
                return files
            print("  need at least one file")
            continue
        p = Path(s.strip('"').strip("'")).expanduser()
        if p.is_file():
            files.append(p)
        else:
            print(f"  not found: {p}")


# --------------------------------------------------------------- fitting ---
def fit_source(path, half_width):
    """Load a spectrum, fit its peaks, return (DataFrame, list of fit dicts)."""
    df = load(path)
    channel = df["channel"].to_numpy(float)
    counts = df["count"].to_numpy(float)

    auto = [int(g) for g in find_peak_guesses(counts)]
    print(f"\n{path.name}: auto-detected peaks near channels {auto}")
    s = ask("  channels to fit (space separated), Enter to use these: ")
    guesses = [float(x) for x in s.replace(",", " ").split()] if s else auto

    fits = []
    for g in guesses:
        try:
            fit, (lo, hi) = fit_peak(channel, counts, g, half_width)
            if not fit.success:
                raise RuntimeError("did not converge")
            fits.append({
                "centroid": fit.param_val("gauss_mu"),
                "unc": fit.param_unc("gauss_mu"),
                "fwhm": fit.param_val("gauss_fwhm"),
                "fwhm_unc": 2 * np.sqrt(2 * np.log(2)) * fit.param_unc("gauss_sigma"),
                "area": fit.param_val("gauss_amp"),
                "area_unc": fit.param_unc("gauss_amp"),
            })
        except Exception as exc:
            print(f"  peak near {g:.0f}: fit failed ({exc}), skipped")
    return df, fits


def collect_points(files):
    """Fit each file and ask for the expected energy of each peak."""
    half_width = ask_float(
        f"\nFit window half-width in channels [Enter = {HALF_WIDTH}]: ", HALF_WIDTH)
    points, spectra = [], {}
    for path in files:
        df, fits = fit_source(path, half_width)
        spectra[path] = df
        print(f"  {'peak':>4} {'centroid (ch)':>14} {'FWHM (ch)':>10} {'area':>12}")
        for i, f in enumerate(fits, 1):
            print(f"  {i:>4} {f['centroid']:>10.2f} ± {f['unc']:<5.2f}"
                  f" {f['fwhm']:>6.1f} {f['area']:>12.0f}")
        for i, f in enumerate(fits, 1):
            e = ask_float(f"  expected energy (keV) of peak {i} "
                          f"@ ch {f['centroid']:.1f} [Enter = skip]: ")
            if e is not None:
                points.append({"source": path.stem, "centroid_ch": f["centroid"],
               "centroid_unc": f["unc"], "energy_kev": e,
               "fwhm_ch": f["fwhm"], "fwhm_ch_unc": f["fwhm_unc"],
               "area": f["area"], "area_unc": f["area_unc"]})
    return pd.DataFrame(points), spectra


# ----------------------------------------------------------- calibration ---
def fit_calibration(pts, order):
    coeffs_high_first = np.polyfit(pts["centroid_ch"], pts["energy_kev"], order)
    return coeffs_high_first[::-1]  # c0, c1, c2 ... (lowest order first)


def evaluate(coeffs, ch):
    return np.polynomial.polynomial.polyval(ch, coeffs)


def calibration_errors(pts, coeffs):
    """Uncertainty of the fitted calibration.

    Two estimates of the coefficient covariance are made:
      * stat    - propagated from the peak-centroid errors (converted to keV with
                  the local slope of the fit). Works even for an exactly
                  determined fit.
      * scatter - from the spread of the points about the fit,
                  s^2 (A^T A)^-1; only exists when there are more points than
                  coefficients.
    The one with the larger variance is used as the calibration uncertainty.

    Returns (cov, info) with cov the coefficient covariance matrix (c0, c1, ...).
    """
    P = np.polynomial.polynomial
    x = pts["centroid_ch"].to_numpy(float)
    y = pts["energy_kev"].to_numpy(float)
    n, p = len(x), len(coeffs)
    A = np.vander(x, p, increasing=True)
    bread = np.linalg.inv(A.T @ A)

    sig_e = np.abs(P.polyval(x, P.polyder(coeffs))) * pts["centroid_unc"].to_numpy(float)
    cov_stat = bread @ A.T @ np.diag(sig_e**2) @ A @ bread

    resid = y - A @ coeffs
    ndf = n - p
    cov_scatter = (resid @ resid / ndf) * bread if ndf > 0 else None

    cov, used = cov_stat, "centroid errors"
    if cov_scatter is not None and np.trace(cov_scatter) > np.trace(cov_stat):
        cov, used = cov_scatter, "scatter of points"

    info = {
        "ndf": ndf,
        "rms_residual_kev": float(np.sqrt(np.mean(resid**2))),
        "chi2": float(np.sum((resid / sig_e) ** 2)) if ndf > 0 else None,
        "uncertainty_source": used,
    }
    return cov, info


def energy_unc(cov, ch):
    """1-sigma energy uncertainty (keV) of the calibration at channel(s) ch."""
    ch = np.atleast_1d(np.asarray(ch, float))
    V = np.vander(ch, cov.shape[0], increasing=True)
    return np.sqrt(np.einsum("ij,jk,ik->i", V, cov, V))


def plot_calibration(pts, coeffs, out):
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    xs = np.linspace(0, max(8192, pts["centroid_ch"].max() * 1.05), 400)
    ax.plot(xs, evaluate(coeffs, xs), color="0.4", lw=1, label="fit")
    for src, g in pts.groupby("source"):
        ax.errorbar(g["centroid_ch"], g["energy_kev"], xerr=g["centroid_unc"],
                    fmt="o", label=src, zorder=3)
        resid = g["energy_kev"] - evaluate(coeffs, g["centroid_ch"])
        axr.errorbar(g["centroid_ch"], resid, xerr=g["centroid_unc"], fmt="o")
    axr.axhline(0, color="0.4", lw=1)
    ax.set_ylabel("Energy (keV)")
    axr.set_ylabel("Residual (keV)")
    axr.set_xlabel("Channel number")
    terms = " + ".join(f"{c:.4g}·ch^{i}" if i > 1 else (f"{c:.4g}·ch" if i else f"{c:.4g}")
                       for i, c in enumerate(coeffs))
    ax.set_title(f"E = {terms}", fontsize=10)
    ax.legend()
    for a in (ax, axr):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


# ------------------------------------------------------------------ main ---
def main():
    files = ask_files()
    pts, spectra = collect_points(files)

    n = len(pts)
    if n < 2:
        print(f"\nNeed at least 2 peaks with energies to calibrate (got {n}).")
        return

    default = 1 if n < 3 else 2
    order = ask_float(f"\n{n} points. Polynomial order 1-{n - 1} [default {min(default, n - 1)}]: ",
                      min(default, n - 1))
    order = int(min(max(order, 1), n - 1))

    coeffs = fit_calibration(pts, order)
    cov, info = calibration_errors(pts, coeffs)
    coeff_unc = np.sqrt(np.diag(cov))
    pts["fit_energy_kev"] = evaluate(coeffs, pts["centroid_ch"])
    pts["residual_kev"] = pts["energy_kev"] - pts["fit_energy_kev"]
    pts["calibration_unc_kev"] = energy_unc(cov, pts["centroid_ch"])

    print("\nCalibration points:")
    print(pts.round(3).to_string(index=False))
    print("\nenergy (keV) = c0 + c1*ch + c2*ch^2 + ...")
    for i, (c, u) in enumerate(zip(coeffs, coeff_unc)):
        print(f"  c{i} = {c:.6g} ± {u:.2g}")
    print(f"RMS residual: {info['rms_residual_kev']:.4f} keV")
    if info["ndf"] > 0:
        print(f"chi2/ndf (vs centroid errors): {info['chi2']:.2f}/{info['ndf']}"
              f" = {info['chi2'] / info['ndf']:.2f}")
    else:
        print("(exactly determined - residuals are zero by construction and the "
              "errors below\n only reflect the centroid errors; add more peaks to test the fit)")
    print(f"Calibration uncertainty taken from: {info['uncertainty_source']}")

    pts.to_csv("calibration_points.csv", index=False)
    Path("calibration.json").write_text(json.dumps({
        "order": order,
        "coefficients_low_to_high": list(coeffs),
        "coefficient_uncertainties": list(coeff_unc),
        "covariance": cov.tolist(),
        "rms_residual_kev": info["rms_residual_kev"],
        "ndf": info["ndf"],
        "chi2": info["chi2"],
        "uncertainty_source": info["uncertainty_source"],
    }, indent=2))
    fig = plot_calibration(pts, coeffs, "calibration.png")
    print("\nWrote calibration_points.csv, calibration.json, calibration.png")

    if (ask("\nWrite recalibrated copies of your spectra? [y/N]: ", "n") or "n").lower().startswith("y"):
        for path, df in spectra.items():
            out = path.with_name(path.stem + "_calibrated.csv")
            df = df.copy()
            df["energy"] = evaluate(coeffs, df["channel"])
            df.to_csv(out, index=False)
            print(f"  wrote {out}")

    plt.show()


if __name__ == "__main__":
    main()
