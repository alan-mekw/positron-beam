"""Energy calibration from a folder of spectrum CSVs.

Each CSV needs the columns  channel, energy_keV, counts  (the format written by
spc_to_csv.py).  For every file the script:

  1. looks up the known gamma energies for that source (from the filename or CLI),
  2. finds each peak near its expected position (using the energy_keV column in
     the CSV as a rough first guess),
  3. fits a Gaussian + linear background to get the peak centroid in channels,
  4. fits  E = a0 + a1*ch (+ a2*ch^2)  to all (centroid, known energy) pairs,
  5. plots the calibration curve with residuals and the individual peak fits.

Usage
-----
    python energy_calibration.py samples/                  # whole folder
    python energy_calibration.py na.csv cs.csv co.csv      # specific files
    python energy_calibration.py samples/ --order 2        # quadratic fit
    python energy_calibration.py samples/ --per-file       # one fit per file too
    python energy_calibration.py mystery.csv --energies 661.657   # override

Filenames are matched to sources by prefix (case-insensitive): "na_run1.csv",
"Na22.csv" and "na.csv" all match "na".  Add your own sources in SOURCES below.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --------------------------------------------------------------------------
# Known sources: filename prefix -> gamma energies in keV.
# Edit freely.  Only list lines that show up as clean full-energy peaks.
# --------------------------------------------------------------------------
SOURCES = {
    "na":  [511.0, 1274.537],                       # Na-22
    "cs":  [661.657],                               # Cs-137
    "co":  [1173.228, 1332.492],                    # Co-60
    "am":  [59.541],                                # Am-241
    "ba":  [80.998, 302.853, 356.013, 383.849],     # Ba-133
    "co57": [122.061, 136.474],                     # Co-57
    "k40": [1460.82],                               # K-40 (background)
    "eu":  [121.78, 244.70, 344.28, 778.90, 964.08, 1112.08, 1408.01],  # Eu-152
}

SEARCH_KEV = 15.0   # how far from the expected energy to look for the peak max


# --------------------------------------------------------------------------
# Peak fitting
# --------------------------------------------------------------------------
def gauss_bg(x, area, mu, sigma, b0, b1):
    g = area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return g + b0 + b1 * (x - mu)


def fit_peak(df, e_expected, search_kev=SEARCH_KEV):
    """Fit one peak. Returns dict with centroid (channel) and fit info, or None."""
    ch = df["channel"].to_numpy()
    cts = df["counts"].to_numpy().astype(float)
    en = df["energy_keV"].to_numpy()

    # rough channel of the expected energy, via the CSV's own energy column
    ch_guess = np.interp(e_expected, en, ch)
    dch = np.median(np.diff(en))                     # keV per channel
    r = int(round(search_kev / dch))
    lo, hi = max(int(ch_guess) - r, 0), min(int(ch_guess) + r, len(ch) - 1)
    if hi - lo < 10:
        return None

    smooth = np.convolve(cts, np.ones(3) / 3, mode="same")
    ipk = lo + int(np.argmax(smooth[lo:hi + 1]))

    # estimate FWHM from the half-max crossings above local background
    bg = np.median(np.r_[cts[max(ipk - 3 * r, 0):max(ipk - r, 1)],
                         cts[ipk + r:ipk + 3 * r]]) if ipk + r < len(cts) else cts[lo:hi].min()
    half = bg + (smooth[ipk] - bg) / 2
    left = ipk
    while left > lo and smooth[left] > half:
        left -= 1
    right = ipk
    while right < hi and smooth[right] > half:
        right += 1
    fwhm_ch = max(right - left, 3)

    w = int(max(4 * fwhm_ch, 8))
    a, b = max(ipk - w, 0), min(ipk + w + 1, len(ch))
    x, y = ch[a:b].astype(float), cts[a:b]
    sig0 = fwhm_ch / 2.355
    p0 = [max(y.max() - y.min(), 1) * sig0 * np.sqrt(2 * np.pi), ipk, sig0,
          np.median(np.r_[y[:5], y[-5:]]), 0.0]
    sigma_y = np.sqrt(np.maximum(y, 1.0))            # Poisson weights
    try:
        popt, pcov = curve_fit(gauss_bg, x, y, p0=p0, sigma=sigma_y,
                               absolute_sigma=True, maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    perr = np.sqrt(np.diag(pcov))
    if not (a <= popt[1] <= b) or popt[2] <= 0 or not np.isfinite(perr[1]):
        return None
    return dict(e_known=e_expected, centroid=popt[1], centroid_err=perr[1],
                sigma_ch=abs(popt[2]), fwhm_keV=2.355 * abs(popt[2]) * dch,
                area=popt[0], popt=popt, x=x, y=y)


# --------------------------------------------------------------------------
# Calibration fit
# --------------------------------------------------------------------------
def fit_calibration(centroids, energies, errs, order):
    """Weighted polynomial fit E(ch). Returns coeffs (a0, a1, ...) and residuals."""
    centroids, energies = np.asarray(centroids), np.asarray(energies)
    order = min(order, len(centroids) - 1)
    if order < 1:
        return None
    # centroid errors are tiny, so floor them to avoid one point dominating
    w = 1.0 / np.maximum(np.asarray(errs), 0.05)
    c = np.polynomial.polynomial.polyfit(centroids, energies, order, w=w)
    resid = energies - np.polynomial.polynomial.polyval(centroids, c)
    return c, resid, order


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def match_source(path, override=None):
    if override:
        return override, "CLI"
    stem = path.stem.lower()
    # longest prefix wins, so "co57_x" matches "co57" rather than "co"
    for key in sorted(SOURCES, key=len, reverse=True):
        if stem.startswith(key):
            return SOURCES[key], key
    return None, None


def collect_files(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        files += sorted(p.glob("*.csv")) if p.is_dir() else [p]
    return files


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_calibration(points, cal_all, per_file, out):
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    names = list(dict.fromkeys(points["file"]))
    for i, name in enumerate(names):
        sub = points[points["file"] == name]
        ax.errorbar(sub["centroid"], sub["e_known"], xerr=sub["centroid_err"],
                    fmt="o", color=colors[i % 10], label=name, capsize=2)

    if cal_all is not None:
        c, resid, order = cal_all
        xs = np.linspace(0, points["centroid"].max() * 1.05, 300)
        ys = np.polynomial.polynomial.polyval(xs, c)
        terms = " + ".join(f"{v:.5g}" if k == 0 else f"{v:.5g}·ch" if k == 1
                           else f"{v:.3g}·ch^{k}" for k, v in enumerate(c))
        ax.plot(xs, ys, "k-", lw=1, label=f"E = {terms}")
        points = points.assign(resid=resid)
        for i, name in enumerate(names):
            sub = points[points["file"] == name]
            axr.plot(sub["centroid"], sub["resid"], "o", color=colors[i % 10])

    for name, (c, _, _) in per_file.items():
        xs = np.linspace(0, points["centroid"].max() * 1.05, 50)
        ax.plot(xs, np.polynomial.polynomial.polyval(xs, c), "--", lw=0.8,
                label=f"fit: {name}")

    axr.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Known energy (keV)")
    axr.set_ylabel("Residual (keV)")
    axr.set_xlabel("Peak centroid (channel)")
    ax.set_title("Energy calibration")
    ax.grid(alpha=0.3)
    axr.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_peak_fits(fits, out):
    n = len(fits)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (label, f) in zip(axes.ravel(), fits):
        ax.step(f["x"], f["y"], where="mid", color="C0", lw=1)
        xx = np.linspace(f["x"][0], f["x"][-1], 300)
        ax.plot(xx, gauss_bg(xx, *f["popt"]), "r-", lw=1)
        ax.axvline(f["centroid"], color="k", ls=":", lw=0.8)
        ax.set_title(f"{label}  ({f['e_known']:.1f} keV)", fontsize=9)
        ax.set_xlabel("channel", fontsize=8)
        ax.set_ylabel("counts", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="CSV files and/or folders of CSVs")
    ap.add_argument("--energies", type=float, nargs="+",
                    help="known energies (keV), overriding the SOURCES lookup for all files")
    ap.add_argument("--order", type=int, default=1, help="polynomial order (default 1)")
    ap.add_argument("--per-file", action="store_true", help="also fit each file separately")
    ap.add_argument("--search", type=float, default=SEARCH_KEV,
                    help=f"search window around expected energy, keV (default {SEARCH_KEV})")
    ap.add_argument("--outdir", default=".", help="where to write plots / tables")
    ap.add_argument("--no-show", action="store_true", help="don't open plot windows")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, fits = [], []
    for path in collect_files(args.inputs):
        energies, key = match_source(path, args.energies)
        if energies is None:
            print(f"skip {path.name}: no source matches (use --energies or edit SOURCES)")
            continue
        df = pd.read_csv(path)
        for e in energies:
            f = fit_peak(df, e, args.search)
            if f is None:
                print(f"  {path.name}: no usable peak near {e} keV")
                continue
            rows.append(dict(file=path.name, source=key, e_known=e,
                             centroid=f["centroid"], centroid_err=f["centroid_err"],
                             fwhm_keV=f["fwhm_keV"], area=f["area"]))
            fits.append((path.stem, f))
            print(f"  {path.name}: {e:9.3f} keV -> ch {f['centroid']:.2f} "
                  f"± {f['centroid_err']:.3f}  (FWHM {f['fwhm_keV']:.2f} keV)")

    if len(rows) < 2:
        raise SystemExit("Need at least 2 fitted peaks to build a calibration.")

    points = pd.DataFrame(rows)
    cal_all = fit_calibration(points["centroid"], points["e_known"],
                              points["centroid_err"], args.order)
    c, resid, order = cal_all
    points["residual_keV"] = resid

    print(f"\nCombined calibration (order {order}), E[keV] = sum a_k * ch^k")
    for k, v in enumerate(c):
        print(f"  a{k} = {v:.8g}")
    print(f"  RMS residual = {np.sqrt(np.mean(resid**2)):.4f} keV")
    if len(points) <= order + 1:
        print("  (as many parameters as points: the fit is exact, so residuals say nothing;"
              " add more peaks/sources)")

    per_file = {}
    if args.per_file:
        for name, sub in points.groupby("file"):
            r = fit_calibration(sub["centroid"], sub["e_known"], sub["centroid_err"], args.order)
            if r is not None:
                per_file[name] = r
                print(f"\n{name}: " + ", ".join(f"a{k}={v:.6g}" for k, v in enumerate(r[0])))

    points.to_csv(outdir / "calibration_points.csv", index=False)
    (outdir / "calibration.json").write_text(json.dumps(
        dict(order=order, coefficients=list(map(float, c)),
             rms_residual_keV=float(np.sqrt(np.mean(resid ** 2)))), indent=2))
    plot_calibration(points, cal_all, per_file, outdir / "calibration_curve.png")
    plot_peak_fits(fits, outdir / "peak_fits.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
