#!/usr/bin/env python3
"""Plot an HPGe spectrum (channel vs counts) and optionally fit peaks.

Works with either the .Spc file or the CSV written by spc_to_csv.py.
Peaks are fitted with becquerel (Gaussian + linear background) and the fits
are drawn over the spectrum.

Usage:
    python plot_spectrum.py INPUT                      # just plot
    python plot_spectrum.py INPUT --fit                # auto-find and fit peaks
    python plot_spectrum.py INPUT --peaks 2224 5772    # fit peaks near these channels
    python plot_spectrum.py INPUT --peaks 2224 5772 --fit-csv fits.csv -o spec.png
    python plot_spectrum.py INPUT -x energy            # energy (keV) on the x axis

Peak positions (--peaks) are always given in channels, whichever x axis is shown.
"""

import argparse
from pathlib import Path

import becquerel as bq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from spc_to_csv import read_spc, spectrum_to_dataframe


def load(path) -> pd.DataFrame:
    if Path(path).suffix.lower() == ".csv":
        return pd.read_csv(path)
    return spectrum_to_dataframe(read_spc(path))


def find_peak_guesses(counts, n_peaks=6, min_channel=300):
    """Rough peak positions: most prominent peaks in the smoothed spectrum."""
    smooth = np.convolve(counts, np.ones(9) / 9, mode="same")
    idx, props = find_peaks(smooth, prominence=0.02 * smooth.max(), width=3)
    keep = idx >= min_channel
    idx, prom = idx[keep], props["prominences"][keep]
    return sorted(idx[np.argsort(prom)[::-1][:n_peaks]])


def fit_peak(channel, counts, guess, half_width):
    """Gaussian + linear background fit around `guess` (channel units)."""
    unc = np.sqrt(np.maximum(counts, 1.0))
    lo, hi = guess - half_width, guess + half_width
    fit = bq.Fitter(["gauss", "line"], x=channel, y=counts, y_unc=unc, roi=(lo, hi))

    # starting guesses from the data in the window
    m = (channel >= lo) & (channel <= hi)
    x, y = channel[m], counts[m]
    bkg = np.median(np.concatenate([y[:5], y[-5:]]))
    mu0 = x[np.argmax(y)]
    sigma0 = max(half_width / 8, 2.0)
    fit.params["gauss_mu"].set(value=mu0)
    fit.params["gauss_sigma"].set(value=sigma0)
    fit.params["gauss_amp"].set(value=(y.max() - bkg) * sigma0 * np.sqrt(2 * np.pi))
    fit.params["line_m"].set(value=0)
    fit.params["line_b"].set(value=bkg)
    fit.fit()
    return fit, (lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help=".Spc file or CSV from spc_to_csv.py")
    ap.add_argument("-x", "--xaxis", choices=["channel", "energy"], default="channel",
                    help="x axis: channel number (default) or energy in keV")
    ap.add_argument("--linear", action="store_true", help="linear y axis (default: log)")
    ap.add_argument("--fit", action="store_true", help="auto-detect peaks and fit them")
    ap.add_argument("--peaks", type=float, nargs="+", metavar="CH",
                    help="approximate channel(s) of peaks to fit (implies --fit)")
    ap.add_argument("--half-width", type=float, default=60,
                    help="half-width of the fit window in channels (default 60)")
    ap.add_argument("--fit-csv", help="write fit results to this CSV")
    ap.add_argument("-o", "--output", help="save figure here instead of showing it")
    args = ap.parse_args()

    df = load(args.input)
    channel = df["channel"].to_numpy(dtype=float)
    counts = df["count"].to_numpy(dtype=float)

    use_energy = args.xaxis == "energy"
    if use_energy:
        if "energy" not in df.columns:
            raise SystemExit("No 'energy' column in this file; use --xaxis channel.")
        energy = df["energy"].to_numpy(float)
        kev_per_ch = np.gradient(energy, channel)  # for converting centroid errors

    def to_x(ch):
        """Channel -> value on the chosen x axis (fits are done in channels)."""
        return np.interp(ch, channel, energy) if use_energy else np.asarray(ch, float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(to_x(channel), counts, where="mid", lw=0.8, label="data", zorder=1)

    if args.peaks or args.fit:
        guesses = args.peaks or find_peak_guesses(counts)
        rows = []
        for g in guesses:
            try:
                fit, (lo, hi) = fit_peak(channel, counts, g, args.half_width)
            except Exception as exc:  # fit failures shouldn't kill the plot
                print(f"peak near channel {g:.0f}: fit failed ({exc})")
                continue
            if not fit.success:
                print(f"peak near channel {g:.0f}: fit did not converge")
                continue
            val = {k: fit.param_val(k) for k in fit.param_names}
            unc = {k: fit.param_unc(k) for k in fit.param_names}
            xs = np.linspace(lo, hi, 400)
            model = lambda x: fit.eval(np.atleast_1d(x), params=fit.result.params)
            ax.plot(to_x(xs), model(xs), color="tab:red", lw=1.5, zorder=3,
                    label="fit" if not rows else None)
            mu_x = float(to_x(val["gauss_mu"]))
            if use_energy:
                mu_x_unc = float(np.interp(val["gauss_mu"], channel, kev_per_ch)) * unc["gauss_mu"]
                label = f"{mu_x:.2f} ± {mu_x_unc:.2f} keV"
            else:
                label = f"{mu_x:.1f} ± {unc['gauss_mu']:.1f}"
            ax.annotate(label, (mu_x, model(val["gauss_mu"])[0]),
                        xytext=(0, 12), textcoords="offset points", ha="center",
                        fontsize=8, color="tab:red")
            rows.append({
                "centroid_ch": val["gauss_mu"], "centroid_unc": unc["gauss_mu"],
                "sigma_ch": val["gauss_sigma"], "fwhm_ch": val["gauss_fwhm"],
                "area": val["gauss_amp"], "area_unc": unc["gauss_amp"],
                "roi_lo": lo, "roi_hi": hi,
                **({"centroid_kev": mu_x, "centroid_kev_unc": mu_x_unc} if use_energy else {}),
            })
        if rows:
            res = pd.DataFrame(rows)
            print(res.round(3).to_string(index=False))
            if args.fit_csv:
                res.to_csv(args.fit_csv, index=False)
                print(f"Wrote {args.fit_csv}")

    ax.set_xlabel("Energy (keV)" if use_energy else "Channel number")
    ax.set_ylabel("Counts")
    ax.set_xlim(to_x(channel.min()), to_x(channel.max()))
    if args.linear:
        ax.set_ylim(bottom=0)
    else:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5, top=ax.get_ylim()[1] * 4)  # room for labels
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
