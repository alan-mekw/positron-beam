#!/usr/bin/env python3
"""
HPGe energy calibration: channel -> energy (keV), written to calibration.json.

Expected layout:

    calibration-data/
        Ba133-Pb-1655-NoCoin.Spc
        Co60-Pb-83322-NoCoin.Spc
        ...
        IDAHO-LABS/
            Ba133-ID.csv        (columns: peak,energy,error)
            Co60-ID.csv
            ...

A spectrum is matched to its reference CSV by the source name at the start of
the file name (text before the first '-'), e.g. Co60-Pb-83322-NoCoin.Spc ->
IDAHO-LABS/Co60-ID.csv.

Method:
  1. Read every .Spc (2816 byte header, then int32 counts per channel).
  2. Seed a rough gain from the two strong Co60 lines (or --gain/--offset).
  3. For every reference energy, predict its channel and fit a Gaussian plus
     linear background in a window around it to get the centroid.
  4. Weighted polynomial fit E(ch), with iterative outlier rejection.
  5. Write calibration.json (and optionally a plot).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

HEADER_BYTES = 2816
EXPECTED_CHANNELS = 8192


def read_spc(path):
    raw = Path(path).read_bytes()
    n = (len(raw) - HEADER_BYTES) // 4
    if n < 1:
        raise ValueError(f"{path}: file too small to hold channel data")
    if n != EXPECTED_CHANNELS:
        print(f"warning: {Path(path).name} has {n} channels, expected {EXPECTED_CHANNELS}",
              file=sys.stderr)
    return np.frombuffer(raw, dtype="<i4", count=n, offset=HEADER_BYTES).astype(float)


def read_reference(path):
    """Return list of (energy, error) from a peak,energy,error csv."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    data = np.atleast_1d(data)
    names = list(data.dtype.names)
    e_col = next((n for n in names if "energ" in n.lower() or "kev" in n.lower()), None)
    if e_col is None:
        raise ValueError(f"{path}: no energy column, header is {names}")
    err_col = next((n for n in names if any(k in n.lower() for k in ("err", "unc", "sigma"))), None)
    if err_col is None:
        print(f"warning: {Path(path).name} has no error column (header {names}), using 0",
              file=sys.stderr)
        errs = np.zeros(len(data))
    else:
        errs = data[err_col]
    return [(float(e), float(err)) for e, err in zip(data[e_col], errs)]


def gauss_lin(x, amp, mu, sigma, b0, b1):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * (x - mu)


def fit_peak(counts, guess, half_window):
    """Gaussian + linear background fit around a predicted channel.
    Returns dict with mu, mu_err, fwhm, or None if the fit is not trustworthy."""
    lo = int(round(guess)) - half_window
    hi = int(round(guess)) + half_window + 1
    if lo < 0 or hi > len(counts):
        return None
    x = np.arange(lo, hi, dtype=float)
    y = counts[lo:hi]

    # start from the local maximum close to the prediction
    near = np.abs(x - guess) <= half_window / 2
    mu0 = x[near][np.argmax(y[near])]
    edge = np.concatenate([y[:3], y[-3:]])
    b0 = float(np.mean(edge))
    amp0 = float(y.max() - b0)
    if amp0 <= 0:
        return None

    p0 = [amp0, mu0, 3.0, b0, 0.0]
    bounds = ([0, guess - half_window / 2, 0.5, -np.inf, -np.inf],
              [np.inf, guess + half_window / 2, half_window / 2, np.inf, np.inf])
    sig = np.sqrt(np.maximum(y, 1.0))
    try:
        popt, pcov = curve_fit(gauss_lin, x, y, p0=p0, sigma=sig,
                               absolute_sigma=True, bounds=bounds, maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    if not np.all(np.isfinite(pcov)):
        return None

    ndf = len(x) - len(popt)
    chi2red = float(np.sum(((y - gauss_lin(x, *popt)) / sig) ** 2) / ndf)
    mu_err = float(np.sqrt(pcov[1, 1]))
    if chi2red > 1:
        mu_err *= np.sqrt(chi2red)
    return {"mu": float(popt[1]), "mu_err": mu_err,
            "fwhm": float(2.3548 * abs(popt[2])), "chi2red": chi2red}


def seed_from_co60(counts, e_low, e_high):
    """Rough linear calibration from the two strongest peaks of a Co60 spectrum."""
    idx, _ = find_peaks(counts, prominence=0.05 * counts.max())
    if len(idx) < 2:
        raise RuntimeError("could not find two peaks in the Co60 spectrum to seed the gain")
    top = sorted(idx[np.argsort(counts[idx])[-2:]])
    slope = (e_high - e_low) / (top[1] - top[0])
    offset = e_low - slope * top[0]
    return offset, slope


def poly_fit(ch, ch_err, en, en_err, order, sys_err):
    """Weighted polynomial fit E = sum c_k ch^k. Uncertainty on E includes the
    centroid error propagated through the local slope."""
    c = np.polyfit(ch, en, order)[::-1]  # unweighted start, increasing order
    for _ in range(3):
        slope = np.polyval(np.polyder(c[::-1]), ch)
        sigma = np.sqrt(en_err ** 2 + (slope * ch_err) ** 2 + sys_err ** 2)
        A = np.vander(ch, order + 1, increasing=True)
        Aw = A / sigma[:, None]
        c, *_ = np.linalg.lstsq(Aw, en / sigma, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    resid = en - A @ c
    chi2 = float(np.sum((resid / sigma) ** 2))
    return c, cov, resid, sigma, chi2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", nargs="?", default="calibration-data")
    ap.add_argument("-o", "--output", default="calibration.json")
    ap.add_argument("--ref-dir", default=None, help="folder with *-ID.csv (default: <data_dir>/IDAHO-LABS)")
    ap.add_argument("--order", type=int, default=1, help="polynomial order (1 = linear)")
    ap.add_argument("--window", type=int, default=15, help="half width of the peak fit window in channels")
    ap.add_argument("--gain", type=float, default=None, help="keV/channel seed (skips the Co60 seed)")
    ap.add_argument("--offset", type=float, default=0.0, help="keV offset seed, used with --gain")
    ap.add_argument("--sys-err", type=float, default=0.0, help="systematic floor on each point, keV")
    ap.add_argument("--clip", type=float, default=4.0, help="reject points beyond this many robust sigmas (0 = off)")
    ap.add_argument("--max-ch-err", type=float, default=None,
                    help="reject peaks whose centroid error exceeds this many channels")
    ap.add_argument("--max-fit-chi2", type=float, default=None,
                    help="reject peaks whose Gaussian fit chi2/ndf exceeds this value")
    ap.add_argument("--influence", action="store_true",
                    help="print a leave-one-out table showing how much each peak pulls the calibration")
    ap.add_argument("--plot", default=None, help="write a calibration/residual plot to this file")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ref_dir = Path(args.ref_dir) if args.ref_dir else data_dir / "IDAHO-LABS"

    spectra = {}
    for f in sorted(data_dir.glob("*.[Ss][Pp][Cc]")):
        spectra[f.name.split("-")[0]] = (f, read_spc(f))
    if not spectra:
        sys.exit(f"no .Spc files found in {data_dir}")

    refs = {}
    for f in sorted(ref_dir.glob("*-ID.csv")):
        refs[f.name.split("-")[0]] = read_reference(f)
    for src in spectra:
        if src not in refs:
            print(f"warning: no reference csv for {src}, skipping it", file=sys.stderr)
    for src in refs:
        if src not in spectra:
            print(f"note: reference csv for {src} but no spectrum, skipping it", file=sys.stderr)

    if args.gain is not None:
        offset0, slope0 = args.offset, args.gain
    elif "Co60" in spectra and "Co60" in refs:
        e = sorted(en for en, _ in refs["Co60"])
        offset0, slope0 = seed_from_co60(spectra["Co60"][1], e[0], e[-1])
    else:
        sys.exit("no Co60 spectrum to seed the gain; pass --gain (keV/channel)")
    print(f"seed calibration: E = {offset0:.3f} + {slope0:.5f} * ch")

    points = []
    for src, (path, counts) in spectra.items():
        if src not in refs:
            continue
        energies = [en for en, _ in refs[src]]
        for en, en_err in refs[src]:
            guess = (en - offset0) / slope0
            # unresolved neighbours would bias a single Gaussian fit
            others = [(e2 - offset0) / slope0 for e2 in energies if e2 != en]
            if any(abs(g - guess) < 1.5 * args.window for g in others):
                print(f"skip {src} {en:.1f} keV: another reference line within the fit window")
                continue
            res = fit_peak(counts, guess, args.window)
            if res is None:
                print(f"skip {src} {en:.1f} keV: fit failed near channel {guess:.0f}")
                continue
            points.append({"source": src, "energy_keV": en, "energy_err_keV": en_err,
                           "centroid_ch": res["mu"], "centroid_err_ch": res["mu_err"],
                           "fwhm_ch": res["fwhm"], "fit_chi2red": res["chi2red"],
                           "used": True})

    # peaks that fail the quality cuts stay in the output but are not used in the fit
    eligible = np.ones(len(points), bool)
    for i, p in enumerate(points):
        if args.max_ch_err is not None and p["centroid_err_ch"] > args.max_ch_err:
            p["reject_reason"] = f"centroid error {p['centroid_err_ch']:.3f} ch > {args.max_ch_err}"
        elif args.max_fit_chi2 is not None and p["fit_chi2red"] > args.max_fit_chi2:
            p["reject_reason"] = f"fit chi2/ndf {p['fit_chi2red']:.2f} > {args.max_fit_chi2}"
        else:
            continue
        eligible[i] = False

    if eligible.sum() < args.order + 1:
        sys.exit(f"only {int(eligible.sum())} usable peaks, need at least {args.order + 1}")

    ch = np.array([p["centroid_ch"] for p in points])
    ch_err = np.array([p["centroid_err_ch"] for p in points])
    en = np.array([p["energy_keV"] for p in points])
    en_err = np.array([p["energy_err_keV"] for p in points])

    keep = eligible.copy()
    for _ in range(10):
        c, cov, resid_k, sigma_k, chi2 = poly_fit(ch[keep], ch_err[keep], en[keep], en_err[keep],
                                                   args.order, args.sys_err)
        if args.clip <= 0 or keep.sum() <= args.order + 3:
            break
        resid_all = en - np.vander(ch, args.order + 1, increasing=True) @ c
        mad = 1.4826 * np.median(np.abs(resid_all[keep] - np.median(resid_all[keep])))
        bad = keep & (np.abs(resid_all) > args.clip * max(mad, 1e-6))
        if not bad.any():
            break
        keep[np.argmax(np.abs(resid_all) * bad)] = False

    ndf = int(keep.sum() - (args.order + 1))
    chi2red = chi2 / ndf if ndf > 0 else float("nan")
    scale = np.sqrt(chi2red) if ndf > 0 and chi2red > 1 else 1.0
    coef_err = np.sqrt(np.diag(cov)) * scale

    resid_all = en - np.vander(ch, args.order + 1, increasing=True) @ c
    for p, r, k in zip(points, resid_all, keep):
        p["residual_keV"] = float(r)
        p["used"] = bool(k)
        if not k and "reject_reason" not in p:
            p["reject_reason"] = "outlier in calibration fit"

    print(f"\n{'src':6s} {'E_ref':>9s} {'centroid':>10s} {'err':>7s} {'resid keV':>10s}  used")
    for p in points:
        print(f"{p['source']:6s} {p['energy_keV']:9.3f} {p['centroid_ch']:10.2f} "
              f"{p['centroid_err_ch']:7.3f} {p['residual_keV']:10.3f}  {'yes' if p['used'] else 'REJECTED: ' + p['reject_reason']}")
    print(f"\nE(ch) = " + " + ".join(f"({ci:.6g})*ch^{k}" for k, ci in enumerate(c)))
    print(f"chi2/ndf = {chi2:.1f}/{ndf}" + (f" = {chi2red:.2f}" if ndf > 0 else ""))
    if scale > 1:
        print(f"note: chi2/ndf > 1, coefficient errors scaled by {scale:.2f}. "
              "Check for gain drift between measurements or try --order 2.")

    # leave-one-out: refit without each used peak and see what changes
    if keep.sum() >= args.order + 2:
        used = np.flatnonzero(keep)
        xs_rng = np.linspace(ch[keep].min(), ch[keep].max(), 200)
        vand = np.vander(xs_rng, args.order + 1, increasing=True)
        base_curve = vand @ c
        for j, i in enumerate(used):
            m = keep.copy()
            m[i] = False
            c2, _, _, _, chi2_2 = poly_fit(ch[m], ch_err[m], en[m], en_err[m], args.order, args.sys_err)
            ndf2 = int(m.sum() - (args.order + 1))
            points[i]["influence"] = {
                "norm_residual": float(resid_k[j] / sigma_k[j]),
                "delta_chi2": float(chi2 - chi2_2),
                "chi2_per_ndf_without": float(chi2_2 / ndf2) if ndf2 > 0 else None,
                "delta_coefficients": [float(a - b) for a, b in zip(c, c2)],
                "max_shift_keV": float(np.max(np.abs(base_curve - vand @ c2))),
            }
        if args.influence:
            rows = sorted((p for p in points if "influence" in p),
                          key=lambda p: -p["influence"]["delta_chi2"])
            print("\nleave-one-out influence (largest chi2 contribution first)")
            print(f"{'src':6s} {'E_ref':>9s} {'resid/sig':>10s} {'d_chi2':>9s} {'chi2/ndf w/o':>13s} {'max shift keV':>14s}")
            for p in rows:
                inf = p["influence"]
                w = f"{inf['chi2_per_ndf_without']:.2f}" if inf["chi2_per_ndf_without"] is not None else "n/a"
                print(f"{p['source']:6s} {p['energy_keV']:9.3f} {inf['norm_residual']:10.2f} "
                      f"{inf['delta_chi2']:9.1f} {w:>13s} {inf['max_shift_keV']:14.4f}")
            print("d_chi2 = chi2 drop when that peak is removed (large = it is dragging the fit). "
                  "max shift = largest change in the calibration across the fitted range if it is removed.")
    elif args.influence:
        print("\nnot enough used peaks for a leave-one-out table")

    out = {
        "model": "E_keV = sum_k coefficients[k] * channel^k",
        "order": args.order,
        "coefficients": [float(x) for x in c],
        "coefficient_errors": [float(x) for x in coef_err],
        "chi2": chi2,
        "ndf": ndf,
        "chi2_per_ndf": float(chi2red) if ndf > 0 else None,
        "error_scale_applied": float(scale),
        "n_channels": int(max(len(v[1]) for v in spectra.values())),
        "peaks": points,
    }
    if args.order == 1:
        out["offset_keV"] = float(c[0])
        out["keV_per_channel"] = float(c[1])
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        for src in sorted({p["source"] for p in points}):
            m = np.array([p["source"] == src for p in points])
            a1.plot(ch[m], en[m], "o", label=src)
            a2.plot(ch[m], resid_all[m], "o")
        xs = np.linspace(0, ch.max() * 1.05, 200)
        a1.plot(xs, np.vander(xs, args.order + 1, increasing=True) @ c, "k-", lw=0.8)
        a1.set_ylabel("energy (keV)")
        a1.legend()
        a2.axhline(0, color="k", lw=0.8)
        a2.set_ylabel("residual (keV)")
        a2.set_xlabel("channel")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
