#!/usr/bin/env python3
"""
Doppler broadening of the 511 keV annihilation line: sample 1 vs sample 2
(e.g. W vs Pb target), HPGe spectra.

Re-uses the .Spc reader, calibration, per-spectrum gain correction and peak
fit from positron_efficiency.py (keep both files in the same folder).

For each spectrum it computes, for the 511 keV line AND the 1173/1332 keV Co-60
lines (which have no Doppler broadening, so they show the instrumental
resolution):

  * Gaussian FWHM (with fit uncertainty)
  * S parameter  = net counts in a central window / net counts in whole peak
  * W parameter  = net counts in the two wings    / net counts in whole peak
Uncertainties on S and W come from a Poisson Monte-Carlo (so background
subtraction is included). Differences (sample2 - sample1) are given with
significance.  It also estimates the Doppler-only FWHM by subtracting the
instrumental resolution in quadrature (model dependent - see caveats printed).

Outputs: doppler_profiles.png, doppler_parameters.png,
         doppler_profile.csv, doppler_summary.csv

Usage:
  python doppler_broadening.py Co60-W-163112.Spc Co60-Pb-max.Spc \
      --cal calibration.json --labels W Pb --outdir .
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from positron_efficiency import (analyse, load_cal, E_511, E_1173, E_1332)

LINES = {"511": E_511, "1173": E_1173, "1332": E_1332}


# ---------------------------------------------------------- integration ----
def edges_of(E):
    mid = 0.5 * (E[:-1] + E[1:])
    dE = E[1] - E[0]
    return np.concatenate([[E[0] - dE / 2], mid, [E[-1] + dE / 2]])


def integrate(counts, edges, lo, hi):
    """Counts between lo and hi keV, with fractional channels at the edges."""
    cum = np.concatenate([[0.0], np.cumsum(counts)])
    return np.interp(hi, edges, cum) - np.interp(lo, edges, cum)


class Background:
    """Linear background from two side bands, as a density (counts/keV)."""

    def __init__(self, counts, edges, E0, near=10.0, far=15.0):
        w = far - near
        dl = integrate(counts, edges, E0 - far, E0 - near) / w
        dr = integrate(counts, edges, E0 + near, E0 + far) / w
        mid = 0.5 * (near + far)
        self.E0 = E0
        self.m = (dr - dl) / (2 * mid)
        self.c = 0.5 * (dl + dr)

    def integral(self, lo, hi):
        a, b = lo - self.E0, hi - self.E0
        return self.c * (b - a) + 0.5 * self.m * (b * b - a * a)


def sw_params(counts, edges, E0, cfg):
    bkg = Background(counts, edges, E0)
    T, s, w1, w2 = cfg["total"], cfg["s_half"], cfg["w_in"], cfg["w_out"]

    def net(lo, hi):
        return integrate(counts, edges, lo, hi) - bkg.integral(lo, hi)

    tot = net(E0 - T, E0 + T)
    cen = net(E0 - s, E0 + s)
    wng = net(E0 - w2, E0 - w1) + net(E0 + w1, E0 + w2)
    return cen / tot, wng / tot, tot


def mc_errors(counts, edges, E0, cfg, n=400, seed=1):
    rng = np.random.default_rng(seed)
    base = np.clip(counts, 0, None)
    out = np.array([sw_params(rng.poisson(base).astype(float), edges, E0, cfg)[:2]
                    for _ in range(n)])
    return out


# ------------------------------------------------------------ profiles -----
def profile(counts, edges, E0, cfg, step=0.5):
    """Background-subtracted peak profile, normalised to unit area in +-total."""
    bkg = Background(counts, edges, E0)
    T = cfg["total"]
    be = np.arange(-T, T + 1e-9, step)
    raw = np.array([integrate(counts, edges, E0 + a, E0 + b) for a, b in zip(be[:-1], be[1:])])
    bg = np.array([bkg.integral(E0 + a, E0 + b) for a, b in zip(be[:-1], be[1:])])
    net = raw - bg
    area = net.sum()
    return 0.5 * (be[:-1] + be[1:]), net / area, np.sqrt(np.clip(raw, 1, None)) / area


# ------------------------------------------------- instrumental resolution -
def doppler_fwhm(f511, f1173, f1332):
    """Doppler-only FWHM from quadrature subtraction of the instrument FWHM
    extrapolated from the Co-60 lines. Returns {model: (fwhm, err)}."""
    F = f511["fwhm"]
    res = {}
    # A: FWHM^2 proportional to E
    bA = 0.5 * (f1173["fwhm"] ** 2 / E_1173 + f1332["fwhm"] ** 2 / E_1332)
    res["A: FWHM^2 ~ E"] = bA * E_511
    # B: FWHM^2 = a + bE through the two Co-60 points (only if a >= 0)
    b = (f1332["fwhm"] ** 2 - f1173["fwhm"] ** 2) / (E_1332 - E_1173)
    a = f1173["fwhm"] ** 2 - b * E_1173
    if a >= 0 and b > 0:
        res["B: FWHM^2 = a + bE"] = a + b * E_511
    out = {}
    for k, inst2 in res.items():
        d2 = F ** 2 - inst2
        out[k] = (np.sqrt(d2), F * f511["fwhm_err"] / np.sqrt(d2)) if d2 > 0 else (np.nan, np.nan)
    return out


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spectra", nargs=2, help="sample1 sample2 (.Spc)")
    ap.add_argument("--cal", default="calibration.json")
    ap.add_argument("--labels", nargs=2, default=["W", "Pb"])
    ap.add_argument("--no-recal", action="store_true")
    ap.add_argument("--s-half", type=float, default=1.0, help="S window half-width (keV)")
    ap.add_argument("--w-in", type=float, default=2.5, help="W wing inner edge (keV from centroid)")
    ap.add_argument("--w-out", type=float, default=5.0, help="W wing outer edge (keV from centroid)")
    ap.add_argument("--total", type=float, default=8.0, help="half-width of total-peak region (keV)")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    cfg = dict(s_half=a.s_half, w_in=a.w_in, w_out=a.w_out, total=a.total)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cal = load_cal(a.cal)
    res = [analyse(p, cal, recal=not a.no_recal) for p in a.spectra]
    labels = a.labels

    # ---- per-spectrum, per-line parameters
    table = {}
    for lab, r in zip(labels, res):
        edges = edges_of(r["E"])
        for ln in LINES:
            f = r["fits"][ln]
            S, W, tot = sw_params(r["counts"], edges, f["mu"], cfg)
            mc = mc_errors(r["counts"], edges, f["mu"], cfg)
            table[(lab, ln)] = dict(S=S, dS=mc[:, 0].std(), W=W, dW=mc[:, 1].std(),
                                    fwhm=f["fwhm"], dfwhm=f["fwhm_err"], mu=f["mu"],
                                    area=tot, mc=mc)

    print(f"\nWindows: S = +-{cfg['s_half']} keV; W = {cfg['w_in']}-{cfg['w_out']} keV each side; "
          f"total = +-{cfg['total']} keV; linear background from 10-15 keV side bands")
    print("\n%-4s %-5s %9s %10s %16s %16s" % ("", "line", "centroid", "FWHM(keV)", "S", "W"))
    for lab in labels:
        for ln in LINES:
            t = table[(lab, ln)]
            print("%-4s %-5s %9.2f %6.2f+-%.2f %9.4f+-%.4f %9.4f+-%.4f"
                  % (lab, ln, t["mu"], t["fwhm"], t["dfwhm"], t["S"], t["dS"], t["W"], t["dW"]))

    l1, l2 = labels
    print(f"\nDifference ({l2} - {l1}):")
    summary = []
    for ln in LINES:
        t1, t2 = table[(l1, ln)], table[(l2, ln)]
        row = {"line_keV": ln}
        for q, dq in (("S", "dS"), ("W", "dW"), ("fwhm", "dfwhm")):
            d = t2[q] - t1[q]
            e = np.hypot(t1[dq], t2[dq])
            row[q] = d
            row["d" + q] = e
        # paired MC for S and W difference isn't needed: samples are independent
        summary.append(row)
        tag = "511 (Doppler + instrument)" if ln == "511" else f"{ln} (instrument only)"
        print(f"  {tag:<28} dS = {row['S']:+.4f}+-{row['dS']:.4f} ({row['S'] / row['dS']:+.1f} sigma)  "
              f"dW = {row['W']:+.4f}+-{row['dW']:.4f} ({row['W'] / row['dW']:+.1f} sigma)  "
              f"dFWHM = {row['fwhm']:+.2f}+-{row['dfwhm']:.2f} keV")

    print("\nDoppler-only FWHM at 511 keV (instrument resolution subtracted in quadrature):")
    dop = {}
    for lab, r in zip(labels, res):
        f = r["fits"]
        dop[lab] = doppler_fwhm(f["511"], f["1173"], f["1332"])
        for k, (v, e) in dop[lab].items():
            print(f"  {lab:<3} model {k:<20} {v:.2f} +- {e:.2f} keV")
    for k in dop[l1]:
        if k in dop[l2]:
            v1, e1 = dop[l1][k]
            v2, e2 = dop[l2][k]
            print(f"  {l2}-{l1} model {k:<20} {v2 - v1:+.2f} +- {np.hypot(e1, e2):.2f} keV")
    print("  (uses only the statistical FWHM error; the spread between models is the "
          "dominant systematic - see notes)")

    # ---- CSVs
    with open(outdir / "doppler_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "line_keV", "centroid_keV", "FWHM_keV", "FWHM_err", "S", "S_err", "W", "W_err"])
        for lab in labels:
            for ln in LINES:
                t = table[(lab, ln)]
                w.writerow([lab, ln, f"{t['mu']:.3f}", f"{t['fwhm']:.3f}", f"{t['dfwhm']:.3f}",
                            f"{t['S']:.5f}", f"{t['dS']:.5f}", f"{t['W']:.5f}", f"{t['dW']:.5f}"])

    prof = []
    for lab, r in zip(labels, res):
        edges = edges_of(r["E"])
        prof.append(profile(r["counts"], edges, r["fits"]["511"]["mu"], cfg))
    x = prof[0][0]
    (y1, e1), (y2, e2) = (prof[0][1], prof[0][2]), (prof[1][1], prof[1][2])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y2 / y1
        eratio = ratio * np.hypot(e1 / y1, e2 / y2)
    with open(outdir / "doppler_profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dE_keV", f"{l1}_norm", f"{l1}_err", f"{l2}_norm", f"{l2}_err", f"ratio_{l2}_over_{l1}", "ratio_err"])
        for row in zip(x, y1, e1, y2, e2, ratio, eratio):
            w.writerow([f"{v:.5f}" for v in row])

    # ---- plot 1: profiles
    col = {l1: "tab:blue", l2: "tab:red"}
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for lab, (xx, yy, ee) in zip(labels, prof):
        for k in (0, 1):
            ax[k].errorbar(xx, yy, ee, fmt="o-", ms=3, lw=0.8, color=col[lab], label=lab)
    ax[0].set_ylabel("Normalised counts / 0.5 keV")
    ax[0].set_title("511 keV annihilation line, background-subtracted, unit area")
    ax[0].legend()
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Normalised counts / 0.5 keV (log)")
    m = (y1 > 2 * e1) & (y2 > 2 * e2)
    ax[2].errorbar(x[m], ratio[m], eratio[m], fmt="ko", ms=3, lw=0.8)
    ax[2].axhline(1, color="gray", ls="--")
    ax[2].set_ylabel(f"{l2} / {l1}")
    ax[2].set_xlabel("E - E$_{centroid}$ (keV)")
    for a_ in ax:
        for edge in (-cfg["s_half"], cfg["s_half"]):
            a_.axvline(edge, color="green", lw=0.6, ls=":")
        for edge in (-cfg["w_out"], -cfg["w_in"], cfg["w_in"], cfg["w_out"]):
            a_.axvline(edge, color="orange", lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(outdir / "doppler_profiles.png", dpi=150)

    # ---- plot 2: S, W, FWHM for 511 and Co-60 reference lines
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    xs = np.arange(len(LINES))
    for ax_, (q, dq, ttl) in zip(axs, (("S", "dS", "S parameter"), ("W", "dW", "W parameter"),
                                       ("fwhm", "dfwhm", "FWHM (keV)"))):
        for i, lab in enumerate(labels):
            vals = [table[(lab, ln)][q] for ln in LINES]
            errs = [table[(lab, ln)][dq] for ln in LINES]
            ax_.bar(xs + (i - 0.5) * 0.38, vals, 0.38, yerr=errs, capsize=3, color=col[lab], label=lab)
        ax_.set_xticks(xs)
        ax_.set_xticklabels([f"{ln} keV" + ("\n(Doppler)" if ln == "511" else "\n(instr. only)") for ln in LINES])
        ax_.set_title(ttl)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(outdir / "doppler_parameters.png", dpi=150)

    print(f"\nSaved to {outdir}: doppler_profiles.png, doppler_parameters.png, "
          "doppler_profile.csv, doppler_summary.csv")


if __name__ == "__main__":
    main()
