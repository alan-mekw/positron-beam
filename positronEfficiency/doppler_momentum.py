#!/usr/bin/env python3
"""
Doppler broadening of the 511 keV line expressed as electron MOMENTUM:
sample 1 vs sample 2 (e.g. W vs Pb), HPGe spectra.

Physics
-------
An annihilating e+e- pair with longitudinal momentum p_L (along the detector
axis) shifts each 511 keV photon by  dE = p_L c / 2,  so

        p_L / (m0 c) = 2 dE / (m0 c^2) = 2 dE / 511 keV
        1 keV of dE   = 3.914e-3 m0c  = 0.536 atomic units (m0c = 137.036 a.u.)

Re-uses the .Spc reader, calibration, per-spectrum gain correction and peak
fit from positron_efficiency.py, and the background / integration helpers from
doppler_broadening.py (keep all files in the same folder).

For each spectrum it
  * converts the background-subtracted 511 keV peak to a momentum distribution
    n(p_L), folded about the fitted centroid (n(+p) and n(-p) averaged) and
    normalised to unit area for p_L >= 0,
  * overlays the instrumental resolution (Gaussian, extrapolated from the
    Co-60 lines, which have no Doppler broadening),
  * forms the RATIO CURVE n_2(p)/n_1(p) with a chi^2 test against 1,
  * reports FWHM in momentum (total and Doppler-only after subtracting the
    instrument in quadrature), mean |p_L|, and S / W in momentum windows.

Outputs: momentum_profiles.png, momentum_profile.csv, momentum_summary.csv

Usage:
  python doppler_momentum.py Co60-W-163112.Spc Co60-Pb-max.Spc \
      --cal calibration.json --labels W Pb --outdir .
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from positron_efficiency import analyse, load_cal, E_511, E_1173, E_1332
from doppler_broadening import (edges_of, integrate, Background, sw_params,
                                mc_errors, doppler_fwhm)

KEV_TO_MC = 2.0 / E_511 * 1e3        # dE [keV] -> p_L [1e-3 m0c]
MC_TO_AU = 137.036e-3                # 1e-3 m0c -> atomic units
MC_TO_KEV = 1.0 / KEV_TO_MC


# ------------------------------------------------------ momentum profile ----
def momentum_profile(counts, edges, E0, pmax, dp):
    """Folded, background-subtracted n(|p_L|), unit area over p_L >= 0.

    p in units of 1e-3 m0c. Returns centres, density [per 1e-3 m0c], error.
    """
    bkg = Background(counts, edges, E0)
    nb = int(round(pmax / dp))
    pe = np.arange(0, nb + 1) * dp                 # |p| bin edges
    de = pe * MC_TO_KEV                            # same, in keV of dE

    def binned(sign):
        raw = np.array([integrate(counts, edges, E0 + sign * lo, E0 + sign * hi)
                        if sign > 0 else
                        integrate(counts, edges, E0 - hi, E0 - lo)
                        for lo, hi in zip(de[:-1], de[1:])])
        bg = np.array([bkg.integral(E0 + lo, E0 + hi) if sign > 0 else
                       bkg.integral(E0 - hi, E0 - lo)
                       for lo, hi in zip(de[:-1], de[1:])])
        return raw, raw - bg

    rr, nr = binned(+1)
    rl, nl = binned(-1)
    net = nr + nl                                  # folded net counts
    var = np.clip(rr, 1, None) + np.clip(rl, 1, None)
    area = net.sum()
    y = net / area / dp
    e = np.sqrt(var) / area / dp
    return 0.5 * (pe[:-1] + pe[1:]), y, e


def gauss_halfnorm(p, fwhm):
    """Unit-area-for-p>=0 Gaussian (the folded resolution function)."""
    s = fwhm / 2.3548
    return 2.0 / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (p / s) ** 2)


def mean_abs_p(p, y, e, dp):
    m = np.sum(p * y) * dp
    return m, np.sqrt(np.sum((p * e) ** 2)) * dp


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spectra", nargs=2, help="sample1 sample2 (.Spc)")
    ap.add_argument("--cal", default="calibration.json")
    ap.add_argument("--labels", nargs=2, default=["W", "Pb"])
    ap.add_argument("--no-recal", action="store_true")
    ap.add_argument("--pmax", type=float, default=30.0,
                    help="max |p_L| shown / used (1e-3 m0c); 30 ~ 7.7 keV")
    ap.add_argument("--dp", type=float, default=1.5, help="momentum bin (1e-3 m0c)")
    ap.add_argument("--s-p", type=float, default=3.9, help="S window |p_L| < this (1e-3 m0c)")
    ap.add_argument("--w-in-p", type=float, default=9.8, help="W window inner |p_L|")
    ap.add_argument("--w-out-p", type=float, default=19.6, help="W window outer |p_L|")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    l1, l2 = a.labels

    # S/W windows in keV (the integration helpers work in energy)
    cfg = dict(s_half=a.s_p * MC_TO_KEV, w_in=a.w_in_p * MC_TO_KEV,
               w_out=a.w_out_p * MC_TO_KEV, total=a.pmax * MC_TO_KEV)

    cal = load_cal(a.cal)
    res = [analyse(p, cal, recal=not a.no_recal) for p in a.spectra]

    # ---- profiles, widths, S/W
    info = {}
    for lab, r in zip(a.labels, res):
        edges = edges_of(r["E"])
        f511 = r["fits"]["511"]
        p, y, e = momentum_profile(r["counts"], edges, f511["mu"], a.pmax, a.dp)
        S, W, _ = sw_params(r["counts"], edges, f511["mu"], cfg)
        mc = mc_errors(r["counts"], edges, f511["mu"], cfg)
        dop = doppler_fwhm(f511, r["fits"]["1173"], r["fits"]["1332"])
        m, dm = mean_abs_p(p, y, e, a.dp)
        info[lab] = dict(p=p, y=y, e=e, S=S, dS=mc[:, 0].std(), W=W, dW=mc[:, 1].std(),
                         fwhm_p=f511["fwhm"] * KEV_TO_MC, dfwhm_p=f511["fwhm_err"] * KEV_TO_MC,
                         dop={k: (v * KEV_TO_MC, e_ * KEV_TO_MC) for k, (v, e_) in dop.items()},
                         meanp=m, dmeanp=dm, mu=f511["mu"], fits=r["fits"])
        # instrument FWHM at 511 (model A: FWHM^2 ~ E) in momentum units
        f73, f32 = r["fits"]["1173"]["fwhm"], r["fits"]["1332"]["fwhm"]
        bA = 0.5 * (f73 ** 2 / E_1173 + f32 ** 2 / E_1332)
        info[lab]["inst_p"] = np.sqrt(bA * E_511) * KEV_TO_MC

    # ---- print
    print(f"\nConversion: p_L = 2 dE / c  ->  1 keV = {KEV_TO_MC:.3f}e-3 m0c "
          f"= {KEV_TO_MC * MC_TO_AU:.3f} a.u.   (momentum unit: 1e-3 m0c)")
    print(f"Windows: S = |p_L| < {a.s_p}; W = {a.w_in_p}-{a.w_out_p}; "
          f"total = +-{a.pmax}  (1e-3 m0c)")
    print(f"\n{'':4s} {'511 centroid':>12s} {'FWHM_p':>14s} {'inst. FWHM_p':>13s} "
          f"{'<|p_L|>':>14s} {'S':>16s} {'W':>16s}")
    for lab in a.labels:
        d = info[lab]
        print(f"{lab:4s} {d['mu']:9.2f}keV {d['fwhm_p']:7.2f}+-{d['dfwhm_p']:.2f} "
              f"{d['inst_p']:13.2f} {d['meanp']:8.3f}+-{d['dmeanp']:.3f} "
              f"{d['S']:9.4f}+-{d['dS']:.4f} {d['W']:9.4f}+-{d['dW']:.4f}")

    print(f"\nDoppler-only FWHM in momentum (instrument subtracted in quadrature), 1e-3 m0c:")
    for lab in a.labels:
        for k, (v, e_) in info[lab]["dop"].items():
            print(f"  {lab:<3} model {k:<20} {v:.2f} +- {e_:.2f}   ({v * MC_TO_AU:.2f} a.u.)")
    d1, d2 = info[l1], info[l2]
    summ = []
    print(f"\nDifference ({l2} - {l1}):")
    for q, dq, nm in (("S", "dS", "S"), ("W", "dW", "W"), ("fwhm_p", "dfwhm_p", "FWHM_p"),
                      ("meanp", "dmeanp", "<|p_L|>")):
        v, e_ = d2[q] - d1[q], np.hypot(d1[dq], d2[dq])
        print(f"  {nm:<8} {v:+.4f} +- {e_:.4f}  ({v / e_:+.1f} sigma)")
        summ.append((nm, v, e_))
    for k in d1["dop"]:
        if k in d2["dop"]:
            v = d2["dop"][k][0] - d1["dop"][k][0]
            e_ = np.hypot(d1["dop"][k][1], d2["dop"][k][1])
            print(f"  Doppler FWHM_p [{k}] {v:+.3f} +- {e_:.3f}")
            summ.append((f"dopFWHM_p[{k}]", v, e_))

    # ---- ratio curve
    p, y1, e1, y2, e2 = d1["p"], d1["y"], d1["e"], d2["y"], d2["e"]
    ok = (y1 > 2 * e1) & (y2 > 2 * e2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y2 / y1
        eratio = ratio * np.hypot(e1 / y1, e2 / y2)
    chi2 = np.sum(((ratio[ok] - 1) / eratio[ok]) ** 2)
    ndf = ok.sum()
    print(f"\nRatio curve {l2}/{l1}: chi^2 vs flat ratio of 1 = {chi2:.1f} for {ndf} bins "
          f"(chi^2/ndf = {chi2 / max(ndf, 1):.2f})")
    lo = ok & (p < a.w_in_p)
    hi = ok & (p > a.w_in_p)
    if lo.any() and hi.any():
        print(f"  mean ratio low-p (|p|<{a.w_in_p}) = {ratio[lo].mean():.3f}; "
              f"high-p (|p|>{a.w_in_p}) = {ratio[hi].mean():.3f}   "
              "(a broader line gives ratio < 1 at low p and > 1 at high p)")

    # ---- CSV
    with open(outdir / "momentum_profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pL_1e-3_m0c", "pL_au", f"{l1}_density", f"{l1}_err",
                    f"{l2}_density", f"{l2}_err", f"ratio_{l2}_over_{l1}", "ratio_err"])
        for row in zip(p, p * MC_TO_AU, y1, e1, y2, e2, ratio, eratio):
            w.writerow([f"{v:.6g}" for v in row])
    with open(outdir / "momentum_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "FWHM_p", "FWHM_p_err", "inst_FWHM_p", "mean_abs_p",
                    "mean_abs_p_err", "S", "S_err", "W", "W_err"])
        for lab in a.labels:
            d = info[lab]
            w.writerow([lab] + [f"{v:.5g}" for v in (
                d["fwhm_p"], d["dfwhm_p"], d["inst_p"], d["meanp"], d["dmeanp"],
                d["S"], d["dS"], d["W"], d["dW"])])
        w.writerow([])
        w.writerow([f"difference {l2}-{l1}", "value", "err"])
        for nm, v, e_ in summ:
            w.writerow([nm, f"{v:.5g}", f"{e_:.5g}"])

    # ---- plot
    col = {l1: "tab:blue", l2: "tab:red"}
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    pp = np.linspace(0, a.pmax, 300)
    for lab in a.labels:
        d = info[lab]
        for k in (0, 1):
            ax[k].errorbar(d["p"], d["y"], d["e"], fmt="o", ms=3, lw=0.8,
                           color=col[lab], label=f"{lab}  (FWHM {d['fwhm_p']:.1f})")
            ax[k].plot(pp, gauss_halfnorm(pp, d["inst_p"]), "--", lw=1, color=col[lab],
                       alpha=0.6, label=f"{lab} instrument only ({d['inst_p']:.1f})")
    ax[0].set_ylabel("n($p_L$)  [per $10^{-3}$ $m_0c$]")
    ax[0].set_title("511 keV line as longitudinal momentum distribution\n"
                    "(folded, background-subtracted, unit area for $p_L\\geq0$)")
    ax[0].legend(fontsize=8)
    ax[1].set_yscale("log")
    top = max(info[l]["y"].max() for l in a.labels)
    ax[1].set_ylim(top * 3e-3, top * 2)
    ax[1].set_ylabel("n($p_L$)  (log)")
    ax[2].errorbar(p[ok], ratio[ok], eratio[ok], fmt="ko", ms=3, lw=0.8)
    ax[2].axhline(1, color="gray", ls="--")
    ax[2].set_ylabel(f"ratio curve {l2} / {l1}")
    ax[2].set_xlabel("$p_L$  ($10^{-3}$ $m_0c$)")
    sec = ax[0].secondary_xaxis("top", functions=(lambda x: x * MC_TO_AU,
                                                  lambda x: x / MC_TO_AU))
    sec.set_xlabel("$p_L$ (a.u.)")
    for a_ in ax:
        a_.axvline(a.s_p, color="green", lw=0.6, ls=":")
        for edge in (a.w_in_p, a.w_out_p):
            a_.axvline(edge, color="orange", lw=0.6, ls=":")
    ax[2].set_xlim(0, a.pmax)
    fig.tight_layout()
    fig.savefig(outdir / "momentum_profiles.png", dpi=150)
    print(f"\nSaved to {outdir}: momentum_profiles.png, momentum_profile.csv, momentum_summary.csv")


if __name__ == "__main__":
    main()
