"""
Gate efficiency and related figures of merit for the 511 keV line, from one UNGATED and one GATED spectrum.

Inputs (each run): <name>.csv with columns channel,count  (from spc_to_csv.py)
                   <name>.json with livetime_s, realtime_s (optional; otherwise pass live_u / live_g)

Main entry point:
    res, table = analyse_gate("data/Na22_ungated.csv", "data/Na22_gated.csv", guess_channel=2240)

What it returns (all with uncertainties):
    gate efficiency  eps = (gated net 511 rate) / (ungated net 511 rate)
    peak position / FWHM in each run and their consistency
    signal-to-background in +-1 FWHM, and the improvement from gating
    background rejection factor, and significance per sqrt(hour) (figure of merit)
Assumes the two runs are independent acquisitions (so their errors add in quadrature).
"""
import json
from math import erf, sqrt
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.optimize as scpo
import matplotlib.pyplot as plt

FWHM_K = 2 * np.sqrt(2 * np.log(2))          # 2.3548


# ----------------------------------------------------------------- loading --
def load_run(csv_path, live_time=None, use="live"):
    """Returns channel, counts, time_s. Time comes from <name>.json (use='live' or 'real') unless live_time is given."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if live_time is None:
        info = json.loads(csv_path.with_suffix(".json").read_text())
        live_time = float(info["livetime_s" if use == "live" else "realtime_s"])
    return df["channel"].to_numpy(float), df["count"].to_numpy(float), float(live_time)


# ------------------------------------------------------------------ fitting --
def fit_peak_free(channel, counts, guess, search=40):
    """Gaussian + linear background, position and width free but bounded. Chi-squared with sqrt(N) errors
    (fine for the >~100 counts per bin-region you get from Na-22; use a Poisson fit for very weak peaks)."""
    m = (channel >= guess - search) & (channel <= guess + search)
    x0, y0 = channel[m], counts[m]
    sm = np.convolve(y0, np.ones(5) / 5, mode="same")
    i0 = int(np.argmax(sm)); g0 = x0[i0]
    bg = np.median(np.concatenate([sm[:5], sm[-5:]]))
    half_max = bg + 0.5 * (sm[i0] - bg)
    lo, hi = i0, i0
    while lo > 0 and sm[lo] > half_max: lo -= 1
    while hi < len(sm) - 1 and sm[hi] > half_max: hi += 1
    fw0 = max(float(hi - lo), 4.0)
    half = max(3.0 * fw0, 15.0)
    w = (channel >= g0 - half) & (channel <= g0 + half)
    x, y = channel[w], counts[w]
    f = lambda x, A, mu, s, b0, b1: A / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / s) ** 2) + b0 + b1 * (x - mu)
    s0 = fw0 / FWHM_K
    p, cov = scpo.curve_fit(f, x, y, p0=[max(y.sum() - bg * len(y), 1.0), g0, s0, bg, 0.0],
                            bounds=([0, g0 - fw0, 0.4 * s0, -np.inf, -np.inf], [np.inf, g0 + fw0, 2.5 * s0, np.inf, np.inf]),
                            sigma=np.sqrt(np.maximum(y, 1)), absolute_sigma=True, maxfev=20000)
    e = np.sqrt(np.diag(cov))
    chi2 = float(np.sum(((y - f(x, *p)) / np.sqrt(np.maximum(y, 1))) ** 2) / (len(x) - 5))
    return dict(area=p[0], area_err=e[0], mu=p[1], mu_err=e[1], sigma=abs(p[2]), fwhm=abs(p[2]) * FWHM_K,
                fwhm_err=e[2] * FWHM_K, b0=p[3], b1=p[4], red_chi2=chi2, x=x, y=y, f=f, popt=p)


def signal_background(fit, nfwhm=1.0):
    """Signal and background counts inside +-nfwhm*FWHM of the centroid (whole channels)."""
    xs = np.arange(np.ceil(fit["mu"] - nfwhm * fit["fwhm"]), np.floor(fit["mu"] + nfwhm * fit["fwhm"]) + 1)
    B = float(np.sum(fit["b0"] + fit["b1"] * (xs - fit["mu"])))
    frac = erf(nfwhm * FWHM_K / np.sqrt(2))                      # fraction of Gaussian inside the window
    S = fit["area"] * frac
    S_err = fit["area_err"] * frac
    return S, S_err, max(B, 1e-9), sqrt(max(B, 1e-9))


# --------------------------------------------------------------- geometry ---
def solid_angle_fraction(radius, distance):
    """Fraction of 4*pi subtended by a disc of given radius, on axis, at the given distance (same units)."""
    return 0.5 * (1 - distance / np.hypot(distance, radius))


# --------------------------------------------------------------- main -------
def analyse_gate(ungated_csv, gated_csv, guess_channel, live_u=None, live_g=None, use="live",
                 search=40, nfwhm=1.0, plot=True, geometry=None):
    """geometry (optional): dict(ge_radius=2.6, ge_dist=3.5, nai_radius=1.6, nai_dist=2.5) in cm, to print a
    geometric expectation for the gate efficiency (excludes window fraction and photopeak fraction)."""
    chu, nu, Tu = load_run(ungated_csv, live_u, use)
    chg, ng, Tg = load_run(gated_csv, live_g, use)
    fu, fg = fit_peak_free(chu, nu, guess_channel, search), fit_peak_free(chg, ng, guess_channel, search)

    Ru, Ru_e = fu["area"] / Tu, fu["area_err"] / Tu
    Rg, Rg_e = fg["area"] / Tg, fg["area_err"] / Tg
    eps = Rg / Ru
    eps_e = eps * np.hypot(Rg_e / Rg, Ru_e / Ru)

    Su, Su_e, Bu, Bu_e = signal_background(fu, nfwhm); Sg, Sg_e, Bg, Bg_e = signal_background(fg, nfwhm)
    sbu, sbg = Su / Bu, Sg / Bg
    sbu_e = sbu * np.hypot(Su_e / Su, Bu_e / Bu); sbg_e = sbg * np.hypot(Sg_e / Sg, Bg_e / Bg)
    impr = sbg / sbu; impr_e = impr * np.hypot(sbu_e / sbu, sbg_e / sbg)
    rej = (Bu / Tu) / (Bg / Tg); rej_e = rej * np.hypot(Bu_e / Bu, Bg_e / Bg)
    zu, zg = fu["area"] / fu["area_err"], fg["area"] / fg["area_err"]
    fom_u, fom_g = zu / np.sqrt(Tu / 3600), zg / np.sqrt(Tg / 3600)

    dmu = fg["mu"] - fu["mu"]; dmu_e = np.hypot(fg["mu_err"], fu["mu_err"])
    wr = fg["fwhm"] / fu["fwhm"]; wr_e = wr * np.hypot(fg["fwhm_err"] / fg["fwhm"], fu["fwhm_err"] / fu["fwhm"])

    rows = [
        ("Live time ungated / gated (s)", Tu, Tg),
        ("Net 511 area (counts)", fu["area"], fg["area"]),
        ("  uncertainty", fu["area_err"], fg["area_err"]),
        ("Net 511 rate (1/s)", Ru, Rg),
        ("Centroid (ch)", fu["mu"], fg["mu"]),
        ("FWHM (ch)", fu["fwhm"], fg["fwhm"]),
        (f"Background counts in +-{nfwhm} FWHM", Bu, Bg),
        ("S/B in that window", sbu, sbg),
        ("Significance Z = area/err", zu, zg),
        ("Z per sqrt(hour)", fom_u, fom_g),
        ("Reduced chi2 of peak fit", fu["red_chi2"], fg["red_chi2"]),
    ]
    table = pd.DataFrame(rows, columns=["quantity", "ungated", "gated"])
    res = dict(gate_efficiency=eps, gate_efficiency_err=eps_e, sb_improvement=impr, sb_improvement_err=impr_e,
               background_rejection=rej, background_rejection_err=rej_e,
               centroid_shift_ch=dmu, centroid_shift_err=dmu_e, width_ratio=wr, width_ratio_err=wr_e,
               fom_ungated=fom_u, fom_gated=fom_g)

    print(table.round(3).to_string(index=False))
    print(f"\nGate efficiency   = {eps:.4f} +/- {eps_e:.4f}   (fraction of ungated 511 counts kept)")
    print(f"S/B improvement   = {impr:.2f} +/- {impr_e:.2f}   (gated S/B divided by ungated S/B)")
    print(f"Background rejection (rate under peak, ungated / gated) = {rej:.1f} +/- {rej_e:.1f}")
    print(f"Centroid shift gated - ungated = {dmu:+.2f} +/- {dmu_e:.2f} ch;  FWHM ratio = {wr:.3f} +/- {wr_e:.3f}  (both should be ~0 and ~1)")
    print(f"Z per sqrt(hour): ungated {fom_u:.1f}, gated {fom_g:.1f}  ->  gating {'helps' if fom_g > fom_u else 'costs'} on this figure of merit")
    if abs(dmu) > 3 * dmu_e or abs(wr - 1) > 3 * wr_e:
        print("WARNING: peak position or width differs between runs by more than 3 sigma (drift, or a different line is being fitted).")
    if max(fu["red_chi2"], fg["red_chi2"]) > 3:
        print("WARNING: poor peak fit (reduced chi2 > 3): check the window and the plot.")
    if abs(Tu - Tg) / Tu > 0.02:
        print("NOTE: live times differ; rates were used, so the efficiency is still valid.")
    if geometry:
        og = solid_angle_fraction(geometry["ge_radius"], geometry["ge_dist"])
        on = solid_angle_fraction(geometry["nai_radius"], geometry["nai_dist"])
        print(f"Geometry: Ge covers {100*og:.1f}% of 4pi, NaI {100*on:.1f}%; geometric ratio min(1, NaI/Ge) = {min(1, on/og):.2f}"
              " (before window and photopeak fractions, so the measured efficiency should be lower)")

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, f, t, lab in [(ax[0], fu, Tu, "ungated"), (ax[1], fg, Tg, "gated")]:
            a.step(f["x"], f["y"], where="mid", color="C0", label=lab)
            xs = np.linspace(f["x"].min(), f["x"].max(), 400)
            a.plot(xs, f["f"](xs, *f["popt"]), "r", label="fit")
            a.plot(xs, f["b0"] + f["b1"] * (xs - f["mu"]), "k--", lw=1, label="background")
            a.set_xlabel("Channel"); a.set_ylabel("Counts"); a.set_title(f"{lab}: area {f['area']:.0f} +/- {f['area_err']:.0f}"); a.legend()
        plt.tight_layout()
    return res, table
