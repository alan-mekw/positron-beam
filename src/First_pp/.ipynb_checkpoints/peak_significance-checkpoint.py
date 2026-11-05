"""
Significance of a small peak at a KNOWN energy (e.g. 511 keV), with and without a target.
Inputs per run: data/<name>.csv (channel, count, energy) and data/<name>.json (livetime_s) from spc_to_csv.py.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.optimize as scpo
import scipy.stats as sps
import matplotlib.pyplot as plt

P = np.polynomial.polynomial


def load_run(csv_path):
    """Returns channel, counts, livetime_s for one run (expects <name>.csv and <name>.json side by side)."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    info = json.loads(csv_path.with_suffix(".json").read_text())
    return df["channel"].to_numpy(float), df["count"].to_numpy(float), float(info["livetime_s"])


def channel_of(E_kev, coeffs, ch_max=8192):
    """Invert energy = sum c_i ch^i for the channel of a given energy (works for any order)."""
    c = np.array(coeffs, float).copy()
    c[0] -= E_kev
    roots = P.polyroots(c)
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots > 0) & (roots < ch_max)]
    if len(roots) == 0:
        raise ValueError("energy not within channel range")
    return float(roots[0])


def check_lines(channel, counts, coeffs, energies=(1173.23, 1332.49), search=15):
    """Quick check that the calibration matches this run: strongest channel near each expected line."""
    for E in energies:
        pred = channel_of(E, coeffs)
        m = (channel > pred - search) & (channel < pred + search)
        found = channel[m][np.argmax(counts[m])]
        print(f"{E:8.2f} keV: predicted ch {pred:8.1f}, strongest bin {found:8.0f}, offset {found - pred:+.1f} ch")


def peak_fit(channel, counts, mu, sigma, half_window=10):
    """Gaussian (fixed position mu, fixed width sigma, area >= 0) on a linear background, Poisson likelihood.
    Returns area, area_err (from Wilks: area / Z), Z (significance), and the window data/fit for plotting."""
    lo, hi = mu - half_window * sigma, mu + half_window * sigma
    m = (channel >= lo) & (channel <= hi)
    x, y = channel[m], counts[m]

    def model(p, area):
        g = area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return np.maximum(p[0] + p[1] * (x - mu), 1e-9) + g

    nll = lambda p, area: np.sum(model(p, area) - y * np.log(model(p, area)))
    p0 = [max(y.mean(), 0.5), 0.0]
    null = scpo.minimize(nll, p0, args=(0.0,), method="Nelder-Mead")
    full = scpo.minimize(lambda q: nll(q[:2], q[2]), list(null.x) + [max(y.sum() - y.mean() * len(y), 1.0)],
                         method="L-BFGS-B", bounds=[(1e-6, None), (None, None), (0, None)])
    area = float(full.x[2])
    ts = max(2 * (null.fun - full.fun), 0.0)
    Z = float(np.sqrt(ts))
    err = area / Z if Z > 0 else np.nan
    return dict(area=area, area_err=err, Z=Z, p_value=float(sps.norm.sf(Z)), x=x, y=y,
                bkg=lambda xx: full.x[0] + full.x[1] * (xx - mu),
                full=lambda xx: full.x[0] + full.x[1] * (xx - mu) + area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xx - mu) / sigma) ** 2))


def rate_excess(ch_t, n_t, L_t, ch_b, n_b, L_b, mu, sigma, nsig=2.0):
    """Counts in +-nsig*sigma around mu, converted to rates (per live second), tin minus blank."""
    wt = (ch_t >= mu - nsig * sigma) & (ch_t <= mu + nsig * sigma)
    wb = (ch_b >= mu - nsig * sigma) & (ch_b <= mu + nsig * sigma)
    Nt, Nb = n_t[wt].sum(), n_b[wb].sum()
    r = Nt / L_t - Nb / L_b
    err = np.sqrt(Nt / L_t ** 2 + Nb / L_b ** 2)
    return r, err, r / err


def difference_spectrum(ch_t, n_t, L_t, n_b, L_b, rebin=4):
    """Rate difference per channel (tin - blank), rebinned for visibility. Returns channel, diff, error."""
    n = (len(ch_t) // rebin) * rebin
    t = n_t[:n].reshape(-1, rebin).sum(1); b = n_b[:n].reshape(-1, rebin).sum(1)
    c = ch_t[:n].reshape(-1, rebin).mean(1)
    return c, t / L_t - b / L_b, np.sqrt(t / L_t ** 2 + b / L_b ** 2)


# ----------------------------------------------------- per-run calibration --
def centroid_near(channel, counts, guess, search=60, verbose=False):
    """Locate a peak near `guess` and fit Gaussian + linear background.
    The start width is measured from the data, and the fit is bounded so it cannot run away.
    Returns centroid, centroid_err, fwhm_ch, or None if the fit is not trustworthy."""
    m = (channel >= guess - search) & (channel <= guess + search)
    x0, y0 = channel[m], counts[m]
    sm = np.convolve(y0, np.ones(5) / 5, mode="same")                  # smoothing: ignore single-bin spikes
    i0 = int(np.argmax(sm)); g0 = x0[i0]
    bg = np.median(np.concatenate([sm[:5], sm[-5:]]))
    half_max = bg + 0.5 * (sm[i0] - bg)
    lo, hi = i0, i0
    while lo > 0 and sm[lo] > half_max: lo -= 1
    while hi < len(sm) - 1 and sm[hi] > half_max: hi += 1
    fw0 = max(float(hi - lo), 4.0)                                        # estimated FWHM in channels
    half = max(3.0 * fw0, 15.0)
    w = (channel >= g0 - half) & (channel <= g0 + half)
    x, y = channel[w], counts[w]
    f = lambda x, A, mu, s, b0, b1: A / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / s) ** 2) + b0 + b1 * (x - mu)
    s0 = fw0 / 2.3548
    try:
        p, cov = scpo.curve_fit(f, x, y, p0=[max(y.sum() - bg * len(y), 1.0), g0, s0, bg, 0.0],
                                bounds=([0, g0 - fw0, 0.4 * s0, -np.inf, -np.inf], [np.inf, g0 + fw0, 2.5 * s0, np.inf, np.inf]),
                                sigma=np.sqrt(np.maximum(y, 1)), absolute_sigma=True, maxfev=20000)
    except Exception as exc:
        if verbose: print("  fit failed:", exc)
        return None
    mu_err = float(np.sqrt(cov[1, 1])); fwhm = float(p[2] * 2.3548)
    if mu_err > 1.0 or fwhm > 2.0 * fw0 or fwhm < 0.5 * fw0:
        if verbose: print(f"  rejected fit near {guess:.0f}: centroid {p[1]:.1f} +/- {mu_err:.2f}, FWHM {fwhm:.1f} (expected ~{fw0:.1f})")
        return None
    return float(p[1]), mu_err, fwhm


def run_calibration(channel, counts, coeffs, energies=(1173.23, 1332.49), search=60, verbose=True):
    """Per-run linear map channel = m*E + c from lines found in THIS spectrum (guesses from the old calibration).
    Returns (m, c, table) and a function ch_of(E). Add more energies (e.g. 609.32, 1460.82) if visible."""
    rows = []
    for E in energies:
        r = centroid_near(channel, counts, channel_of(E, coeffs), search=search, verbose=verbose)
        if r is None:
            if verbose: print(f"  {E} keV: fit failed, skipped")
            continue
        rows.append((E, *r))
    t = pd.DataFrame(rows, columns=["E_keV", "centroid", "centroid_err", "fwhm_ch"])
    if len(t) < 2:
        raise ValueError("need at least 2 fitted lines")
    m, c = np.polyfit(t["E_keV"], t["centroid"], 1)
    t["resid_ch"] = t["centroid"] - (m * t["E_keV"] + c)
    if verbose:
        print(t.round(2).to_string(index=False)); print(f"  channel = {m:.4f} * E + {c:.2f}   (channel of 0 keV: {c:.1f})")
    return m, c, t, (lambda E: m * E + c)