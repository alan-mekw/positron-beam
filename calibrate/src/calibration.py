"""Build the channel -> energy calibration from all sources, with a chi-squared fit."""
import numpy as np
from scipy.stats import chi2 as chi2_dist

from src.peakFinding import find_all_peaks


def collect_points(sources):
    """Gather fitted (channel, energy) points from every source.

    Returns channels, channel_errs, energies, energy_errs, labels and fits
    (a dict of peak fits per source name).
    """
    all_ch, all_ch_err, all_en, all_en_err, labels, fits = [], [], [], [], [], {}
    for source in sources:
        ch, ch_err, en, en_err, fits[source["name"]] = find_all_peaks(source)
        all_ch.extend(ch)
        all_ch_err.extend(ch_err)
        all_en.extend(en)
        all_en_err.extend(en_err)
        labels.extend([source["name"]] * len(ch))
    return (np.array(all_ch), np.array(all_ch_err),
            np.array(all_en), np.array(all_en_err), labels, fits)


def fit_calibration(channels, channel_errs, energies, energy_errs=None, order=1, n_iter=50):
    """Chi-squared polynomial fit energy = f(channel), with errors on both axes.

    The centroid error is converted to an energy error using the local slope
    (effective-variance method):  sigma_eff^2 = sigma_E^2 + (f'(ch) * sigma_ch)^2.
    Since the slope depends on the fit, the weighted least squares is iterated to
    convergence.

    Returns a dict with
        coeffs      : polynomial coefficients, highest power first (np.polyval order)
        cov, errs   : covariance matrix / 1-sigma errors of coeffs (absolute errors)
        errs_scaled : errs * sqrt(max(1, chi2/ndf)); use this if chi2/ndf > 1, i.e. when
                      the quoted point errors look underestimated
        chi2, ndf, red_chi2, p_value : goodness of fit (nan where ndf <= 0)
        residuals, sigma_eff : data - fit, and the effective error on each point
    """
    x = np.asarray(channels, float)
    y = np.asarray(energies, float)
    sx = np.asarray(channel_errs, float)
    sy = np.zeros_like(y) if energy_errs is None else np.asarray(energy_errs, float)

    n_par = order + 1
    if len(x) < n_par:
        raise ValueError(f"Need at least {n_par} points for an order-{order} fit, got {len(x)}")

    A = np.vander(x, n_par)                       # columns: x^order ... x^0
    coeffs = np.polyfit(x, y, order)              # unweighted starting point
    for _ in range(n_iter):
        slope = np.polyval(np.polyder(coeffs), x)
        sigma_eff = np.sqrt(sy ** 2 + (slope * sx) ** 2)
        w = 1.0 / sigma_eff ** 2
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        new = cov @ (A.T @ (w * y))
        done = np.allclose(new, coeffs, rtol=1e-12, atol=1e-14)
        coeffs = new
        if done:
            break

    residuals = y - A @ coeffs
    chi2 = float(np.sum((residuals / sigma_eff) ** 2))
    ndf = len(x) - n_par
    red = chi2 / ndf if ndf > 0 else np.nan
    errs = np.sqrt(np.diag(cov))
    return {
        "coeffs": coeffs, "cov": cov, "errs": errs,
        "errs_scaled": errs * np.sqrt(max(1.0, red)) if ndf > 0 else errs,
        "chi2": chi2, "ndf": ndf, "red_chi2": red,
        "p_value": chi2_dist.sf(chi2, ndf) if ndf > 0 else np.nan,
        "residuals": residuals, "sigma_eff": sigma_eff, "order": order,
    }


def channel_to_energy(channel, cal, channel_err=0.0):
    """Energy and its 1-sigma error for a channel (or array of channels).

    Includes the calibration-parameter uncertainty (with correlations) and, optionally,
    the uncertainty on the channel itself.
    """
    ch = np.atleast_1d(np.asarray(channel, float))
    J = np.vander(ch, cal["order"] + 1)           # dE/dcoeff
    energy = J @ cal["coeffs"]
    var = np.einsum("ij,jk,ik->i", J, cal["cov"], J)
    var = var + (np.polyval(np.polyder(cal["coeffs"]), ch) * channel_err) ** 2
    return energy, np.sqrt(var)
