"""Locate peak centroids in a spectrum, using the approximate calibration as a guide.

Peaks are fitted by minimising chi-squared with Poisson uncertainties on the counts,
so every centroid comes with a statistical error and a goodness-of-fit.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist


def energy_to_channel(energy, approx_cal, n_channels):
    """Channel whose approximate energy is closest to `energy`."""
    channels = np.arange(n_channels)
    approx_energy = np.polyval(approx_cal[::-1], channels)
    return int(np.argmin(np.abs(approx_energy - energy)))


def _gauss(x, amp, mu, sigma, bkg):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bkg


def poisson_sigma(counts):
    """1-sigma Poisson error on a number of counts: sqrt(N), floored at 1 so empty bins
    do not get zero error (which would give them infinite weight in chi-squared)."""
    return np.sqrt(np.maximum(counts, 1.0))


def chi_squared(y, model, sigma):
    """Sum of ((data - model) / sigma)^2."""
    return float(np.sum(((y - model) / sigma) ** 2))


def find_peak(counts, guess, half_window=30):
    """Chi-squared fit of a gaussian + flat background near `guess`.

    The fit is done twice: first weighting with the observed counts, then re-weighting
    with the first model's prediction. Weighting by the observed counts alone biases the
    fit low (bins that fluctuate down get more weight), so the second pass removes that.

    Returns a dict with
        popt, perr : parameters (amp, mu, sigma, bkg) and their 1-sigma statistical errors
        perr_scaled: perr * sqrt(max(1, chi2/ndf)). If chi2/ndf >> 1 the model does not
                     describe the peak shape (real peaks have tails), so the purely
                     statistical error is too optimistic; this is the conservative one
        pcov       : parameter covariance matrix (absolute, i.e. not rescaled by chi2/ndf)
        chi2, ndf, red_chi2, p_value : goodness of fit
        lo, hi     : fit window (channels)
    """
    lo, hi = max(guess - half_window, 0), min(guess + half_window, len(counts))
    x = np.arange(lo, hi)
    y = counts[lo:hi]
    top = x[np.argmax(y)]                      # highest bin in the window
    p0 = [y.max() - y.min(), top, 2.0, y.min()]

    sigma = poisson_sigma(y)
    for _ in range(2):
        popt, pcov = curve_fit(_gauss, x, y, p0=p0, sigma=sigma, absolute_sigma=True)
        p0 = popt
        sigma = poisson_sigma(_gauss(x, *popt))  # model-based errors for the next pass

    chi2 = chi_squared(y, _gauss(x, *popt), sigma)
    ndf = len(x) - len(popt)
    perr = np.sqrt(np.diag(pcov))
    return {
        "popt": popt, "perr": perr, "pcov": pcov,
        "perr_scaled": perr * np.sqrt(max(1.0, chi2 / ndf)),
        "chi2": chi2, "ndf": ndf,
        "red_chi2": chi2 / ndf if ndf > 0 else np.nan,
        "p_value": chi2_dist.sf(chi2, ndf) if ndf > 0 else np.nan,
        "lo": lo, "hi": hi,
    }


def find_all_peaks(source, half_window=30):
    """Return (channels, channel_errs (rescaled by sqrt(chi2/ndf) if > 1), energies, energy_errs, fits) for one source.

    Each entry of `fits` is the dict from `find_peak` plus the accepted `energy`.
    """
    channels, channel_errs, energies, energy_errs, fits = [], [], [], [], []
    for energy, energy_err in zip(source["energies"], source["energy_errs"]):
        guess = energy_to_channel(energy, source["approx_cal"], len(source["counts"]))
        fit = find_peak(source["counts"], guess, half_window)
        fit["energy"] = energy
        channels.append(fit["popt"][1])
        channel_errs.append(fit["perr_scaled"][1])
        energies.append(energy)
        energy_errs.append(energy_err)
        fits.append(fit)
    return (np.array(channels), np.array(channel_errs),
            np.array(energies), np.array(energy_errs), fits)
