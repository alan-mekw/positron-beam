"""Locate peak centroids in a spectrum, using the approximate calibration as a guide."""
import numpy as np
from scipy.optimize import curve_fit


def energy_to_channel(energy, approx_cal, n_channels):
    """Channel whose approximate energy is closest to `energy`."""
    channels = np.arange(n_channels)
    approx_energy = np.polyval(approx_cal[::-1], channels)
    return int(np.argmin(np.abs(approx_energy - energy)))


def _gauss(x, amp, mu, sigma, bkg):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bkg


def find_peak(counts, guess, half_window=30):
    """Fit a gaussian + flat background near `guess`.

    Returns (popt, lo, hi): fit parameters (amp, mu, sigma, bkg) and the window used.
    """
    lo, hi = max(guess - half_window, 0), min(guess + half_window, len(counts))
    x = np.arange(lo, hi)
    y = counts[lo:hi]
    top = x[np.argmax(y)]                      # highest bin in the window
    p0 = [y.max() - y.min(), top, 2.0, y.min()]
    popt, _ = curve_fit(_gauss, x, y, p0=p0)
    return popt, lo, hi


def find_all_peaks(source, half_window=30):
    """Return (channels, energies, fits) of the fitted peaks for one calibration source.

    Each entry of `fits` is a dict with the gaussian parameters and the fit window.
    """
    channels, energies, fits = [], [], []
    for energy in source["energies"]:
        guess = energy_to_channel(energy, source["approx_cal"], len(source["counts"]))
        popt, lo, hi = find_peak(source["counts"], guess, half_window)
        channels.append(popt[1])
        energies.append(energy)
        fits.append({"popt": popt, "lo": lo, "hi": hi, "energy": energy})
    return np.array(channels), np.array(energies), fits
