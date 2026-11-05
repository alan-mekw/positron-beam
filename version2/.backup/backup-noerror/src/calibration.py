"""Build the channel -> energy calibration from all sources."""
import numpy as np

from src.peakFinding import find_all_peaks


def collect_points(sources):
    """Gather fitted (channel, energy) points from every source.

    Returns channels, energies, labels and fits (a dict of peak fits per source name).
    """
    all_ch, all_en, labels, fits = [], [], [], {}
    for source in sources:
        ch, en, fits[source["name"]] = find_all_peaks(source)
        all_ch.extend(ch)
        all_en.extend(en)
        labels.extend([source["name"]] * len(ch))
    return np.array(all_ch), np.array(all_en), labels, fits


def fit_calibration(channels, energies, order=1):
    """Polynomial fit energy = f(channel). Returns coefficients (highest power first)."""
    return np.polyfit(channels, energies, order)
