"""Plot the calibration."""
import matplotlib.pyplot as plt
import numpy as np

from src.peakFinding import _gauss


def plot_calibration(channels, energies, labels, coeffs, save_path=None):
    fig, (ax, res) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    x = np.linspace(0, channels.max() * 1.05, 200)
    ax.plot(x, np.polyval(coeffs, x), "k-", label="fit")
    for name in dict.fromkeys(labels):                    # one marker set per source
        m = np.array(labels) == name
        ax.plot(channels[m], energies[m], "o", label=name)
    ax.set_ylabel("Energy (keV)")
    ax.set_title("HPGe calibration: channel vs energy")
    ax.legend()
    ax.grid(alpha=0.3)

    res.axhline(0, color="k", lw=1)
    res.plot(channels, energies - np.polyval(coeffs, channels), "o")
    res.set_xlabel("Channel")
    res.set_ylabel("Residual (keV)")
    res.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_spectra(sources, fits, save_path=None, pad=60):
    """One row per source: full spectrum, then a zoom on each peak with its gaussian fit."""
    n_zoom = max(len(fits[s["name"]]) for s in sources)
    fig, axes = plt.subplots(len(sources), 1 + n_zoom, squeeze=False,
                             figsize=(5 * (1 + n_zoom), 4 * len(sources)))
    for row, source in zip(axes, sources):
        counts = source["counts"]
        channels = np.arange(len(counts))
        peak_fits = fits[source["name"]]

        ax = row[0]
        ax.semilogy(channels, np.where(counts > 0, counts, np.nan), lw=0.8)
        for f in peak_fits:
            ax.axvline(f["popt"][1], color="r", ls="--", lw=0.8)
        ax.set_title(f"{source['name']} spectrum")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")

        for ax, f in zip(row[1:], peak_fits):
            amp, mu, sigma, bkg = f["popt"]
            lo, hi = int(mu - pad), int(mu + pad)
            ax.step(channels[lo:hi], counts[lo:hi], where="mid", label="data")
            xs = np.linspace(f["lo"], f["hi"], 300)
            ax.plot(xs, _gauss(xs, *f["popt"]), "r-", label="gaussian fit")
            ax.set_title(f"{f['energy']} keV: centroid {mu:.2f}, sigma {abs(sigma):.2f}")
            ax.set_xlabel("Channel")
            ax.legend()
        for ax in row[1 + len(peak_fits):]:
            ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
