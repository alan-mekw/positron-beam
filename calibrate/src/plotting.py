"""Plot the calibration."""
import matplotlib.pyplot as plt
import numpy as np

from src.calibration import channel_to_energy
from src.peakFinding import _gauss, poisson_sigma


def plot_calibration(channels, channel_errs, energies, energy_errs, labels, cal, save_path=None):
    fig, (ax, res) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    x = np.linspace(0, channels.max() * 1.05, 200)
    y, y_err = channel_to_energy(x, cal)
    ax.plot(x, y, "k-", label="fit")
    ax.fill_between(x, y - y_err, y + y_err, color="k", alpha=0.2, label=r"fit $\pm1\sigma$")
    for name in dict.fromkeys(labels):                    # one marker set per source
        m = np.array(labels) == name
        ax.errorbar(channels[m], energies[m], xerr=channel_errs[m], yerr=energy_errs[m],
                    fmt="o", capsize=3, label=name)
    ax.set_ylabel("Energy (keV)")
    ax.set_title(rf"HPGe calibration: $\chi^2$/ndf = {cal['chi2']:.2f}/{cal['ndf']}")
    ax.legend()
    ax.grid(alpha=0.3)

    res.axhline(0, color="k", lw=1)
    res.errorbar(channels, cal["residuals"], yerr=cal["sigma_eff"], fmt="o", capsize=3)
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
            lo, hi = max(int(mu - pad), 0), min(int(mu + pad), len(counts))
            ax.errorbar(channels[lo:hi], counts[lo:hi], yerr=poisson_sigma(counts[lo:hi]),
                        fmt=".", color="C0", lw=0.8, label="data")
            xs = np.linspace(f["lo"], f["hi"], 300)
            ax.plot(xs, _gauss(xs, *f["popt"]), "r-", label="gaussian fit")
            ax.set_title(f"{f['energy']} keV: centroid {mu:.2f} $\\pm$ {f['perr_scaled'][1]:.2f}, "
                         f"sigma {abs(sigma):.2f}\n"
                         rf"$\chi^2$/ndf = {f['chi2']:.1f}/{f['ndf']}", fontsize=10)
            ax.set_xlabel("Channel")
            ax.legend()
        for ax in row[1 + len(peak_fits):]:
            ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
