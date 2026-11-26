"""Test whether gain/offset drift between source measurements explains a high chi-squared.

Compares three calibration fits on the same peak centroids:
  1. single line:                  E = a1*ch + a0                (what main.py does)
  2. shared slope, offset/source:  E = a1*ch + a0[source]        (tests zero drift)
  3. single quadratic:             E = a2*ch^2 + a1*ch + a0      (tests nonlinearity)

Place next to main.py (project root) and run:  python drift_check.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_dist

from src.getData import CALIBRATION_DIR, load_calibration_data
from src.calibration import collect_points, fit_calibration


def fit_calibration_offsets(channels, channel_errs, energies, energy_errs, labels, n_iter=50):
    """Chi-squared fit with a shared slope and one free offset per source.

    Same effective-variance method as calibration.fit_calibration:
    sigma_eff^2 = sigma_E^2 + (a1 * sigma_ch)^2, iterated because a1 comes from the fit.
    coeffs = [a1, a0_source1, a0_source2, ...] in the order of `names`.
    """
    x = np.asarray(channels, float)
    y = np.asarray(energies, float)
    sx = np.asarray(channel_errs, float)
    sy = np.zeros_like(y) if energy_errs is None else np.asarray(energy_errs, float)
    lab = np.asarray(labels)

    names = list(dict.fromkeys(labels))
    n_par = 1 + len(names)
    if len(x) < n_par:
        raise ValueError(f"Need at least {n_par} points for this fit, got {len(x)}")

    A = np.column_stack([x] + [(lab == n).astype(float) for n in names])
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(n_iter):
        sigma_eff = np.sqrt(sy ** 2 + (coeffs[0] * sx) ** 2)
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
    return {"names": names, "coeffs": coeffs, "cov": cov, "errs": np.sqrt(np.diag(cov)),
            "errs_scaled": (np.sqrt(np.diag(cov)) * np.sqrt(max(1.0, red)) if ndf > 0
                            else np.sqrt(np.diag(cov))),
            "chi2": chi2, "ndf": ndf, "red_chi2": red,
            "p_value": chi2_dist.sf(chi2, ndf) if ndf > 0 else np.nan,
            "residuals": residuals, "sigma_eff": sigma_eff}


def main():
    fits_dir = CALIBRATION_DIR / "fits"
    fits_dir.mkdir(exist_ok=True)

    sources = load_calibration_data()
    channels, channel_errs, energies, energy_errs, labels, fits = collect_points(sources)
    lab = np.array(labels)

    # 1. single line
    base = fit_calibration(channels, channel_errs, energies, energy_errs, order=1)
    print("1. Single line (E = a1*ch + a0)")
    print(f"   a1 = {base['coeffs'][0]:.6f}, a0 = {base['coeffs'][1]:.4f} keV, "
          f"chi2/ndf = {base['chi2']:.2f}/{base['ndf']}")

    # 2. shared slope, offset per source
    off = fit_calibration_offsets(channels, channel_errs, energies, energy_errs, labels)
    names = off["names"]
    print("\n2. Shared slope, free offset per source (E = a1*ch + a0[source])")
    print(f"   a1 = {off['coeffs'][0]:.6f} +/- {off['errs'][0]:.6f} keV/channel")
    for i, n in enumerate(names, start=1):
        print(f"   a0[{n}] = {off['coeffs'][i]:.4f} +/- {off['errs'][i]:.4f} keV")
    print(f"   chi2/ndf = {off['chi2']:.2f}/{off['ndf']}", end="")
    if off["ndf"] > 0:
        print(f"  (reduced chi2 = {off['red_chi2']:.2f}, p = {off['p_value']:.3f})")
    else:
        print("  (no degrees of freedom)")
    if off["ndf"] > 0 and off["red_chi2"] > 1:
        print("   errors scaled by sqrt(chi2/ndf) (use these as the quoted uncertainties):")
        print(f"     a1 +/- {off['errs_scaled'][0]:.6f} keV/channel")
        for i, n in enumerate(names, start=1):
            print(f"     a0[{n}] +/- {off['errs_scaled'][i]:.4f} keV")
    print(f"   Offset of each source relative to {names[0]}:")
    for i in range(1, len(names)):
        d = off["coeffs"][i + 1] - off["coeffs"][1]
        var = off["cov"][i + 1, i + 1] + off["cov"][1, 1] - 2 * off["cov"][1, i + 1]
        print(f"     {names[i]}: {d:+.4f} +/- {np.sqrt(var):.4f} keV "
              f"({d / off['coeffs'][0]:+.3f} channels)")

    # 3. quadratic
    quad = None
    if len(channels) >= 3:
        quad = fit_calibration(channels, channel_errs, energies, energy_errs, order=2)
        print("\n3. Single quadratic (E = a2*ch^2 + a1*ch + a0)")
        print(f"   chi2/ndf = {quad['chi2']:.2f}/{quad['ndf']}")

    print("\nHow to read this:")
    print(" - (2) drops to ~1 and (3) does not: offset drift between measurements.")
    print(" - (3) drops to ~1 and (2) does not: nonlinearity, use order=2 in main.py.")
    print(" - neither drops: peak shape (tails/multiplets) or underestimated errors; "
          "check the per-peak chi2/ndf printed by main.py.")

    pulls = off["residuals"] / off["sigma_eff"]
    print("\nPer-point pulls (per-source-offset fit):")
    for l, e, ch, r, p in zip(labels, energies, channels, off["residuals"], pulls):
        print(f"  {l:6s} {e:9.2f} keV  ch {ch:8.2f}  resid {r:+.3f} keV  pull {p:+.1f}")

    print("\nChi2 after dropping the largest-pull point, one at a time (per-source offsets):")
    keep = np.ones(len(channels), bool)
    for _ in range(2):
        idx = np.flatnonzero(keep)
        f = fit_calibration_offsets(channels[idx], channel_errs[idx], energies[idx],
                                    energy_errs[idx], lab[idx])
        worst = idx[np.argmax(np.abs(f["residuals"] / f["sigma_eff"]))]
        keep[worst] = False
        idx = np.flatnonzero(keep)
        f = fit_calibration_offsets(channels[idx], channel_errs[idx], energies[idx],
                                    energy_errs[idx], lab[idx])
        print(f"  dropped {labels[worst]} {energies[worst]:.2f} keV -> "
              f"chi2/ndf = {f['chi2']:.2f}/{f['ndf']}")

    fig, axes = plt.subplots(3 if quad else 2, 1, figsize=(7, 9 if quad else 7), sharex=True)
    panels = [(base, rf"Single line: $\chi^2$/ndf = {base['chi2']:.2f}/{base['ndf']}"),
              (off, rf"Per-source offsets: $\chi^2$/ndf = {off['chi2']:.2f}/{off['ndf']}")]
    if quad:
        panels.append((quad, rf"Quadratic: $\chi^2$/ndf = {quad['chi2']:.2f}/{quad['ndf']}"))
    for ax, (res, title) in zip(axes, panels):
        ax.axhline(0, color="k", lw=1)
        for n in names:
            m = lab == n
            ax.errorbar(channels[m], res["residuals"][m], yerr=res["sigma_eff"][m],
                        fmt="o", capsize=3, label=n)
        ax.set_ylabel("Residual (keV)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Channel")
    fig.tight_layout()
    fig.savefig(fits_dir / "drift_residuals.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
