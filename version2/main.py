import numpy as np

from src.getData import CALIBRATION_DIR, load_calibration_data
from src.calibration import collect_points, fit_calibration
from src.plotting import plot_calibration, plot_spectra

# Reuse the per-source-offset fit from drift_check without duplicating the code.
from drift_check import fit_calibration_offsets


def main():
    fits_dir = CALIBRATION_DIR / "fits"
    fits_dir.mkdir(exist_ok=True)

    sources = load_calibration_data()
    print("Loaded:", ", ".join(s["name"] for s in sources))

    channels, channel_errs, energies, energy_errs, labels, fits = collect_points(sources)
    print("\nPeak fits (gaussian + flat background, Poisson chi-squared):")
    for name, peak_fits in fits.items():
        for f in peak_fits:
            print(f"{name} {f['energy']} keV: channel {f['popt'][1]:.3f} +/- {f['perr_scaled'][1]:.3f} "
                  f"(stat. {f['perr'][1]:.3f}), "
                  f"chi2/ndf = {f['chi2']:.1f}/{f['ndf']} (p = {f['p_value']:.2f})")

    # ---- Candidate models, all on the same points -----------------------------------
    candidates = {}

    lin = fit_calibration(channels, channel_errs, energies, energy_errs, order=1)
    candidates["linear"] = {
        "cal": lin,
        "n_par": 2,
        "desc": "E = a1*ch + a0",
    }

    off = fit_calibration_offsets(channels, channel_errs, energies, energy_errs, labels)
    # Wrap the per-source-offset result in a calibration-like dict so downstream
    # code (plotting.channel_to_energy) can consume it. We use the *first* source's
    # offset as the reference intercept; per-source offsets differ only by a constant.
    off_cal = {
        "coeffs": np.array([off["coeffs"][0], off["coeffs"][1]]),
        "cov":    off["cov"][np.ix_([0, 1], [0, 1])],
        "errs":   off["errs"][:2],
        "errs_scaled": off["errs_scaled"][:2],
        "chi2": off["chi2"], "ndf": off["ndf"],
        "red_chi2": off["red_chi2"], "p_value": off["p_value"],
        "residuals": off["residuals"], "sigma_eff": off["sigma_eff"],
        "order": 1,
    }
    candidates["per-source offsets"] = {
        "cal": off_cal,
        "n_par": 1 + len(off["names"]),
        "desc": "E = a1*ch + a0[source]",
        "raw": off,
    }

    if len(channels) >= 3:
        quad = fit_calibration(channels, channel_errs, energies, energy_errs, order=2)
        candidates["quadratic"] = {
            "cal": quad,
            "n_par": 3,
            "desc": "E = a2*ch^2 + a1*ch + a0",
        }

    # ---- Compare --------------------------------------------------------------------
    n_pts = len(channels)
    print("\nModel comparison (same points, same errors):")
    print(f"  N points = {n_pts}")
    for name, c in candidates.items():
        cal = c["cal"]
        print(f"  {name:18s} {c['desc']:28s}  "
              f"chi2/ndf = {cal['chi2']:.3f}/{cal['ndf']}  "
              f"(red = {cal['red_chi2']:.3f}, p = {cal['p_value']:.3f})"
              if cal["ndf"] > 0 else
              f"  {name:18s} {c['desc']:28s}  chi2 = {cal['chi2']:.3f} (no ndf)")

    # Pick by p-value (goodness of fit), tie-broken by fewer parameters (Occam).
    scored = []
    for name, c in candidates.items():
        cal = c["cal"]
        if cal["ndf"] > 0:
            scored.append((cal["p_value"], -c["n_par"], name))
    if scored:
        _, _, best_name = max(scored)
    else:
        best_name = "linear"
    best = candidates[best_name]
    cal = best["cal"]
    print(f"\nBest model by p-value (fewer params on ties): {best_name}")
    print(f"  {best['desc']}")
    print(f"  chi2/ndf = {cal['chi2']:.3f}/{cal['ndf']}", end="")
    if cal["ndf"] > 0:
        print(f"  (red = {cal['red_chi2']:.3f}, p = {cal['p_value']:.3f})")
        if cal["red_chi2"] > 1:
            print(f"  errors scaled by sqrt(chi2/ndf): "
                  f"a1 +/- {cal['errs_scaled'][0]:.6f} keV/channel, "
                  f"a0 +/- {cal['errs_scaled'][1]:.4f} keV")
    else:
        print()

    a1, a0 = cal["coeffs"][0], cal["coeffs"][-1]
    e1, e0 = cal["errs"][0], cal["errs"][-1]
    print(f"  a1 = {a1:.6f} +/- {e1:.6f} keV/channel")
    print(f"  a0 = {a0:.4f} +/- {e0:.4f} keV")
    if "raw" in best:  # per-source offsets: also report each source's own intercept
        raw = best["raw"]
        print("  Per-source offsets (relative to "
              f"{raw['names'][0]}):")
        for i in range(1, len(raw["names"])):
            d = raw["coeffs"][i + 1] - raw["coeffs"][1]
            var = (raw["cov"][i + 1, i + 1] + raw["cov"][1, 1]
                   - 2 * raw["cov"][1, i + 1])
            print(f"    {raw['names'][i]}: {d:+.4f} +/- {np.sqrt(var):.4f} keV")

    # ---- Plots (use the winning calibration) ----------------------------------------
    plot_calibration(channels, channel_errs, energies, energy_errs, labels, cal,
                     save_path=fits_dir / "calibration.png")
    plot_spectra(sources, fits, save_path=fits_dir / "spectrum.png")


if __name__ == "__main__":
    main()
