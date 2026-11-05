from src.getData import CALIBRATION_DIR, load_calibration_data
from src.calibration import collect_points, fit_calibration
from src.plotting import plot_calibration, plot_spectra


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

    cal = fit_calibration(channels, channel_errs, energies, energy_errs, order=1)
    a1, a0 = cal["coeffs"]
    e1, e0 = cal["errs"]
    print("\nCalibration: E = a1 * channel + a0")
    print(f"  a1 = {a1:.6f} +/- {e1:.6f} keV/channel")
    print(f"  a0 = {a0:.4f} +/- {e0:.4f} keV")
    print(f"  chi2/ndf = {cal['chi2']:.2f}/{cal['ndf']}")
    if cal["ndf"] > 0:
        print(f"  reduced chi2 = {cal['red_chi2']:.2f}, p = {cal['p_value']:.3f}")
        if cal["red_chi2"] > 1:
            print(f"  errors scaled by sqrt(chi2/ndf): a1 +/- {cal['errs_scaled'][0]:.6f}, "
                  f"a0 +/- {cal['errs_scaled'][1]:.4f}")
    else:
        print("  (no degrees of freedom: the line passes through the points exactly, so "
              "chi2 cannot test the fit; add more peaks/sources)")

    plot_calibration(channels, channel_errs, energies, energy_errs, labels, cal,
                     save_path=fits_dir / "calibration.png")
    plot_spectra(sources, fits, save_path=fits_dir / "spectrum.png")


if __name__ == "__main__":
    main()
