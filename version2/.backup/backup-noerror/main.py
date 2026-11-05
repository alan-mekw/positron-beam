from src.getData import CALIBRATION_DIR, load_calibration_data
from src.calibration import collect_points, fit_calibration
from src.plotting import plot_calibration, plot_spectra


def main():
    fits_dir = CALIBRATION_DIR / "fits"
    fits_dir.mkdir(exist_ok=True)

    sources = load_calibration_data()
    print("Loaded:", ", ".join(s["name"] for s in sources))

    channels, energies, labels, fits = collect_points(sources)
    for c, e, l in zip(channels, energies, labels):
        print(f"{l}: channel {c:.2f} -> {e} keV")

    coeffs = fit_calibration(channels, energies, order=1)
    print(f"Calibration: E = {coeffs[0]:.6f} * channel + {coeffs[1]:.4f} keV")

    plot_calibration(channels, energies, labels, coeffs, save_path=fits_dir / "calibration.png")
    plot_spectra(sources, fits, save_path=fits_dir / "spectrum.png")


if __name__ == "__main__":
    main()
