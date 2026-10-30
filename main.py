import becquerel as bq
import pandas as pd

spec = bq.Spectrum.from_file("test.Chn")

df = pd.DataFrame({
    "channel": range(len(spec.counts)),
    "counts": spec.counts.value if hasattr(spec.counts, "value") else spec.counts,
})

# Optional: energy column if the file contains a calibration
if spec.is_calibrated:
    df["energy_keV"] = spec.bin_centers_kev

df.to_csv("sample.csv", index=False)
