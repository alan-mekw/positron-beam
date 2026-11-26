import numpy as np
from src.getData import load_calibration_data
from src.peakFinding import energy_to_channel

for s in load_calibration_data():
    counts = s["counts"]
    print(f"\n{s['name']}: n_channels = {len(counts)}, total counts = {counts.sum():.0f}, "
          f"approx_cal = {s['approx_cal']}")
    for e in s["energies"]:
        g = energy_to_channel(e, s["approx_cal"], len(counts))
        lo, hi = max(g - 30, 0), min(g + 30, len(counts))
        w = counts[lo:hi]
        top = lo + int(np.argmax(w))
        print(f"  {e:9.3f} keV -> guess ch {g:5d} | window max {w.max():8.0f} at ch {top:5d} "
              f"| window min {w.min():8.0f}")
