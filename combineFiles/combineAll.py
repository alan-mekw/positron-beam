"""Sum run*.Spc files into combined.Spc.  Usage: python sum_spc.py "C:\\GammaData\\overnight" """
import sys
from pathlib import Path
import becquerel as bq
import numpy as np

folder = Path(sys.argv[1])
files = sorted(folder.glob("run*.[sS]pc"))
if not files:
    sys.exit("No run*.Spc files found.")

spectra = [bq.Spectrum.from_file(str(f)) for f in files]
if len({len(s.counts_vals) for s in spectra}) != 1:
    sys.exit("Files have different channel counts.")
total = sum(s.counts_vals for s in spectra)

# Use the first file as a template and overwrite its counts and times
data = files[0].read_bytes()
for kind in ("<u4", "<f4"):  # integer or real SPC
    old = spectra[0].counts_vals.astype(kind).tobytes()
    if data.count(old) == 1:
        start = data.index(old)
        break
else:
    sys.exit("Could not locate the counts in the template file.")

head, tail = data[:start], data[start + len(old):]
live0, real0 = spectra[0].livetime, spectra[0].realtime
for first, new in ((live0, sum(s.livetime for s in spectra)),
                   (real0, sum(s.realtime for s in spectra))):
    old_t, new_t = (np.float32(t).tobytes() for t in (first, new))
    if live0 == real0 or old_t not in head + tail:
        sys.exit("Could not safely update live/real time.")
    head, tail = head.replace(old_t, new_t), tail.replace(old_t, new_t)

out = folder / "combined.Spc"
out.write_bytes(head + total.astype(kind).tobytes() + tail)
print(f"Combined {len(files)} files into {out}")
