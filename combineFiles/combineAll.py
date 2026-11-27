"""Usage: python3 combineAll.py runs/   (writes runs/combined.Spc)"""
import struct
import sys
from pathlib import Path
import numpy as np

folder = Path(sys.argv[1])
files = sorted(folder.glob("run*.[sS]pc"))
if not files:
    sys.exit("No run*.Spc files found.")

total, real, live = 0, 0.0, 0.0
for f in files:
    data = f.read_bytes()
    kind = struct.unpack_from("<h", data, 2)[0]            # 1 = integer SPC
    record, _, channels = struct.unpack_from("<3h", data, 60)
    if kind != 1:
        sys.exit(f"{f.name}: not an integer SPC file")
    start = (record - 1) * 128                             # spectrum position
    total = total + np.frombuffer(data, "<u4", channels, start).astype(np.uint64)
    r, l = struct.unpack_from("<2f", data, 90)             # real, live time
    real, live = real + r, live + l

# The first file is the template for calibration and other header details
out = bytearray(files[0].read_bytes())
start = (struct.unpack_from("<h", out, 60)[0] - 1) * 128
out[start:start + 4 * len(total)] = total.astype("<u4").tobytes()
struct.pack_into("<2f", out, 90, real, live)
out[294:314] = f"{real:10.0f}{live:10.0f}".encode()        # text copy of times
(folder / "combined.Spc").write_bytes(out)
print(f"{len(files)} files, {total.sum()} counts, real {real:.2f} s, live {live:.2f} s")
