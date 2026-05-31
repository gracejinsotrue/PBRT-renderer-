"""
make_white_furnace.py
Generates a tiny constant-white RGBE HDR file for the white furnace test.

Run from any directory:
    python make_white_furnace.py

Output: validation_tests/hair_furnace_test/textures/white_furnace.hdr
"""

import os
import struct

# RGBE encoding of (1.0, 1.0, 1.0):
#   frexp(1.0) → mantissa=0.5, exp=1
#   exponent byte = exp + 128 = 129
#   each channel byte = floor(channel / 2^1 * 256) = 128
# Decodes back as: (128/256) * 2^(129-128) = 0.5 * 2 = 1.0 ✓

W, H = 2, 2

header = (
    b"#?RADIANCE\n"
    b"FORMAT=32-bit_rle_rgbe\n"
    b"\n"
    + f"-Y {H} +X {W}\n".encode()
)

# Each pixel encodes to exactly (1.0, 1.0, 1.0)
pixel  = bytes([128, 128, 128, 129])
pixels = pixel * (W * H)

script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(
    script_dir, "validation_tests", "hair_furnace_test", "textures", "white_furnace.hdr"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "wb") as f:
    f.write(header)
    f.write(pixels)

print(f"Written: {out_path}")
print(f"  {W}x{H} pixels, all (1.0, 1.0, 1.0) — constant white environment.")
