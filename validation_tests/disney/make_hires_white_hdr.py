r"""
make_hires_white_hdr.py
Generate a 64x128 white HDR for the furnace tests.

Why: the existing 2x2 white_furnace.hdr is so coarse that envmap importance
sampling without UV jittering (see Shaders.hlsl SampleEnvmap) produces a ~6%
bias on diffuse materials. A finer envmap shrinks each pixel's solid angle,
making the unjittered pixel-center samples a better approximation of the
continuous distribution.

This is a diagnostic. The real fix is to add UV jittering in SampleEnvmap.
But this confirms the diagnosis: if running the furnace tests against this
finer envmap dramatically reduces the bias, the bug is unjittered envmap NEE.

Output: validation_tests/disney/textures/white_hires.hdr
"""
import os

W, H = 128, 64

# RGBE encoding of (1.0, 1.0, 1.0): bytes=(128,128,128,129)
# decodes as (128/256) * 2^(129-128) = 1.0 per channel
header = (
    b"#?RADIANCE\n"
    b"FORMAT=32-bit_rle_rgbe\n"
    b"\n" + f"-Y {H} +X {W}\n".encode()
)
pixel = bytes([128, 128, 128, 129])

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "textures", "white_hires.hdr")
os.makedirs(os.path.dirname(out), exist_ok=True)

with open(out, "wb") as f:
    f.write(header)
    f.write(pixel * (W * H))

print(f"wrote {out}  ({W}x{H} constant white)")
