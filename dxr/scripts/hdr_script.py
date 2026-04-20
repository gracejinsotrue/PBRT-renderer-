#!/usr/bin/env python3
# Generate a constant-white HDR environment map for the white-furnace test.

# Produces a 2x1 RGBE .hdr file where every pixel has radiance (1, 1, 1).


import sys
import os


def rgbe_encode(r, g, b):
    """Encode a float RGB triple to Radiance 4-byte RGBE."""
    m = max(r, g, b)
    if m < 1e-32:
        return (0, 0, 0, 0)
    # frexp: m = mantissa * 2^exp, mantissa in [0.5, 1)
    import math
    mantissa, exponent = math.frexp(m)
    scale = mantissa * 256.0 / m
    return (
        int(r * scale),
        int(g * scale),
        int(b * scale),
        exponent + 128,
    )


def write_hdr(path, width, height, rgb_value=(1.0, 1.0, 1.0)):
    """Write a constant-color Radiance .hdr image."""
    encoded = rgbe_encode(*rgb_value)
    pixel_bytes = bytes(encoded)

    with open(path, "wb") as f:
        # Radiance header
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n")
        f.write(b"\n")
        f.write(f"-Y {height} +X {width}\n".encode("ascii"))
        for _ in range(height):
            for _ in range(width):
                f.write(pixel_bytes)

    print(f"Wrote {path}: {width}x{height}, constant RGB={rgb_value}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "textures/white_furnace.hdr"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_hdr(out, width=512, height=256, rgb_value=(1.0, 1.0, 1.0)) 