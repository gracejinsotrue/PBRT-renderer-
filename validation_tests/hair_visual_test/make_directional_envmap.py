"""
Generate a simple directional HDR environment map for the hair visual test.

Creates a 512x256 latlong HDR with:
  - A bright directional "sun" from the right side (+X direction) at slight elevation
  - A soft blue-gray sky gradient everywhere else

Run from the repo root:
    python validation_tests/hair_visual_test/make_directional_envmap.py

Writes: validation_tests/hair_visual_test/textures/white_furnace.hdr
(named white_furnace.hdr so the hardcoded envmap path in DXRApp.cpp picks it up)
"""

import os
import numpy as np

OUTPUT = os.path.join(os.path.dirname(__file__), "textures", "white_furnace.hdr")

W, H = 512, 256

# Build UV grid: u in [0,1], v in [0,1]
u = (np.arange(W) + 0.5) / W   # (W,)
v = (np.arange(H) + 0.5) / H   # (H,)
uu, vv = np.meshgrid(u, v)      # (H, W)

# Lat-long to direction:  phi = 2*pi*u (azimuth), theta = pi*v (polar from +Y up)
phi   = 2.0 * np.pi * uu   # azimuth
theta = np.pi * vv          # polar from top

sin_t = np.sin(theta)
cos_t = np.cos(theta)
sin_p = np.sin(phi)
cos_p = np.cos(phi)

# World direction in Y-up:  dx = sin_t*cos_p, dy = cos_t, dz = sin_t*sin_p
dx = sin_t * cos_p   # (H, W)
dy = cos_t
dz = sin_t * sin_p

# Sun direction: from the right-front (+X side), slight upward elevation
import math
sun_elev  = math.radians(20)    # 20 degrees above horizon
sun_azim  = math.radians(90)    # 90° azimuth = +X direction in lat-long

sun_dx = math.cos(sun_elev) * math.cos(sun_azim)
sun_dy = math.sin(sun_elev)
sun_dz = math.cos(sun_elev) * math.sin(sun_azim)

# Cosine similarity to sun
dot_sun = dx * sun_dx + dy * sun_dy + dz * sun_dz  # (H, W)

# Sky gradient: blue-tinted gray, brighter near top
sky_r = 0.5 + 0.2 * (1.0 - vv)
sky_g = 0.55 + 0.2 * (1.0 - vv)
sky_b = 0.7  + 0.2 * (1.0 - vv)

# Sun disk: very tight lobe (cosine^2000 gives ~2° angular radius)
sun_intensity = 50.0
sun_mask = np.maximum(dot_sun, 0.0) ** 2000
sun_r = sky_r + sun_intensity * sun_mask
sun_g = sky_g + sun_intensity * sun_mask
sun_b = sky_b + sun_intensity * 0.9 * sun_mask   # slightly warm sun

img = np.stack([sun_r, sun_g, sun_b], axis=-1).astype(np.float32)  # (H, W, 3)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# Write as Radiance HDR (.hdr) using numpy — simple implementation
# RGBE encoding
def float_to_rgbe(pixels):
    """Convert float32 RGB (H,W,3) to RGBE uint8 (H,W,4)."""
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    m = np.maximum(np.maximum(r, g), b)
    valid = m > 1e-32
    exp = np.zeros_like(m, dtype=np.int32)
    mantissa = np.ones_like(m)
    exp[valid] = np.floor(np.log2(m[valid])).astype(np.int32) + 1
    scale = np.where(valid, np.ldexp(1.0, -exp) * 256.0, 0.0)
    re = np.clip((r * scale).astype(np.int32), 0, 255).astype(np.uint8)
    ge = np.clip((g * scale).astype(np.int32), 0, 255).astype(np.uint8)
    be = np.clip((b * scale).astype(np.int32), 0, 255).astype(np.uint8)
    ee = np.clip((exp + 128).astype(np.int32), 0, 255).astype(np.uint8)
    return np.stack([re, ge, be, ee], axis=-1)

rgbe = float_to_rgbe(img)  # (H, W, 4)

with open(OUTPUT, 'wb') as f:
    # HDR header
    f.write(b'#?RADIANCE\n')
    f.write(b'FORMAT=32-bit_rle_rgbe\n')
    f.write(b'\n')
    f.write(f'-Y {H} +X {W}\n'.encode())
    # Write uncompressed RGBE scanlines
    for y in range(H):
        row = rgbe[y]  # (W, 4)
        f.write(row.tobytes())

print(f"Wrote {OUTPUT}  ({W}x{H}, sun from +X at 20° elevation, intensity={sun_intensity})")
