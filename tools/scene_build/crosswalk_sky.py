# Procedural clear-sky envmap for the crosswalk scene.
#
# No sun is painted in: direct sunlight comes from the geometric disk built by
# sun_disk.py, which can hold a true 0.53 deg angular size. A 2k equirect cannot
# (its texel is ~0.18 deg), and sun angular size is what sets dapple penumbra.
#
# Keep envmapScale at 1.0 and expose with evCompensation, otherwise envmapScale
# silently changes the sun-to-sky ratio baked in here.
#
#   python tools/scene_build/crosswalk_sky.py <out.hdr> [sky_irradiance]
import numpy as np, cv2, sys, os, math

OUT = sys.argv[1] if len(sys.argv) > 1 else "sky.hdr"
SKY_E = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # target horizontal irradiance

W, H = 2048, 1024
ZEN = np.array([0.22, 0.40, 0.95], np.float32)
HOR = np.array([0.72, 0.82, 1.00], np.float32)
GND = np.array([0.10, 0.095, 0.085], np.float32)

# Envmap convention (Envmap.hlsli): u = atan2(z,x)/2pi, v = acos(y)/pi. v=0 zenith, 0.5 horizon.
v = (np.arange(H, dtype=np.float32) + 0.5) / H
theta = v * math.pi
up = theta <= math.pi / 2

t = np.clip(theta / (math.pi / 2), 0, 1)
hue = ZEN[None, :] * (1 - (t ** 0.6))[:, None] + HOR[None, :] * (t ** 0.6)[:, None]
bright = 1.0 + 0.9 * (t ** 1.2)
row = hue * bright[:, None]
row[~up] = GND[None, :] * 0.5

img = np.repeat(row[:, None, :], W, axis=1).astype(np.float32)

# normalise the upper hemisphere to the requested horizontal irradiance:
# E = sum L cos(theta) sin(theta) dtheta dphi
dth, dph = math.pi / H, 2 * math.pi / W
lum = row @ np.array([0.2126, 0.7152, 0.0722], np.float32)
E = float((lum[up] * np.cos(theta[up]) * np.sin(theta[up])).sum() * dth * dph * W)
img *= SKY_E / E
lum2 = (row * SKY_E / E) @ np.array([0.2126, 0.7152, 0.0722], np.float32)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
cv2.imwrite(OUT, img[..., ::-1].copy())   # cv2 wants BGR
print("wrote %s  %dx%d" % (OUT, W, H))
print("sky horizontal irradiance %.4f (was %.4f before scaling)" % (SKY_E, E))
print("zenith radiance %.4f  horizon radiance %.4f" % (lum2[0], lum2[H // 2 - 1]))
