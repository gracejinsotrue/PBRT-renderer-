# Measure the painted sun's irradiance against the sky dome's in sky_custom.hdr.
# What matters is INTEGRATED energy, not peak value: a huge SUN_RGB with a sub-pixel
# SIGMA contributes almost nothing. Aim for ~5; below ~1 there is no visible dapple.
#   python tools/analysis/sm_sunratio.py [path/to/sky.hdr]
import cv2, numpy as np, math, sys

path = sys.argv[1] if len(sys.argv) > 1 else "scenes/san_miguel/sky_custom.hdr"
img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).astype(np.float64)
H, W, _ = img.shape
b, g, r = img[..., 0], img[..., 1], img[..., 2]
L = 0.2126 * r + 0.7152 * g + 0.0722 * b

theta = (np.arange(H) + 0.5) / H * math.pi                      # 0 = zenith
dw = np.repeat(((2 * math.pi / W) * (math.pi / H) * np.sin(theta))[:, None], W, axis=1)
cos = np.clip(np.cos(theta), 0, None)[:, None]
sun = L > max(np.median(L) * 50, 1000)

E_tot = (L * cos * dw)[:H // 2].sum()                           # upper hemisphere only
E_sun = (L * cos * dw * sun)[:H // 2].sum()
E_sky = E_tot - E_sun
print(f"{path}  {W}x{H}")
print(f"  E_sun = {E_sun:8.3f}   E_sky = {E_sky:8.3f}   sun/sky = {E_sun / max(E_sky, 1e-9):.2f}")
print("  (sky.hdr as downloaded = 1.91; clear-sky target ~5; below ~1 = no dapple)")
