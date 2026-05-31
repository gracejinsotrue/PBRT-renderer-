r"""
analyze_furnace.py
Reads an EXR from the white furnace test and checks mean luminance ≈ 1.0.

Usage:
    python analyze_furnace.py path\to\snapshot_N.exr

Install (run once):
    pip install numpy openexr
"""

import sys
import os
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python analyze_furnace.py path\\to\\snapshot_N.exr")
    sys.exit(1)

exr_path = sys.argv[1]
if not os.path.exists(exr_path):
    print(f"ERROR: File not found: {exr_path}")
    sys.exit(1)

# --- Load EXR ---
img = None

# Attempt 1: cv2 — most reliable EXR reader on Windows/conda
try:
    import cv2
    raw = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if raw is not None and raw.ndim == 3:
        img = raw[..., ::-1].astype(np.float32)   # BGR -> RGB
        print("[loader] cv2")
    elif raw is not None:
        img = raw.astype(np.float32)
        print("[loader] cv2 (grayscale)")
except Exception as e:
    print(f"[cv2 loader failed: {e}]")

# Attempt 2: OpenEXR (pip install openexr)
if img is None:
    try:
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(exr_path)
        dw = f.header()['dataWindow']
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        R = np.frombuffer(f.channel('R', pt), dtype=np.float32).reshape(H, W)
        G = np.frombuffer(f.channel('G', pt), dtype=np.float32).reshape(H, W)
        B = np.frombuffer(f.channel('B', pt), dtype=np.float32).reshape(H, W)
        img = np.stack([R, G, B], axis=-1)
        print("[loader] OpenEXR")
    except ImportError:
        pass
    except Exception as e:
        print(f"[OpenEXR loader failed: {e}]")

if img is None:
    print("ERROR: Could not load EXR.")
    print("  Install cv2:    conda install opencv  or  pip install opencv-python")
    print("  Install OpenEXR: pip install openexr")
    sys.exit(1)

if img.ndim == 2:
    img = np.stack([img, img, img], axis=-1)


R, G, B = img[..., 0], img[..., 1], img[..., 2]
lum = 0.2126 * R + 0.7152 * G + 0.0722 * B

mean_lum = float(lum.mean())
std_lum  = float(lum.std())
min_lum  = float(lum.min())
max_lum  = float(lum.max())

print()
print(f"  Image        : {img.shape[1]} x {img.shape[0]}")
print(f"  Mean lum     : {mean_lum:.4f}   (white furnace target = 1.000)")
print(f"  Std dev      : {std_lum:.4f}   (MC noise; lower with more samples)")
print(f"  Min / Max    : {min_lum:.4f} / {max_lum:.4f}")
print()

# --- Pass / fail ---
delta = abs(mean_lum - 1.0)
if delta < 0.03:
    print(f"  PASS  mean={mean_lum:.4f}, within 3% of 1.0 — hair BCSDF conserves energy.")
elif mean_lum < 0.97:
    print(f"  FAIL  mean={mean_lum:.4f} < 0.97 — hair BCSDF is absorbing energy.")
    print("        Likely cause: residual lobe (p>=3) under-weighted in HairAp.")
else:
    print(f"  FAIL  mean={mean_lum:.4f} > 1.03 — hair BCSDF is creating energy.")
    print("        Likely cause: normalization error in HairMp or HairNp.")
