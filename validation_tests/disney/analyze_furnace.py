r"""
analyze_furnace.py
Disney white-furnace analyzer. For each furnace_*.xml render, masks the sphere
pixels with an analytic ray-sphere test, reports mean reflectance, and (if a
matching ref_*.exr is present) shows our_mean / ref_mean per channel.

The white furnace doesn't say "Disney must integrate to 1" — Burley diffuse and
single-scatter GGX both deviate from 1 by design. It DOES say "our deviation
must match Mitsuba's deviation". That's the apples-to-apples test.

Usage:
    python analyze_furnace.py snapshot_512.exr [ref_furnace_*.exr]
"""

import sys
import os

# Must be set before cv2 import — conda OpenCV ships EXR disabled by default.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np


def load_exr(path):
    import cv2
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if raw is None:
        print(f"ERROR: could not load {path}")
        sys.exit(1)
    return raw[..., :3][..., ::-1].astype(np.float32)


def sphere_mask(W, H, fov_deg=45.0, cam_pos=(0, 0, 4), sphere_R=1.0):
    """Pixels whose camera ray hits the unit sphere at origin.
    Matches the cmp_*.xml camera: look_at origin=(0,0,4) target=(0,0,0) up=(0,1,0)."""
    fy = np.tan(np.deg2rad(fov_deg) / 2.0)
    fx = fy * (W / H)
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    u = (i + 0.5) / W * 2.0 - 1.0
    v = 1.0 - (j + 0.5) / H * 2.0
    dx, dy, dz = u * fx, v * fy, -np.ones_like(u)
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= n; dy /= n; dz /= n
    cx, cy, cz = cam_pos
    b = cx * dx + cy * dy + cz * dz
    c = cx * cx + cy * cy + cz * cz - sphere_R * sphere_R
    return (b * b - c) > 0.0


def stats_on_mask(img, mask):
    pixels = img[mask]
    return {
        "n": int(mask.sum()),
        "mean": pixels.mean(axis=0),
        "min": pixels.min(axis=0),
        "max": pixels.max(axis=0),
        "lum_mean": float((0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]).mean()),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ours_path = sys.argv[1]
    ref_path = sys.argv[2] if len(sys.argv) > 2 else None

    ours = load_exr(ours_path)
    H, W = ours.shape[:2]
    mask = sphere_mask(W, H)

    print(f"Image          : {W} x {H}    sphere pixels = {int(mask.sum())} / {W*H}")
    print()

    s = stats_on_mask(ours, mask)
    print(f"[ours]   {ours_path}")
    print(f"  sphere mean R,G,B = {s['mean']}")
    print(f"  luminance mean    = {s['lum_mean']:.4f}")
    print(f"  range R,G,B  min={s['min']}  max={s['max']}")
    print()

    if ref_path:
        ref = load_exr(ref_path)
        if ref.shape != ours.shape:
            print(f"WARN: shape mismatch ours={ours.shape} ref={ref.shape}; reslicing mask")
            mask_ref = sphere_mask(ref.shape[1], ref.shape[0])
            r = stats_on_mask(ref, mask_ref)
        else:
            r = stats_on_mask(ref, mask)

        print(f"[ref]    {ref_path}")
        print(f"  sphere mean R,G,B = {r['mean']}")
        print(f"  luminance mean    = {r['lum_mean']:.4f}")
        print()

        ratio = s["mean"] / np.maximum(r["mean"], 1e-10)
        lum_ratio = s["lum_mean"] / max(r["lum_mean"], 1e-10)
        print(f"[ratio = ours / ref]")
        print(f"  R = {ratio[0]:.4f}    G = {ratio[1]:.4f}    B = {ratio[2]:.4f}")
        print(f"  Luminance ratio   = {lum_ratio:.4f}")
        print()

        # Verdict
        max_dev = max(abs(ratio - 1.0).max(), abs(lum_ratio - 1.0))
        if max_dev < 0.02:
            print(f"  PASS  max channel deviation = {max_dev*100:.2f}% (<2%)")
        elif max_dev < 0.05:
            print(f"  CLOSE max channel deviation = {max_dev*100:.2f}% (<5%); investigate if cmp_*.xml also off")
        else:
            print(f"  FAIL  max channel deviation = {max_dev*100:.2f}% (>=5%); BRDF math diverges from Mitsuba")


if __name__ == "__main__":
    main()
