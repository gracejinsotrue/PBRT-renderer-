r"""
compare_exr.py
Apples-to-apples comparison of two linear-HDR EXR renders (e.g. nori-dxr vs Mitsuba).

Both EXR files MUST be linear scene-referred HDR (no tonemap, no gamma).
nori-dxr's SaveSnapshotEXR() writes such an EXR (raw g_accum / sampleCount).
Mitsuba 3's mi.render() + mi.Bitmap(img).write('x.exr') writes such an EXR.

Usage:
    python compare_exr.py <ours.exr> <reference.exr> [--out diff_dir] [--exposure 0]

Outputs (when --out is provided):
    diff_dir/ours_srgb.png         tonemapped (Reinhard + sRGB) version of ours
    diff_dir/ref_srgb.png          same tonemap applied to reference
    diff_dir/abs_diff.png          per-pixel |ours - ref| visualised
    diff_dir/false_color.png       false-color signed diff (red = ours brighter, blue = ref brighter)

Stats printed:
    - per-channel mean, max, ratio
    - linear RMSE
    - relMSE (Mitsuba-style: variance-aware error metric)
    - tonemapped PSNR (for sanity check, not the truth)

Both Reinhard tonemap and PSNR computations use the SAME curve as nori-dxr's
display path, so when the renders truly match, the false-color image is mid-gray.
"""

import argparse
import os
import sys

# Must be set before cv2 import — conda OpenCV ships EXR disabled by default.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np


def load_exr(path):
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required. pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    # OpenEXR support requires the build-time flag; cv2 from pip is built with it.
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if raw is None:
        print(f"ERROR: could not load {path}", file=sys.stderr)
        sys.exit(1)
    if raw.ndim != 3 or raw.shape[2] < 3:
        print(f"ERROR: expected 3-channel EXR, got shape {raw.shape}", file=sys.stderr)
        sys.exit(1)
    # OpenCV stores BGR, convert to RGB
    return raw[..., :3][..., ::-1].astype(np.float32)


def reinhard_srgb(img, exposure_ev=0.0):
    """Same display curve nori-dxr uses internally so the eye-test is apples-to-apples."""
    x = img * (2.0 ** exposure_ev)
    x = np.maximum(x, 0.0)
    x = x / (1.0 + x)
    return np.power(x, 1.0 / 2.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", help="EXR from nori-dxr (linear HDR)")
    ap.add_argument("reference", help="EXR from Mitsuba / ground truth (linear HDR)")
    ap.add_argument("--out", default=None, help="Output directory for diff PNGs")
    ap.add_argument("--exposure", type=float, default=0.0, help="Display EV stops for PNG previews only")
    args = ap.parse_args()

    ours = load_exr(args.ours)
    ref = load_exr(args.reference)

    if ours.shape != ref.shape:
        print(f"ERROR: shape mismatch  ours={ours.shape}  ref={ref.shape}", file=sys.stderr)
        sys.exit(1)

    H, W, _ = ours.shape
    print(f"Resolution      : {W} x {H}")
    print(f"Ours  range     : [{ours.min():.4f}, {ours.max():.4f}]   mean={ours.mean():.4f}")
    print(f"Ref   range     : [{ref.min():.4f}, {ref.max():.4f}]   mean={ref.mean():.4f}")
    print()

    # Per-channel stats in LINEAR space (this is the truth)
    chan = ["R", "G", "B"]
    print(f"{'channel':>8}  {'ours_mean':>10}  {'ref_mean':>10}  {'ratio':>8}  {'mean|diff|':>10}  {'max|diff|':>10}")
    for c in range(3):
        ours_mean = float(ours[..., c].mean())
        ref_mean = float(ref[..., c].mean())
        ratio = ours_mean / ref_mean if ref_mean > 1e-10 else float("inf")
        diff = np.abs(ours[..., c] - ref[..., c])
        print(f"{chan[c]:>8}  {ours_mean:>10.4f}  {ref_mean:>10.4f}  {ratio:>8.4f}  {diff.mean():>10.4f}  {diff.max():>10.4f}")
    print()

    # Linear RMSE — primary scalar metric
    rmse = float(np.sqrt(((ours - ref) ** 2).mean()))
    print(f"Linear RMSE     : {rmse:.5f}")

    # Mitsuba-style relMSE: ((ours - ref)^2 / (ref^2 + eps)).mean()
    rel_mse = float((((ours - ref) ** 2) / (ref * ref + 1e-2)).mean())
    print(f"relMSE          : {rel_mse:.5f}    (variance-normalised, lower is better)")

    # Per-pixel luminance ratio at sphere center hints at F0 saturation issues
    cy, cx = H // 2, W // 2
    print(f"Center pixel    : ours={ours[cy, cx]}  ref={ref[cy, cx]}")
    print()

    # Tonemapped PSNR — secondary sanity check
    ours_srgb = reinhard_srgb(ours, args.exposure)
    ref_srgb = reinhard_srgb(ref, args.exposure)
    mse_srgb = float(((ours_srgb - ref_srgb) ** 2).mean())
    if mse_srgb > 1e-10:
        psnr = 10.0 * np.log10(1.0 / mse_srgb)
        print(f"Tonemap PSNR    : {psnr:.2f} dB    (>30 dB = visually close; <20 dB = real differences)")
    else:
        print("Tonemap PSNR    : inf (images are byte-identical after tonemap)")

    # Image outputs
    if args.out:
        try:
            import cv2
        except ImportError:
            print("opencv-python required to write PNGs", file=sys.stderr)
            sys.exit(1)

        os.makedirs(args.out, exist_ok=True)

        def save_rgb(name, img_rgb):
            img8 = np.clip(img_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out, name), img8[..., ::-1])  # RGB->BGR

        save_rgb("ours_srgb.png", ours_srgb)
        save_rgb("ref_srgb.png", ref_srgb)
        save_rgb("abs_diff.png", np.clip(np.abs(ours - ref), 0, 1))

        # False color: per-channel signed diff scaled 5x and remapped to mid-gray
        signed = (ours - ref) * 5.0
        false_color = np.clip(0.5 + 0.5 * signed, 0, 1)
        save_rgb("false_color.png", false_color)

        print()
        print(f"Wrote PNGs to {args.out}/")


if __name__ == "__main__":
    main()
