r"""
validate_volumetrics.py
=======================
GPU-side quantitative validation tests for the volumetric rendering
implementation in  dxr/shaders/Volume.hlsl  and  dxr/shaders/Shaders.hlsl.

How it works
------------
Each test is a carefully designed DXR scene whose correct pixel value is
known analytically.  We run nori-dxr.exe on the scene, wait for it to
auto-save an EXR (it does this when m_targetSamples frames accumulate),
then compare the mean luminance to the analytic expectation.

Tests
-----
1. White furnace (energy conservation)
   Scene  : pure scattering volume (sigmaA=0, sigmaS=1, g=0), white env.
   Physics: no absorption  →  every photon survives  →  E[pixel] = 1.0.
   Catches: energy leaks in NullCollisionFreeFlight weight arithmetic,
            incorrect throughput accumulation, missing null-collision terms.

2. Beer-Lambert slab (transmittance)
   Scene  : pure absorbing slab (sigmaA=2, sigmaS=0), thickness d=1, white env.
   Physics: E[pixel] = exp(-sigmaA * d) = exp(-2) ≈ 0.13534.
   Catches: biased free-flight sampling, wrong extinction in homogeneous path,
            wrong albedo=0 path-kill logic.

Usage
-----
    python validate_volumetrics.py [options]

    --dxr-exe PATH   Path to nori-dxr.exe.
                     Default: dxr\build\Release\nori-dxr.exe
    --timeout N      Seconds to wait per render (default: 300).

Requirements
------------
    pip install numpy opencv-python
    (or pip install numpy openexr as a fallback)
"""

import argparse
import math
import os
import struct
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(REPO, "validation_tests", "volumetric_tests")

DXR_DEFAULT = os.path.join(REPO, "dxr", "build", "Release", "nori-dxr.exe")


# ---------------------------------------------------------------------------
# White-furnace HDR generator
# ---------------------------------------------------------------------------

def make_white_furnace_hdr(out_path):
    """
    Write a 2x2 constant-white (RGB=1.0) RGBE HDR file.
    Encoding: mantissa = 0.5, exponent byte = 129  →  decoded = 0.5 * 2^(129-128) = 1.0
    """
    W, H = 2, 2
    header = (
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n"
        b"\n"
        + f"-Y {H} +X {W}\n".encode()
    )
    pixel = bytes([128, 128, 128, 129])  # R=G=B=1.0 in RGBE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(pixel * (W * H))


# ---------------------------------------------------------------------------
# EXR loader (tries cv2, then OpenEXR)
# ---------------------------------------------------------------------------

def load_exr(path):
    """Load an EXR file; return (H, W, 3) float32 numpy array."""
    import numpy as np

    # cv2 is the most reliable EXR reader on Windows
    try:
        import cv2
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is not None and raw.ndim == 3:
            return raw[..., ::-1].astype(np.float32)   # BGR -> RGB
        if raw is not None:
            return np.stack([raw] * 3, axis=-1).astype(np.float32)
    except Exception:
        pass

    # Fallback: OpenEXR
    try:
        import OpenEXR, Imath
        f = OpenEXR.InputFile(path)
        dw = f.header()["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        import numpy as np2
        R = np2.frombuffer(f.channel("R", pt), dtype="float32").reshape(H, W)
        G = np2.frombuffer(f.channel("G", pt), dtype="float32").reshape(H, W)
        B = np2.frombuffer(f.channel("B", pt), dtype="float32").reshape(H, W)
        return np2.stack([R, G, B], axis=-1)
    except ImportError:
        pass
    except Exception:
        pass

    raise RuntimeError(
        f"Cannot load '{path}'.\n"
        "Install: pip install opencv-python   or   pip install openexr"
    )


# ---------------------------------------------------------------------------
# Run one DXR test
# ---------------------------------------------------------------------------

def run_dxr_test(dxr_exe, xml_path, label, expected, tolerance, sample_count, timeout):
    """
    Run nori-dxr.exe on xml_path, wait for snapshot_{sample_count}.exr,
    then check mean luminance against expected ± tolerance.
    Returns (passed: bool, detail: str).
    """
    import numpy as np

    exr_path = os.path.join(os.path.dirname(xml_path), f"snapshot_{sample_count}.exr")

    # Remove stale output so we can detect a fresh write
    if os.path.exists(exr_path):
        os.remove(exr_path)

    try:
        proc = subprocess.Popen(
            [dxr_exe, xml_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, f"nori-dxr.exe not found: {dxr_exe}"

    start = time.time()
    while True:
        elapsed = time.time() - start
        ret = proc.poll()
        if ret is not None:
            break
        if elapsed > timeout:
            proc.kill()
            return False, f"timeout after {timeout} s"
        time.sleep(1.0)

    if not os.path.exists(exr_path):
        err = proc.stderr.read().decode(errors="replace").strip()
        return False, f"EXR not saved (exit {ret}). stderr: {err[:200]}"

    try:
        img = load_exr(exr_path)
    except RuntimeError as e:
        return False, str(e)

    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    mean = float(lum.mean())
    std  = float(lum.std())

    passed = abs(mean - expected) <= tolerance
    detail = (
        f"mean={mean:.4f}  expected={expected:.4f}  tol=±{tolerance:.4f}  std={std:.4f}"
    )
    return passed, detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU volumetric validation tests for nori-dxr."
    )
    parser.add_argument("--dxr-exe", default=DXR_DEFAULT,
                        help="Path to nori-dxr.exe")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Max seconds to wait per render (default: 300)")
    args = parser.parse_args()

    if not os.path.exists(args.dxr_exe):
        print(f"ERROR: nori-dxr.exe not found at '{args.dxr_exe}'")
        print("Build it first:")
        print("  & 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe'")
        print("    dxr\\build\\nori-dxr.vcxproj /p:Configuration=Release /m")
        return 1

    tests = [
        {
            "label"   : "White furnace  (sigmaA=0, sigmaS=1, g=0  →  mean=1.0)",
            "dir"     : os.path.join(TESTS, "white_furnace"),
            "xml"     : os.path.join(TESTS, "white_furnace", "vol_white_furnace.xml"),
            "expected": 1.0,
            "tol"     : 0.04,   # generous: 512 spp Monte Carlo noise on a 64x64 image
            "spp"     : 512,
        },
        {
            "label"   : f"Beer-Lambert slab (sigmaA=2, d=1  →  mean=exp(-2)≈{math.exp(-2):.4f})",
            "dir"     : os.path.join(TESTS, "beer_lambert"),
            "xml"     : os.path.join(TESTS, "beer_lambert", "vol_beer_lambert.xml"),
            "expected": math.exp(-2.0),
            "tol"     : 0.01,
            "spp"     : 512,
        },
    ]

    print("\n=== Volumetric GPU Validation Tests ===\n")
    print(f"  nori-dxr: {args.dxr_exe}\n")

    results = []

    for t in tests:
        # Generate the white-furnace HDR in the test's textures/ subfolder.
        # The DXR app looks for  textures/white_furnace.hdr  relative to the
        # scene XML's directory (hardcoded in DXRApp::CreateTextures).
        hdr = os.path.join(t["dir"], "textures", "white_furnace.hdr")
        if not os.path.exists(hdr):
            print(f"  Generating {hdr}")
            make_white_furnace_hdr(hdr)

        print(f"  Running: {t['label']}")
        print(f"           ({t['spp']} spp, may take 1-3 min) ...", flush=True)

        passed, detail = run_dxr_test(
            dxr_exe      = args.dxr_exe,
            xml_path     = t["xml"],
            label        = t["label"],
            expected     = t["expected"],
            tolerance    = t["tol"],
            sample_count = t["spp"],
            timeout      = args.timeout,
        )
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {detail}\n")
        results.append((t["label"], passed, detail))

    # Summary
    print("=== Summary ===\n")
    passed_count = sum(1 for _, p, _ in results if p)
    for label, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {label}")

    print(f"\n  {passed_count} / {len(results)} tests passed.")

    if passed_count < len(results):
        print("\n  FAILED tests indicate a bug in Volume.hlsl or Shaders.hlsl.")
        print("  Check the mean vs expected values above for the direction of the error.")
        return 1

    print("\n  All tests PASSED.  GPU volumetric rendering is quantitatively correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
