r"""
render_mitsuba_ref.py
Renders albedo-texture validation reference images using Mitsuba 3.

Generates:
  textures/sky_gradient.hdr   — 128×64 sky-to-ground gradient envmap (horizontally uniform)
  ref_textured.exr            — sphere WITH  albedo texture  (512 spp, path tracer)
  ref_flat.exr                — sphere WITHOUT texture, flat grey (512 spp, path tracer)

Both images use the same camera, envmap, and sphere geometry as the matching
nori-dxr scenes (sphere_textured.xml / sphere_flat.xml).

Usage:
    pip install mitsuba pillow
    python render_mitsuba_ref.py

Then render the nori-dxr scenes (from this directory) and compare:

    cd C:\Users\gjin3\Desktop\nori-26sp\validation_tests\texture_test

    ..\..\dxr\build\Release\nori-dxr.exe sphere_textured.xml --headless
    Copy-Item snapshot_512.exr snapshot_textured_512.exr
    python ..\compare_exr.py snapshot_textured_512.exr ref_textured.exr --out diff_textured

    ..\..\dxr\build\Release\nori-dxr.exe sphere_flat.xml --headless
    Copy-Item snapshot_512.exr snapshot_flat_512.exr
    python ..\compare_exr.py snapshot_flat_512.exr ref_flat.exr --out diff_flat

Key design choices that keep the two renderers comparable
---------------------------------------------------------
* Horizontally-uniform envmap (sky_gradient.hdr):
    nori-dxr and Mitsuba 3 use different azimuth-zero conventions for
    equirectangular images, causing a ~2× brightness mismatch with sunset.hdr.
    The sky_gradient.hdr is azimuthally constant so no azimuth offset matters.
* Albedo texture loaded as sRGB in BOTH renderers:
    - nori-dxr  : loadIfNotEmpty(gd.albedoTexture, /*isSRGB=*/true)
                  -> DXGI_FORMAT_R8G8B8A8_UNORM_SRGB (hardware sRGB decode)
    - Mitsuba 3 : bitmap with raw=False (default) — Mitsuba also applies sRGB
                  gamma decode, matching nori-dxr's hardware sRGB path.
* V-coordinate flip applied in both:
    - nori-dxr  : uploads UV as (u, 1-v) for all non-hair meshes (DXRApp.cpp)
    - Mitsuba 3 : OBJ loader default flip_tex_coords=True flips V in the mesh
  Both renderers therefore sample the texture at the same (u, 1-v_obj) coords.
"""

import math
import os

import numpy as np

try:
    import mitsuba as mi
except ImportError:
    raise SystemExit("Mitsuba 3 not found. Install with: pip install mitsuba")

try:
    mi.set_variant("cuda_ad_rgb")
except Exception:
    mi.set_variant("llvm_ad_rgb")
print(f"[mitsuba] variant = {mi.variant()}")

HERE         = os.path.dirname(os.path.abspath(__file__))
SPHERE_OBJ   = os.path.abspath(os.path.join(HERE, "..", "blender_default_sphere.obj"))
GRADIENT_HDR = os.path.join(HERE, "textures", "sky_gradient.hdr")
ALBEDO_PNG   = os.path.join(HERE, "textures", "rocky-rugged-terrain_1_albedo.png")

assert os.path.exists(SPHERE_OBJ), f"Missing required file: {SPHERE_OBJ}"
assert os.path.exists(ALBEDO_PNG),  f"Missing required file: {ALBEDO_PNG}"


# ---------------------------------------------------------------------------
# 1.  Generate the sky-to-ground gradient envmap (Radiance HDR)
# ---------------------------------------------------------------------------

def _rgbe(r: float, g: float, b: float):
    """Encode one RGB triplet as a 4-tuple of uint8 (RGBE)."""
    max_c = max(r, g, b)
    if max_c < 1e-32:
        return (0, 0, 0, 0)
    m, e = math.frexp(max_c)
    v = m * 256.0 / max_c
    return (
        min(255, int(r * v)),
        min(255, int(g * v)),
        min(255, int(b * v)),
        max(0, min(255, e + 128)),
    )


def _rle_channel(data: list) -> bytes:
    """New-style per-channel RLE used in Radiance HDR scanlines."""
    out = []
    i = 0
    n = len(data)
    while i < n:
        j = i + 1
        while j < n and data[j] == data[i] and (j - i) < 127:
            j += 1
        run = j - i
        if run > 2 or i + run >= n:
            out.append(128 + run)
            out.append(data[i])
            i = j
        else:
            k = j
            while (k < n
                   and (k + 1 >= n or data[k] != data[k + 1])
                   and (k - i) < 127):
                k += 1
            non_run_len = k - i
            out.append(non_run_len)
            out.extend(data[i:k])
            i = k
    return bytes(out)


def generate_sky_gradient_hdr(path: str, W: int = 128, H: int = 64) -> None:
    """
    Write a Radiance HDR containing a sky-to-ground gradient.
    HORIZONTALLY UNIFORM — every pixel in a row has the same colour.
    """
    keyframes = [
        (0.00, (2.5,  2.8,  4.0 )),
        (0.45, (1.5,  1.1,  0.6 )),
        (0.55, (0.30, 0.20, 0.10)),
        (1.00, (0.05, 0.04, 0.02)),
    ]

    def lerp_color(v):
        for k in range(len(keyframes) - 1):
            v0, c0 = keyframes[k]
            v1, c1 = keyframes[k + 1]
            if v0 <= v <= v1:
                t = (v - v0) / (v1 - v0) if v1 > v0 else 0.0
                return tuple(c0[i] * (1 - t) + c1[i] * t for i in range(3))
        return keyframes[-1][1]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        header = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n" % (H, W)
        f.write(header.encode("ascii"))
        for y in range(H):
            v = (y + 0.5) / H
            cr, cg, cb = lerp_color(v)
            rv, gv, bv, ev = _rgbe(cr, cg, cb)
            f.write(bytes([2, 2, (W >> 8) & 0xFF, W & 0xFF]))
            for ch_val in (rv, gv, bv, ev):
                f.write(_rle_channel([ch_val] * W))

    print(f"[setup] sky gradient : {path}  ({W}x{H})")


generate_sky_gradient_hdr(GRADIENT_HDR)


# ---------------------------------------------------------------------------
# 2.  Shared scene builder
# ---------------------------------------------------------------------------

CAMERA = mi.ScalarTransform4f.look_at(
    origin=[3, 2, 4], target=[0, 0, 0], up=[0, 1, 0]
)
W, H, SPP = 512, 512, 512


def build_scene(with_texture: bool) -> dict:
    if with_texture:
        # raw=False (default): Mitsuba applies sRGB gamma decode,
        # matching nori-dxr's DXGI_FORMAT_R8G8B8A8_UNORM_SRGB hardware path.
        bsdf = {
            "type": "diffuse",
            "reflectance": {
                "type": "bitmap",
                "filename": ALBEDO_PNG,
                "raw": False,
            },
        }
    else:
        bsdf = {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": [0.7, 0.7, 0.7]},
        }

    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 16},
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": CAMERA,
            "film": {
                "type": "hdrfilm",
                "width": W, "height": H,
                "pixel_format": "rgb",
                "component_format": "float32",
            },
            "sampler": {"type": "independent", "sample_count": SPP},
        },
        "envmap": {"type": "envmap", "filename": GRADIENT_HDR},
        "sphere": {
            "type": "obj",
            "filename": SPHERE_OBJ,
            # flip_tex_coords=True (Mitsuba OBJ default) flips V, matching
            # nori-dxr's upload-time V flip (1.0f - UV(1,v) in DXRApp.cpp).
            "bsdf": bsdf,
        },
    }


# ---------------------------------------------------------------------------
# 3.  Render both scenes
# ---------------------------------------------------------------------------

TESTS = [
    ("textured", True,  "sphere with albedo texture"),
    ("flat",     False, "sphere without texture — flat grey baseline"),
]

for name, with_tex, desc in TESTS:
    out = os.path.join(HERE, f"ref_{name}.exr")
    print(f"[render] {name:10s}  {W}x{H} @ {SPP} spp  ({desc})")
    scene = mi.load_dict(build_scene(with_tex))
    img   = mi.render(scene)
    mi.Bitmap(img).write(out)
    print(f"         -> {out}")

print("\nAll reference renders complete.")
print()
print("Next steps (run from this directory):")
print("  ..\\..\\dxr\\build\\Release\\nori-dxr.exe sphere_textured.xml --headless")
print("  Copy-Item snapshot_512.exr snapshot_textured_512.exr")
print("  ..\\..\\dxr\\build\\Release\\nori-dxr.exe sphere_flat.xml --headless")
print("  Copy-Item snapshot_512.exr snapshot_flat_512.exr")
print("  python ..\\compare_exr.py snapshot_flat_512.exr ref_flat.exr --out diff_flat")
print("  python ..\\compare_exr.py snapshot_textured_512.exr ref_textured.exr --out diff_textured")
print("  python ..\\compare_exr.py snapshot_textured_512.exr snapshot_flat_512.exr --out diff_tex_vs_flat")
