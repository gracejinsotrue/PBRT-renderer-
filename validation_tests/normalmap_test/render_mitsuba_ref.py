r"""
render_mitsuba_ref.py
Renders normal-map validation reference images using Mitsuba 3.

Generates:
  textures/wavy_normal.png    — 512×512 sinusoidal tangent-space normal map
  textures/sky_gradient.hdr   — 128×64 sky-to-ground gradient envmap (horizontally uniform)
  ref_normalmap.exr           — cube WITH  normal map  (512 spp, path tracer)
  ref_flat.exr                — cube WITHOUT normal map (512 spp, path tracer)

Both images use the same camera, envmap, diffuse BSDF, and cube geometry as the
matching nori-dxr scenes (cube_normalmap.xml / cube_flat.xml).

Usage:
    pip install mitsuba pillow          # opencv-python also works as fallback
    python render_mitsuba_ref.py

Then render the nori-dxr scenes (from this directory) and compare:

    cd C:\Users\gjin3\Desktop\nori-26sp\validation_tests\normalmap_test

    ..\..\dxr\build\Release\nori-dxr.exe cube_normalmap.xml --headless
    Copy-Item snapshot_512.exr snapshot_normalmap_512.exr
    python ..\compare_exr.py snapshot_normalmap_512.exr ref_normalmap.exr --out diff_normalmap

    ..\..\dxr\build\Release\nori-dxr.exe cube_flat.xml --headless
    Copy-Item snapshot_512.exr snapshot_flat_512.exr
    python ..\compare_exr.py snapshot_flat_512.exr ref_flat.exr --out diff_flat

Key design choices that keep the two renderers comparable
---------------------------------------------------------
* Horizontally-uniform envmap (sky_gradient.hdr):
    nori-dxr and Mitsuba 3 use different azimuth-zero conventions for
    equirectangular images, causing a ~2x brightness mismatch with sunset.hdr.
    The sky_gradient.hdr is azimuthally constant — every pixel in a row is the
    same colour — so no azimuth offset can affect results.  The sky-to-ground
    gradient still provides strong directional (elevation-dependent) lighting
    that makes normal mapping clearly visible.
* Normal map loaded LINEAR (no sRGB decode) in both renderers:
    - nori-dxr  : loadIfNotEmpty(gd.normalTexture, /*isSRGB=*/false) -> DXGI_FORMAT_R8G8B8A8_UNORM
    - Mitsuba 3 : bitmap with raw=True
* V-coordinate flip applied in both:
    - nori-dxr  : uploads UV as (u, 1-v) for non-hair meshes (DXRApp.cpp line ~1037)
    - Mitsuba 3 : OBJ loader default flip_tex_coords=True flips V in the mesh
  Both renderers therefore sample the texture and compute TBN from the same
  (u, 1-v_obj) coordinates, so the bump pattern and tangent frame match.
* TBN computed from per-triangle UV derivatives in both renderers.
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

HERE     = os.path.dirname(os.path.abspath(__file__))
CUBE_OBJ     = os.path.abspath(os.path.join(HERE, "..", "blender_default_sphere.obj"))
GRADIENT_HDR = os.path.join(HERE, "textures", "sky_gradient.hdr")
NMAP_PNG     = os.path.join(HERE, "textures", "rocky-rugged-terrain_1_normal-ogl.png")

assert os.path.exists(CUBE_OBJ), f"Missing required file: {CUBE_OBJ}"


# ---------------------------------------------------------------------------
# 1.  Generate the synthetic normal map
# ---------------------------------------------------------------------------

def generate_wavy_normal(path: str, size: int = 512, cycles: int = 4,
                          amplitude: float = 0.3) -> None:
    """
    Sinusoidal tangent-space normal map (OpenGL / DX convention):
        nx = A * sin(2*pi * u * cycles)
        ny = A * sin(2*pi * v * cycles)
        nz = sqrt(1 - nx^2 - ny^2)   (then re-normalised to unit length)

    Encoded as uint8 RGB:
        R = nx*0.5 + 0.5,  G = ny*0.5 + 0.5,  B = nz*0.5 + 0.5

    Amplitude 0.3 gives a maximum tilt of ~17.5° from the surface normal —
    clearly visible but not so extreme that edge-case TBN differences matter.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    u = np.linspace(0.0, 1.0, size, endpoint=False)
    v = np.linspace(0.0, 1.0, size, endpoint=False)
    uu, vv = np.meshgrid(u, v)

    nx = amplitude * np.sin(2.0 * np.pi * uu * cycles)
    ny = amplitude * np.sin(2.0 * np.pi * vv * cycles)
    nz = np.sqrt(np.maximum(1.0 - nx**2 - ny**2, 0.0))

    # Re-normalise (nz as computed above is already unit if nx^2+ny^2 <= 1,
    # but we re-normalise for safety after any floating-point rounding).
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= length
    ny /= length
    nz /= length

    r = np.clip(nx * 0.5 + 0.5, 0.0, 1.0)
    g = np.clip(ny * 0.5 + 0.5, 0.0, 1.0)
    b = np.clip(nz * 0.5 + 0.5, 0.0, 1.0)

    img = (np.stack([r, g, b], axis=-1) * 255.0).round().astype(np.uint8)

    try:
        from PIL import Image
        Image.fromarray(img, "RGB").save(path)
    except ImportError:
        import cv2
        # cv2 uses BGR channel order
        cv2.imwrite(path, img[:, :, ::-1])

    print(f"[setup] normal map  : {path}  ({size}x{size}, {cycles} cycles, A={amplitude})")


# rocky-rugged-terrain_1_normal-ogl.png is already present in textures/.
assert os.path.exists(NMAP_PNG), f"Missing: {NMAP_PNG}"
print(f"[setup] normal map  : {NMAP_PNG}")


# ---------------------------------------------------------------------------
# 1b.  Generate the sky-to-ground gradient envmap (Radiance HDR)
# ---------------------------------------------------------------------------

def _rgbe(r: float, g: float, b: float):
    """Encode one RGB triplet as a 4-tuple of uint8 (RGBE)."""
    max_c = max(r, g, b)
    if max_c < 1e-32:
        return (0, 0, 0, 0)
    m, e = math.frexp(max_c)   # max_c = m * 2^e, m in [0.5, 1.0)
    v = m * 256.0 / max_c      # = 256 / 2^e
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
            # collect a non-run region
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
    Write a Radiance HDR file containing a sky-to-ground gradient.

    The image is HORIZONTALLY UNIFORM — every pixel in a row has the same
    colour — so neither renderer's azimuth convention matters.  The vertical
    (elevation) variation gives strong directional lighting that makes normal
    mapping clearly visible on side faces of the cube.

    Keyframes (V = 0 is the top of the image = +Y direction = sky):
        V = 0.00  sky blue   (2.5, 2.8, 4.0)
        V = 0.45  horizon    (1.5, 1.1, 0.6)
        V = 0.55  gnd-top    (0.30, 0.20, 0.10)
        V = 1.00  ground     (0.05, 0.04, 0.02)
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

            # New-style scanline header
            f.write(bytes([2, 2, (W >> 8) & 0xFF, W & 0xFF]))

            # All W pixels in this row share the same RGBE value → one long run each channel
            for ch_val in (rv, gv, bv, ev):
                row_data = [ch_val] * W
                f.write(_rle_channel(row_data))

    print(f"[setup] sky gradient: {path}  ({W}x{H})")


generate_sky_gradient_hdr(GRADIENT_HDR)


# ---------------------------------------------------------------------------
# 2.  Shared scene builder
# ---------------------------------------------------------------------------

CAMERA = mi.ScalarTransform4f.look_at(
    origin=[3, 2, 4], target=[0, 0, 0], up=[0, 1, 0]
)
W, H, SPP = 512, 512, 512


def build_scene(with_normalmap: bool) -> dict:
    base_bsdf = {
        "type": "diffuse",
        "reflectance": {"type": "rgb", "value": [0.7, 0.7, 0.7]},
    }

    if with_normalmap:
        bsdf = {
            "type": "normalmap",
            # raw=True: treat the PNG as a linear-space bitmap (no sRGB gamma
            # decode), matching nori-dxr which loads the normal texture with
            # isSRGB=false -> DXGI_FORMAT_R8G8B8A8_UNORM.
            "normalmap": {
                "type": "bitmap",
                "filename": NMAP_PNG,
                "raw": True,
            },
            "bsdf": base_bsdf,
        }
    else:
        bsdf = base_bsdf

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
        "cube": {
            "type": "obj",
            "filename": CUBE_OBJ,
            # flip_tex_coords=True (Mitsuba default) flips V, matching
            # nori-dxr's upload-time V flip (1.0f - UV(1,v)).
            "bsdf": bsdf,
        },
    }


# ---------------------------------------------------------------------------
# 3.  Render both scenes
# ---------------------------------------------------------------------------

TESTS = [
    ("normalmap", True,  "cube with wavy normal map"),
    ("flat",      False, "cube without normal map (reference baseline)"),
]

for name, with_nm, desc in TESTS:
    out = os.path.join(HERE, f"ref_{name}.exr")
    print(f"[render] {name:12s}  {W}x{H} @ {SPP} spp  ({desc})")
    scene = mi.load_dict(build_scene(with_nm))
    img   = mi.render(scene)
    mi.Bitmap(img).write(out)
    print(f"         -> {out}")

print("\nAll reference renders complete.")
print()
print("Compare against nori-dxr (run from this directory):")
print("  ..\\..\\dxr\\build\\Release\\nori-dxr.exe cube_normalmap.xml --headless")
print("  Copy-Item snapshot_512.exr snapshot_normalmap_512.exr")
print("  python ..\\compare_exr.py snapshot_normalmap_512.exr ref_normalmap.exr --out diff_normalmap")
print()
print("  ..\\..\\dxr\\build\\Release\\nori-dxr.exe cube_flat.xml --headless")
print("  Copy-Item snapshot_512.exr snapshot_flat_512.exr")
print("  python ..\\compare_exr.py snapshot_flat_512.exr ref_flat.exr    --out diff_flat")
