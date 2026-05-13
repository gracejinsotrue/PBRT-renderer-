r"""
render_mitsuba_ref.py
Renders IBL (image-based lighting) validation reference images using Mitsuba 3.

Generates:
  ref_ibl.exr          — diffuse sphere lit by sunset.hdr envmap (512 spp)
  ref_mirror.exr        — mirror sphere lit by sunset.hdr (per-direction envmap lookup)
  ref_furnace_ibl.exr   — white diffuse sphere under constant white envmap (energy conservation)

Azimuth convention fix
-----------------------
nori-dxr evaluates the envmap with:
    phi = atan2(d.z, d.x)   ->  U=0 maps to the +X direction
Mitsuba evaluates it with:
    phi = atan2(-d.z, -d.x) ->  U=0 maps to the -X direction
(i.e., a 180-degree azimuth offset)

For a horizontally-uniform HDR (like sky_gradient) this doesn't matter.
For a real HDR (like sunset.hdr) it would cause a brightness mismatch
wherever the illumination is azimuthally non-uniform.

Fix: apply a 180-degree Y-axis rotation to Mitsuba's envmap via `to_world`.
This shifts Mitsuba's U=0 from -X to +X, making both renderers sample the
same hemisphere for every outgoing direction.

Validation logic
-----------------
1. ref_ibl.exr      — Mitsuba with sunset.hdr + 180° rotation correction
2. nori-dxr renders sphere_ibl.xml --headless -> snapshot_ibl_512.exr
3. compare_exr.py snapshot_ibl_512.exr ref_ibl.exr --out diff_ibl
   Expected: RMSE < 0.05, PSNR > 35 dB  (near-perfect, pure IBL scene)

Usage:
    cd C:\Users\gjin3\Desktop\nori-26sp\validation_tests\ibl_test
    python render_mitsuba_ref.py
    ..\..\dxr\build\Release\nori-dxr.exe sphere_ibl.xml --headless
    Copy-Item snapshot_512.exr snapshot_ibl_512.exr
    python ..\compare_exr.py snapshot_ibl_512.exr ref_ibl.exr --out diff_ibl
"""

import os

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
SUNSET_HDR   = os.path.join(HERE, "textures", "sunset.hdr")
WHITE_HDR    = os.path.abspath(os.path.join(HERE, "..", "hair_furnace_test", "textures", "white_furnace.hdr"))

assert os.path.exists(SPHERE_OBJ), f"Missing: {SPHERE_OBJ}"
assert os.path.exists(SUNSET_HDR), f"Missing: {SUNSET_HDR}"
assert os.path.exists(WHITE_HDR),  f"Missing: {WHITE_HDR}"

# Mitsuba's envmap plugin requires >= 2x3 resolution.
# white_furnace.hdr is 2x2, so we generate a 4x4 RGBE white HDR for Mitsuba.
WHITE_HDR_4X4 = os.path.join(HERE, "white_4x4.hdr")
if not os.path.exists(WHITE_HDR_4X4):
    W4, H4 = 4, 4
    header = (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {H4} +X {W4}\n".encode())
    pixel  = bytes([128, 128, 128, 129])   # (128/256) * 2^(129-128) = 1.0 in RGBE
    with open(WHITE_HDR_4X4, "wb") as f:
        f.write(header + pixel * (W4 * H4))
    print(f"[setup] wrote {WHITE_HDR_4X4}")

# ---------------------------------------------------------------------------
# Shared camera / resolution
# ---------------------------------------------------------------------------

CAMERA = mi.ScalarTransform4f.look_at(
    origin=[3, 2, 4], target=[0, 0, 0], up=[0, 1, 0]
)
W, H = 512, 512

def make_scene(bsdf_dict, envmap_path, use_rotation=True, spp=512):
    """Build a Mitsuba scene dict with the given BSDF and envmap."""
    envmap_entry = {"type": "envmap", "filename": envmap_path}
    if use_rotation:
        envmap_entry["to_world"] = mi.ScalarTransform4f.rotate([0, 1, 0], 270)
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
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "envmap": envmap_entry,
        "sphere": {
            "type": "obj",
            "filename": SPHERE_OBJ,
            "bsdf": bsdf_dict,
        },
    }


# ---------------------------------------------------------------------------
# Test 1: diffuse sphere, sunset.hdr  (original test)
# ---------------------------------------------------------------------------
scene_dict = make_scene(
    bsdf_dict={"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.7, 0.7, 0.7]}},
    envmap_path=SUNSET_HDR,
    use_rotation=True,
    spp=512,
)
out = os.path.join(HERE, "ref_ibl.exr")
print(f"[render] diffuse IBL  {W}x{H} @ 512 spp  (sunset.hdr)")
scene = mi.load_dict(scene_dict)
img   = mi.render(scene)
mi.Bitmap(img).write(out)
print(f"         -> {out}")

# ---------------------------------------------------------------------------
# Test 2: mirror sphere, sunset.hdr
# Every pixel reflects a specific envmap direction. Tests per-direction UV
# correctness and the MIS code path for specular BSDFs.
# Mitsuba equivalent: smooth conductor with specular_reflectance overridden to
# (1,1,1) so Fresnel = 1 everywhere, matching nori-dxr's perfect mirror BSDF.
# ---------------------------------------------------------------------------
scene_mirror = make_scene(
    bsdf_dict={
        "type": "conductor",
        "specular_reflectance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
    },
    envmap_path=SUNSET_HDR,
    use_rotation=True,
    spp=512,
)
out_mirror = os.path.join(HERE, "ref_mirror.exr")
print(f"[render] mirror IBL   {W}x{H} @ 512 spp  (sunset.hdr)")
scene = mi.load_dict(scene_mirror)
img   = mi.render(scene)
mi.Bitmap(img).write(out_mirror)
print(f"         -> {out_mirror}")

# ---------------------------------------------------------------------------
# Test 3: white diffuse sphere, constant white envmap — furnace test.
# With L_i = 1 everywhere and albedo = 1, rendering equation gives L_o = 1.
# The sphere should be invisible against the background. Tests that the CDF
# samples the full sphere of directions without bias and the PDF is normalised.
# Rotation is irrelevant for a constant envmap.
# ---------------------------------------------------------------------------
scene_furnace = make_scene(
    bsdf_dict={"type": "diffuse", "reflectance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}},
    envmap_path=WHITE_HDR_4X4,
    use_rotation=False,
    spp=512,
)
out_furnace = os.path.join(HERE, "ref_furnace_ibl.exr")
print(f"[render] furnace IBL  {W}x{H} @ 512 spp  (white_furnace.hdr)")
scene = mi.load_dict(scene_furnace)
img   = mi.render(scene)
mi.Bitmap(img).write(out_furnace)
print(f"         -> {out_furnace}")

print("\nAll reference renders complete.")
print()
print("Next: run nori-dxr for each scene, then compare_exr.py:")
print("  ..\\..\\dxr\\build\\Release\\nori-dxr.exe sphere_ibl.xml")
print("  ..\\..\\dxr\\build\\Release\\nori-dxr.exe sphere_mirror.xml")
print("  ..\\..\\dxr\\build\\Release\\nori-dxr.exe sphere_furnace_ibl.xml")
