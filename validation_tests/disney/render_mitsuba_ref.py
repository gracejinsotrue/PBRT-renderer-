r"""
render_mitsuba_ref.py
Renders all Disney reference EXRs using Mitsuba 3 `principled` BSDF.

For every nori-dxr scene cmp_*.xml / furnace_*.xml, this writes a matching
ref_*.exr in the SAME directory so compare_exr.py can pair them up.

To keep geometry identical, Mitsuba loads the SAME sphere.obj as nori-dxr
(rather than the procedural 'sphere' shape) so any sphere-size mismatch is
removed from the comparison.

Run:
    pip install mitsuba opencv-python
    python render_mitsuba_ref.py

Output (per scene):
    ref_<name>.exr   linear scene-referred HDR

Then compare:
    python ..\compare_exr.py snapshot_<N>.exr ref_<name>.exr --out diff_<name>
"""

import os
import mitsuba as mi

# CUDA wavefront if available, CPU LLVM otherwise.
try:
    mi.set_variant("cuda_ad_rgb")
except Exception:
    mi.set_variant("llvm_ad_rgb")
print(f"[mitsuba] variant = {mi.variant()}")

HERE = os.path.dirname(os.path.abspath(__file__))
SPHERE_OBJ = os.path.join(HERE, "sphere.obj")
SUNSET_HDR = os.path.join(HERE, "textures", "sunset.hdr")
# 4x4 white HDR generated on first run — clears Mitsuba's 2x3 minimum resolution.
# Sample-identical to the 2x2 white_furnace.hdr nori-dxr uses (constant 1,1,1).
WHITE_HDR = os.path.join(HERE, "ref_white_4x4.hdr")


def ensure_white_hdr():
    if os.path.exists(WHITE_HDR):
        return
    # RGBE encoding of (1,1,1): byte=128, exp=129 -> (128/256) * 2^(129-128) = 1.0
    W, H = 4, 4
    header = (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {H} +X {W}\n".encode())
    pixel = bytes([128, 128, 128, 129])
    with open(WHITE_HDR, "wb") as f:
        f.write(header + pixel * (W * H))
    print(f"[setup] wrote {WHITE_HDR}")


ensure_white_hdr()

for p in (SPHERE_OBJ, SUNSET_HDR, WHITE_HDR):
    assert os.path.exists(p), f"missing {p}"


def build_scene(bsdf_params, envmap, width, height, samples, bsdf_type="principled"):
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 16},
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]
            ),
            "film": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
                "pixel_format": "rgb",
                "component_format": "float32",
            },
            "sampler": {"type": "independent", "sample_count": samples},
        },
        "light": {
            "type": "envmap",
            "filename": envmap,
            # Corrects the 270° azimuth offset between nori-dxr's phi convention
            # (phi = atan2(d.z, d.x)) and Mitsuba's standard convention.
            # Only matters for non-constant envmaps; white furnace HDR is unaffected.
            "to_world": mi.ScalarTransform4f.rotate([0, 1, 0], 270),
        },
        # Procedural unit sphere at origin (guaranteed outward normals).
        # nori-dxr's sphere.obj is also a unit sphere at origin (verified: r=1.0,
        # 1986 verts) — but its normals are inward-facing, which Mitsuba renders
        # as black (back-facing primary hit -> zero BSDF). nori-dxr is more
        # permissive here. Using the procedural sphere keeps the comparison fair.
        "sphere": {
            "type": "sphere",
            "bsdf": {"type": bsdf_type, **bsdf_params},
        },
    }


# Each entry: (output_name, envmap, width, height, samples, bsdf_params)
TESTS = [
    # --- sunset comparison scenes (match cmp_*.xml) ---
    ("metallic_blue", SUNSET_HDR, 512, 512, 1024, {
        "base_color": {"type": "rgb", "value": [0.10, 0.30, 0.90]},
        "roughness": 0.2, "metallic": 0.9, "specular": 1.0, "spec_tint": 0.5,
        "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0,
        "clearcoat": 0.0, "clearcoat_gloss": 0.5,
    }),
    ("rough_red_clearcoat", SUNSET_HDR, 512, 512, 1024, {
        "base_color": {"type": "rgb", "value": [0.85, 0.05, 0.05]},
        "roughness": 0.6, "metallic": 0.0, "specular": 0.5, "spec_tint": 0.0,
        "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0,
        "clearcoat": 1.0, "clearcoat_gloss": 0.9,
    }),
    ("sheen_fabric", SUNSET_HDR, 512, 512, 1024, {
        "base_color": {"type": "rgb", "value": [0.588, 0.310, 0.639]},
        "roughness": 0.85, "metallic": 0.0, "specular": 0.05, "spec_tint": 0.0,
        "anisotropic": 0.0, "sheen": 0.5, "sheen_tint": 0.5,
        "clearcoat": 0.0, "clearcoat_gloss": 0.5,
    }),

    # --- furnace scenes (match furnace_*.xml) ---
    ("furnace_diffuse", WHITE_HDR, 256, 256, 512, {
        "base_color": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        "roughness": 0.5, "metallic": 0.0, "specular": 0.0, "spec_tint": 0.0,
        "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0,
        "clearcoat": 0.0, "clearcoat_gloss": 0.0,
    }),
    ("furnace_metallic_smooth", WHITE_HDR, 256, 256, 512, {
        "base_color": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        "roughness": 0.1, "metallic": 1.0, "specular": 1.0, "spec_tint": 0.0,
        "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0,
        "clearcoat": 0.0, "clearcoat_gloss": 0.0,
    }),
    ("furnace_metallic_rough", WHITE_HDR, 256, 256, 512, {
        "base_color": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        "roughness": 0.5, "metallic": 1.0, "specular": 1.0, "spec_tint": 0.0,
        "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0,
        "clearcoat": 0.0, "clearcoat_gloss": 0.0,
    }),
    ("furnace_clearcoat", WHITE_HDR, 256, 256, 512, {
        "base_color": {"type": "rgb", "value": [0.0, 0.0, 0.0]},
        "roughness": 0.5, "metallic": 0.0, "specular": 0.0, "spec_tint": 0.0,
        "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0,
        "clearcoat": 1.0, "clearcoat_gloss": 0.5,
    }),
]

# Pure Lambertian sanity test (uses Mitsuba's `diffuse` BSDF, not `principled`).
LAMBERTIAN_TEST = ("furnace_lambertian", WHITE_HDR, 256, 256, 512,
                   {"reflectance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}}, "diffuse")


def main():
    for name, envmap, w, h, samples, params in TESTS:
        out = os.path.join(HERE, f"ref_{name}.exr")
        print(f"[render] {name}  ({w}x{h}, {samples} spp)  -> {out}")
        scene = mi.load_dict(build_scene(params, envmap, w, h, samples))
        img = mi.render(scene)
        mi.Bitmap(img).write(out)

    # Lambertian sanity (uses `diffuse` BSDF, not `principled`)
    name, envmap, w, h, samples, params, bsdf_type = LAMBERTIAN_TEST
    out = os.path.join(HERE, f"ref_{name}.exr")
    print(f"[render] {name}  ({w}x{h}, {samples} spp, bsdf={bsdf_type})  -> {out}")
    scene = mi.load_dict(build_scene(params, envmap, w, h, samples, bsdf_type=bsdf_type))
    img = mi.render(scene)
    mi.Bitmap(img).write(out)
    print("done.")


if __name__ == "__main__":
    main()
