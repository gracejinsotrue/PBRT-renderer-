"""Re-render the 3 visual Disney references with the 270-degree envmap rotation fix."""
import os
import mitsuba as mi

try:
    mi.set_variant("cuda_ad_rgb")
except Exception:
    mi.set_variant("llvm_ad_rgb")
print(f"[mitsuba] variant = {mi.variant()}")

HERE = os.path.dirname(os.path.abspath(__file__))
SUNSET_HDR = os.path.join(HERE, "textures", "sunset.hdr")

TESTS = [
    ("metallic_blue", {
        "base_color": {"type": "rgb", "value": [0.10, 0.30, 0.90]},
        "roughness": 0.2, "metallic": 0.9, "specular": 1.0, "spec_tint": 0.5,
        "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0,
        "clearcoat": 0.0, "clearcoat_gloss": 0.5,
    }),
    ("rough_red_clearcoat", {
        "base_color": {"type": "rgb", "value": [0.85, 0.05, 0.05]},
        "roughness": 0.6, "metallic": 0.0, "specular": 0.5, "spec_tint": 0.0,
        "anisotropic": 0.0, "sheen": 0.0, "sheen_tint": 0.0,
        "clearcoat": 1.0, "clearcoat_gloss": 0.9,
    }),
    ("sheen_fabric", {
        "base_color": {"type": "rgb", "value": [0.588, 0.310, 0.639]},
        "roughness": 0.85, "metallic": 0.0, "specular": 0.05, "spec_tint": 0.0,
        "anisotropic": 0.0, "sheen": 0.5, "sheen_tint": 0.5,
        "clearcoat": 0.0, "clearcoat_gloss": 0.5,
    }),
]

for name, params in TESTS:
    out = os.path.join(HERE, f"ref_{name}.exr")
    print(f"[render] {name} -> {out}")
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 16},
        "sensor": {
            "type": "perspective", "fov": 45,
            "to_world": mi.ScalarTransform4f.look_at(
                origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]
            ),
            "film": {
                "type": "hdrfilm", "width": 512, "height": 512,
                "pixel_format": "rgb", "component_format": "float32",
            },
            "sampler": {"type": "independent", "sample_count": 1024},
        },
        "light": {
            "type": "envmap", "filename": SUNSET_HDR,
            "to_world": mi.ScalarTransform4f.rotate([0, 1, 0], 270),
        },
        "sphere": {"type": "sphere", "bsdf": {"type": "principled", **params}},
    })
    img = mi.render(scene)
    mi.Bitmap(img).write(out)
    print(f"  done.")

print("All done.")
