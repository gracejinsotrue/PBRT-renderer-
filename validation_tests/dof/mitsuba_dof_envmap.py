r"""
Render DOF validation reference images with Mitsuba 3.
Matches nori-dxr scenes in dof_envmap_*.xml.
Run from the validation_tests/dof/ directory:
    C:\Users\gjin3\anaconda3\python.exe mitsuba_dof_envmap.py
"""
import os
import mitsuba as mi

try:
    mi.set_variant("cuda_ad_rgb")
except Exception:
    mi.set_variant("llvm_ad_rgb")
print(f"[mitsuba] variant = {mi.variant()}")

HERE = os.path.dirname(os.path.abspath(__file__))
SUNSET_HDR = os.path.join(HERE, "textures", "sunset.hdr")
CUBE_OBJ   = os.path.join(HERE, "blender_default_cube.obj").replace("\\", "/")

COMMON_BSDF = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.2, 0.2]}}

def base_scene(sensor):
    return {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 16},
        "sensor": sensor,
        "envmap": {
            "type": "envmap",
            "filename": SUNSET_HDR,
            "to_world": mi.ScalarTransform4f.rotate([0, 1, 0], 270),
        },
        "cube": {
            "type": "obj",
            "filename": CUBE_OBJ,
            "bsdf": COMMON_BSDF,
        },
    }

def film_sampler(spp=1024):
    return {
        "film": {
            "type": "hdrfilm", "width": 512, "height": 512,
            "pixel_format": "rgb", "component_format": "float32",
        },
        "sampler": {"type": "independent", "sample_count": spp},
    }

CAMERA_ORIGIN = [0, 1, 3]
CAMERA_TARGET  = [0, 1, 0]
CAMERA_UP      = [0, 1, 0]
FOV            = 45.0
FOCUS_DIST     = 3.0

RENDERS = [
    # (output_name, sensor_type, aperture_radius)
    ("ref_dof_pinhole",  "perspective", 0.0 ),
    ("ref_dof_main",     "thinlens",    0.15),
    ("ref_dof_small",    "thinlens",    0.05),
]

for name, sensor_type, aperture in RENDERS:
    out = os.path.join(HERE, f"{name}.exr")
    print(f"[render] {name} -> {out}")

    to_world = mi.ScalarTransform4f.look_at(
        origin=CAMERA_ORIGIN, target=CAMERA_TARGET, up=CAMERA_UP
    )

    if sensor_type == "perspective":
        sensor = {"type": "perspective", "fov": FOV, "to_world": to_world, **film_sampler()}
    else:
        sensor = {
            "type": "thinlens",
            "fov": FOV,
            "aperture_radius": aperture,
            "focus_distance": FOCUS_DIST,
            "to_world": to_world,
            **film_sampler(),
        }

    scene = mi.load_dict(base_scene(sensor))
    img = mi.render(scene)
    mi.Bitmap(img).write(out)
    print(f"  done.")

print("All Mitsuba renders complete.")
