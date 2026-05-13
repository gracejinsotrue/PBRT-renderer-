"""
render_vol_mitsuba.py
Renders the volumetric visual reference images using Mitsuba 3.

Generates (all in this directory):
  ref_vol_g03.exr   — main visual comparison (sigmaT=0.5, albedo=0.8, g=0.3)
  ref_vol_g0.exr    — phase ablation, isotropic (g=0)
  ref_vol_g08.exr   — phase ablation, forward scatter (g=0.8)

Camera and medium match vol_visual_g03.xml / vol_visual_g0.xml / vol_visual_g08.xml.

Envmap azimuth convention:
  nori-dxr:  phi = atan2(d.z, d.x) — U=0 maps to +X
  Mitsuba 3: default U=0 maps to -X
  Fix: rotate envmap 270 degrees around Y so both match.

Usage:
  cd C:\\Users\\gjin3\\Desktop\\nori-26sp\\validation_tests\\volumetric_tests\\vol_visual
  C:\\Users\\gjin3\\anaconda3\\python.exe render_vol_mitsuba.py
"""
import os, sys

try:
    import mitsuba as mi
except ImportError:
    sys.exit("Mitsuba 3 not found.  Install with: pip install mitsuba")

try:
    mi.set_variant("cuda_ad_rgb")
except Exception:
    mi.set_variant("llvm_ad_rgb")
print(f"[mitsuba] variant = {mi.variant()}")

HERE       = os.path.dirname(os.path.abspath(__file__))
SUNSET_HDR = os.path.join(HERE, "textures", "sunset.hdr")
SPHERE_OBJ = os.path.join(HERE, "sphere.obj")

assert os.path.exists(SUNSET_HDR), f"Missing: {SUNSET_HDR}"
assert os.path.exists(SPHERE_OBJ), f"Missing: {SPHERE_OBJ}"

# Camera matches nori-dxr: origin=(0,0,15), target=(0,0,0), up=(0,1,0), FOV=45
CAM_XFORM = mi.ScalarTransform4f.look_at(
    origin=[0, 0, 15], target=[0, 0, 0], up=[0, 1, 0]
)

# Envmap: 270-degree Y rotation to match nori-dxr's phi=atan2(z,x) convention
ENVMAP_ROT = mi.ScalarTransform4f.rotate([0, 1, 0], 270)

def render_scene(g_value, out_path):
    print(f"[render] g={g_value:.1f} -> {out_path}")
    scene = mi.load_dict({
        "type": "scene",
        # volpath handles heterogeneous + homogeneous media, MIS
        "integrator": {
            "type": "volpath",
            "max_depth": 32,
        },
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": CAM_XFORM,
            "film": {
                "type": "hdrfilm",
                "width": 256, "height": 256,
                "pixel_format": "rgb",
                "component_format": "float32",
            },
            "sampler": {"type": "independent", "sample_count": 512},
        },
        # Sunset environment map with azimuth-convention fix
        "envlight": {
            "type": "envmap",
            "filename": SUNSET_HDR,
            "to_world": ENVMAP_ROT,
        },
        # Cube [-11,11]^3 matches nori-dxr volume bbox exactly.
        # Camera at z=15 is outside so rays enter medium at z=11 face.
        # Using cube (not sphere) ensures identical path lengths to nori-dxr.
        "fog_bound": {
            "type": "cube",
            "to_world": mi.ScalarTransform4f.scale([11, 11, 11]),
            "bsdf": {"type": "null"},
            "interior": {
                "type": "homogeneous",
                "sigma_t": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
                "albedo":  {"type": "rgb", "value": [0.8, 0.8, 0.8]},
                "phase": {"type": "hg", "g": g_value},
            },
        },
        # Diffuse sphere at origin (radius 1), matches sphere.obj
        "sphere_shape": {
            "type": "obj",
            "filename": SPHERE_OBJ,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
            },
        },
    })
    img = mi.render(scene)
    mi.Bitmap(img).write(out_path)
    print(f"  done -> {out_path}")


render_scene(0.3, os.path.join(HERE, "ref_vol_g03.exr"))
render_scene(0.0, os.path.join(HERE, "ref_vol_g0.exr"))
render_scene(0.8, os.path.join(HERE, "ref_vol_g08.exr"))

print("\n[done]  All volumetric reference images written.")
