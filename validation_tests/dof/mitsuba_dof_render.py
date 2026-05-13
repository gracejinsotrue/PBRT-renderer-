"""
Render the DOF test scene in Mitsuba 3 for validation against nori-dxr.
Usage: python mitsuba_dof_render.py <output.exr> [<scene_dir>]
"""

import sys
import os

# Try different Mitsuba variants
try:
    import mitsuba as mi
except ImportError:
    print("ERROR: mitsuba 3 required. Install via: pip install mitsuba", file=sys.stderr)
    sys.exit(1)

try:
    mi.set_variant('cuda_ad_rgb')
    print("[mitsuba] Using cuda_ad_rgb variant", file=sys.stderr)
except:
    try:
        mi.set_variant('llvm_mono_ad_rgb')
        print("[mitsuba] Using llvm_mono_ad_rgb variant", file=sys.stderr)
    except:
        try:
            mi.set_variant('scalar_rgb')
            print("[mitsuba] Using scalar_rgb variant", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Could not set any Mitsuba variant: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.executable} {sys.argv[0]} <output.exr> [<scene_dir>]", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]
    scene_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    # Use forward slashes for Mitsuba XML paths (works on all platforms)
    cube_path = os.path.join(scene_dir, 'blender_default_cube.obj').replace('\\', '/')
    light_path = os.path.join(scene_dir, 'light_plane.obj').replace('\\', '/')

    if not os.path.exists(cube_path):
        print(f"ERROR: {cube_path} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(light_path):
        print(f"ERROR: {light_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load OBJ directly — preserves per-face flat normals (s 0 / f v/vt/vn in the OBJ).
    # PLY conversion would discard normals and cause Mitsuba to compute smoothed normals,
    # producing incorrect shading at cube edges vs. nori's flat-shaded OBJ.
    scene_xml = f"""<scene version="2.0.0">
    <integrator type="direct" />

    <sensor type="thinlens">
        <float name="fov" value="50"/>
        <float name="aperture_radius" value="0.1"/>
        <float name="focus_distance" value="4.0"/>
        <transform name="to_world">
            <lookat origin="0, 1, 3" target="0, 1, 0" up="0, 1, 0"/>
        </transform>
        <film type="hdrfilm">
            <integer name="width" value="512"/>
            <integer name="height" value="512"/>
        </film>
        <sampler type="independent">
            <integer name="sample_count" value="256"/>
        </sampler>
    </sensor>

    <shape type="obj">
        <string name="filename" value="{cube_path}"/>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="0.8, 0.2, 0.2"/>
        </bsdf>
    </shape>

    <shape type="obj">
        <string name="filename" value="{light_path}"/>
        <emitter type="area">
            <rgb name="radiance" value="10, 10, 10"/>
        </emitter>
    </shape>
</scene>"""

    print("[mitsuba] Creating scene with thinlens DOF (aperture=0.1, focus=4.0), OBJ meshes...", file=sys.stderr)
    scene = mi.load_string(scene_xml)

    print("[mitsuba] Rendering (256 samples, 512x512)...", file=sys.stderr)
    img = mi.render(scene)

    print(f"[mitsuba] Saving to {output_path}...", file=sys.stderr)
    mi.Bitmap(img).write(output_path)
    print("[mitsuba] Done.", file=sys.stderr)


if __name__ == '__main__':
    main()
