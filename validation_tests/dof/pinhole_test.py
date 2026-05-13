"""Quick pinhole comparison to isolate scene-level mismatch from DOF."""
import mitsuba as mi
import tempfile, os, numpy as np

mi.set_variant('cuda_ad_rgb')

def make_ply(verts, faces, path):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n')
        f.write(f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        for v in verts:
            f.write(f'{v[0]} {v[1]} {v[2]}\n')
        for fi in faces:
            f.write(f'3 {fi[0]} {fi[1]} {fi[2]}\n')

def load_obj(path):
    verts, faces = [], []
    for line in open(path):
        if line.startswith('v '):
            verts.append(list(map(float, line.split()[1:4])))
        elif line.startswith('f '):
            fi = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
            if len(fi) == 3:
                faces.append(fi)
            elif len(fi) == 4:
                faces += [[fi[0], fi[1], fi[2]], [fi[0], fi[2], fi[3]]]
    return verts, faces

script_dir = os.path.dirname(os.path.abspath(__file__))
cv, cf = load_obj(os.path.join(script_dir, 'blender_default_cube.obj'))
lv, lf = load_obj(os.path.join(script_dir, 'light_plane.obj'))

with tempfile.TemporaryDirectory() as td:
    cube_ply = os.path.join(td, 'cube.ply')
    light_ply = os.path.join(td, 'light.ply')
    make_ply(cv, cf, cube_ply)
    make_ply(lv, lf, light_ply)

    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'direct'},
        'sensor': {
            'type': 'perspective',
            'fov': 50.0,
            'to_world': mi.ScalarTransform4f.look_at([0, 1, 3], [0, 1, 0], [0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 512, 'height': 512},
            'sampler': {'type': 'independent', 'sample_count': 256}
        },
        'cube': {
            'type': 'ply',
            'filename': cube_ply,
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.2, 0.2]}}
        },
        'light': {
            'type': 'ply',
            'filename': light_ply,
            'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [10.0, 10.0, 10.0]}}
        }
    })

    img = mi.render(scene)
    out = os.path.join(script_dir, 'pinhole_mitsuba.exr')
    mi.Bitmap(img).write(out)
    arr = np.array(mi.Bitmap(out))
    print(f'Mitsuba pinhole: R={arr[...,0].mean():.4f} G={arr[...,1].mean():.4f} B={arr[...,2].mean():.4f}  mean={arr.mean():.4f}')
    print(f'Saved: {out}')
