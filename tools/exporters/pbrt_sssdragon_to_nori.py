# pbrt-v4 "sssdragon" -> Nori scene converter.
#
# Companion piece to the head2 skin renders: a face makes subsurface
# scattering subtle, a backlit translucent dragon makes it obvious.
#
# Three things here that tools/exporters/pbrt_bmw_to_nori.py does not handle, which is why
# this is its own script rather than a flag on that one:
#   * dragon.ply.gz is gzipped, BIG-endian, and carries no normals
#     (7.2M triangles, so it is read with numpy rather than struct)
#   * the dragon shape has a real CTM on it (translate/rotate/rotate/scale)
#     which has to be baked into the vertices, since our OBJ <transform> would
#     work but keeping the two meshes in one space is simpler
#   * the scene applies `Scale -1 1 1` before LookAt, a deliberate mirror,
#     which happens to cancel Nori's own screen-left/right flip - so unlike
#     the BMW this one can use a plain <lookat> and still match pbrt
#
# usage: python tools/exporters/pbrt_sssdragon_to_nori.py <path/to/pbrt-v4-scenes/sssdragon>
import os, sys, gzip, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root: tools/<group>/ -> ..
_args = [a for a in sys.argv[1:] if not a.startswith('-')]
SRC = _args[0] if _args else os.path.join(REPO, "pbrt-v4-scenes", "sssdragon")
OUT = os.path.join(REPO, "scenes", "sssdragon")
MESH = os.path.join(OUT, "meshes")
TEX = os.path.join(OUT, "textures")

# reuse the ply/envmap/material helpers from the BMW converter
sys.path.insert(0, HERE)
import importlib.util
_s = importlib.util.spec_from_file_location('bmw', os.path.join(HERE, 'pbrt_bmw_to_nori.py'))
bmw = importlib.util.module_from_spec(_s)
sys.argv = [sys.argv[0]]           # keep its module-level arg parsing quiet
_s.loader.exec_module(bmw)


def read_header(fh):
    """Returns (nverts, nfaces, properties, big_endian)."""
    nv = nf = 0
    props, element, big = [], None, False
    while True:
        line = fh.readline().decode('ascii', 'replace').strip()
        t = line.split()
        if not t:
            continue
        if t[0] == 'format':
            big = 'big_endian' in t[1]
        elif t[0] == 'element':
            element = t[1]
            if element == 'vertex':
                nv = int(t[2])
            elif element == 'face':
                nf = int(t[2])
        elif t[0] == 'property' and element == 'vertex':
            props.append(t[2])
        elif t[0] == 'end_header':
            return nv, nf, props, big


def read_ply_np(path):
    """Binary PLY -> (positions Nx3 float32, faces Mx3 int32). Handles either
    endianness and an all-triangle `list uchar int` face record."""
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rb') as fh:
        nv, nf, props, big = read_header(fh)
        e = '>' if big else '<'
        stride = len(props)
        V = np.frombuffer(fh.read(nv * stride * 4), dtype=e + 'f4').reshape(nv, stride)
        pos = np.ascontiguousarray(V[:, :3]).astype(np.float64)

        # face records are 1-byte count + count*int32; this scene is all
        # triangles, so the record is a fixed 13 bytes and can be read flat
        rec = np.dtype([('n', 'u1'), ('v', e + 'i4', 3)])
        assert rec.itemsize == 13, rec.itemsize
        F = np.frombuffer(fh.read(nf * 13), dtype=rec, count=nf)
        bad = int((F['n'] != 3).sum())
        if bad:
            raise SystemExit('%s: %d non-triangle faces, unsupported' % (path, bad))
        faces = F['v'].astype(np.int64)
    return pos, faces, props, V


def write_obj_np(path, pos, faces, normals=None, uvs=None):
    """Stream a large mesh out as OBJ. Formatting 10M+ lines through numpy
    string ops rather than a Python loop keeps this to a few seconds."""
    with open(path, 'w', buffering=1 << 22) as o:
        for i in range(0, len(pos), 500000):
            c = pos[i:i + 500000]
            o.write('\n'.join('v %.6g %.6g %.6g' % (x, y, z) for x, y, z in c))
            o.write('\n')
        if uvs is not None:
            o.write('\n'.join('vt %.6g %.6g' % (u, v) for u, v in uvs) + '\n')
        if normals is not None:
            o.write('\n'.join('vn %.5g %.5g %.5g' % (x, y, z) for x, y, z in normals) + '\n')
        f1 = faces + 1
        for i in range(0, len(f1), 500000):
            c = f1[i:i + 500000]
            if normals is not None and uvs is not None:
                o.write('\n'.join('f %d/%d/%d %d/%d/%d %d/%d/%d'
                                  % (a, a, a, b, b, b, cc, cc, cc) for a, b, cc in c))
            else:
                # no normals in the source; the OBJ loader will compute
                # angle-weighted smooth ones, and omitting them roughly
                # halves a 250 MB file
                o.write('\n'.join('f %d %d %d' % (a, b, cc) for a, b, cc in c))
            o.write('\n')


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# (x, y, z) -> (x, z, -y): brings a Z-up scene into the Y-up world our GPU
# camera assumes. Applied to geometry rather than to the camera, because the
# camera cannot express the resulting roll.
ZUP_TO_YUP = rot('x', -90)


def main():
    os.makedirs(MESH, exist_ok=True)
    os.makedirs(TEX, exist_ok=True)

    # ---- dragon: CTM is Translate * Rotate(90,x) * Rotate(-90,y) * Scale(0.02)
    print('reading dragon.ply.gz (7.2M triangles, gzipped big-endian) ...')
    pos, faces, _, _ = read_ply_np(os.path.join(SRC, 'geometry', 'dragon.ply.gz'))
    print('  %d verts, %d tris   source bbox %s .. %s'
          % (len(pos), len(faces), pos.min(0).round(3), pos.max(0).round(3)))

    # pbrt CTM for this shape, post-multiplied in file order:
    #   Translate 0.2 0.3 0.78 * Rotate(90,x) * Rotate(-90,y) * Scale(0.02)
    # then ZUP_TO_YUP on top of it. This scene is Z-up (its ground sits at
    # z~0 and the camera's up is mostly +z), but our GPU camera rebuilds its
    # basis from yaw/pitch alone - DXRApp_Camera.cpp does
    # right = (cos yaw, 0, -sin yaw) - so world up is hardwired to +Y and any
    # roll in the scene's up vector is silently dropped. Rotating the geometry
    # is the fix; leaving it Z-up renders the dragon lying on its side.
    M = ZUP_TO_YUP @ rot('x', 90) @ rot('y', -90) @ (np.eye(3) * 0.02)
    pos = pos @ M.T + ZUP_TO_YUP @ np.array([0.2, 0.3, 0.78])
    lo, hi = pos.min(0), pos.max(0)
    print('  placed bbox %s .. %s  (size %s)'
          % (lo.round(3), hi.round(3), (hi - lo).round(3)))

    print('writing meshes/dragon.obj ...')
    write_obj_np(os.path.join(MESH, 'dragon.obj'), pos, faces)
    print('  %.0f MB' % (os.path.getsize(os.path.join(MESH, 'dragon.obj')) / 1e6))
    diag = float(np.linalg.norm(hi - lo))

    # ---- ground plane, no transform; this one does carry normals and UVs
    gpos, gfaces, gprops, gV = read_ply_np(os.path.join(SRC, 'geometry', 'meshes_0.ply'))
    gpos = gpos @ ZUP_TO_YUP.T
    gn = gV[:, 3:6].astype(np.float64) @ ZUP_TO_YUP.T
    guv = gV[:, 6:8].astype(np.float64)
    write_obj_np(os.path.join(MESH, 'ground.obj'), gpos, gfaces, normals=gn, uvs=guv)
    print('ground: %d verts, %d tris' % (len(gpos), len(gfaces)))

    print('converting envmap (equal-area octahedral -> equirect) ...')
    bmw.convert_envmap(os.path.join(SRC, 'textures', 'small_rural_road_equiarea.exr'),
                       os.path.join(TEX, 'small_rural_road.hdr'))

    # ---- camera.
    #
    # NOT pbrt's. Its camera is
    #   LookAt 3.69558 -3.46243 3.25463  3.04072 -2.85176 2.80939
    #          up -0.317366 0.312466 0.895346
    # which carries real roll (the up vector is nowhere near an axis), and our
    # GPU camera cannot express roll at all - see the ZUP_TO_YUP note above.
    # So this frames the dragon itself: a three-quarter view from the head
    # side, elevated, with the long axis (now Z) running across the frame.
    W, H = 1366, 1024
    fov = 36.0
    centre = (lo + hi) * 0.5
    az, el = math.radians(30.0), math.radians(15.0)
    dist = (0.5 * 1.35 * float(max(hi - lo))) / math.tan(math.radians(fov / 2))
    eye = tuple(centre + dist * np.array([math.cos(el) * math.cos(az),
                                          math.sin(el),
                                          math.cos(el) * math.sin(az)]))
    look = tuple(centre)
    up = (0.0, 1.0, 0.0)
    print('camera: dist %.2f, origin %s' % (dist, np.round(eye, 3)))

    # pbrt's coateddiffuse default reflectance is 0.5; map it the same way the
    # BMW converter does so the ground matches that scene's conventions
    grnd = bmw.coated_diffuse_albedo((0.5, 0.5, 0.5))
    grough = bmw.remap_roughness(0.2)

    # radii echo pbrt's own dragon_10 / _50 / _250 sweep: same geometry, same
    # light, only the mean free path changes
    for tag, radius in (('opaque', diag * 0.002),
                        ('jade', diag * 0.010),
                        ('translucent', diag * 0.040)):
        x = ["<?xml version='1.0' encoding='utf-8'?>", '',
             '<!-- Stanford dragon with the random-walk BSSRDF, converted from',
             '     mmp/pbrt-v4-scenes/sssdragon by tools/exporters/pbrt_sssdragon_to_nori.py.',
             '     radius %.4f = %.1f%% of the model diagonal (%.3f).' % (radius, radius / diag * 100, diag),
             '',
             '     Geometry is rotated from the source scene\'s Z-up into Y-up,',
             '     and the camera is ours rather than pbrt\'s, because our GPU',
             '     camera rebuilds its basis from yaw/pitch and so cannot',
             '     represent a rolled up vector. See tools/exporters/pbrt_sssdragon_to_nori.py. -->', '',
             '<scene>',
             '\t<string name="envmap" value="textures/small_rural_road.hdr"/>',
             '\t<float name="envmapScale" value="1.0"/>',
             '\t<float name="evCompensation" value="0.0"/>', '',
             '\t<camera type="perspective">',
             '\t\t<float name="fov" value="%.3f"/>' % fov,
             '\t\t<transform name="toWorld">',
             '\t\t\t<lookat target="%g, %g, %g" origin="%g, %g, %g" up="%g, %g, %g"/>'
             % (look + eye + up),
             '\t\t</transform>',
             '\t\t<integer name="width" value="%d"/>' % W,
             '\t\t<integer name="height" value="%d"/>' % H,
             '\t</camera>', '',
             '\t<sampler type="independent">',
             '\t\t<integer name="sampleCount" value="512"/>',
             '\t</sampler>', '',
             '\t<!-- dragon: 7.2M tris, random-walk subsurface -->',
             '\t<mesh type="obj">',
             '\t\t<string name="filename" value="meshes/dragon.obj"/>',
             '\t\t<bsdf type="subsurface">',
             '\t\t\t<color name="albedo" value="0.80 0.55 0.35"/>',
             '\t\t\t<float name="radius"    value="%.5f"/>' % radius,
             '\t\t\t<float name="intIOR"    value="1.5"/>',
             '\t\t\t<float name="extIOR"    value="1.0"/>',
             '\t\t\t<float name="g"         value="0.0"/>',
             '\t\t\t<float name="roughness" value="0.15"/>',
             '\t\t\t<float name="specular"  value="0.5"/>',
             '\t\t</bsdf>',
             '\t</mesh>', '',
             '\t<!-- ground: pbrt coateddiffuse, roughness 0.2 -->',
             '\t<mesh type="obj">',
             '\t\t<string name="filename" value="meshes/ground.obj"/>',
             '\t\t<bsdf type="disney">',
             '\t\t\t<color name="baseColor" value="%.6f %.6f %.6f"/>' % grnd,
             '\t\t\t<float name="roughness" value="%.4f"/>' % grough,
             '\t\t\t<float name="metallic" value="0.0"/>',
             '\t\t\t<float name="specular" value="0.5"/>',
             '\t\t</bsdf>',
             '\t</mesh>', '', '</scene>']
        p = os.path.join(OUT, 'dragon_%s.xml' % tag)
        open(p, 'w').write('\n'.join(x) + '\n')
        print('wrote %s  (radius %.5f)' % (p, radius))

    print('camera fov %.2f horizontal (pbrt %.2f on the short axis) at %dx%d'
          % (fov, 28.841503, W, H))


if __name__ == '__main__':
    main()
