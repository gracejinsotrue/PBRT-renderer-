# pbrt-v4 "bmw-m6" -> Nori scene converter.
#
# Reads mmp/pbrt-v4-scenes/bmw-m6 (binary-LE plymesh geometry + an equal-area
# octahedral envmap) and writes scenes/bmw_m6/: one OBJ per pbrt named
# material, an equirectangular .hdr envmap, and scene.xml.
#
# usage: python _pbrt_bmw_to_nori.py <path/to/pbrt-v4-scenes/bmw-m6>
import os, re, sys, struct, math

HERE = os.path.dirname(os.path.abspath(__file__))
_args = [a for a in sys.argv[1:] if not a.startswith('-')]
SRC  = _args[0] if _args else os.path.join(HERE, "pbrt-v4-scenes", "bmw-m6")
OUT  = os.path.join(HERE, "scenes", "bmw_m6")
MESH = os.path.join(OUT, "meshes")
TEX  = os.path.join(OUT, "textures")

# Normal-incidence reflectance of aluminium; pbrt uses the measured
# metal-Al eta/k spectra, we only have an RGB basecolor to work with.
AL = (0.9130, 0.9210, 0.9250)


# ---------------------------------------------------------------- pbrt parsing

def tokenize(text):
    # strip comments, then split on whitespace keeping "quoted strings" and [ ]
    text = re.sub(r'#[^\n]*', '', text)
    return re.findall(r'"[^"]*"|\[|\]|[^\s\[\]"]+', text)


def parse_pbrt(path):
    """Return (materials, shapes, camera). Enough of pbrt for this one scene."""
    toks = tokenize(open(path).read())
    i, n = 0, len(toks)
    materials, shapes, camera = {}, [], {}
    current_mat = None

    def read_params(j):
        """Read `"type name" value...` pairs until the next directive."""
        params = {}
        while j < n and toks[j].startswith('"') and ' ' in toks[j][1:-1]:
            decl = toks[j][1:-1].split()
            ptype, pname = decl[0], decl[1]
            j += 1
            vals = []
            if j < n and toks[j] == '[':
                j += 1
                while toks[j] != ']':
                    vals.append(toks[j].strip('"'))
                    j += 1
                j += 1
            else:
                vals.append(toks[j].strip('"'))
                j += 1
            params[pname] = (ptype, vals)
        return params, j

    while i < n:
        t = toks[i]
        if t == 'MakeNamedMaterial':
            name = toks[i + 1].strip('"')
            params, i = read_params(i + 2)
            materials[name] = params
        elif t == 'NamedMaterial':
            current_mat = toks[i + 1].strip('"')
            i += 2
        elif t == 'Shape':
            kind = toks[i + 1].strip('"')
            params, i = read_params(i + 2)
            if kind == 'plymesh':
                shapes.append((current_mat, params['filename'][1][0]))
        elif t == 'LookAt':
            v = [float(x) for x in toks[i + 1:i + 10]]
            camera['eye'], camera['look'], camera['up'] = v[0:3], v[3:6], v[6:9]
            i += 10
        elif t == 'Camera':
            params, i = read_params(i + 2)
            if 'fov' in params:
                camera['fov'] = float(params['fov'][1][0])
        elif t == 'Film':
            params, i = read_params(i + 2)
            camera['width']  = int(params['xresolution'][1][0])
            camera['height'] = int(params['yresolution'][1][0])
        elif t == 'LightSource':
            params, i = read_params(i + 2)
            camera.setdefault('envmap', params['filename'][1][0])
        else:
            i += 1
    return materials, shapes, camera


# ------------------------------------------------------------- material mapper

def f(params, key, default):
    return float(params[key][1][0]) if key in params else default


def remap_roughness(r):
    """pbrt roughness -> our roughness, both expressed as GGX alpha.

    pbrt-v4 defaults to remaproughness=true and uses alpha = sqrt(roughness);
    shaders/Disney.hlsli uses alpha = roughness^2. Feeding pbrt's numbers
    straight through would make everything a near-perfect mirror (the floor's
    0.0104 is alpha 0.102 in pbrt but alpha 0.0001 here).
    """
    return r ** 0.25


# Cosine-weighted average Fresnel reflectance of a smooth eta=1.5 interface,
# seen from outside and (by the 1/eta^2 radiance-compression identity) inside.
F_EXT = 0.0918
F_INT = 1.0 - (1.0 - F_EXT) / (1.5 ** 2)


def coated_diffuse_albedo(R):
    """Diffuse throughput of a Lambertian base under a smooth dielectric coat.

    pbrt's `coateddiffuse` is a real layered BxDF: light refracts through the
    coat, and on the way back out ~60% of it is totally internally reflected,
    so the stack returns much less than `reflectance`. Disney has no layering,
    so handing it the raw reflectance makes every coated surface 1.2-1.7x too
    bright -- and in an enclosed studio like this one that compounds over
    several bounces. Pre-multiplying the basecolor by this factor puts the
    diffuse lobe back at pbrt's energy; the coat's own specular reflection is
    what Disney's dielectric lobe at specular=0.5 already provides.
    """
    return tuple((1.0 - F_EXT) * (1.0 - F_INT) * c / (1.0 - c * F_INT) for c in R)


def rgb(params, key, default):
    if key not in params:
        return default
    v = params[key][1]
    return (float(v[0]), float(v[1]), float(v[2]))


def to_nori_bsdf(name, params, materials, indent='\t\t'):
    """Map a pbrt-v4 material to the closest BSDF this renderer implements."""
    kind = params['type'][1][0]
    L = []

    def disney(base, rough, metal=0.0, spec=0.5, coat=0.0, coatgloss=1.0):
        L.append('<bsdf type="disney">')
        L.append('\t<color name="baseColor" value="%.6f %.6f %.6f"/>' % base)
        L.append('\t<float name="roughness" value="%.4f"/>' % rough)
        L.append('\t<float name="metallic" value="%.3f"/>' % metal)
        L.append('\t<float name="specular" value="%.3f"/>' % spec)
        if coat > 0.0:
            L.append('\t<float name="clearcoat" value="%.3f"/>' % coat)
            L.append('\t<float name="clearcoatGloss" value="%.3f"/>' % coatgloss)
        L.append('</bsdf>')

    if kind == 'diffuse':
        # Pure Lambertian: Disney with a fully rough, non-specular lobe.
        disney(rgb(params, 'reflectance', (0.5, 0.5, 0.5)), 1.0, 0.0, 0.0)

    elif kind == 'coateddiffuse':
        # Smooth dielectric interface over a Lambertian base == plastic, which
        # is what Disney gives you at metallic=0 with the coat roughness.
        r = f(params, 'roughness', f(params, 'uroughness', 0.1))
        base = coated_diffuse_albedo(rgb(params, 'reflectance', (0.5, 0.5, 0.5)))
        disney(base, remap_roughness(r), 0.0, 0.5)

    elif kind == 'conductor':
        r = f(params, 'roughness', f(params, 'uroughness', 0.0))
        disney(AL, remap_roughness(r), 1.0, 0.5)

    elif kind == 'coatedconductor':
        # Rough metal under a smooth clearcoat (the wheel rims).
        r = f(params, 'conductor.roughness', 0.05)
        ir = f(params, 'interface.roughness', 0.0)
        disney(AL, remap_roughness(r), 1.0, 0.5,
               coat=1.0, coatgloss=1.0 - min(remap_roughness(ir), 1.0))

    elif kind == 'dielectric':
        L.append('<bsdf type="dielectric">')
        L.append('\t<float name="intIOR" value="%.4f"/>' % f(params, 'eta', 1.5))
        L.append('\t<float name="extIOR" value="1.000277"/>')
        L.append('</bsdf>')

    elif kind == 'mix':
        # No stochastic material mixing here; blend the two reflectances by
        # `amount`, which in pbrt-v4 is the probability of the *second* one.
        a = f(params, 'amount', 0.5)
        m0, m1 = params['materials'][1][0], params['materials'][1][1]

        def effective(m):
            c = rgb(materials[m], 'reflectance', (0.5,) * 3)
            return coated_diffuse_albedo(c) if \
                materials[m]['type'][1][0] == 'coateddiffuse' else c

        c0, c1 = effective(m0), effective(m1)
        base = tuple((1.0 - a) * c0[k] + a * c1[k] for k in range(3))
        r0 = remap_roughness(f(materials[m0], 'roughness', 1.0))
        r1 = remap_roughness(f(materials[m1], 'roughness', 1.0))

        # pbrt picks one of the two materials per shading point, so a coat on
        # only one of them is present for only that fraction of the surface.
        # Collapsing to a single always-coated Disney lobe would give the
        # blend far too much specular; scale F0 by the coated weight instead.
        coated = sum(w for w, m in ((1.0 - a, m0), (a, m1))
                     if materials[m]['type'][1][0] in ('coateddiffuse',
                                                       'coatedconductor'))
        disney(base, (1.0 - a) * r0 + a * r1, 0.0, 0.5 * coated)

    else:
        raise SystemExit('unhandled pbrt material type %r on %r' % (kind, name))

    return '\n'.join(indent + l for l in L)


# ------------------------------------------------------------------ ply -> obj

def read_ply(path):
    """Binary-little-endian PLY with float x/y/z, nx/ny/nz and optional u/v."""
    with open(path, 'rb') as fh:
        nv, nf, props = 0, 0, []
        element = None
        while True:
            line = fh.readline().decode('ascii').strip()
            t = line.split()
            if not t:
                continue
            if t[0] == 'format' and t[1] != 'binary_little_endian':
                raise SystemExit('%s: only binary_little_endian supported' % path)
            elif t[0] == 'element':
                element = t[1]
                if element == 'vertex':
                    nv = int(t[2])
                elif element == 'face':
                    nf = int(t[2])
            elif t[0] == 'property' and element == 'vertex':
                props.append(t[2])
            elif t[0] == 'end_header':
                break

        has_uv = 'u' in props
        stride = len(props)
        verts = struct.unpack('<%df' % (nv * stride), fh.read(nv * stride * 4))

        faces = []
        data = fh.read()
        off = 0
        for _ in range(nf):
            k = data[off]
            off += 1
            idx = struct.unpack_from('<%di' % k, data, off)
            off += 4 * k
            for j in range(1, k - 1):          # triangle fan
                faces.append((idx[0], idx[j], idx[j + 1]))
    return verts, stride, has_uv, faces, nv


def write_obj(ply_paths, out_path):
    """Concatenate several PLYs into one OBJ, keeping normals and UVs."""
    meshes = [read_ply(p) for p in ply_paths]

    # Nori's OBJ loader indexes the texcoord array for *every* vertex as soon
    # as the file declares any `vt` at all, so a group mixing UV'd and un-UV'd
    # meshes has to pad the latter with dummy coordinates. That keeps v/vt/vn
    # aligned one-to-one and lets every face use the same index three times.
    group_uv = any(m[2] for m in meshes)

    vbase = 0
    total_tris = 0
    with open(out_path, 'w') as o:
        for verts, stride, has_uv, faces, nv in meshes:
            V, N, T = [], [], []
            for i in range(nv):
                b = i * stride
                V.append('v %.6g %.6g %.6g' % (verts[b], verts[b + 1], verts[b + 2]))
                N.append('vn %.5g %.5g %.5g' % (verts[b + 3], verts[b + 4], verts[b + 5]))
                if has_uv:
                    T.append('vt %.6g %.6g' % (verts[b + 6], verts[b + 7]))
                elif group_uv:
                    T.append('vt 0 0')
            o.write('\n'.join(V) + '\n')
            if T:
                o.write('\n'.join(T) + '\n')
            o.write('\n'.join(N) + '\n')
            F = []
            for (a, b_, c) in faces:
                a += vbase + 1; b_ += vbase + 1; c += vbase + 1
                if group_uv:
                    F.append('f %d/%d/%d %d/%d/%d %d/%d/%d' % (a, a, a, b_, b_, b_, c, c, c))
                else:
                    F.append('f %d//%d %d//%d %d//%d' % (a, a, b_, b_, c, c))
            o.write('\n'.join(F) + '\n')
            vbase += nv
            total_tris += len(faces)
    return total_tris


# ------------------------------------------------------------------- envmap

def equal_area_sphere_to_square(dx, dy, dz):
    """pbrt-v4 EqualAreaSphereToSquare, vectorised over numpy arrays."""
    import numpy as np
    x, y, z = np.abs(dx), np.abs(dy), np.abs(dz)
    r = np.sqrt(np.maximum(0.0, 1.0 - z))
    a = np.maximum(x, y)
    b = np.minimum(x, y)
    b = np.where(a == 0.0, 0.0, b / np.where(a == 0.0, 1.0, a))
    phi = np.arctan(b) * (2.0 / math.pi)
    phi = np.where(x < y, 1.0 - phi, phi)
    v = phi * r
    u = r - v
    south = dz < 0.0
    u2 = np.where(south, 1.0 - v, u)
    v2 = np.where(south, 1.0 - u, v)
    u2 = np.copysign(u2, dx)
    v2 = np.copysign(v2, dy)
    return 0.5 * (u2 + 1.0), 0.5 * (v2 + 1.0)


def convert_envmap(exr_path, hdr_path, width=4096, height=2048):
    """Equal-area octahedral (pbrt) -> equirectangular, with the scene's
    `Rotate 80 0 1 0` / `Rotate -90 1 0 0` light-to-world baked in."""
    import numpy as np, cv2
    try:
        import OpenEXR, Imath
    except ImportError:
        raise SystemExit(
            "This step needs the OpenEXR python bindings, which are not "
            "installed for %s.\nOpenCV 5 dropped its EXR decoder, so there is "
            "no fallback. Either `pip install OpenEXR` for this interpreter, "
            "or re-run with one that has it." % sys.executable)
    # OpenCV 5 ships without the EXR codec, so read through OpenEXR directly.
    exr = OpenEXR.InputFile(exr_path)
    dw = exr.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    ch = [np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(H, W)
          for c in ('B', 'G', 'R')]                    # BGR, to match cv2.imwrite
    img = np.stack(ch, axis=-1)                        # top row first

    # Directions this renderer's equirect lookup expects: it does
    # phi = atan2(d.z, d.x), theta = acos(d.y), uv = (phi/2pi, theta/pi).
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    phi = (u * 2.0 * math.pi)[None, :]
    theta = (v * math.pi)[:, None]
    ones = np.ones((height, width))
    d = np.stack([np.sin(theta) * np.cos(phi) * ones,
                  np.cos(theta) * ones,
                  np.sin(theta) * np.sin(phi) * ones], axis=-1)

    def Ry(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rx(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    R = Ry(math.radians(80.0)) @ Rx(math.radians(-90.0))   # light -> world
    dl = d.reshape(-1, 3) @ R                              # world -> light (R^T d)

    su, sv = equal_area_sphere_to_square(dl[:, 0], dl[:, 1], dl[:, 2])
    px = np.clip((su * W).astype(np.int32), 0, W - 1)
    py = np.clip((sv * H).astype(np.int32), 0, H - 1)
    out = img[py, px].reshape(height, width, 3).astype(np.float32)
    cv2.imwrite(hdr_path, out)
    return out


# ------------------------------------------------------------------- camera

def lookat_matrix(eye, look, up):
    """Row-major camera-to-world for Nori, with the horizontal axis negated.

    Nori's <lookat> and pbrt's LookAt produce the same three basis vectors, but
    the two cameras disagree on where the first one points on screen, so using
    <lookat> directly gives a mirror image of the pbrt render.
    """
    def sub(a, b):   return [a[i] - b[i] for i in range(3)]
    def cross(a, b): return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
                             a[0]*b[1] - a[1]*b[0]]
    def norm(v):
        l = math.sqrt(sum(c * c for c in v))
        return [c / l for c in v]

    d = norm(sub(look, eye))
    left = norm(cross(norm(up), d))
    newup = norm(cross(d, left))

    rows = [['%.8g' % -left[i], '%.8g' % newup[i], '%.8g' % d[i], '%.8g' % eye[i]]
            for i in range(3)]
    rows.append(['0', '0', '0', '1'])
    return ' '.join(c for r in rows for c in r)


# ----------------------------------------------------------------------- main

def main():
    os.makedirs(MESH, exist_ok=True)
    os.makedirs(TEX, exist_ok=True)

    pbrt = os.path.join(SRC, 'bmw-m6.pbrt')
    materials, shapes, cam = parse_pbrt(pbrt)
    print('materials: %d   shapes: %d' % (len(materials), len(shapes)))

    # group the plys by material, preserving first-use order
    groups, order = {}, []
    for mat, ply in shapes:
        if mat not in groups:
            groups[mat] = []
            order.append(mat)
        groups[mat].append(os.path.join(SRC, ply.replace('\\', '/')))

    # --xml-only reuses the meshes and envmap already on disk; only the
    # materials and camera are rewritten. Handy when tuning the BSDF mapping.
    xml_only = '--xml-only' in sys.argv

    total = 0
    tris = {}
    for mat in order:
        obj = os.path.join(MESH, '%s.obj' % mat)
        if xml_only:
            n, carry = 0, b'\n'          # carry 2 bytes so a chunk boundary
            with open(obj, 'rb') as fh:  # can't split an "\nf " marker
                for chunk in iter(lambda: fh.read(1 << 22), b''):
                    buf = carry + chunk
                    n += buf.count(b'\nf ')
                    carry = buf[-2:]
            tris[mat] = n
        else:
            tris[mat] = write_obj(groups[mat], obj)
        total += tris[mat]
        print('  %-20s %3d ply -> %8d tris' % (mat, len(groups[mat]), tris[mat]))
    print('total triangles: %d' % total)

    if not xml_only:
        print('converting envmap (equal-area octahedral -> equirect) ...')
        convert_envmap(os.path.join(SRC, cam['envmap'].replace('\\', '/')),
                       os.path.join(TEX, 'sunflowers.hdr'))

    # pbrt's fov is the angle subtended by the *shorter* image axis; ours is
    # always horizontal, so widen it by the aspect ratio when width > height.
    W, H = cam['width'], cam['height']
    half = math.tan(math.radians(cam['fov'] * 0.5))
    fov = 2.0 * math.degrees(math.atan(half * (W / H if W > H else 1.0)))

    to_world = lookat_matrix(cam['eye'], cam['look'], cam['up'])

    x = ["<?xml version='1.0' encoding='utf-8'?>", '',
         '<!-- BMW M6, converted from mmp/pbrt-v4-scenes/bmw-m6 by',
         '     _pbrt_bmw_to_nori.py. Geometry (c) its original authors, see',
         '     BLENDSWAP_LICENSE.txt. -->', '',
         '<scene>',
         '\t<string name="envmap" value="textures/sunflowers.hdr"/>',
         '\t<float name="envmapScale" value="1.0"/>',
         '\t<float name="evCompensation" value="0.0"/>',
         '\t<float name="bloomIntensity" value="0.08"/>',
         '\t<float name="bloomThreshold" value="1.2"/>',
         '\t<float name="bloomKnee" value="0.4"/>', '',
         '\t<camera type="perspective">',
         '\t\t<float name="fov" value="%.3f"/>' % fov,
         '\t\t<!-- pbrt: LookAt %g %g %g  %g %g %g  %g %g %g. Written as a matrix'
         % tuple(cam['eye'] + cam['look'] + cam['up']),
         '\t\t     rather than <lookat> because both renderers build the same',
         '\t\t     basis from it, but pbrt sends camera +x to screen right while',
         '\t\t     our sampleToCamera sends it to screen left, which mirrors the',
         '\t\t     image; the first column is negated to undo that. -->',
         '\t\t<transform name="toWorld">',
         '\t\t\t<matrix value="%s"/>' % to_world,
         '\t\t</transform>',
         '\t\t<integer name="width" value="%d"/>' % W,
         '\t\t<integer name="height" value="%d"/>' % H,
         '\t</camera>', '',
         '\t<sampler type="independent">',
         '\t\t<integer name="sampleCount" value="256"/>',
         '\t</sampler>', '']

    for mat in order:
        x.append('\t<!-- %s: pbrt "%s", %d tris -->'
                 % (mat, materials[mat]['type'][1][0], tris[mat]))
        x.append('\t<mesh type="obj">')
        x.append('\t\t<string name="filename" value="meshes/%s.obj"/>' % mat)
        x.append(to_nori_bsdf(mat, materials[mat], materials))
        x.append('\t</mesh>')
        x.append('')
    x.append('</scene>')

    with open(os.path.join(OUT, 'scene.xml'), 'w') as fh:
        fh.write('\n'.join(x) + '\n')
    print('wrote %s' % os.path.join(OUT, 'scene.xml'))
    print('camera: fov %.2f (pbrt %.1f on the short axis) at %dx%d' % (fov, cam['fov'], W, H))


if __name__ == '__main__':
    main()
