# pbrt-v4 "sportscar" -> Nori scene converter.
#
# Reads mmp/pbrt-v4-scenes/sportscar (binary-LE plymesh geometry + an
# equal-area octahedral envmap) and writes scenes/sportscar/: one OBJ per pbrt
# named material, an equirectangular .hdr envmap, and scene.xml.
#
# Differences from the bmw-m6 converter, which this borrows its proven pieces
# from (camera basis, roughness remap, coated-diffuse energy compensation,
# octahedral projection):
#
#   * sportscar nests AttributeBegin blocks and puts a real transform on them
#     -- the whole car sits under a +90 deg rotation about X that takes the
#     model from Y-up to the scene's Z-up, and the ground plane carries a
#     non-uniform scale. bmw-m6 had none of that, so its parser ignored the
#     CTM entirely. Here the stack is tracked and baked into the vertices.
#   * six materials are pbrt "measured" (RGL tensor-file BSDFs). Evaluating
#     those means implementing the Dupuy & Jakob interpolant, so each is
#     approximated by a Disney lobe instead; see MEASURED below.
#
# usage: python tools/exporters/pbrt_sportscar_to_nori.py [path/to/pbrt-v4-scenes/sportscar]
#        --preview   960x540 at 96 spp instead of the full-size scene
#        --xml-only  reuse the existing OBJs, only re-emit the scene XML
#        --view NAME one of VIEWS below (default 'pbrt', the file's own LookAt)
import os, re, sys, math, struct, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root: tools/<group>/ -> ..
_args = [a for a in sys.argv[1:] if not a.startswith('-')]
SRC = _args[0] if _args else os.path.join(REPO, "pbrt-v4-scenes", "sportscar")
OUT = os.path.join(REPO, "scenes", "sportscar")
MESH = os.path.join(OUT, "meshes")
TEX = os.path.join(OUT, "textures")
PREVIEW = '--preview' in sys.argv
XML_ONLY = '--xml-only' in sys.argv   # skip the OBJ pass; only re-emit scene.xml
VIEW = (sys.argv[sys.argv.index('--view') + 1]
        if '--view' in sys.argv else 'pbrt')

WIDTH, HEIGHT, SPP = (960, 540, 96) if PREVIEW else (1600, 900, 512)

# Alternate cameras, in *renderer* space (Y-up, after WORLD_FIX), so they are
# written straight through rather than routed via fix_point. Measured off the
# converted meshes: the car is 4.56 long x 2.26 wide x 1.09 tall, centred on
# the origin, nose at +Z (headlights z=+1.68), tail at -Z (exhaust z=-2.16),
# open cockpit at z=-0.05. 'pbrt' keeps the LookAt from the scene file: a low
# front three-quarter off the +X/+Z corner.
VIEWS = {
    'pbrt': None,
    # Elevated rear three-quarter: tail, diffuser and rear deck, looking down
    # into the open cockpit. 36 deg off the tail axis, 28 deg above horizontal.
    #
    # Framing here is constrained by the backdrop, not by taste. The sky
    # envmap is pure black at and below 0 deg elevation (measured: rows 1023+
    # of the 2048-row equirect are identically zero), so any downward ray that
    # overshoots the backdrop renders black -- and the backdrop is only 11.6 m
    # wide in X against 30 m in Z. Three things follow, and slackening any one
    # of them puts a black wedge in a top corner:
    #
    #   * the camera looks toward -X, which has 6.54 m of plane in front of it
    #     against 5.05 m the other way;
    #   * the lens is longer than pbrt's 44.6 deg, which shrinks how far the
    #     frame corners splay off-axis;
    #   * the tilt is steep enough that every corner ray lands on the plane
    #     (worst-case reach 17.3 m, inside the backdrop in both axes).
    #
    # Distance is set from the lens so the car reads the same size as in the
    # pbrt view: 11.47 m at 26 deg matches 6.45 m at 44.63 deg.
    'rear3q': dict(eye=(6.05, 5.93, -8.44), look=(0.10, 0.55, -0.25),
                   up=(0.0, 1.0, 0.0), fov=26.0),
}
if VIEW not in VIEWS:
    raise SystemExit('unknown --view %r; try one of %s'
                     % (VIEW, ', '.join(sorted(VIEWS))))

# Reuse the bmw converter's helpers rather than copying them: they encode
# fixes that took measurement to get right (the negated camera axis, the
# roughness convention gap, the layered-coat energy loss).
_spec = importlib.util.spec_from_file_location(
    'pbrt_bmw', os.path.join(HERE, 'pbrt_bmw_to_nori.py'))
_bmw = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]
_spec.loader.exec_module(_bmw)
sys.argv = _argv

tokenize = _bmw.tokenize
lookat_matrix = _bmw.lookat_matrix
remap_roughness = _bmw.remap_roughness
coated_diffuse_albedo = _bmw.coated_diffuse_albedo
equal_area_sphere_to_square = _bmw.equal_area_sphere_to_square
f, rgb = _bmw.f, _bmw.rgb

AL = _bmw.AL   # aluminium normal-incidence reflectance

# sportscar is authored Z-up (the pbrt camera's up is ~+Z). This renderer's
# camera is Y-up: it decomposes the camera-to-world matrix into yaw/pitch
# about +Y and rebuilds from those, which silently drops the roll a Z-up
# basis carries and lands the image on its side. Rather than fight that,
# rotate the whole scene -90 deg about X on the way out, so world +Z becomes
# world +Y. Applied to geometry, camera and envmap alike.
WORLD_FIX = [[1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]


def fix_point(p):
    return (p[0], p[2], -p[1])


# The six RGL measured BSDFs, approximated by Disney lobes.
#
# These are not guesses: every number below was read out of the .bsdf files
# themselves (RGL "tensor_file" containers). Each holds a `spectra` table of
# 195 wavelengths over 8 incident elevations and a 32x32 grid of outgoing
# directions, plus the microfacet `ndf`. So:
#
#   baseColor  = the normal-incidence spectrum, averaged over outgoing
#                directions and converted to linear sRGB through the CIE 1931
#                matching functions, white-balanced so a flat spectrum is
#                neutral (the sRGB matrix is D65-referenced, integrating a
#                flat spectrum gives illuminant E, and skipping the adaptation
#                tints everything magenta).
#   roughness  = from the NDF peak. GGX has D(0) = 1/(pi a^2), and this
#                renderer's Disney takes alpha = roughness^2, so
#                roughness = sqrt(1 / (pi * max(ndf)))^0.5.
#
# Two sanity checks that the readout is calibrated: the ceiling paint is
# described upstream as a "custom color match to Kodak 18% gray card" and
# comes out neutral at luminance 0.166, and the A4 paper comes out faintly
# blue, which is what optical brightening agents in office paper do.
#
# What this CANNOT reproduce: two of the six are colour-shift films, and their
# hue swings with angle (17.8 deg of hue drift for the body, 38.4 for the
# suspension). Disney's baseColor is a constant, so the render keeps the
# near-normal colour and loses the shift. See the note in the scene XML.
#
# (baseColor, roughness, metallic, clearcoat)
MEASURED = {
    # The car body: a chameleon vinyl wrap, "TeckWrap Blue Agat MCH03". Deep
    # blue face-on, drifting toward teal at grazing -- only the blue survives
    # here. alpha 0.0022 makes this very nearly a mirror, which is most of
    # why the reference render looks so much glossier than a guess would.
    'cc_blue_agat_spec': ((0.055, 0.231, 0.526), 0.047, 0.55, 1.00),
    # "Metallic paint from the L3-37 robot (Solo: A Star Wars Story), ILM".
    # Neutral silver, 0.3 deg of hue drift, so a constant colour is honest.
    'ilm_l3_37_metallic_spec': ((0.676, 0.686, 0.673), 0.183, 1.00, 0.00),
    # Ultra-flat waterborne ceiling paint on the brake rotors.
    'laika_ceiling_paint_18_gray_spec': ((0.158, 0.168, 0.173), 0.816, 0.00, 0.00),
    # White acrylic felt (Edukit A4) on the seats.
    'acrylic_felt_white_spec': ((0.585, 0.620, 0.775), 0.757, 0.00, 0.00),
    # "TeckWrap Silk Blue VCH502N" on the suspension: the strongest colour
    # shifter of the six at 38.4 deg, and the one we lose the most of.
    'vch_silk_blue_spec': ((0.015, 0.083, 0.449), 0.301, 0.00, 0.00),
    # White A4 paper, 160 gsm -- the studio floor.
    'paper_white_spec': ((0.816, 0.824, 0.972), 0.779, 0.00, 0.00),
}


# ---------------------------------------------------------------- pbrt parsing

def mat_identity():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def mat_from_pbrt(vals):
    """pbrt writes 4x4 matrices column-major; return them row-major."""
    m = [float(v) for v in vals]
    return [[m[4 * j + i] for j in range(4)] for i in range(4)]


def parse_pbrt(path, materials=None, shapes=None, camera=None, ctm=None,
               stack=None):
    """Parse one pbrt file, following Include, tracking the transform stack."""
    materials = {} if materials is None else materials
    shapes = [] if shapes is None else shapes
    camera = {} if camera is None else camera
    ctm = mat_identity() if ctm is None else ctm
    stack = [] if stack is None else stack

    toks = tokenize(open(path).read())
    i, n = 0, len(toks)
    current_mat = [None]

    def read_params(j):
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

    mat_stack = []
    while i < n:
        t = toks[i]
        if t == 'MakeNamedMaterial':
            name = toks[i + 1].strip('"')
            params, i = read_params(i + 2)
            materials[name] = params
        elif t == 'NamedMaterial':
            current_mat[0] = toks[i + 1].strip('"')
            i += 2
        elif t == 'AttributeBegin':
            stack.append([r[:] for r in ctm])
            mat_stack.append(current_mat[0])
            i += 1
        elif t == 'AttributeEnd':
            ctm = stack.pop()
            current_mat[0] = mat_stack.pop()
            i += 1
        elif t in ('Transform', 'ConcatTransform'):
            j = i + 1
            vals = []
            if toks[j] == '[':
                j += 1
                while toks[j] != ']':
                    vals.append(toks[j])
                    j += 1
                j += 1
            else:
                vals = toks[j:j + 16]
                j += 16
            M = mat_from_pbrt(vals)
            ctm = M if t == 'Transform' else mat_mul(ctm, M)
            i = j
        elif t == 'Include':
            inc = toks[i + 1].strip('"')
            parse_pbrt(os.path.join(SRC, inc), materials, shapes, camera,
                       ctm, stack)
            i += 2
        elif t == 'Shape':
            kind = toks[i + 1].strip('"')
            params, i = read_params(i + 2)
            if kind == 'plymesh':
                shapes.append((current_mat[0], params['filename'][1][0],
                               [r[:] for r in ctm]))
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
            camera['width'] = int(params['xresolution'][1][0])
            camera['height'] = int(params['yresolution'][1][0])
        elif t == 'LightSource':
            params, i = read_params(i + 2)
            camera.setdefault('envmap', params['filename'][1][0])
        else:
            i += 1
    return materials, shapes, camera


# ------------------------------------------------------------- material mapper

def to_nori_bsdf(name, params, indent='\t\t'):
    kind = params['type'][1][0]
    L = []

    def disney(base, rough, metal=0.0, spec=0.5, coat=0.0, coatgloss=1.0):
        L.append('<bsdf type="disney">')
        L.append('\t<color name="baseColor" value="%.6f %.6f %.6f"/>' % tuple(base))
        L.append('\t<float name="roughness" value="%.4f"/>' % rough)
        L.append('\t<float name="metallic" value="%.3f"/>' % metal)
        L.append('\t<float name="specular" value="%.3f"/>' % spec)
        if coat > 0.0:
            L.append('\t<float name="clearcoat" value="%.3f"/>' % coat)
            L.append('\t<float name="clearcoatGloss" value="%.3f"/>' % coatgloss)
        L.append('</bsdf>')

    if kind == 'diffuse':
        disney(rgb(params, 'reflectance', (0.5, 0.5, 0.5)), 1.0, 0.0, 0.0)

    elif kind == 'coateddiffuse':
        r = f(params, 'roughness', f(params, 'uroughness', 0.1))
        base = coated_diffuse_albedo(rgb(params, 'reflectance', (0.5,) * 3))
        disney(base, remap_roughness(min(r, 1.0)), 0.0, 0.5)

    elif kind == 'conductor':
        r = f(params, 'roughness', f(params, 'uroughness', 0.0))
        disney(AL, remap_roughness(r), 1.0, 0.5)

    elif kind == 'dielectric':
        L.append('<bsdf type="dielectric">')
        L.append('\t<float name="intIOR" value="%.4f"/>' % f(params, 'eta', 1.5))
        L.append('\t<float name="extIOR" value="1.000277"/>')
        L.append('</bsdf>')

    elif kind == 'measured':
        stem = os.path.splitext(os.path.basename(
            params['filename'][1][0]))[0]
        if stem not in MEASURED:
            raise SystemExit('no approximation for measured BSDF %r' % stem)
        base, rough, metal, coat = MEASURED[stem]
        disney(base, rough, metal, 0.5, coat=coat,
               coatgloss=0.97 if coat else 1.0)

    else:
        raise SystemExit('unhandled pbrt material type %r on %r' % (kind, name))

    return '\n'.join(indent + l for l in L)


# ------------------------------------------------------------------ ply -> obj

def read_ply(path):
    """Binary-LE PLY. sportscar names its texcoords s/t, not u/v."""
    with open(path, 'rb') as fh:
        nv, nf, props, element = 0, 0, [], None
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

        has_uv = ('u' in props) or ('s' in props)
        has_n = 'nx' in props
        stride = len(props)
        verts = struct.unpack('<%df' % (nv * stride), fh.read(nv * stride * 4))

        faces, data, off = [], fh.read(), 0
        for _ in range(nf):
            k = data[off]
            off += 1
            idx = struct.unpack_from('<%di' % k, data, off)
            off += 4 * k
            for j in range(1, k - 1):
                faces.append((idx[0], idx[j], idx[j + 1]))
    return verts, stride, has_uv, has_n, faces, nv


def xform_point(M, p):
    return (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]*p[2] + M[0][3],
            M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]*p[2] + M[1][3],
            M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]*p[2] + M[2][3])


def normal_matrix(M):
    """Inverse transpose of the upper 3x3, so normals survive a non-uniform
    scale (the ground plane has one)."""
    a = [row[:3] for row in M[:3]]
    det = (a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
           - a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
           + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]))
    if abs(det) < 1e-20:
        return a
    inv = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = [[a[x][y] for y in range(3) if y != j] for x in range(3) if x != i]
            c = r[0][0] * r[1][1] - r[0][1] * r[1][0]
            inv[i][j] = ((-1) ** (i + j)) * c / det      # already transposed
    return [[inv[j][i] for j in range(3)] for i in range(3)]   # inverse-transpose


def xform_normal(N, v):
    n = (N[0][0]*v[0] + N[0][1]*v[1] + N[0][2]*v[2],
         N[1][0]*v[0] + N[1][1]*v[1] + N[1][2]*v[2],
         N[2][0]*v[0] + N[2][1]*v[1] + N[2][2]*v[2])
    l = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    return n if l < 1e-20 else (n[0]/l, n[1]/l, n[2]/l)


def write_obj(entries, out_path):
    """entries: [(ply_path, ctm)]. The CTM is baked into the vertices."""
    meshes = [(read_ply(p), M) for p, M in entries]
    group_uv = any(m[0][2] for m in meshes)

    vbase, total = 0, 0
    with open(out_path, 'w') as o:
        for (verts, stride, has_uv, has_n, faces, nv), M in meshes:
            NM = normal_matrix(M)
            V, N, T = [], [], []
            for i in range(nv):
                b = i * stride
                p = xform_point(M, (verts[b], verts[b+1], verts[b+2]))
                V.append('v %.6g %.6g %.6g' % p)
                if has_n:
                    nn = xform_normal(NM, (verts[b+3], verts[b+4], verts[b+5]))
                    N.append('vn %.5g %.5g %.5g' % nn)
                if has_uv:
                    k = b + (6 if has_n else 3)
                    T.append('vt %.6g %.6g' % (verts[k], verts[k+1]))
                elif group_uv:
                    T.append('vt 0 0')
            o.write('\n'.join(V) + '\n')
            if T:
                o.write('\n'.join(T) + '\n')
            if N:
                o.write('\n'.join(N) + '\n')
            F = []
            for (a, b_, c) in faces:
                a += vbase + 1; b_ += vbase + 1; c += vbase + 1
                if N and T:
                    F.append('f %d/%d/%d %d/%d/%d %d/%d/%d'
                             % (a, a, a, b_, b_, b_, c, c, c))
                elif N:
                    F.append('f %d//%d %d//%d %d//%d' % (a, a, b_, b_, c, c))
                else:
                    F.append('f %d %d %d' % (a, b_, c))
            o.write('\n'.join(F) + '\n')
            vbase += nv
            total += len(faces)
    return total


# ------------------------------------------------------------------- envmap

def read_exr(path):
    """Minimal OpenEXR reader: single-part scanline, ZIP/ZIPS/uncompressed,
    FLOAT channels. Written out rather than leaning on the OpenEXR bindings,
    which are not installed here and which OpenCV 5 no longer substitutes for
    (it dropped its EXR decoder). sky.exr is 512x512 ZIP, which is a few lines
    of zlib plus EXR's byte predictor and interleave."""
    import numpy as np
    import zlib
    d = open(path, 'rb').read()
    if d[:4] != b'\x76\x2f\x31\x01':
        raise SystemExit('not an EXR: %s' % path)
    off = 8
    chans, comp, dw = [], None, None
    while True:
        e = d.index(b'\0', off); name = d[off:e].decode(); off = e + 1
        if not name:
            break
        e = d.index(b'\0', off); off = e + 1                    # attribute type
        size = struct.unpack('<i', d[off:off + 4])[0]; off += 4
        val = d[off:off + size]; off += size
        if name == 'channels':
            i = 0
            while i < len(val) and val[i] != 0:
                j = val.index(b'\0', i); cn = val[i:j].decode(); i = j + 1
                chans.append((cn, struct.unpack('<i', val[i:i + 4])[0]))
                i += 16
        elif name == 'compression':
            comp = val[0]
        elif name == 'dataWindow':
            dw = struct.unpack('<4i', val)
    if comp not in (0, 2, 3):
        raise SystemExit('EXR compression %d not supported' % comp)
    if any(pt != 2 for _, pt in chans):
        raise SystemExit('only FLOAT channels supported')

    W = dw[2] - dw[0] + 1
    H = dw[3] - dw[1] + 1
    rows_per_block = 1 if comp in (0, 2) else 16
    nblocks = (H + rows_per_block - 1) // rows_per_block
    offsets = struct.unpack('<%dQ' % nblocks, d[off:off + 8 * nblocks])

    names = [c for c, _ in chans]            # header stores them alphabetically
    planes = {c: np.zeros((H, W), np.float32) for c in names}
    rowbytes = W * 4 * len(names)

    for bo in offsets:
        y0 = struct.unpack('<i', d[bo:bo + 4])[0] - dw[1]
        size = struct.unpack('<i', d[bo + 4:bo + 8])[0]
        raw = d[bo + 8:bo + 8 + size]
        nrows = min(rows_per_block, H - y0)
        if comp == 0 or size == rowbytes * nrows:
            buf = raw                                       # stored as-is
        else:
            b = bytearray(zlib.decompress(raw))
            for i in range(1, len(b)):                      # undo predictor
                b[i] = (b[i - 1] + b[i] - 128) & 0xFF
            half = (len(b) + 1) // 2                        # undo interleave
            out = bytearray(len(b))
            out[0::2] = b[:half]
            out[1::2] = b[half:]
            buf = bytes(out)
        p = 0
        for r in range(nrows):
            for c in names:
                planes[c][y0 + r] = np.frombuffer(buf, np.float32, W, p)
                p += W * 4
    return planes, W, H


def convert_envmap(exr_path, hdr_path, width=4096, height=2048):
    """Equal-area octahedral (pbrt) -> equirectangular, with the sky light's
    `Scale -1 1 1` / `Rotate 90 0 0 1` light-to-world baked in."""
    import numpy as np, cv2
    planes, W, H = read_exr(exr_path)
    img = np.stack([planes['B'], planes['G'], planes['R']], axis=-1)  # cv2 order

    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    phi = (u * 2.0 * math.pi)[None, :]
    theta = (v * math.pi)[:, None]
    ones = np.ones((height, width))
    d = np.stack([np.sin(theta) * np.cos(phi) * ones,
                  np.cos(theta) * ones,
                  np.sin(theta) * np.sin(phi) * ones], axis=-1)

    # light-to-world = Scale(-1,1,1) @ Rotate(90 deg, z), which works out to a
    # plain x/y swap and is its own inverse.
    R = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    # d is in *our* world, which is the pbrt world rotated by WORLD_FIX, so
    # undo that before going into light space: dl = R^T Wfix^T d.
    Wf = np.array([r[:3] for r in WORLD_FIX[:3]])
    dl = d.reshape(-1, 3) @ (Wf @ R)

    su, sv = equal_area_sphere_to_square(dl[:, 0], dl[:, 1], dl[:, 2])
    px = np.clip((su * W).astype(np.int32), 0, W - 1)
    py = np.clip((sv * H).astype(np.int32), 0, H - 1)
    out = img[py, px].reshape(height, width, 3).astype(np.float32)
    cv2.imwrite(hdr_path, out)
    return out


# ----------------------------------------------------------------------- main

def main():
    os.makedirs(MESH, exist_ok=True)
    os.makedirs(TEX, exist_ok=True)

    scene_file = os.path.join(SRC, 'sportscar-sky.pbrt')
    if not os.path.isfile(scene_file):
        raise SystemExit('not found: %s\npass the sportscar dir as argv[1]'
                         % scene_file)
    materials, shapes, cam = parse_pbrt(scene_file)
    print('parsed %d materials, %d shapes' % (len(materials), len(shapes)))

    # group shapes by material, keeping each shape's own CTM
    groups = {}
    for mat, ply, M in shapes:
        groups.setdefault(mat, []).append((os.path.join(SRC, ply), M))

    blocks = []
    for mat, entries in sorted(groups.items()):
        if mat not in materials:
            print('  skipping %r: no such material' % mat)
            continue
        obj = os.path.join(MESH, '%s.obj' % re.sub(r'[^\w.-]', '_', mat))
        if XML_ONLY and os.path.exists(obj):
            tris = -1
        else:
            # premultiply the Z-up -> Y-up fix onto each shape's own CTM
            tris = write_obj([(p, mat_mul(WORLD_FIX, M)) for p, M in entries], obj)
        kind = materials[mat]['type'][1][0]
        print('  %-34s %2d ply  %7d tris  (%s)'
              % (mat, len(entries), tris, kind))
        blocks.append(
            '\t<!-- %s: pbrt "%s", %d tris -->\n'
            '\t<mesh type="obj">\n'
            '\t\t<string name="filename" value="meshes/%s"/>\n'
            '%s\n'
            '\t</mesh>' % (mat, kind, tris, os.path.basename(obj),
                           to_nori_bsdf(mat, materials[mat])))

    env_out = os.path.join(TEX, 'sky.hdr')
    if not os.path.exists(env_out):
        print('converting envmap (equal-area octahedral -> equirect) ...')
        convert_envmap(os.path.join(SRC, cam['envmap']), env_out)

    W, H = WIDTH, HEIGHT
    # pbrt's fov is the angle subtended by the *shorter* image axis; ours is
    # horizontal, so widen it when the film is landscape.
    half = math.tan(math.radians(cam['fov'] * 0.5))
    fov = 2.0 * math.degrees(math.atan(half * (W / H if W > H else 1.0)))
    v = VIEWS[VIEW]
    if v is None:
        to_world = lookat_matrix(fix_point(cam['eye']), fix_point(cam['look']),
                                 fix_point(cam['up']))
        cam_note = ('pbrt: LookAt %g %g %g  %g %g %g  %g %g %g'
                    % tuple(cam['eye'] + cam['look'] + cam['up']))
    else:
        # already in renderer space; no fix_point
        fov = v['fov']
        to_world = lookat_matrix(v['eye'], v['look'], v['up'])
        cam_note = ('view %r: eye %g %g %g  look %g %g %g (renderer space)'
                    % ((VIEW,) + tuple(v['eye']) + tuple(v['look'])))

    xml = [
        "<?xml version='1.0' encoding='utf-8'?>", '',
        '<!-- pbrt-v4 sportscar (sky variant), converted by',
        '     tools/exporters/pbrt_sportscar_to_nori.py - edit that, not this file.',
        '',
        '     Lit purely by the environment: the five area-light planes in',
        '     sportscar-sky.pbrt have their Shape lines commented out upstream,',
        '     so they emit nothing. Six of the car materials are pbrt "measured"',
        '     RGL BSDFs approximated by Disney lobes; see MEASURED in the',
        '     converter for what each became and why. -->',
        '<scene>',
        '\t<string name="envmap" value="textures/sky.hdr"/>',
        '\t<float name="envmapScale" value="1.0"/>',
        '\t<float name="evCompensation" value="0.0"/>',
        '\t<float name="bloomIntensity" value="0.06"/>',
        '\t<float name="bloomThreshold" value="1.4"/>',
        '\t<float name="bloomKnee" value="0.4"/>', '',
        '\t<camera type="perspective">',
        '\t\t<float name="fov" value="%.4f"/>' % fov,
        '\t\t<!-- %s. A matrix rather' % cam_note,
        '\t\t     than <lookat>: both renderers build the same basis, but pbrt',
        '\t\t     sends camera +x to screen right and ours sends it left, which',
        '\t\t     mirrors the image. The first column is negated to undo it. -->',
        '\t\t<transform name="toWorld">',
        '\t\t\t<matrix value="%s"/>' % to_world,
        '\t\t</transform>',
        '\t\t<integer name="width" value="%d"/>' % W,
        '\t\t<integer name="height" value="%d"/>' % H,
        '\t</camera>', '',
        '\t<sampler type="independent">',
        '\t\t<integer name="sampleCount" value="%d"/>' % SPP,
        '\t</sampler>', '',
    ]
    xml.extend(blocks)
    xml.append('</scene>')

    out = os.path.join(OUT, 'scene.xml' if VIEW == 'pbrt'
                       else 'scene_%s.xml' % VIEW)
    open(out, 'w').write('\n'.join(xml) + '\n')
    print('\nwrote %s' % out)
    print('camera: fov %.2f (pbrt %.1f on the short axis) at %dx%d, %d spp'
          % (fov, cam['fov'], W, H, SPP))


if __name__ == '__main__':
    main()
