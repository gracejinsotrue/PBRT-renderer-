# pbrt-v4 "head" -> Nori scene converter, and the SSS-vs-Disney comparison set.
#
# Source: https://github.com/mmp/pbrt-v4-scenes/tree/master/head
#   geometry/head.ply            9223 verts / 17674 tris, binary LE, with UVs
#   textures/head_albedomap.png  the skin albedo the pbrt material feeds to
#                                Material "subsurface" as `reflectance`
#
# The pbrt scene renders the head exactly one way (a random-walk BSSRDF under a
# sky HDRI). This script keeps that material as the reference and adds the
# three other corners of a 2x2:
#
#                     sky HDRI              dark room + one area light
#   Disney BRDF   head_sky_disney.xml       head_dark_disney.xml
#   random walk   head_sky_sss.xml          head_dark_sss.xml
#
# Everything except the skin BSDF and the lighting block is identical across
# the four, so any difference in the images is the thing being compared.
#
# usage: python tools/exporters/pbrt_head_to_nori.py [path/to/pbrt-v4-scenes/head] [--preview]
import os, sys, math, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root: tools/<group>/ -> ..
_args = [a for a in sys.argv[1:] if not a.startswith('-')]
OUT = os.path.join(REPO, "scenes", "head")
SRC = _args[0] if _args else os.path.join(OUT, "_src")
MESH = os.path.join(OUT, "meshes")
TEX = os.path.join(OUT, "textures")
PREVIEW = '--preview' in sys.argv
# --xml-only reuses the mesh already on disk; the PN pass takes about a minute
# and rewrites 40 MB, which is wasted when only the light or a BSDF changed.
XML_ONLY = '--xml-only' in sys.argv

# reuse the tested PLY reader / OBJ writer from the BMW port
sys.path.insert(0, HERE)
import importlib.util
_s = importlib.util.spec_from_file_location('bmw', os.path.join(HERE, 'pbrt_bmw_to_nori.py'))
bmw = importlib.util.module_from_spec(_s)
_argv, sys.argv = sys.argv, [sys.argv[0]]   # keep its module-level arg parsing quiet
_s.loader.exec_module(bmw)
sys.argv = _argv


# --------------------------------------------------------------- geometry

# head.ply is 17674 triangles for a whole head -- about a 7 mm edge -- carrying
# normals smooth enough to make it LOOK like a much finer mesh. Those two facts
# fight each other under a single hard light: measured over the asset, a face
# normal sits 6.2 deg from its own corner normals on average and 32 deg at the
# 99th percentile, and on 6% of triangles at least one corner's shading normal
# says "facing the light" while the face it belongs to says "facing away".
#
# NEE offsets its shadow ray along the GEOMETRIC normal (Emitter.hlsli) but
# decides visibility from the SHADING normal, so every one of those triangles
# starts its shadow ray on the wrong side of its own surface and self-occludes.
# Under an envmap nobody notices - light arrives from everywhere. Under one
# area light it is a hard, triangle-shaped sawtooth right down the terminator,
# and it does not soften when you enlarge the light, because it is a geometry
# disagreement rather than a penumbra.
#
# The fix is to make the geometry agree with the normals rather than to move
# the light: PN triangles (Vlachos et al. 2001) replace each flat triangle with
# the cubic Bezier patch that interpolates its corner positions AND corner
# normals, so the tessellated surface actually curves the way the shading
# normals claim. Differentiating that patch for the shading normals (rather
# than using Vlachos's quadratic normal field, see patch_normal) takes the
# mean discrepancy from 6.2 deg to 1.1 and the sawtooth with it. It also rounds
# off the silhouette of the skull, which was visibly faceted against the sky.
PN_LEVEL = 4


def pn_subdivide(P, Nrm, UV, F, level):
    """PN-triangle tessellation. Returns (verts, normals, uvs, faces).

    Vertices are emitted per input triangle rather than welded; the patch is
    C0 across shared edges and the normal field is continuous there too, so
    the seams do not show and the bookkeeping stays simple.
    """
    p1, p2, p3 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    n1, n2, n3 = Nrm[F[:, 0]], Nrm[F[:, 1]], Nrm[F[:, 2]]
    t1, t2, t3 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]

    def w(pa, pb, na):
        return np.einsum('ij,ij->i', pb - pa, na)[:, None]

    b300, b030, b003 = p1, p2, p3
    b210 = (2 * p1 + p2 - w(p1, p2, n1) * n1) / 3.0
    b120 = (2 * p2 + p1 - w(p2, p1, n2) * n2) / 3.0
    b021 = (2 * p2 + p3 - w(p2, p3, n2) * n2) / 3.0
    b012 = (2 * p3 + p2 - w(p3, p2, n3) * n3) / 3.0
    b102 = (2 * p3 + p1 - w(p3, p1, n3) * n3) / 3.0
    b201 = (2 * p1 + p3 - w(p1, p3, n1) * n1) / 3.0
    E = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0
    Vc = (p1 + p2 + p3) / 3.0
    b111 = E + (E - Vc) / 2.0

    def patch(u, v):
        ww = 1.0 - u - v
        return (b300 * u ** 3 + b030 * v ** 3 + b003 * ww ** 3
                + b210 * (3 * u * u * v) + b120 * (3 * u * v * v)
                + b201 * (3 * u * u * ww) + b021 * (3 * v * v * ww)
                + b102 * (3 * u * ww * ww) + b012 * (3 * v * ww * ww)
                + b111 * (6 * u * v * ww))

    def patch_normal(u, v, e=1e-4):
        """The TRUE normal of the cubic patch, by central differences.

        Vlachos also defines a separate quadratic normal field, and using it is
        what a GPU tessellator does. Here it is the wrong choice: that field is
        not the derivative of the patch, so it disagrees with the tessellated
        geometry by a mean 3.9 deg (p99 26 deg) at level 4 -- i.e. it rebuilds
        the very mismatch the subdivision is meant to remove. Differentiating
        the patch instead drops that to 1.1 deg mean, p99 5.6. The tangent
        plane at a corner is still the input normal by construction, so the
        surface keeps the original scan's shading and stays continuous across
        patch edges.
        """
        du = (patch(u + e, v) - patch(u - e, v)) / (2 * e)
        dv = (patch(u, v + e) - patch(u, v - e)) / (2 * e)
        m = np.cross(du, dv)
        return m / np.maximum(np.linalg.norm(m, axis=1), 1e-20)[:, None]

    # barycentric lattice, ordered so a row-major walk gives the sub-triangles
    grid, index = [], {}
    for i in range(level + 1):
        for j in range(level + 1 - i):
            index[(i, j)] = len(grid)
            grid.append((i, j, level - i - j))
    nv = len(grid)

    verts = np.empty((len(F), nv, 3))
    norms = np.empty((len(F), nv, 3))
    uvs = np.empty((len(F), nv, 2))
    for k, (i, j, kk) in enumerate(grid):
        u, v, ww = i / level, j / level, kk / level
        verts[:, k] = patch(u, v)
        norms[:, k] = patch_normal(u, v)
        uvs[:, k] = t1 * u + t2 * v + t3 * ww

    tris = []
    for i in range(level):
        for j in range(level - i):
            tris.append((index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]))
            if j + i < level - 1:
                tris.append((index[(i + 1, j)], index[(i + 1, j + 1)],
                             index[(i, j + 1)]))
    tris = np.array(tris)
    base = (np.arange(len(F)) * nv)[:, None, None]
    faces = (tris[None, :, :] + base).reshape(-1, 3)
    return (verts.reshape(-1, 3), norms.reshape(-1, 3),
            uvs.reshape(-1, 2), faces)


def write_obj_pn(ply_path, out_path, level=PN_LEVEL):
    """head.ply -> PN-subdivided OBJ with unit normals."""
    import numpy as np
    verts, stride, has_uv, faces, nv = bmw.read_ply(ply_path)
    a = np.array(verts, np.float64).reshape(nv, stride)
    P, Nrm = a[:, 0:3], a[:, 3:6]
    Nrm = Nrm / np.maximum(np.linalg.norm(Nrm, axis=1), 1e-20)[:, None]
    UV = a[:, 6:8] if has_uv else np.zeros((nv, 2))
    F = np.array(faces)
    if level <= 1:
        V2, N2, T2, F2 = P, Nrm, UV, F
    else:
        V2, N2, T2, F2 = pn_subdivide(P, Nrm, UV, F, level)
    V2 = V2 * SCENE_SCALE            # metres -> centimetres, see SCENE_SCALE
    with open(out_path, 'w') as o:
        o.write("# %s, PN-subdivided level %d by tools/exporters/pbrt_head_to_nori.py\n"
                % (os.path.basename(ply_path), level))
        o.write('\n'.join('v %.6g %.6g %.6g' % tuple(p) for p in V2) + '\n')
        o.write('\n'.join('vt %.6g %.6g' % tuple(t) for t in T2) + '\n')
        o.write('\n'.join('vn %.5f %.5f %.5f' % tuple(n) for n in N2) + '\n')
        o.write('\n'.join('f %d/%d/%d %d/%d/%d %d/%d/%d'
                          % (f[0] + 1, f[0] + 1, f[0] + 1,
                             f[1] + 1, f[1] + 1, f[1] + 1,
                             f[2] + 1, f[2] + 1, f[2] + 1) for f in F2) + '\n')
    return len(F2)


def write_obj_unit_normals(ply_path, out_path):
    """head.ply -> OBJ, normalising the vertex normals on the way through.

    bmw.write_obj copies nx/ny/nz straight out of the PLY, which is fine for
    every other pbrt asset we have ported. head.ply is not: its normals are
    unnormalised and their lengths run from 3.6e3 to 5.5e7, a spread of four
    orders of magnitude BETWEEN ADJACENT VERTICES. Barycentric interpolation
    then gets dominated by whichever corner happens to be longest, so the
    shading normal snaps from vertex to vertex instead of varying smoothly and
    the head renders with hard, triangle-aligned steps wherever the light grazes
    it -- a sawtooth across the temple that looks like a shadow terminator bug
    but survives moving the light, because it is the normals, not the shadow.
    """
    verts, stride, has_uv, faces, nv = bmw.read_ply(ply_path)
    with open(out_path, 'w') as o:
        o.write("# %s, normals unitised - see tools/exporters/pbrt_head_to_nori.py\n"
                % os.path.basename(ply_path))
        for i in range(nv):
            b = i * stride
            o.write('v %.6g %.6g %.6g\n' % (verts[b], verts[b + 1], verts[b + 2]))
        if has_uv:
            for i in range(nv):
                b = i * stride
                o.write('vt %.6g %.6g\n' % (verts[b + 6], verts[b + 7]))
        for i in range(nv):
            b = i * stride
            n = norm([verts[b + 3], verts[b + 4], verts[b + 5]])
            o.write('vn %.6f %.6f %.6f\n' % tuple(n))
        for (a, b_, c) in faces:
            a += 1; b_ += 1; c += 1
            if has_uv:
                o.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n'
                        % (a, a, a, b_, b_, b_, c, c, c))
            else:
                o.write('f %d//%d %d//%d %d//%d\n' % (a, a, b_, b_, c, c))
    return len(faces)


# --------------------------------------------------------------- camera
#
# head.pbrt asks for a 1920x1080 frame and then throws most of it away:
#
#   "float cropwindow" [.3 .8 .15 .7]
#
# so the picture that scene actually produces is the 960x594 window in the
# middle-left of that frame. Our film has no crop, so the crop is folded into
# the camera instead: re-aim along the crop centre and narrow the fov to the
# crop's half-width. That is an on-axis approximation of pbrt's off-axis
# frustum -- the two differ by well under a pixel at this crop offset.
PBRT_EYE = (0.322839, 0.0534825, 0.504299)
PBRT_LOOK = (-0.140808, -0.162727, -0.354936)
PBRT_UP = (0.0355799, 0.964444, -0.261882)
PBRT_FOV = 30.0                 # pbrt: angle on the SHORTER axis
CROP = (0.3, 0.8, 0.15, 0.7)
FILM = (1920, 1080)

# pbrt's crop clips the crown of the skull dead flat against the top edge, and
# its 1.62:1 letterbox cannot hold a head and its neck at once. Opening the fov
# a touch recovers the crown; the frame is then made taller than pbrt's and the
# axis tilted so the bottom edge lands where the neck meets the shoulders.
#
# Measured off the mesh in camera units (tan-space, so scale-free): the crown
# projects to +0.201, and slicing the mesh by height shows the neck holding a
# steady 6.2-6.6 cm half-width from world y -16 to -9 before the trapezius
# flares out to 9.6, then 13, then 17 cm - that flare starts at camy -0.213.
# Bracketing exactly those two puts the half-height at 0.2133, i.e. a 1.45:1
# frame, with the axis 3.91 deg above pbrt's. Making the frame taller rather
# than widening the fov keeps the head the same size on screen.
FOV_SCALE = 1.30
TILT_DEG = 3.91
ASPECT = 1.45             # None to keep pbrt's crop aspect


def norm(v):
    l = math.sqrt(sum(c * c for c in v))
    return [c / l for c in v]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def camera():
    """Returns (origin, target, hfov_deg, width, height)."""
    aspect = FILM[0] / FILM[1]
    t = math.tan(math.radians(PBRT_FOV * 0.5))

    # pbrt screen window for aspect > 1 is x in [-aspect, aspect], y in [-1, 1],
    # and raster y runs downward, so screen_y = 1 - 2 * ndc_y.
    x0 = -aspect + CROP[0] * 2 * aspect
    x1 = -aspect + CROP[1] * 2 * aspect
    y0 = 1.0 - 2.0 * CROP[2]
    y1 = 1.0 - 2.0 * CROP[3]
    cx, hx = 0.5 * (x0 + x1), 0.5 * abs(x1 - x0)
    cy, hy = 0.5 * (y0 + y1), 0.5 * abs(y1 - y0)

    # pbrt camera basis: columns [right, up, dir], +z forward
    d = norm([PBRT_LOOK[i] - PBRT_EYE[i] for i in range(3)])
    right = norm(cross(norm(PBRT_UP), d))
    up = cross(d, right)

    # crop-centre ray, in camera space then world space
    cdir = (cx * t, cy * t, 1.0)
    wdir = norm([cdir[0] * right[i] + cdir[1] * up[i] + cdir[2] * d[i]
                 for i in range(3)])

    # tilt about the world-horizontal axis through the view direction, so the
    # frame stays level (our GPU camera has no roll anyway)
    horiz = norm(cross((0.0, 1.0, 0.0), wdir))
    vert = cross(wdir, horiz)
    a = math.radians(TILT_DEG)
    wdir = norm([wdir[i] * math.cos(a) + vert[i] * math.sin(a)
                 for i in range(3)])

    hfov = 2.0 * math.degrees(math.atan(hx * t * FOV_SCALE))
    ar = ASPECT if ASPECT else hx / hy
    width = 512 if PREVIEW else 1280
    height = int(round(width / ar))
    # focaldistance 0.573 in head.pbrt, i.e. the target sits on the face
    target = [PBRT_EYE[i] + wdir[i] * 0.573 for i in range(3)]
    return ([c * SCENE_SCALE for c in PBRT_EYE],
            [c * SCENE_SCALE for c in target], hfov, width, height)


# The whole scene is authored in CENTIMETRES rather than the source asset's
# metres. This is not cosmetic: SubsurfaceWalk is started at
# `hitPos - Ng * 0.001` (Shaders.hlsl), a hardcoded WORLD-SPACE offset, and at
# metre scale pbrt's 0.97 mm mean free path makes that offset 1.03 mean free
# paths deep. The walk therefore begins inside the medium instead of at the
# boundary, scatters more before it can escape, and loses energy - worst in the
# channels with the lowest single-scattering albedo, which is why the head came
# out both dark and red.
#
# Measured on a white-furnace sphere (see tools/analysis/sss_furnace.py), holding
# mfp/diameter fixed at 0.00115 so the physics is identical and only the world
# units change:
#
#   albedo 0.55 0.30 0.22, index-matched, theory says 0.553 0.299 0.219
#     metre scale  (offset 0.43 mfp)  -> 0.406 0.172 0.115
#     x100 scale   (offset 0.004 mfp) -> 0.507 0.259 0.187
#
# so the units alone are worth roughly +25% / +50% / +62% per channel, and they
# pull the red cast down with them. Nothing else about the scene changes:
# geometry, camera, light size and light distance all scale together, so solid
# angles and therefore radiances are untouched.
SCENE_SCALE = 100.0

ORIGIN, TARGET, FOV, W, H = camera()
SPP = 64 if PREVIEW else 1024

# pbrt's LookAt has a rolled up vector (about -15 deg). Our GPU camera rebuilds
# its basis from yaw/pitch alone (DXRApp_Camera.cpp) and cannot carry roll, so
# the roll is dropped and the frame is written as a plain <lookat> with world
# up. Nori's <lookat> and pbrt's LookAt build the same basis but disagree on
# which way camera +x points on screen, so this frame is pbrt's mirrored.


# --------------------------------------------------------------- lighting

def rotation_for(feature_az_in_hdri, wanted_world_az):
    """EvalEnvmap looks up atan2(d.z, d.x) + envmapRotation (radians, added to
    phi directly). To drag an HDRI feature to a chosen world azimuth, rotate by
    the difference."""
    return math.radians(feature_az_in_hdri - wanted_world_az)


# head.pbrt lights the head with small_rural_road, which is a full outdoor
# capture: its lower hemisphere is a tarmac road with grass and treeline, and
# that is what the first version of these renders put behind the head. Swapped
# for a Poly Haven "puresky" instead -- sky only, the ground replaced by a
# smooth gradient, so nothing below the horizon reads as a place.
#
# evening_road_01 out of the 59 puresky captures for two reasons.
#
#  * Its sun is low, 12.6 deg. A low sun rakes the face instead of dropping the
#    brow into its own shadow, and grazing light is what makes subsurface
#    transport visible: the longer the path through the ear and the wing of the
#    nose, the more of the light leaves somewhere other than where it entered.
#  * Its sun is DIFFUSED - peak radiance 22, a 22:1 contrast against the sky.
#    The clear-sky pureskies have a real solar disc at 10^5, and that is
#    unusable here for the same reason a small area light was: nothing does
#    next-event estimation from the point where a subsurface walk exits, so the
#    only way a walk finds the sun is for its exit ray to land on a disc
#    covering 1e-5 of the sphere. Tried syferfontein_18d_clear (peak 242000)
#    first and the face came back as salt-and-pepper fireflies with blown
#    specular patches. A diffused sun keeps the direction and drops the variance.
#
# Sun azimuth in the file is 216 deg; 100 deg puts it three-quarters front-left
# of a face that points at 59 deg. Its lower hemisphere is a bright gradient
# rather than a dark one, so the backdrop behind the head stays sky-coloured.
SKY = dict(envmap="textures/puresky.hdr", scale=1.0, ev=-0.8,
           rotation=rotation_for(216.0, 100.0))

# The dark pair is lit by a soft disc PAINTED INTO AN ENVIRONMENT MAP rather
# than by an area-light quad, and that is a workaround for a renderer bug, not
# a stylistic choice.
#
# The bug: a mesh area light produces a hard, triangle-aligned staircase at the
# terminator instead of a penumbra. It is not this asset. Reproduced on a bare
# 102400-triangle sphere with a plain <bsdf type="diffuse"> and a single 5x5
# quad at 9 units (scenes/_sss_furnace/ball.xml): the terminator comes out as a
# stair-stepped edge following the tessellation. Ruled out, by measurement:
#   * the mesh - a sphere does it too, and only 0.26% of this head's triangles
#     have a shading normal that disagrees in sign with their face normal
#   * tessellation - PN level 1, 4 and 8 (17k to 1.1M triangles) all show it
#   * scene scale - SCENE_SCALE 1, 10 and 100 are pixel-identical
#   * light size - 0.12 m and 0.50 m give the SAME edge, which alone proves it
#     is not a penumbra; a 4x larger source must give a 4x wider transition
# It only clears when the source is bigger than its own distance (ratio > ~2),
# i.e. when the light wraps so far round that no terminator exists at all - and
# that is flat, shapeless light.
#
# The envmap path does not have the bug: same geometry, same direction, clean
# gradient. It has its own next-event estimator (EnvmapDirectIllumination,
# sampling the luminance CDF) which is a separate code path from
# MISDirectIllumination. So the key is a 14 deg disc with a smootherstep edge
# burned into a 2048x1024 equirect map, which is a large studio softbox. Everything outside it sits at 0.004 - almost
# nothing, but a strictly black envmap gives the sampling CDF a zero integral.
#
# The disc sits at azimuth 125, elevation 30: front and slightly camera-left of
# a face pointing at 59 deg, so the light lands ON THE FACE with the shadow
# falling away toward the ear. It is 114 deg off the view axis, far outside a
# 34 deg frame, so the background stays black.
KEY_AZ, KEY_EL = 125.0, 30.0
# 22 deg rather than a tighter, brighter disc: skin at alpha 0.05 is glossy
# enough to mirror the source, and a 14 deg disc came back as a hard white
# blob on the forehead with the map's texel structure visible inside it.
KEY_RADIUS_DEG = 32.0
KEY_PEAK = 5.6
KEY_FLOOR = 0.004
KEY_TINT = (1.0, 0.80, 0.64)          # warm, ~3200 K

STUDIO = dict(envmap="textures/studio.hdr", scale=1.0, ev=0.0, rotation=0.0)

HEAD_CENTER = tuple(c * SCENE_SCALE for c in (0.0, -0.01, -0.05))


def write_studio_hdr(path, width=2048, height=1024):
    """Near-black equirect environment with one soft warm disc."""
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    phi = u * 2.0 * math.pi
    theta = v * math.pi
    # the renderer's lookup is phi = atan2(d.z, d.x), theta = acos(d.y)
    d = np.stack(np.broadcast_arrays(
        np.sin(theta)[:, None] * np.cos(phi)[None, :],
        np.cos(theta)[:, None] * np.ones(width)[None, :],
        np.sin(theta)[:, None] * np.sin(phi)[None, :]), axis=-1)
    a, e = math.radians(KEY_AZ), math.radians(KEY_EL)
    L = np.array([math.cos(a) * math.cos(e), math.sin(e),
                  math.sin(a) * math.cos(e)])
    ang = np.degrees(np.arccos(np.clip((d * L).sum(-1), -1.0, 1.0)))
    # smootherstep rather than a hard disc, so the shadow gets a real penumbra
    t = np.clip(1.0 - ang / KEY_RADIUS_DEG, 0.0, 1.0)
    img = np.full((height, width, 3), KEY_FLOOR, np.float32)
    img += ((t * t * t * (t * (t * 6 - 15) + 10))[..., None] * KEY_PEAK
            * np.array(KEY_TINT, np.float32))
    import cv2
    cv2.imwrite(path, img[..., ::-1].astype(np.float32))   # cv2 wants BGR
    sr = 2.0 * math.pi * (1.0 - math.cos(math.radians(KEY_RADIUS_DEG)))
    return sr, KEY_PEAK * math.pi * math.sin(math.radians(KEY_RADIUS_DEG)) ** 2


# --------------------------------------------------------------- materials

# head.pbrt:
#   Material "subsurface"
#     "texture reflectance" [ "albedomap" ]      <- textures/head_albedomap.png
#     "rgb mfp" [ 0.0012953 0.00095238 0.00067114 ]
#     "float eta" 1.33   "float uroughness/vroughness" 0.05  (remap off)
#
# Our BSSRDF takes a SCALAR mean free path (Subsurface.hlsli samples free
# flight from one sigma_t so the walk needs no spectral MIS) and recovers all
# of the colour from the per-channel single-scattering albedo, which it derives
# from the blurred reflectance texture. So the three mfp channels collapse to
# their mean, and the red-travels-furthest behaviour pbrt gets from the mfp
# spread we get instead from alpha: SubsurfaceParams turns the reflectance into
# an alpha whose blue channel is absorbed fastest.
#
# tint stays 1 -- unlike the head2 asset, whose walk needed the medium albedo
# lifted, this reflectance map is already a diffuse-albedo map and pbrt's own
# material feeds it in raw.
MFP = (0.0012953, 0.00095238, 0.00067114)          # metres, from head.pbrt
RADIUS = sum(MFP) / 3.0 * SCENE_SCALE

# Specular lobe, matched across the two arms so the comparison is about
# transport and not gloss. head.pbrt sets uroughness/vroughness 0.05 with
# remaproughness off, i.e. 0.05 IS the microfacet alpha -- but 0.05 is a wet
# lacquer, and pbrt gets away with it only because it lights the head with a
# soft sky, where a near-mirror lobe has nothing compact to reflect. Under a
# studio key it turns the scan's forehead creases into hard bright brackets, so
# this is opened to 0.16. Both arms read the same constant, so the comparison
# stays a comparison of transport. Both of our BSDFs take
# a perceptual roughness and square it (Disney.hlsli: "alpha = roughness
# squared so that roughness feels linear"), and the subsurface path additionally
# scales roughness by lerp(1, 0.65, saturate(specular * 2)) before squaring --
# 0.755 at specular 0.35. So the same alpha needs two different numbers.
#
# Leaving the walk at a literal 0.05 is what made the first dark render show a
# crisp white rectangle on the ear: alpha 0.0025 is a mirror, and a mirror
# reflects a recognisable picture of the light quad.
SPEC_ALPHA = 0.16
SPEC_SCALE = 0.755
ROUGH_WALK = math.sqrt(SPEC_ALPHA) / SPEC_SCALE
ROUGH_DISNEY = math.sqrt(SPEC_ALPHA)

ALBEDO_TEX = '\t\t\t<string name="albedoTexture" value="textures/head_albedomap.png"/>'


def skin_walk(radius=RADIUS):
    return """\t\t<bsdf type="subsurface">
\t\t\t<color name="albedo" value="0.62 0.44 0.36"/>
\t\t\t<color name="tint"   value="1.0 1.0 1.0"/>
%s
\t\t\t<float name="radius"    value="%.6g"/>
\t\t\t<float name="intIOR"    value="1.33"/>
\t\t\t<float name="extIOR"    value="1.0"/>
\t\t\t<float name="g"         value="0.0"/>
\t\t\t<float name="roughness" value="%.4f"/>
\t\t\t<float name="specular"  value="0.35"/>
\t\t</bsdf>""" % (ALBEDO_TEX, radius, ROUGH_WALK)


# The comparison arm. Disney has no transport model at all: `subsurface` blends
# in a Hanrahan-Krueger-ish retro-reflective lobe that lifts grazing angles,
# which reads as softness but never moves energy sideways through the mesh.
# Same albedo map, same specular level, so the images differ by the transport.
def skin_disney():
    return """\t\t<bsdf type="disney">
\t\t\t<color name="baseColor" value="0.62 0.44 0.36"/>
%s
\t\t\t<float name="roughness"      value="%.4f"/>
\t\t\t<float name="metallic"       value="0.0"/>
\t\t\t<float name="specular"       value="0.35"/>
\t\t\t<float name="sheen"          value="0.05"/>
\t\t\t<float name="sheenTint"      value="1.0"/>
\t\t\t<float name="subsurface"     value="0.5"/>
\t\t\t<float name="clearcoat"      value="0.05"/>
\t\t\t<float name="clearcoatGloss" value="0.8"/>
\t\t</bsdf>""" % (ALBEDO_TEX, ROUGH_DISNEY)


# --------------------------------------------------------------- assembly

HEADER = """<?xml version='1.0' encoding='utf-8'?>

<!-- %s

     Head geometry and skin albedo from mmp/pbrt-v4-scenes/head; converted by
     tools/exporters/pbrt_head_to_nori.py - edit that, not this file. The camera, sampler and
     mesh are identical across the four scenes in this folder, so the only
     things that vary are the skin BSDF and the light. -->
<scene>
\t<string name="envmap" value="%s"/>
\t<float name="envmapScale" value="%g"/>
\t<float name="envmapRotation" value="%.6f"/>
\t<float name="evCompensation" value="%g"/>

\t<camera type="perspective">
\t\t<float name="fov" value="%.4f"/>
\t\t<transform name="toWorld">
\t\t\t<lookat target="%.6f, %.6f, %.6f" origin="%.6f, %.6f, %.6f" up="0, 1, 0"/>
\t\t</transform>
\t\t<integer name="width" value="%d"/>
\t\t<integer name="height" value="%d"/>
\t</camera>

\t<sampler type="independent">
\t\t<integer name="sampleCount" value="%d"/>
\t</sampler>

\t<mesh type="obj">
\t\t<string name="filename" value="meshes/head.obj"/>
%s
\t</mesh>
"""

def scene(note, skin, light):
    return HEADER % (note, light['envmap'], light['scale'], light['rotation'],
                     light['ev'], FOV, TARGET[0], TARGET[1], TARGET[2],
                     ORIGIN[0], ORIGIN[1], ORIGIN[2], W, H, SPP,
                     skin) + "</scene>\n"


SCENES = {
    'head_sky_disney': ("Disney BRDF under the scene's own sky HDRI. No "
                        "transport: the softness is a shading trick.",
                        skin_disney(), SKY),
    'head_sky_sss': ("Random-walk BSSRDF under the scene's own sky HDRI - "
                     "pbrt's material, ported.", skin_walk(), SKY),
    'head_dark_disney': ("Disney BRDF, black room, one soft key on the face. "
                         "Light stops dead at the surface.",
                         skin_disney(), STUDIO),
    'head_dark_sss': ("Random-walk BSSRDF, black room, one soft key on the "
                      "face - light bleeds into the shadow and the ear glows.",
                      skin_walk(), STUDIO),
}


def main():
    os.makedirs(MESH, exist_ok=True)
    os.makedirs(TEX, exist_ok=True)

    ply = os.path.join(SRC, 'head.ply')
    if not os.path.exists(ply):
        ply = os.path.join(SRC, 'geometry', 'head.ply')
    obj = os.path.join(MESH, 'head.obj')
    if XML_ONLY and os.path.exists(obj):
        print('head.obj: reusing %s' % obj)
    else:
        tris = write_obj_pn(ply, obj)
        print('head.ply -> %s  (%d tris, PN level %d)' % (obj, tris, PN_LEVEL))

    tex = os.path.join(TEX, 'head_albedomap.png')
    if not os.path.exists(tex):
        for cand in (os.path.join(SRC, 'head_albedomap.png'),
                     os.path.join(SRC, 'textures', 'head_albedomap.png')):
            if os.path.exists(cand):
                shutil.copy(cand, tex)
                break
    print('albedo map: %s' % tex)

    sky = os.path.join(TEX, 'small_rural_road.hdr')
    if not os.path.exists(sky):
        shutil.copy(os.path.join(REPO, 'scenes', 'sssdragon', 'textures',
                                 'small_rural_road.hdr'), sky)
    print('sky envmap: %s' % sky)

    sr, irr = write_studio_hdr(os.path.join(TEX, 'studio.hdr'))
    print('key light: textures/studio.hdr - %g deg disc at az %g el %g '
          '(%.3f sr, irradiance ~%.1f)'
          % (KEY_RADIUS_DEG, KEY_AZ, KEY_EL, sr, irr))

    for name in sorted(SCENES):
        note, skin, light = SCENES[name]
        p = os.path.join(OUT, name + '.xml')
        open(p, 'w').write(scene(note, skin, light))
        print('wrote', p)

    print('\ncamera origin %.4f %.4f %.4f -> target %.4f %.4f %.4f'
          % (tuple(ORIGIN) + tuple(TARGET)))
    print('fov %.3f deg horizontal, %dx%d @ %d spp' % (FOV, W, H, SPP))
    print('sss radius %.6g m (mean of pbrt mfp %s)' % (RADIUS, MFP))


if __name__ == '__main__':
    main()
