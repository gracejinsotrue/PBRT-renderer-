# tools/scene_build/slide_build.py -- procedural water-slide flumes for scenes/pool_store
#
# Sweeps a cross-section along a Catmull-Rom path with parallel-transport frames.
# Handles the closed-tube -> open-flume transition, wall thickness with real rim
# edges, ring flanges, banking, and a scalloped rim. Run inside Blender:
#
#   g = {"SLIDE_WHICH": "orange"}     # or "teal", or "both"
#   exec(open(r"C:\Users\gjin3\Desktop\nori-26sp\tools\scene_build\slide_build.py").read(), g)
#
# Geometry is in Blender world space (Z up, corridor along +Y), fitted to the
# reference by back-projection. Objects are created smooth-shaded with an
# EdgeSplit modifier so the exporter's corner normals keep the rims crisp.

import bpy, numpy as np, math

# ---------------------------------------------------------------- path utils
def catmull(P, n):
    P = np.asarray(P, float)
    Q = np.vstack([P[0] + (P[0] - P[1]), P, P[-1] + (P[-1] - P[-2])])
    segs = len(P) - 1
    ts = np.linspace(0.0, segs, n)
    out = np.zeros((n, 3))
    for i, t in enumerate(ts):
        k = min(int(t), segs - 1); u = t - k
        p0, p1, p2, p3 = Q[k], Q[k + 1], Q[k + 2], Q[k + 3]
        u2 = u * u; u3 = u2 * u
        out[i] = 0.5 * ((2 * p1) + (-p0 + p2) * u +
                        (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 +
                        (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
    return out

def arc_resample(C, n):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    s = np.linspace(0.0, d[-1], n)
    return np.stack([np.interp(s, d, C[:, k]) for k in range(3)], 1), d[-1]

def frames(C):
    """Parallel-transport frames. N is the 'floor of the flume' direction."""
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.zeros_like(T)
    ref = np.array([0.0, 0.0, -1.0])
    n0 = ref - T[0] * np.dot(ref, T[0])
    if np.linalg.norm(n0) < 1e-6:
        n0 = np.array([1.0, 0.0, 0.0]) - T[0] * T[0, 0]
    N[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(T)):
        v = np.cross(T[i - 1], T[i]); s = np.linalg.norm(v)
        if s < 1e-9:
            N[i] = N[i - 1]
        else:
            v /= s; a = math.atan2(s, float(np.dot(T[i - 1], T[i])))
            ca, sa = math.cos(a), math.sin(a)
            N[i] = (N[i - 1] * ca + np.cross(v, N[i - 1]) * sa +
                    v * float(np.dot(v, N[i - 1])) * (1 - ca))
        N[i] -= T[i] * np.dot(N[i], T[i])
        N[i] /= np.linalg.norm(N[i])
    B = np.cross(T, N)
    return T, N, B

def smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)

# ---------------------------------------------------------------- the sweep
def build_flume(name, path, radius, thickness=0.030, M=48, N_ST=None,
                close_to=0.50, open_to=0.62, open_deg=104.0, closed_deg=179.0,
                ell_b=1.0, ell_n=1.0, bank_deg=0.0,
                flange_every=0.0, flange_r=0.045, flange_w=0.055, rad_prof=None,
                scallop_amp=0.0, scallop_n=3.0, scallop_from=0.0, scallop_to=0.30,
                end_lip=0.0, mat=None):
    C0 = catmull(path, 900)
    length = float(np.sum(np.linalg.norm(np.diff(C0, axis=0), axis=1)))
    if N_ST is None:
        N_ST = max(60, int(length / 0.045))
    C, length = arc_resample(C0, N_ST)
    T, Nd, Bd = frames(C)
    t = np.linspace(0.0, 1.0, N_ST)
    s = t * length

    # coverage half-angle: closed tube -> open flume
    A = np.radians(closed_deg + (open_deg - closed_deg) * smoothstep(close_to, open_to, t))
    if scallop_amp > 0.0:
        w = smoothstep(scallop_to, scallop_from, t)          # 1 at the top, 0 lower down
        A = A - np.radians(scallop_amp) * w * (0.5 - 0.5 * np.cos(2 * math.pi * scallop_n * t))

    # radius with flange rings on the closed run
    R = np.full(N_ST, float(radius))
    if flange_every > 0.0:
        k = np.mod(s + flange_every * 0.5, flange_every) - flange_every * 0.5
        bump = np.exp(-(k / flange_w) ** 2) * flange_r
        R = R + bump * (1.0 - smoothstep(close_to - 0.04, open_to, t))

    if rad_prof:
        R = R * np.interp(t, [q[0] for q in rad_prof], [q[1] for q in rad_prof])

    if bank_deg:
        phi = np.radians(bank_deg) * smoothstep(0.0, 1.0, t)
    else:
        phi = np.zeros(N_ST)

    j = np.arange(M)
    inner = np.zeros((N_ST, M, 3)); outer = np.zeros((N_ST, M, 3))
    for i in range(N_ST):
        th = np.linspace(-A[i], A[i], M) + phi[i]
        cn = np.cos(th) * ell_n; cb = np.sin(th) * ell_b
        d = cn[:, None] * Nd[i][None, :] + cb[:, None] * Bd[i][None, :]
        dl = np.linalg.norm(d, axis=1, keepdims=True)
        u = d / dl
        rr = R[i] * dl[:, 0]
        lip = 0.0
        if end_lip and t[i] > 0.90:
            lip = end_lip * smoothstep(0.90, 1.0, t[i])
        inner[i] = C[i] + u * (rr[:, None])
        outer[i] = C[i] + u * ((rr + thickness + lip)[:, None])

    V = np.concatenate([inner.reshape(-1, 3), outer.reshape(-1, 3)], 0)
    OFF = N_ST * M
    def idx(i, jj, out=False):
        return (OFF if out else 0) + i * M + jj
    F = []
    for i in range(N_ST - 1):
        for jj in range(M - 1):
            F.append((idx(i, jj), idx(i, jj + 1), idx(i + 1, jj + 1), idx(i + 1, jj)))          # inner
            F.append((idx(i, jj, 1), idx(i + 1, jj, 1), idx(i + 1, jj + 1, 1), idx(i, jj + 1, 1)))  # outer
        F.append((idx(i, 0), idx(i + 1, 0), idx(i + 1, 0, 1), idx(i, 0, 1)))                     # rim A
        F.append((idx(i, M - 1, 1), idx(i + 1, M - 1, 1), idx(i + 1, M - 1), idx(i, M - 1)))     # rim B
    for jj in range(M - 1):                                                                       # end annuli
        F.append((idx(0, jj), idx(0, jj, 1), idx(0, jj + 1, 1), idx(0, jj + 1)))
        F.append((idx(N_ST - 1, jj + 1), idx(N_ST - 1, jj + 1, 1), idx(N_ST - 1, jj, 1), idx(N_ST - 1, jj)))

    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V], [], F)
    me.update(); me.validate()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if mat is not None:
        ob.data.materials.append(mat)
    m = ob.modifiers.new("EdgeSplit", 'EDGE_SPLIT')
    m.split_angle = math.radians(34.0)
    m.use_edge_angle = True; m.use_edge_sharp = False
    print(f"{name}: len {length:.2f} m, {N_ST} stations x {M}, {len(V)} verts, {len(F)} faces")
    return ob

def legs(name, feet, tops, r=0.035, mat=None):
    """Simple round support posts with a square base plate."""
    V, F = [], []
    for (fx, fy, fz), (tx, ty, tz) in zip(feet, tops):
        n0 = len(V); K = 12
        for k in range(K):
            a = 2 * math.pi * k / K
            V.append((fx + r * math.cos(a), fy + r * math.sin(a), fz))
        for k in range(K):
            a = 2 * math.pi * k / K
            V.append((tx + r * math.cos(a), ty + r * math.sin(a), tz))
        for k in range(K):
            k2 = (k + 1) % K
            F.append((n0 + k, n0 + k2, n0 + K + k2, n0 + K + k))
        b = len(V); h = 0.16
        V += [(fx - h, fy - h, fz), (fx + h, fy - h, fz), (fx + h, fy + h, fz), (fx - h, fy + h, fz),
              (fx - h, fy - h, fz + 0.02), (fx + h, fy - h, fz + 0.02), (fx + h, fy + h, fz + 0.02), (fx - h, fy + h, fz + 0.02)]
        F += [(b + 4, b + 5, b + 6, b + 7), (b, b + 1, b + 5, b + 4), (b + 1, b + 2, b + 6, b + 5),
              (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7)]
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); me.update(); me.validate()
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    if mat is not None: ob.data.materials.append(mat)
    return ob

# ---------------------------------------------------------------- fitted paths
ORANGE_PATH = [
    (1.06, 8.60, 4.60), (1.06, 7.40, 3.55), (1.06, 6.30, 2.75), (1.05, 5.30, 2.05),
    (1.04, 4.50, 1.52), (1.02, 4.08, 1.20), (0.92, 3.94, 0.92), (0.68, 3.96, 0.60),
    (0.34, 4.09, 0.32), (0.02, 4.23, 0.15), (-0.24, 4.35, 0.08),
]
TEAL_PATH = [
    (0.05, 5.80, 3.40), (0.04, 5.76, 2.90), (0.00, 5.72, 2.35), (-0.08, 5.68, 1.75),
    (-0.21, 5.63, 1.15), (-0.40, 5.58, 0.62), (-0.62, 5.52, 0.22), (-0.82, 5.47, 0.00),
    (-1.00, 5.43, -0.09),
]
ORANGE_FEET = [(0.95, 4.60, 0.0), (1.38, 4.35, 0.0)]
ORANGE_TOPS = [(1.00, 4.40, 0.80), (1.30, 4.30, 1.08)]

_which = globals().get("SLIDE_WHICH", "both")
_mo = bpy.data.materials.get("MatSlideOrg")
_mt = bpy.data.materials.get("MatSlideTeal")
_mm = bpy.data.materials.get("MatMetal") or bpy.data.materials.get("MatDeck")

for _n in ("SlideOrange", "SlideTeal", "SlideLegs", "PH_SlideOrange", "PH_SlideTeal"):
    _o = bpy.data.objects.get(_n)
    if _o and (_which == "both" or _n.lower().find(_which) >= 0 or _n.startswith("PH_")):
        bpy.data.objects.remove(_o, do_unlink=True)

if _which in ("orange", "both"):
    build_flume("SlideOrange", ORANGE_PATH, radius=0.430, thickness=0.030, M=56,
                close_to=0.78, open_to=0.86, open_deg=92.0, closed_deg=179.0,
                flange_every=1.10, flange_r=0.048, flange_w=0.050,
                rad_prof=[(0.0,1.0),(0.78,1.0),(0.92,1.08),(1.0,1.14)],
                end_lip=0.035, mat=_mo)
    legs("SlideLegs", ORANGE_FEET, ORANGE_TOPS, mat=_mm)
if _which in ("teal", "both"):
    build_flume("SlideTeal", TEAL_PATH, radius=0.310, thickness=0.022, M=48,
                close_to=1.10, open_to=1.20, open_deg=126.0, closed_deg=126.0,
                ell_b=1.12, ell_n=0.94,
                scallop_amp=26.0, scallop_n=2.5, scallop_from=0.02, scallop_to=0.34,
                mat=_mt)
bpy.ops.wm.save_mainfile()
print("slides built:", [o.name for o in bpy.context.scene.objects if o.name.startswith("Slide")])
