# Pillow stack for scenes/liminal_bed: three rectangular pillows with real corners,
# built as thick sheets (top + bottom joined round a hem ring), same technique as
# the duvet.  The earlier metaball pillows were rounded blobs with no corners and
# read as bars of soap.
#   exec(open(r"C:\Users\gjin3\Desktop\nori-26sp\tools\scene_build\pillow_build.py").read())
import bpy, math
import numpy as np

ZM = 0.808
UVTILE = 0.40
G = globals()
def P(k, d): return float(G[k]) if k in G else d

def vnoise(X, Y, freq, seed, n=64):
    g = np.random.default_rng(seed).random((n, n))
    fx = X * freq; fy = Y * freq
    i0 = np.floor(fx).astype(np.int64); j0 = np.floor(fy).astype(np.int64)
    tx = fx - i0; ty = fy - j0
    tx = tx * tx * (3 - 2 * tx); ty = ty * ty * (3 - 2 * ty)
    a = g[i0 % n, j0 % n]; b = g[(i0 + 1) % n, j0 % n]
    c = g[i0 % n, (j0 + 1) % n]; d = g[(i0 + 1) % n, (j0 + 1) % n]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty

def fbm(X, Y, f, seed, oct=3):
    s = 0.0; a = 1.0; t = 0.0
    for k in range(oct):
        s = s + a * vnoise(X, Y, f * 2 ** k, seed + 17 * k); t += a; a *= 0.5
    return s / t

def pillow_mesh(name, L, W, T, seed, NU=150, NV=110, n_se=5.0):
    """A pillow: superellipse plan (rounded rectangle with real corners), elliptical
    cross-section, flat fabric ears at the corners where the case outruns the fill."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-L / 2, L / 2, NU); ys = np.linspace(-W / 2, W / 2, NV)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    r = ((np.abs(X) / (L / 2)) ** n_se + (np.abs(Y) / (W / 2)) ** n_se) ** (1.0 / n_se)
    rc = np.clip(r, 0.0, 1.0)
    # A stuffed pillow keeps nearly full thickness right out to the edge and then
    # rolls over tightly.  An elliptical section (the earlier version) tapers to a
    # knife edge and reads as folded paper.
    prof = np.sqrt(np.clip(1.0 - rc ** 7.0, 0.0, 1.0))
    h = T * 0.5 * prof
    h = h * (1.0 + 0.07 * np.exp(-(rc ** 2) / 0.35))                    # fuller in the middle
    # down clumps, positive-biased so they push outward
    pf = fbm(X, Y, 9.0, seed + 1, 2)
    top = h + (0.020 * (np.clip(pf, 0.35, 1.0) - 0.35) / 0.65
               + 0.006 * (fbm(X, Y, 20.0, seed + 2, 2) - 0.5) * 2.0) * prof
    bot = -h * 0.85                                                     # a little flatter underneath
    # the plan boundary is the rectangle; beyond the superellipse the two sheets meet: the fabric ear
    top = np.where(r >= 1.0, 0.0, top); bot = np.where(r >= 1.0, 0.0, bot)

    Ntop = NU * NV
    ti = np.arange(Ntop, dtype=np.int64).reshape(NU, NV)
    bi = np.empty((NU, NV), np.int64)
    inner = (NU - 2) * (NV - 2)
    bi[1:-1, 1:-1] = Ntop + np.arange(inner, dtype=np.int64).reshape(NU - 2, NV - 2)
    bi[0, :] = ti[0, :]; bi[-1, :] = ti[-1, :]; bi[:, 0] = ti[:, 0]; bi[:, -1] = ti[:, -1]
    co = np.empty((Ntop + inner, 3), np.float32)
    co[:Ntop, 0] = X.ravel(); co[:Ntop, 1] = Y.ravel(); co[:Ntop, 2] = top.ravel()
    co[Ntop:, 0] = X[1:-1, 1:-1].ravel(); co[Ntop:, 1] = Y[1:-1, 1:-1].ravel(); co[Ntop:, 2] = bot[1:-1, 1:-1].ravel()
    def quads(idx, flip):
        a = idx[:-1, :-1]; b = idx[1:, :-1]; c = idx[1:, 1:]; d = idx[:-1, 1:]
        return np.stack([a, d, c, b] if flip else [a, b, c, d], axis=-1).reshape(-1, 4)
    F = np.concatenate([quads(ti, False), quads(bi, True)], axis=0)
    uv = np.stack([(co[:, 0] + L / 2) / UVTILE, (co[:, 1] + W / 2) / UVTILE], axis=-1).astype(np.float32)
    return co, F, uv

def place(co, rotz, tiltx, cx, cy, zbase, rolly=0.0):
    """roll about y, tilt about x, rotate about z, then set the lowest point to zbase.
    A pillow thrown on a bed is never square to anything: every one gets all three."""
    c, s_ = math.cos(rotz), math.sin(rotz)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], np.float32)
    c2, s2 = math.cos(tiltx), math.sin(tiltx)
    Rx = np.array([[1, 0, 0], [0, c2, -s2], [0, s2, c2]], np.float32)
    c3, s3 = math.cos(rolly), math.sin(rolly)
    Ry = np.array([[c3, 0, s3], [0, 1, 0], [-s3, 0, c3]], np.float32)
    P_ = co @ Ry.T @ Rx.T @ Rz.T
    P_[:, 0] += cx; P_[:, 1] += cy
    P_[:, 2] += zbase - P_[:, 2].min()
    return P_

def build(name, co, F, uv, mat):
    nf = F.shape[0]
    me = bpy.data.meshes.new(name + "_mesh")
    me.vertices.add(co.shape[0]); me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.loops.add(nf * 4); me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
    me.polygons.add(nf); me.polygons.foreach_set("loop_start", np.arange(nf, dtype=np.int32) * 4)
    try: me.polygons.foreach_set("loop_total", np.full(nf, 4, np.int32))
    except Exception: pass
    me.update(calc_edges=True); me.validate(verbose=False)
    me.polygons.foreach_set("use_smooth", np.ones(nf, np.int8))
    uvl = me.uv_layers.new(name="UVMap"); uvl.data.foreach_set("uv", uv[F.ravel()].ravel())
    me.update()
    me.materials.append(mat)
    ob = bpy.data.objects.get(name)
    if ob is None:
        ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    else:
        old = ob.data; ob.data = me
        if old.users == 0: bpy.data.meshes.remove(old)
    ob.matrix_world.identity()
    return co[:, 2].max()

mat = bpy.data.materials["Duvet.001"]        # same fabric as the duvet
L, W, T = 0.74, 0.50, 0.26

# Pillows LEAN, they do not stack flat: the back one nearly upright, the middle
# one resting back against it, the front one resting against that.  Each one's
# bottom edge is down on the bed and its top edge is up against the one behind.
# (tiltx rotates the width axis up and back toward the head, +Y.)
co, F, uv = pillow_mesh("Pillow_3", 0.88, W, T * 0.95, 43)
zC = build("Pillow_3", place(co, -0.16, math.radians(60.0), -0.24, 1.00, ZM - 0.035, rolly=math.radians(3.0)), F, uv, mat)
co, F, uv = pillow_mesh("Pillow_2_Pillow_2_0", L, W, T, 42)
zB = build("Pillow_2_Pillow_2_0", place(co,  0.18, math.radians(43.0),  0.02, 0.80, ZM + 0.02, rolly=math.radians(5.0)), F, uv, mat)
co, F, uv = pillow_mesh("Pillow_1_Pillow_1_0", L, W, T, 41)
zA = build("Pillow_1_Pillow_1_0", place(co, -0.12, math.radians(27.0), -0.06, 0.64, ZM + 0.06, rolly=math.radians(-4.0)), F, uv, mat)
print("pillow tops: %.3f %.3f %.3f  (plinth top %.3f)" % (zA, zB, zC, ZM))
