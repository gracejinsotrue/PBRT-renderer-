# Procedural duvet for scenes/liminal_bed.
#
# A blanket reads as a blanket because of three things, all of which the earlier
# metaball version lacked:
#   1. it is a FINITE PIECE OF CLOTH - there is a hem you can trace all the way round
#   2. its surface is developable-ish - broad arcs meeting at SHARP creases, not
#      doubly-curved everywhere (doubly curved everywhere = rubber / pool noodles)
#   3. it has THICKNESS, so the cut edge shows as a rolled hem and the folds end in
#      rounded tubes where they run off the foot of the bed
# So this builds a thick sheet: a top surface over a bottom surface, joined round a
# rectangular hem.  The fold profile across the bed is a row of半 semicircles meeting
# at cusps - the cusp is what gives a real crease; a sine wave gives corrugated roofing.
#   exec(open(r"C:\Users\gjin3\Desktop\nori-26sp\tools\scene_build\duvet_build.py").read())
import bpy, math
import numpy as np

DUVET_OBJ = globals().get("DUVET_OBJ", "Duvet")
G = globals()
def P(k, d):  return float(G[k]) if k in G else d
def PI(k, d): return int(G[k]) if k in G else d

ZM       = 0.808                 # plinth top (bed floats ~0.55 m over the grass)
PX0, PX1 = -0.700, 0.700         # plinth footprint
PY0, PY1 = -1.160, 1.160

X0, X1  = P("DUVET_X0", -0.960), P("DUVET_X1", 0.930)   # the piece of cloth, in plan: 26 cm past the near edge
Y0, Y1  = P("DUVET_Y0", -1.470), P("DUVET_Y1", 0.400)   # Y1 tucks under the front pillow
CELLMIN = P("DUVET_CELLMIN", 0.280)  # fold pitch range
CELLMAX = P("DUVET_CELLMAX", 0.470)
AMIN    = P("DUVET_AMIN", 0.095)     # thickness left in the crease: down does not pinch to nothing
QMIN    = P("DUVET_QMIN", 0.27)      # cross-seam (box quilt) pitch range along the bed
QMAX    = P("DUVET_QMAX", 0.60)
QFLOOR  = P("DUVET_QFLOOR", 0.60)    # how much loft survives at a cross seam
ALO     = P("DUVET_ALO", 0.265)      # fold height range - keep this SPREAD NARROW:
                                     # a fold much taller than its neighbours is a lump
AHI     = P("DUVET_AHI", 0.365)
THK     = P("DUVET_THK", 0.280)      # thickness where it hangs free of the bed
DROP    = P("DUVET_DROP", 0.340)     # how far it droops past the plinth edge
DROPW   = P("DUVET_DROPW", 0.240)
RH      = P("DUVET_RH", 0.170)       # hem roll radius
WOB     = P("DUVET_WOB", 0.065)      # folds wander instead of running dead straight
WRINK   = P("DUVET_WRINK", 0.40)     # above ~0.6 the crease streaks read as FUR
PUFF    = P("DUVET_PUFF", 1.0)       # down-clump lumpiness, 4-10 cm
UVTILE  = P("DUVET_UVTILE", 0.40)    # metres per fabric texture tile
NU      = PI("DUVET_NU", 340)
NV      = PI("DUVET_NV", 360)
SEED    = PI("DUVET_SEED", 17)

rng = np.random.default_rng(SEED)

def vnoise(X, Y, freq, seed, n=64):
    g = np.random.default_rng(seed).random((n, n))
    fx = X * freq; fy = Y * freq
    i0 = np.floor(fx).astype(np.int64); j0 = np.floor(fy).astype(np.int64)
    tx = fx - i0; ty = fy - j0
    tx = tx * tx * (3.0 - 2.0 * tx); ty = ty * ty * (3.0 - 2.0 * ty)
    a = g[i0 % n, j0 % n]; b = g[(i0 + 1) % n, j0 % n]
    c = g[i0 % n, (j0 + 1) % n]; d = g[(i0 + 1) % n, (j0 + 1) % n]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty

def fbm(X, Y, f, seed, oct=3):
    s = 0.0; a = 1.0; t = 0.0
    for k in range(oct):
        s = s + a * vnoise(X, Y, f * (2 ** k), seed + 17 * k); t += a; a *= 0.5
    return s / t

def sgn(a): return (a - 0.5) * 2.0

def clustered(a, b, n, amt=0.55):
    s = np.linspace(0.0, 1.0, n)
    g = s - amt * np.sin(2.0 * math.pi * s) / (2.0 * math.pi)
    return a + (b - a) * g

xs = clustered(X0, X1, NU)
ys = clustered(Y0, Y1, NV)
X, Y = np.meshgrid(xs, ys, indexing='ij')
XU, YU = X.copy(), Y.copy()          # unwarped, used for the boundary distance
inside_u = np.minimum(np.minimum(XU - PX0, PX1 - XU), np.minimum(YU - PY0, PY1 - YU))
X = X + 0.030 * sgn(fbm(XU, YU, 1.15, 501, 2)) + 0.080 * sgn(fbm(XU, YU, 0.40, 503, 2)) + 0.014 * sgn(fbm(XU, YU, 3.0, 505, 1))
Y = Y + 0.030 * sgn(fbm(XU, YU, 1.15, 502, 2)) + 0.080 * sgn(fbm(XU, YU, 0.40, 504, 2)) + 0.014 * sgn(fbm(XU, YU, 3.0, 506, 1))

# ---- the fold field: irregular cells, each a semicircular roll, cusps between ----
edges = [X0 - 0.22]
while edges[-1] < X1 + 0.22:
    edges.append(edges[-1] + rng.uniform(CELLMIN, CELLMAX))
edges = np.array(edges, np.float64)
amps  = rng.uniform(ALO, AHI, size=len(edges) - 1)

xw = X + WOB * sgn(fbm(X * 0.0 + 3.0, Y, 0.75, 77, 2)) + 0.035 * sgn(fbm(X, Y, 1.9, 91, 2))
k  = np.clip(np.searchsorted(edges, xw) - 1, 0, len(edges) - 2)
w  = (xw - edges[k]) / (edges[k + 1] - edges[k])
prof = np.sqrt(np.clip(1.0 - (2.0 * w - 1.0) ** 2, 0.0, 1.0))     # cusped, not sinusoidal

# each fold swells and fades along its own length
avar = 0.82 + 0.24 * vnoise(k.astype(np.float32) * 0.83, Y, 0.45, 909)
# and the whole duvet is fuller in the middle of the bed than at the head
# Height must stay EVEN along the bed.  Any localised concentration - a Gaussian
# band, a dome, a heap - reads as something lumped under the blanket, however it
# is shaped.  Only mild, aimless variation here.
along = (0.94 + 0.14 * sgn(fbm(X * 0.0 + 5.0, Y, 0.55, 733, 2))) \
        * (1.0 - 0.22 * np.clip((Y - (Y1 - 0.40)) / 0.40, 0.0, 1.0))
H = np.maximum(amps[k] * prof * avar * along, AMIN)

# BOX QUILTING.  The reference duvet is a grid, not channels: a second row of
# seams runs ACROSS the bed and pinches every channel into compartments.  Same
# cusped profile as the channels, shallower (QFLOOR), seams wandering a little.
yedges = [Y0 - 0.25]
while yedges[-1] < Y1 + 0.25:
    yedges.append(yedges[-1] + rng.uniform(QMIN, QMAX))
yedges = np.array(yedges, np.float64)
yq = Y + 0.035 * sgn(fbm(X, Y, 0.9, 641, 2)) + 0.16 * X * 0.0
kq = np.clip(np.searchsorted(yedges, yq) - 1, 0, len(yedges) - 2)
wq = (yq - yedges[kq]) / (yedges[kq + 1] - yedges[kq])
qprof = np.sqrt(np.clip(1.0 - (2.0 * wq - 1.0) ** 2, 0.0, 1.0))
qvar  = 0.80 + 0.40 * vnoise(X, kq.astype(np.float32) * 0.77, 1.0, 651)   # some seams bite deeper
H = H * (QFLOOR + (1.0 - QFLOOR) * qprof ** (0.85 * qvar))
cellvar = 0.82 + 0.36 * vnoise(k.astype(np.float32) * 0.83, kq.astype(np.float32) * 0.77, 1.0, 661)
H = H * cellvar                                   # each box filled differently
H = np.maximum(H, AMIN)
# NO isotropic bump here.  A smooth radial dome added on top of the fold field
# reads as something hidden UNDER the blanket.  Fullness at the head has to come
# through the folds themselves, which is what `along` and the cross family do.

# ---- rolled hem, computed early: folds merge into a continuous rolled edge as
# they reach it, otherwise a cusp landing on the boundary cuts a notch ----
db = np.minimum(np.minimum(XU - X0, X1 - XU), np.minimum(YU - Y0, Y1 - YU))
q = np.clip(db / RH, 0.0, 1.0)
roll = np.sqrt(np.clip(1.0 - (1.0 - q) ** 2, 0.0, 1.0))
merge = np.clip(1.0 - db / 0.32, 0.0, 1.0) ** 1.3
# the top edge of a duvet is thinner than its sides, and a full-height roll there
# is a wall that hides the pillows
headfac = np.clip((YU - (Y1 - 0.28)) / 0.28, 0.0, 1.0)
# the quilt seams pinch the hem too, so the edge scallops at every seam instead
# of running as a ruler line
qbite = 0.20 + 0.45 * vnoise(kq.astype(np.float32) * 0.77, XU * 0.0 + 2.0, 1.0, 671)   # per cross seam
cbite = 0.15 + 0.40 * vnoise(k.astype(np.float32) * 0.83, YU * 0.0 + 4.0, 1.0, 681)    # per channel seam
# cosine versions of the seam profiles: smooth minima, no cusps.  The cusped
# profiles stay for the folds up on the surface; the hem uses these.
cprof  = 0.5 - 0.5 * np.cos(2.0 * math.pi * w)
cqprof = 0.5 - 0.5 * np.cos(2.0 * math.pi * wq)
hem_cross = 1.0 - qbite * (1.0 - cqprof)
hem_chan  = 1.0 - cbite * (1.0 - cprof)
dside = np.minimum(XU - X0, X1 - XU); dend = np.minimum(YU - Y0, Y1 - YU)
wside = np.clip((dend - dside) / 0.18 + 0.5, 0.0, 1.0)     # 1 near the long edges, 0 near the ends
hemf = wside * hem_cross + (1.0 - wside) * hem_chan
sidefac = np.clip(-inside_u / 0.10, 0.0, 1.0)      # 1 where hanging free of the plinth
H = np.maximum(H, (AMIN + (0.235 - 0.185 * headfac - 0.10 * sidefac) * merge) * (0.55 + 0.45 * hemf))
H = H * (1.0 - 0.18 * sidefac)                       # a little thinner where it hangs

# ---- the surface it lies on, drooping once it leaves the plinth ----
inside = np.minimum(np.minimum(X - PX0, PX1 - X), np.minimum(Y - PY0, PY1 - Y))
# how far the fabric hangs varies along the edge: between seams it sags lower,
# at seams it is held up, and on top of that a slow wander so no two stretches match
sagv  = wside * (1.0 - cqprof) * qbite + (1.0 - wside) * (1.0 - cprof) * cbite
dropv = (0.78 + 0.32 * sagv
         + 0.38 * sgn(fbm(X, Y, 0.7, 771, 2))          # long tucks and spills
         + 0.20 * sgn(fbm(X, Y, 2.1, 772, 2)))         # and shorter ones
e = np.clip(-inside, 0.0, None) / (DROPW * (0.65 + 0.7 * np.clip(dropv, 0.3, 1.6)) / 1.0)
zb_free = ZM - DROP * (e * e * (3.0 - 2.0 * e) if True else e)
zb = np.where(inside > 0.0, ZM, ZM - DROP * np.clip(dropv, 0.55, 1.20) * np.clip(e, 0, 1) ** 2 * (3.0 - 2.0 * np.clip(e, 0, 1)))

top = zb + H
bot = np.where(inside > 0.0, zb, np.maximum(top - THK, zb - 0.02))

top = bot + (top - bot) * roll

# ---- fabric wrinkles as GEOMETRY, 2-6 cm ----
# One pixel of the final render covers ~7 mm of duvet, so the mip chain averages
# any normal-map detail finer than ~2 cm away before the shader samples it, and
# box-filtered normals flatten as they mip.  Real geometry does not, so the
# fabric crumple lives here.  Concentrated toward the fold creases (1 - prof),
# which is where a duvet cover actually crumples.
creaseW = 0.45 + 1.15 * (1.0 - prof)
top = top + WRINK * creaseW * (0.0030 * sgn(fbm(X, Y, 19.0, 301, 2))
                               + 0.0008 * sgn(fbm(X, Y, 41.0, 302, 2))) * roll

# ---- down-clump lumpiness, 4-10 cm, 1-2 cm high: THIS is the "texture" a down
# duvet has at this camera distance.  Positive-biased so clumps push outward,
# stronger on the fold tops than down in the creases, kept off the hem roll.
pf1 = fbm(X, Y, 8.5, 321, 2); pf2 = fbm(X, Y, 16.0, 322, 2)
puff = (0.026 * (np.clip(pf1, 0.35, 1.0) - 0.35) / 0.65
        + 0.009 * (np.clip(pf2, 0.40, 1.0) - 0.40) / 0.60)
top = top + PUFF * puff * (0.55 + 0.45 * prof) * roll

# ---- mesh: two sheets sharing one boundary ring -> closed, manifold ----
Ntop = NU * NV
ti = np.arange(Ntop, dtype=np.int64).reshape(NU, NV)
bi = np.empty((NU, NV), dtype=np.int64)
inner = (NU - 2) * (NV - 2)
bi[1:-1, 1:-1] = Ntop + np.arange(inner, dtype=np.int64).reshape(NU - 2, NV - 2)
bi[0, :] = ti[0, :]; bi[-1, :] = ti[-1, :]; bi[:, 0] = ti[:, 0]; bi[:, -1] = ti[:, -1]

footfac = np.clip(1.0 - (YU - Y0) / 0.22, 0.0, 1.0)
nearfac = np.clip(1.0 - (XU - X0) / 0.22, 0.0, 1.0)
farfac  = np.clip(1.0 - (X1 - XU) / 0.22, 0.0, 1.0)
Xl = X - 0.045 * cqprof * nearfac + 0.045 * cqprof * farfac
Yl = Y - 0.060 * cprof * footfac
co = np.empty((Ntop + inner, 3), np.float32)
co[:Ntop, 0] = Xl.ravel(); co[:Ntop, 1] = Yl.ravel(); co[:Ntop, 2] = top.ravel()
co[Ntop:, 0] = Xl[1:-1, 1:-1].ravel(); co[Ntop:, 1] = Yl[1:-1, 1:-1].ravel(); co[Ntop:, 2] = bot[1:-1, 1:-1].ravel()

def quads(idx, flip):
    a = idx[:-1, :-1]; b = idx[1:, :-1]; c = idx[1:, 1:]; d = idx[:-1, 1:]
    return np.stack([a, d, c, b] if flip else [a, b, c, d], axis=-1).reshape(-1, 4)

F = np.concatenate([quads(ti, False), quads(bi, True)], axis=0)
nf = F.shape[0]

me = bpy.data.meshes.new("DuvetSheet")
me.vertices.add(co.shape[0]); me.vertices.foreach_set("co", co.ravel())
me.loops.add(nf * 4); me.loops.foreach_set("vertex_index", F.astype(np.int32).ravel())
me.polygons.add(nf); me.polygons.foreach_set("loop_start", np.arange(nf, dtype=np.int32) * 4)
try: me.polygons.foreach_set("loop_total", np.full(nf, 4, dtype=np.int32))
except Exception: pass
me.update(calc_edges=True); me.validate(verbose=False)
me.polygons.foreach_set("use_smooth", np.ones(nf, np.int8)); me.update()

# UVs by ARC LENGTH over the surface, not by plan position.  A planar projection
# onto a near-vertical fold flank stretches the texture along the slope, and
# isotropic grain turns into long streaks that read as fur.  Arc length is also
# the real material coordinate of gathered cloth.
def arclen_uv(Z):
    ddx = np.diff(X, axis=0); ddz = np.diff(Z, axis=0)
    su = np.concatenate([np.zeros((1, NV), np.float64),
                         np.cumsum(np.sqrt(ddx * ddx + ddz * ddz), axis=0)], axis=0)
    ddy = np.diff(Y, axis=1); ddzy = np.diff(Z, axis=1)
    sv = np.concatenate([np.zeros((NU, 1), np.float64),
                         np.cumsum(np.sqrt(ddy * ddy + ddzy * ddzy), axis=1)], axis=1)
    return su / UVTILE, sv / UVTILE
ut, vt = arclen_uv(top)
ub, vb = arclen_uv(bot)
vuv = np.empty((co.shape[0], 2), np.float32)
vuv[:Ntop, 0] = ut.ravel(); vuv[:Ntop, 1] = vt.ravel()
vuv[Ntop:, 0] = ub[1:-1, 1:-1].ravel(); vuv[Ntop:, 1] = vb[1:-1, 1:-1].ravel()
uvl = me.uv_layers.new(name="UVMap")
uvl.data.foreach_set("uv", vuv[F.ravel()].ravel())
me.update()

ob = bpy.data.objects.get(DUVET_OBJ)
if ob is None:
    ob = bpy.data.objects.new(DUVET_OBJ, bpy.data.meshes.new(DUVET_OBJ + "_tmp"))
    bpy.context.scene.collection.objects.link(ob)
old = ob.data
for m in list(old.materials): me.materials.append(m)
for mod in list(ob.modifiers): ob.modifiers.remove(mod)
ob.data = me; ob.matrix_world.identity()
if old.users == 0: bpy.data.meshes.remove(old)

print("duvet: %d folds, %d verts / %d quads" % (len(amps), co.shape[0], nf))
print("bounds x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f" %
      (X.min(), X.max(), Y.min(), Y.max(), min(top.min(), bot.min()), top.max()))
print("peak above plinth %.3f m" % (top.max() - ZM))
