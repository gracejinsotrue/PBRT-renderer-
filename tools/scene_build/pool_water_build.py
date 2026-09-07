import bpy, numpy as np, math
sc = bpy.context.scene

# ---------------- displaced water ----------------
# Reference streaks run near-horizontal in image => wave crests lateral,
# waves travelling along the corridor. Streaks ~33 x 13 mm in world near field.
PX0, PX1, PY0, PY1, ZW = -1.85, 0.75, -3.0, 20.0, -0.06
DX = 0.012
xs = np.arange(PX0, PX1 + 1e-9, DX)
ys = [PY0]; dy = DX
while ys[-1] < PY1:
    ys.append(ys[-1] + dy)
    if ys[-1] > 6.0: dy = min(0.12, dy*1.006)
ys = np.array(ys); ys[-1] = PY1
X, Y = np.meshgrid(xs, ys, indexing='ij')

def vnoise(X, Y, cell, seed):
    nx = int((PX1-PX0)/cell) + 4; ny = int((PY1-PY0)/cell) + 4
    g = np.random.default_rng(seed).random((nx, ny))
    u = (X-PX0)/cell + 1; v = (Y-PY0)/cell + 1
    i0 = np.clip(np.floor(u).astype(int), 0, nx-2); j0 = np.clip(np.floor(v).astype(int), 0, ny-2)
    fu = u-i0; fv = v-j0
    su = fu*fu*(3-2*fu); sv = fv*fv*(3-2*fv)
    a = g[i0,j0]*(1-su) + g[i0+1,j0]*su
    b = g[i0,j0+1]*(1-su) + g[i0+1,j0+1]*su
    return (a*(1-sv) + b*sv)*2 - 1

def wave(X, Y, lam, amp, ang_deg, ph):
    k = 2*math.pi/lam; a = math.radians(ang_deg)
    return amp*np.sin(k*(X*math.cos(a) + Y*math.sin(a)) + ph)

Z  = np.full_like(X, ZW)
Z += wave(X, Y, 0.80,  0.0060,  88.0, 0.7)
Z += wave(X, Y, 0.26,  0.0038,  79.0, 2.1)
Z += wave(X, Y, 0.130, 0.0017,  97.0, 4.0)
Z += wave(X, Y, 0.075, 0.00065, 84.0, 1.2)
Z += 0.0022*vnoise(X, Y, 0.35, 3) + 0.0009*vnoise(X, Y, 0.14, 7)

nx, ny = len(xs), len(ys)
verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
i = np.arange(nx-1)[:, None]*ny + np.arange(ny-1)[None, :]
quads = np.stack([i, i+ny, i+ny+1, i+1], axis=-1).reshape(-1, 4)

old = bpy.data.objects.get("Water")
matw = old.data.materials[0] if old and old.data.materials else None
if old: bpy.data.objects.remove(old, do_unlink=True)
me = bpy.data.meshes.new("Water")
me.vertices.add(len(verts)); me.vertices.foreach_set("co", verts.ravel())
nf = len(quads)
me.loops.add(nf*4); me.loops.foreach_set("vertex_index", quads.ravel().astype(np.int32))
me.polygons.add(nf)
me.polygons.foreach_set("loop_start", (np.arange(nf)*4).astype(np.int32))
me.polygons.foreach_set("loop_total", np.full(nf, 4, np.int32))
me.polygons.foreach_set("use_smooth", np.ones(nf, np.int32))
me.update(); me.validate()
ob = bpy.data.objects.new("Water", me); sc.collection.objects.link(ob)
if matw: ob.data.materials.append(matw)

gy = np.gradient(Z, axis=1)/np.gradient(Y, axis=1)
gx = np.gradient(Z, axis=0)/np.gradient(X, axis=0)
print("water grid %d x %d = %d verts, %d quads" % (nx, ny, len(verts), nf))
print("z range %.4f..%.4f   rms slope %.2f deg   p99 %.2f deg" % (
    Z.min(), Z.max(),
    math.degrees(math.atan(np.sqrt((gx**2 + gy**2).mean()))),
    math.degrees(math.atan(np.percentile(np.hypot(gx, gy), 99)))))
bpy.ops.wm.save_mainfile()
