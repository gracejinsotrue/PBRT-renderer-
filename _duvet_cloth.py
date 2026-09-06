
# Drapes a slack cloth over the metaball stuffing built by _duvet_build.py.
# Metaballs alone read as soap bubbles: only cloth gives folds and creases.
# Call 1: exec this file.  Then step frames, then run _duvet_cloth_finish.
import bpy, math
import numpy as np

G = globals()
def P(k, d):  return float(G[k]) if k in G else d
def PI(k, d): return int(G[k]) if k in G else d

NX     = PI("CL_NX", 152)          # cloth resolution
NY     = PI("CL_NY", 182)
X0, X1 = -1.190, 1.200             # cloth is oversized: slack is what makes folds
Y0, Y1 = -1.760,  0.880
ZSTART = P("CL_Z", 2.36)           # starts flat just above the stuffing
SEED   = PI("CL_SEED", 5)

# --- cloth sheet -------------------------------------------------------------
xs = np.linspace(X0, X1, NX); ys = np.linspace(Y0, Y1, NY)
Xg, Yg = np.meshgrid(xs, ys, indexing='ij')
rng = np.random.default_rng(SEED)
Zg = np.full_like(Xg, ZSTART) + rng.normal(scale=0.0015, size=Xg.shape)   # break symmetry
co = np.stack([Xg, Yg, Zg], axis=-1).reshape(-1, 3).astype(np.float32)

idx = np.arange(NX * NY, dtype=np.int64).reshape(NX, NY)
a = idx[:-1, :-1]; b = idx[1:, :-1]; c = idx[1:, 1:]; d = idx[:-1, 1:]
faces = np.stack([a, b, c, d], axis=-1).reshape(-1, 4)
nf = faces.shape[0]

me = bpy.data.meshes.new("DuvetClothMesh")
me.vertices.add(co.shape[0]); me.vertices.foreach_set("co", co.ravel())
me.loops.add(nf * 4); me.loops.foreach_set("vertex_index", faces.ravel())
me.polygons.add(nf); me.polygons.foreach_set("loop_start", np.arange(nf, dtype=np.int32) * 4)
try: me.polygons.foreach_set("loop_total", np.full(nf, 4, dtype=np.int32))
except Exception: pass
me.update(calc_edges=True); me.validate(verbose=False)

old = bpy.data.objects.get("DuvetCloth")
if old: 
    m = old.data; bpy.data.objects.remove(old); bpy.data.meshes.remove(m)
ob = bpy.data.objects.new("DuvetCloth", me)
bpy.context.scene.collection.objects.link(ob)

cl = ob.modifiers.new("Cloth", 'CLOTH')
st = cl.settings
st.quality = 6
st.mass = 0.30
st.tension_stiffness = 14; st.compression_stiffness = 14; st.shear_stiffness = 6
st.bending_stiffness = 0.55        # soft, but below ~0.3 the free edge micro-buckles into a sawtooth
st.air_damping = 1.4
st.tension_damping = 8; st.compression_damping = 8; st.shear_damping = 8
st.bending_damping = 0.6
cs = cl.collision_settings
cs.use_collision = True
cs.distance_min = 0.010
cs.collision_quality = 4
cs.friction = 25
cs.use_self_collision = True
cs.self_distance_min = 0.010
cs.self_friction = 5
cl.point_cache.frame_start = 1
cl.point_cache.frame_end = 90

# --- colliders: the stuffing and the plinth ---------------------------------
for nm, fr in (("DuvetBody", 70.0), ("BedBase", 45.0), 
               ("Pillow_1_Pillow_1_0", 45.0), ("Pillow_2_Pillow_2_0", 45.0)):
    o = bpy.data.objects.get(nm)
    if not o: 
        print("collider missing:", nm); continue
    if not any(m.type == 'COLLISION' for m in o.modifiers):
        o.modifiers.new("Collision", 'COLLISION')
    o.collision.thickness_outer = 0.012
    o.collision.thickness_inner = 0.02
    o.collision.cloth_friction = fr
    o.collision.damping = 0.6

bpy.context.scene.frame_set(1)
print("cloth %d verts / %d quads, colliders ready" % (co.shape[0], nf))
