# floating_causeway: Blender scene, everything except the water displacement and
# the foliage. Parametric; see scenes/floating_causeway/PLAN.md for where every
# number came from.
#
# The camera is fitted, not chosen. Measured off the reference: horizon at 63.31%
# of frame height, causeway top edge 14 px below it, waterline 33.5 px below it.
# Those give two fov-independent ratios, z_causeway = 0.538 * h_cam and
# z_figure = 3.79 * h_cam, which put the reference camera 0.45 m above the water
# with a 0.24 m kerb. We cannot keep that: the causeway top crosses the horizon
# when z_top = h_cam, so at 0.45 m the largest possible float is 0.45 m and beyond
# that the slab starts occluding the ridge. Raising the camera to 1.60 m and moving
# the causeway to 41.6 m keeps the reference's 14 px slab face and horizon fraction
# while opening a 36 px gap of daylight underneath.
#
#   exec(open(r"...tools/scene_build/causeway_build.py").read(), {})
import bpy, bmesh, math, os, numpy as np
from mathutils import Matrix

CAM_LOC = (0.0, 0.0, 1.60)
CAM_PITCH = 3.749          # deg UP. The reference horizon sits 63.31% down the frame,
                           # i.e. below centre, and pitching DOWN moves the horizon UP,
                           # so this composition looks up. Blender euler X > 90 is up.
CAM_FOV_H = 60.0
RES = (1920, 818)

CW_DIST = 41.6             # causeway centreline distance
CW_GAP = 0.90              # underside above the water, the float
CW_THICK = 0.35
CW_WIDTH = 2.0
CW_YAW = 6.0               # deg, so the right end recedes as it does in the reference
CW_HALFLEN = 60.0
CW_BLOCK = 1.0             # visible block joints, ~40 px each at this distance
CW_JOINT = 0.02

WATER_R = 3000.0           # flat stand-in; the real surface is the projected grid
BED_Z = -3.5

SUN_ELEV, SUN_AZIM = 12.85, 72.79   # Blender convention, drives sun_disk.py too

# Tree placement is solved from the reference's leaf scale, not eyeballed. Measured
# off the canopy: leaf highlight blobs are 0.135 deg median and the leaf-pattern
# band-pass peaks at 0.17-0.34 deg. The fine library's cards are 7.2 cm at scale 1,
# so 0.072*S/D = 0.0038 rad pins S/D at 0.053. Scale 1.6 at 30 m gives 11.5 cm
# leaves subtending 0.22 deg, and puts the canopy (leaf z 1.73 to 15.38 at scale 1)
# across elevations 2.2 to 37.5 deg, filling the frame's upper right and running off
# the top. The plain oak library would need a leaf 1.7x too big to do the same job.
TREE_LIB = r"assets/crosswalk/tree_oak_fine.blend"

# Leaf size is what pins scale to distance: the fine library's cards are 7.2 cm at
# scale 1, so 0.072*S/D = 0.0038 rad holds S/D near 0.053 for every instance. That
# also fixes the crown half-angle at 17.6 deg and the trunk width at ~106 px, which
# is the whole placement problem: a crown reaching left of azimuth 15 deg drags its
# own trunk into frame, and the reference has only a slim branch at the right edge,
# never a 2 m trunk.
#
# The fix is LEAN. Tipping a tree by L moves its crown centre about 0.5*H*sin(L) to
# one side of its trunk, which is 8-9 deg of azimuth here, so the trunks can all sit
# beyond the 30 deg frame edge while the crowns still reach azimuth 7-8 deg. Leaning
# out over open water is what these trees would really do anyway.
#
# DENSITY, and it was backwards on the first two attempts. Measured in the canopy
# core the reference sits at L p50 0.232 with 15% sky showing through and only 8% of
# pixels dark; three overlapping instances gave p50 0.052 with 0.2% sky and 98% dark.
# A backlit canopy is bright because the sun reaches the far side of the leaves we
# see, and a mass three trees deep self-shadows so thoroughly that almost nothing is
# lit from behind. Translucency is not the brightness knob here (raising it darkens
# the canopy, since the reflection lobe scales by 1-T) - thinning it is.
# BRANCHES ONLY, no trunk. The reference shows a lateral limb entering from the
# right edge with its foliage filling the corner, never a trunk meeting the water.
# Fitting a circle through the reference canopy's left edge at three row bands puts
# its foliage mass at azimuth 14.6 deg, elevation 20.3 deg, angular radius 15.1 deg,
# i.e. centred just above the top of frame. A whole tree cannot sit there: the trunk
# would land at x 0.73 inside the frame, and leaning it far enough to clear the edge
# drags a 220 px trunk diagonally across the corner instead.
#
# Of the six bark meshes only ...0.013 reaches z=0, so it is the trunk and main
# limbs; the rest start at z 2.0 to 9.0. Dropping that one mesh leaves branches and
# foliage with no trunk, which is exactly the reference's read.
#
# Placement solves to the fitted target: leaf mass centre along the axis is about
# 8.6*S, so a 25 deg lean at azimuth 26 and 16 m puts it at azimuth 14.3 deg,
# elevation 20.5 deg with a 19.9 deg radius, and leaves at 0.250 deg against the
# reference's 0.276 deg.
# (azimuth deg, distance m, scale, spin deg, lean deg)
#
# The third entry is solved so its crown CENTRE lands on the sun rather than near
# its rim. With S/D pinned, the crown centre sits at azimuth
# atan2(sin(az) - 0.456 sin(lean), cos(az)) and elevation atan((0.456 D cos(lean)
# - 1.6) / D), so az 24.9, lean 18, D 7.38 puts it at (17.2, 12.9) which is the sun
# to within a fifth of a degree. The first two crowns have the sun 8.2 deg off
# centre, out in the lacy part, which is why it was punching through a clean hole.
# DEPTH IS WHAT DARKENS A CANOPY, so only two crowns cover the sun's line of sight.
# The old (26.0, 16.0, 0.92) crown put its centre at (14.3, 20.5), 8.2 deg off the
# sun, and at 47% of the crown radius the foliage there is already sparse enough to
# leave a clean hole, so it was buying depth without buying occlusion. Dropping it
# and keeping the sun-centred crown trades three leaf layers for two.
TREES = [(31.0, 20.5, 1.18, 215.0, 20.0),
         (24.9, 7.38, 0.391, 145.0, 18.0)]
TRUNK_DROP = {"Oak_Oak_Bark_1_Mat1_0.013"}
LEAF_SPECULAR = 0.12       # 0.5 threw white specular slashes across the backlit cards

# FOREGROUND MASS, cut from the SAME oak foliage as the canopy. It used to be a
# lilac bush pack, and even retinted to the oak's colour (opaque sRGB 0.820/0.747/
# 0.359 against the lilac's 0.797/0.718/0.315, near identical) the leaf itself was
# obviously a different plant.
#
# The oak library cannot simply be scaled down: every one of its six leaf meshes is a
# whole crown 7-12 m across, so a bush-sized instance carries bush-sized leaves. The
# reference's foreground leaves measure 0.414 deg against the canopy's 0.276, i.e.
# BIGGER, because what it shows in the foreground is a nearby branch, not a distant
# tree. So each foreground clump is a SPRAY cut spatially out of a crown: keep the
# leaf cards whose centres fall inside a box, discard the rest, and the cards keep
# their original 7.2 cm size. Scale then follows distance to hold the leaf angular
# size, and the box size sets how big the clump reads.
#
# DISTANCE, NOT AZIMUTH DENSITY, IS THE COVERAGE CONTROL. A clump spans about 9 deg
# of azimuth here, so what decides whether it covers the 62 px strip under the
# causeway (elevation -0.34 to -2.46) is whether its top clears that strip. The
# reference's profile over that strip runs 0.16 leaf at x 0.55 to 0.98 at x 0.83, so
# the left clumps sit lower and the right ones higher.
#
# Azimuths are STRATIFIED: with uniform draws every change to a count reshuffled the
# whole arrangement and opened a new hole somewhere, twice at the sun's azimuth where
# the water glitter then showed through.
#
# (count, azimuth deg, distance m, top elevation deg)
DROP_DOUBLED_CARDS = False   # the "_c1" duplicate leaf sets; keeping them costs nothing here

# Weighted to the bottom right, following the reference: its foreground mass is
# deepest from x 0.75 to 0.95 and thins to almost nothing by x 0.55.
SPRAY_BANDS = [(22, (16.0, 46.0), (4.2, 7.4), (0.6, 2.8)),
               (18, (14.0, 48.0), (3.4, 5.8), (-3.0, -0.4)),
               (18, (12.0, 48.0), (2.8, 4.6), (-8.0, -3.5)),
               (8, (2.0, 14.0), (3.8, 5.4), (-5.5, -2.5)),
               (8, (14.0, 21.0), (6.0, 7.6), (0.8, 2.4))]
SPRAY_LEAF = 0.414          # deg, target leaf angular size (reference foreground)
SPRAY_CARD = 0.072          # m, the fine library's leaf card at scale 1
SPRAY_BOX = (1.35, 2.10)    # m, spatial box cut out of the crown, in crown units
# The crown is a hollow shell of cards with large gaps, so a fixed box lands in empty
# space as often as not: the first pass gave clumps of 3, 10 and 21 faces alongside
# ones of 3800, and the mass read as scattered sprigs floating over the water. The
# box grows until it has caught a real clump.
SPRAY_MIN_CARDS = 520       # leaf cards, i.e. quads, per clump
SPRAY_JITTER = 0.10
SPRAY_SEED = 5

LEAF_TRANSLUCENCY = 0.35

MAT = {
    "Water":    dict(kind="glass", color=(0.35, 0.55, 0.62), ior=1.333),
    "LakeBed":  dict(kind="pbr", color=(0.020, 0.030, 0.030), rough=0.90),
    "Causeway": dict(kind="pbr", color=(0.280, 0.270, 0.250), rough=0.85),
}


def wipe():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    # Appending the tree library each rebuild leaves orphan meshes behind, and they
    # accumulate into hundreds of MB across runs.
    for _ in range(3):
        for c in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
            for d in list(c):
                if d.users == 0:
                    c.remove(d)


def material(name):
    spec = MAT[name]
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if spec["kind"] == "glass":
        s = nt.nodes.new("ShaderNodeBsdfGlass")
        s.inputs["Color"].default_value = (*spec["color"], 1.0)
        s.inputs["IOR"].default_value = spec["ior"]
        s.inputs["Roughness"].default_value = 0.0
    else:
        s = nt.nodes.new("ShaderNodeBsdfPrincipled")
        s.inputs["Base Color"].default_value = (*spec["color"], 1.0)
        s.inputs["Roughness"].default_value = spec["rough"]
        if "Specular IOR Level" in s.inputs:
            s.inputs["Specular IOR Level"].default_value = 0.35
    nt.links.new(s.outputs[0], out.inputs["Surface"])
    return m


def mesh_from(name, verts, faces, matname):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(material(matname))
    bpy.context.collection.objects.link(ob)
    return ob


def plane(name, r, z, matname):
    v = [(-r, -r, z), (r, -r, z), (r, r, z), (-r, r, z)]
    return mesh_from(name, v, [(0, 1, 2, 3)], matname)


def causeway():
    # One mesh of discrete blocks. The joints are ~40 px on screen at 41.6 m, which
    # is what stops the slab reading as an extruded rectangle.
    z0, z1 = CW_GAP, CW_GAP + CW_THICK
    y0, y1 = -CW_WIDTH / 2.0, CW_WIDTH / 2.0
    verts, faces = [], []
    n = int(2 * CW_HALFLEN / CW_BLOCK)
    for i in range(n):
        a = -CW_HALFLEN + i * CW_BLOCK + CW_JOINT * 0.5
        b = a + CW_BLOCK - CW_JOINT
        k = len(verts)
        verts += [(a, y0, z0), (b, y0, z0), (b, y1, z0), (a, y1, z0),
                  (a, y0, z1), (b, y0, z1), (b, y1, z1), (a, y1, z1)]
        faces += [(k, k + 3, k + 2, k + 1), (k + 4, k + 5, k + 6, k + 7),
                  (k, k + 1, k + 5, k + 4), (k + 1, k + 2, k + 6, k + 5),
                  (k + 2, k + 3, k + 7, k + 6), (k + 3, k, k + 4, k + 7)]
    v = np.array(verts, np.float64)
    t = math.radians(CW_YAW)
    c, s = math.cos(t), math.sin(t)
    x, y = v[:, 0].copy(), v[:, 1].copy()
    v[:, 0] = x * c - y * s
    v[:, 1] = x * s + y * c + CW_DIST
    return mesh_from("Causeway", v, faces, "Causeway")


def tree():
    lib = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", TREE_LIB))
    out = []
    for i, (az, dist, scale, rotz, lean) in enumerate(TREES):
        with bpy.data.libraries.load(lib, link=False) as (src, dst):
            dst.objects = [n for n in src.objects
                           if not (DROP_DOUBLED_CARDS and n.endswith("_c1"))
                           and n not in TRUNK_DROP]
        obs = [o for o in dst.objects if o is not None]

        # Lean is applied OUTSIDE the spin so it tips toward world -x (into frame)
        # whatever the spin is; inside, the lean direction would follow the spin.
        place = (Matrix.Translation((dist * math.sin(math.radians(az)),
                                     dist * math.cos(math.radians(az)), 0.0))
                 @ Matrix.Rotation(math.radians(-lean), 4, "Y")
                 @ Matrix.Rotation(math.radians(rotz), 4, "Z")
                 @ Matrix.Scale(scale, 4))

        # Set matrix_world outright and zero the deltas. Baking a transform into the
        # mesh while an object is still parented writes a parent-cancelling local
        # matrix that comes back as a 90 deg rotation once it is unparented.
        local = {o.name: o.matrix_world.copy() for o in obs}
        for o in obs:
            o.name = "T%d_%s" % (i, o.name)
            if o.name not in bpy.context.collection.objects:
                bpy.context.collection.objects.link(o)
            o.parent = None
            o.delta_location = (0.0, 0.0, 0.0)
            o.delta_rotation_euler = (0.0, 0.0, 0.0)
            o.delta_scale = (1.0, 1.0, 1.0)
            o.matrix_world = place @ local[o.name.split("_", 1)[1]]

        # The exporter reads this custom property, not a Principled socket: there is
        # no honest Principled analogue and Transmission Weight would turn the leaves
        # to glass in the viewport. 0.35 is the calibrated crosswalk canopy value.
        for o in obs:
            for sl in o.material_slots:
                if sl.material is not None and "Leav" in sl.material.name:
                    sl.material["nori_translucency"] = LEAF_TRANSLUCENCY
                    bs = next((x for x in sl.material.node_tree.nodes
                               if x.type == "BSDF_PRINCIPLED"), None)
                    if bs is not None and "Specular IOR Level" in bs.inputs:
                        bs.inputs["Specular IOR Level"].default_value = LEAF_SPECULAR
        out.append(obs)
    return out


def _mesh_arrays(me):
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    npoly = len(me.polygons)
    ls = np.empty(npoly, np.int32)
    lt = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    nl = len(me.loops)
    lv = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    uv = np.empty(nl * 2, np.float32)
    me.uv_layers[0].data.foreach_get("uv", uv)
    return co.reshape(-1, 3), ls, lt, lv, uv.reshape(-1, 2)


def cut_spray(arrays, centre, box, name, mat):
    co, ls, lt, lv, uv = arrays
    cen = np.add.reduceat(co[lv], ls, axis=0) / lt[:, None]
    grown = box
    for _ in range(9):
        keep = np.all(np.abs(cen - centre) <= grown * 0.5, axis=1)
        if int(keep.sum()) >= SPRAY_MIN_CARDS:
            break
        grown *= 1.3
    sel = np.nonzero(keep)[0]
    if len(sel) < 40:
        return None

    verts = []
    faces = []
    uvs = []
    remap = {}
    for pi in sel:
        a, t = int(ls[pi]), int(lt[pi])
        f = []
        for k in range(t):
            vi = int(lv[a + k])
            if vi not in remap:
                remap[vi] = len(verts)
                verts.append(co[vi])
            f.append(remap[vi])
            uvs.append(uv[a + k])
        faces.append(f)

    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    lay = me.uv_layers.new(name="UVMap")
    flat = np.asarray(uvs, np.float32).ravel()
    lay.data.foreach_set("uv", flat)
    me.materials.append(mat)
    me.update()
    return me, grown


def foreground():
    lib = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", TREE_LIB))
    with bpy.data.libraries.load(lib, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n.startswith("Oak_Leavs")]
    leaf_obs = [o for o in dst.objects if o is not None and o.type == "MESH"]
    pool = []
    for o in leaf_obs:
        arrays = _mesh_arrays(o.data)
        co = arrays[0]
        pool.append((arrays, co.min(0), co.max(0), o.data.materials[0]))

    rng = np.random.default_rng(SPRAY_SEED)
    plan = [(b, d, e, (k + 0.5) / n) for n, b, d, e in SPRAY_BANDS for k in range(n)]
    out = []
    total = 0
    for i, (az_rng, d_rng, e_rng, frac) in enumerate(plan):
        arrays, mn, mx, mat = pool[i % len(pool)]
        span = az_rng[1] - az_rng[0]
        az = az_rng[0] + span * frac + rng.uniform(-0.35, 0.35) * span / len(plan)
        dist = rng.uniform(*d_rng)
        e_top = rng.uniform(*e_rng)
        box = rng.uniform(*SPRAY_BOX)

        # Cut from the crown's outer shell, where the cards are, not its hollow middle.
        u = rng.uniform(0.0, 2.0 * math.pi)
        rad = 0.5 * min(mx[0] - mn[0], mx[1] - mn[1]) * rng.uniform(0.55, 0.85)
        mid = (mn + mx) * 0.5
        centre = np.array([mid[0] + rad * math.cos(u),
                           mid[1] + rad * math.sin(u),
                           rng.uniform(mn[2] + 0.15 * (mx[2] - mn[2]),
                                       mn[2] + 0.85 * (mx[2] - mn[2]))], np.float32)

        cut = cut_spray(arrays, centre, box, "F%02d" % i, mat)
        if cut is None:
            continue
        me, grown = cut
        total += len(me.polygons)

        scale = (SPRAY_LEAF * math.pi / 180.0) * dist / SPRAY_CARD
        scale *= 1.0 + rng.uniform(-SPRAY_JITTER, SPRAY_JITTER)
        half = 0.5 * grown * scale
        z = CAM_LOC[2] + dist * math.tan(math.radians(e_top)) - half

        ob = bpy.data.objects.new("F%02d" % i, me)
        bpy.context.collection.objects.link(ob)
        ob.matrix_world = (Matrix.Translation((dist * math.sin(math.radians(az)),
                                               dist * math.cos(math.radians(az)), z))
                           @ Matrix.Rotation(rng.uniform(0, 2 * math.pi), 4, "Z")
                           @ Matrix.Rotation(math.radians(rng.uniform(-25, 25)), 4, "Y")
                           @ Matrix.Scale(scale, 4)
                           @ Matrix.Translation((-centre[0], -centre[1], -centre[2])))
        if mat is not None and "Leav" in mat.name:
            mat["nori_translucency"] = LEAF_TRANSLUCENCY
            bs = next((x for x in mat.node_tree.nodes if x.type == "BSDF_PRINCIPLED"), None)
            if bs is not None and "Specular IOR Level" in bs.inputs:
                bs.inputs["Specular IOR Level"].default_value = LEAF_SPECULAR
        out.append(ob)

    for o in leaf_obs:
        bpy.data.meshes.remove(o.data, do_unlink=True)
    print("foreground: %d sprays, %d faces" % (len(out), total))
    return out



wipe()

cam_data = bpy.data.cameras.new("Camera")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.angle_x = math.radians(CAM_FOV_H)
cam = bpy.data.objects.new("Camera", cam_data)
cam.location = CAM_LOC
cam.rotation_euler = (math.radians(90.0 + CAM_PITCH), 0.0, 0.0)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam
bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = RES
bpy.context.scene.render.resolution_percentage = 100

plane("Water", WATER_R, 0.0, "Water")
plane("LakeBed", WATER_R, BED_Z, "LakeBed")
causeway()
tree_sets = tree()
shrub_obs = foreground()

# Preview only. The exporter drops lamps; direct light in the render is the
# geometric disk from sun_disk.py, injected through NORI_MEDIUM_XML.
sl = bpy.data.lights.new("SunPreview", type="SUN")
sl.angle = math.radians(2.0)
so = bpy.data.objects.new("SunPreview", sl)
e, a = math.radians(SUN_ELEV), math.radians(SUN_AZIM)
so.rotation_euler = (math.pi / 2 - e, 0.0, a + math.pi / 2)
bpy.context.collection.objects.link(so)

bpy.context.view_layer.update()

C = (RES[1] / 2.0) / math.tan(math.atan(math.tan(math.radians(CAM_FOV_H / 2)) * RES[1] / RES[0]))
hor = RES[1] / 2.0 + C * math.tan(math.radians(CAM_PITCH))   # pitched up, so below centre
print("C %.1f px/rad   horizon row %.1f of %d = %.2f%%" % (C, hor, RES[1], 100 * hor / RES[1]))
for nm, z in [("slab top", CW_GAP + CW_THICK), ("slab underside", CW_GAP), ("water below", 0.0)]:
    print("  %-15s row %.1f" % (nm, hor + C * (CAM_LOC[2] - z) / CW_DIST))
def screen_bbox(obs):
    xs, ys = [], []
    tot = 0
    for o in obs:
        if o.type != "MESH":
            continue
        n = len(o.data.vertices)
        v = np.empty(n * 3, np.float32)
        o.data.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3) @ np.array(o.matrix_world.to_3x3()).T + np.array(o.matrix_world.translation)
        tot += len(o.data.polygons)
        d = v - np.array(CAM_LOC)
        fwd = np.array([0.0, math.cos(math.radians(CAM_PITCH)), math.sin(math.radians(CAM_PITCH))])
        up = np.array([0.0, -math.sin(math.radians(CAM_PITCH)), math.cos(math.radians(CAM_PITCH))])
        rt = np.array([1.0, 0.0, 0.0])
        z = d @ fwd
        m = z > 0.05
        xs.append(RES[0] / 2.0 + C * (d[m] @ rt) / z[m])
        ys.append(RES[1] / 2.0 - C * (d[m] @ up) / z[m])
    x = np.concatenate(xs); y = np.concatenate(ys)
    return x, y, tot


for i, obs in enumerate(tree_sets):
    tx, ty, tf = screen_bbox(obs)
    print("tree %d: %6d faces  screen x %.2f..%.2f  y %.2f..%.2f"
          % (i, tf, tx.min() / RES[0], tx.max() / RES[0], ty.min() / RES[1], ty.max() / RES[1]))
print("  reference canopy measures x 0.82..1.00, y 0.00..0.45; x 0.68..0.82 is pure sky")
print("  sun at x 0.768 y 0.170; frame edge is azimuth 30 deg so trunks must clear it")
sx_, sy_, sf_ = screen_bbox(shrub_obs)
print("foreground: %d faces  screen x %.2f..%.2f  y %.2f..%.2f   (reference mass x 0.55..1.00, top y 0.56..0.74)"
      % (sf_, sx_.min() / RES[0], sx_.max() / RES[0], sy_.min() / RES[1], sy_.max() / RES[1]))
print("objects:", len(bpy.data.objects))
