# Build the "crosswalk under a plane tree" scene (komorebi reference match).
#
# Run inside Blender via the MCP:
#   g = {"XW_REPO": r"C:\Users\gjin3\Desktop\nori-26sp"}
#   exec(open(r"C:\Users\gjin3\Desktop\nori-26sp\tools\scene_build\crosswalk_build.py").read(), g)
#
# Geometry is left unapplied (no transform_apply) since the exporter reads matrix_world.
# After export, scene.xml still needs evCompensation and the bloom block patched in: the
# exporter writes evCompensation 0.0 and knows nothing about bloom.
import bpy, math, os, mathutils

REPO = globals().get("XW_REPO", r"C:\Users\gjin3\Desktop\nori-26sp")
ASSETS = os.path.join(REPO, "assets", "crosswalk")

# Oak stand-ins for the London plane (Sketchfab "Oak Tree" by Cosmic_dust, CC-BY, summer
# variant). Source is 15.4 m tall with a 12 m crown, 684k faces of which 217k are leaf cards.
# x, y, z-rotation (rad), scale. Placement is set by where each crown's shadow lands: at
# SUN_ELEV a crown throws its shadow (height / tan(elev)) along the anti-sun azimuth. The
# reference's dapple comes from crown EDGES over the crossing, so Tree0 overhangs from the
# left and Tree1 sits back in the hedge with just its near edge reaching the road; the rest
# are far enough up the street (y >= 20) that their shadows clear the curb.
# Two libraries from the same oak. tree_oak_fine.blend has every leaf card shrunk to 60% about
# its own centre plus a second rotated copy of the leaves, so a tree keeps its crown reach and
# trunk girth but carries leaves 0.6x the size: that decouples the three things the near trees
# need (small leaves, crown over the road, a real trunk) which one scale factor could not give.
# Far trees use the plain library; distance already makes their leaves read small.
# Entries: x, y, z-rotation (rad), scale, optional lean (rad, tilts the top toward +x).
NEAR_TREES = [(-7.5, 5.5, 0.7, 1.60, 0.12),   # the visible trunk, ~1 m thick at 20 m, leaning over the road
              (-6.0, -4.0, 3.9, 1.40),         # crown edge reaches x~2.4 and throws the dapple on the near crossing
              (-13.0, 0.0, 1.2, 1.30), (-10.0, -9.0, 5.0, 1.20)]
FAR_TREES = [(5.0, 14.0, 2.3, 1.70), (-13.0, 20.0, 4.1, 1.45), (13.0, 21.0, 5.2, 1.40),
             (0.0, 10.0, 2.9, 1.60), (14.0, 12.0, 0.4, 1.45)]
TREE_SETS = [("tree_oak_fine.blend", NEAR_TREES), ("tree_oak.blend", FAR_TREES)]

ROAD_TILE = 4.0            # metres per texture repeat on the asphalt
WALK_TILE = 2.0            # metres per texture repeat on the pavers

LEAF_TRANSLUCENCY = 0.35   # diffuse transmission through a leaf blade

# Sun high and a little to the RIGHT: with the tree on the left, its crown shadow is thrown
# mostly out of frame, so the road reads as sunlit pavement with dapple over it rather than
# shade with sunflecks. Keep tools/scene_build/sun_disk.py in sync with these two numbers.
SUN_ELEV = 62.0     # degrees above horizon
SUN_AZIM = 70.0     # Blender azimuth, atan2(y,x) in degrees

CAM_LOC  = (1.0, -13.0, 1.55)
CAM_TGT  = (1.0, 1.0, 2.85)   # pitched ~5.3 deg up
CAM_FOV_H = 52.0    # horizontal fov, degrees -> ~33 deg vertical half-angle at 3:4
RES_W, RES_H = 900, 1200

# pedestrian signal: Sketchfab "Pedestrian Traffic Light" by ASA21 (CC-BY), mirrored in the
# library so the housing hangs left of the pole like the reference. Non-uniform scale keeps
# the pole slim while the head sits high. The lamps are painted into its texture, so the
# glow is a small emitter quad laid over the red lamp; bloom does the rest.
SIG_POS = (5.0, 0.9, 0.15)
SIG_SCALE = (0.65, 0.65, 1.30)


def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for blk in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for b in list(blk):
            if b.users == 0: blk.remove(b)

def bsdf_of(m):
    for n in m.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED': return n
    raise RuntimeError("no Principled BSDF in " + m.name)

def mat_pbr(name, base, rough=0.7, metal=0.0, spec=0.5):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = bsdf_of(m)
    p.inputs["Base Color"].default_value = (base[0], base[1], base[2], 1.0)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    for key in ("Specular IOR Level", "Specular"):
        if key in p.inputs: p.inputs[key].default_value = spec; break
    return m

def mat_textured(name, albedo, normal, rough=0.7, spec=0.5):
    # Normal Map strength stays at 1.0: the exporter writes only the file path, so any
    # strength set here would show in the Cycles preview and not in the render.
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; p = bsdf_of(m)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = 0.0
    for key in ("Specular IOR Level", "Specular"):
        if key in p.inputs: p.inputs[key].default_value = spec; break
    ap = os.path.join(ASSETS, "tex", albedo)
    if os.path.isfile(ap):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(ap, check_existing=True)
        t.location = (-620, 220)
        nt.links.new(t.outputs["Color"], p.inputs["Base Color"])
    else:
        print("[tex] missing", ap)
    npath = os.path.join(ASSETS, "tex", normal)
    if os.path.isfile(npath):
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = bpy.data.images.load(npath, check_existing=True)
        tn.image.colorspace_settings.name = 'Non-Color'
        tn.location = (-620, -180)
        nm = nt.nodes.new("ShaderNodeNormalMap"); nm.location = (-320, -180)
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], p.inputs["Normal"])
    return m

def load_lib(path):
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = list(src.objects)
    return [o for o in dst.objects if o is not None]

def place(ob, matrix):
    # Library objects can carry leftover parent-cancelling transforms, and delta transforms
    # ride on top of location/rotation/scale, so set the world matrix outright.
    ob.parent = None
    ob.delta_location = (0.0, 0.0, 0.0)
    ob.delta_rotation_euler = (0.0, 0.0, 0.0)
    ob.delta_scale = (1.0, 1.0, 1.0)
    ob.rotation_mode = 'XYZ'
    ob.matrix_world = matrix

def mat_emit(name, color, strength):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = bsdf_of(m)
    p.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    for k in ("Emission Color", "Emission"):
        if k in p.inputs: p.inputs[k].default_value = (color[0], color[1], color[2], 1.0); break
    p.inputs["Emission Strength"].default_value = strength
    return m

def mesh_from(name, verts, faces, mat, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces); me.update()
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                uvl.data[li].uv = uvs[me.loops[li].vertex_index]
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    if mat: me.materials.append(mat)
    return ob

def quad(name, c, mat, uvs=None):
    return mesh_from(name, list(c), [(0, 1, 2, 3)], mat, uvs)

def quad_tiled(name, c, mat, tile):
    # uv straight from world position, so tiling is seamless across the whole quad
    return quad(name, c, mat, {i: (p[0] / tile, p[1] / tile) for i, p in enumerate(c)})

def box(name, center, size, mat):
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    cx, cy, cz = center
    v = [(cx-hx, cy-hy, cz-hz), (cx+hx, cy-hy, cz-hz), (cx+hx, cy+hy, cz-hz), (cx-hx, cy+hy, cz-hz),
         (cx-hx, cy-hy, cz+hz), (cx+hx, cy-hy, cz+hz), (cx+hx, cy+hy, cz+hz), (cx-hx, cy+hy, cz+hz)]
    f = [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
    return mesh_from(name, v, f, mat)

def cylinder(name, base, radius, height, mat, seg=24):
    cx, cy, cz = base; v = []; f = []
    for i in range(seg):
        a = 2*math.pi*i/seg
        v.append((cx+radius*math.cos(a), cy+radius*math.sin(a), cz))
    for i in range(seg):
        a = 2*math.pi*i/seg
        v.append((cx+radius*math.cos(a), cy+radius*math.sin(a), cz+height))
    for i in range(seg):
        j = (i+1) % seg
        f.append((i, j, seg+j, seg+i))
    f.append(tuple(range(seg-1, -1, -1)))
    f.append(tuple(range(seg, 2*seg)))
    return mesh_from(name, v, f, mat)

def hedge(name, x0, x1, y_front, y_back, h_mean, h_var, mat, seg=200, seed=7):
    # h_var 0 gives the straight top the reference has; the canopy hides the edge anyway
    import random
    rnd = random.Random(seed)
    ph = (rnd.uniform(0, 6.283), rnd.uniform(0, 6.283), rnd.uniform(0, 6.283))
    verts, faces = [], []
    for i in range(seg + 1):
        t = i / seg
        x = x0 + (x1 - x0) * t
        n = (0.55 * math.sin(x * 0.42 + ph[0]) + 0.30 * math.sin(x * 1.13 + ph[1])
             + 0.15 * math.sin(x * 2.70 + ph[2]))
        h = h_mean + h_var * n
        b = len(verts)
        verts += [(x, y_front, 0.0), (x, y_front, h), (x, y_back, h * 0.92)]
        if i:
            p = b - 3
            faces.append((p, b, b + 1, p + 1))          # front face
            faces.append((p + 1, b + 1, b + 2, p + 2))  # top, sloping away
    return mesh_from(name, verts, faces, mat)


clear()

m_asphalt  = mat_textured("Asphalt", "road_albedo.png", "road_normal.png", rough=0.72)
m_paint    = mat_pbr("RoadPaint",(0.52, 0.52, 0.50),   rough=0.62)
m_curb     = mat_pbr("Curb",     (0.30, 0.29, 0.27),   rough=0.82)
m_pavers   = mat_textured("Pavers", "walk_albedo.png", "walk_normal.png", rough=0.85)
m_hedge    = mat_pbr("Hedge",    (0.019, 0.029, 0.011),rough=0.92)
m_pole     = mat_pbr("Pole",     (0.20, 0.20, 0.20),   rough=0.45, metal=0.55)
m_housing  = mat_pbr("Housing",  (0.045, 0.045, 0.048),rough=0.55)
m_red      = mat_emit("SignalRed", (1.0, 0.10, 0.05), 14.0)

# ground plane, road surface
quad_tiled("Road", [(-40,-34,0), (40,-34,0), (40,0.0,0), (-40,0.0,0)], m_asphalt, ROAD_TILE)

# zebra bars: long in x, repeating in y, parallel to the curb (matches the reference)
BAR_W, BAR_GAP, BAR_LEN = 0.62, 0.58, 13.0
verts, faces, y, n = [], [], -14.4, 0
while y < -0.7:
    b = len(verts)
    verts += [(-BAR_LEN/2, y, 0.004), (BAR_LEN/2, y, 0.004),
              (BAR_LEN/2, y+BAR_W, 0.004), (-BAR_LEN/2, y+BAR_W, 0.004)]
    faces.append((b, b+1, b+2, b+3))
    y += BAR_W + BAR_GAP; n += 1
mesh_from("CrosswalkStripes", verts, faces, m_paint)

# curb + sidewalk
box("Curb", (0, 0.20, 0.075), (80, 0.40, 0.15), m_curb)
quad_tiled("Sidewalk", [(-40,0.40,0.15), (40,0.40,0.15), (40,7.0,0.15), (-40,7.0,0.15)], m_pavers, WALK_TILE)

# hedge mass behind the sidewalk: reads near-black in the reference
hedge("Hedge", -40.0, 40.0, 7.0, 10.4, 5.4, 0.0, m_hedge)

# pedestrian signal
sig_blend = os.path.join(ASSETS, "signal.blend")
if os.path.isfile(sig_blend):
    for so in load_lib(sig_blend):
        ob = so.copy(); ob.name = "Signal"
        bpy.context.collection.objects.link(ob)
        place(ob, mathutils.Matrix.Translation(SIG_POS)
                  @ mathutils.Matrix.Diagonal((SIG_SCALE[0], SIG_SCALE[1], SIG_SCALE[2], 1.0)))
    # red lamp in library space: housing front at y=-0.437, lamps centred x~-0.40, red at z~3.55
    lx = SIG_POS[0] - 0.40 * SIG_SCALE[0]
    ly = SIG_POS[1] - 0.437 * SIG_SCALE[1] - 0.006
    lz = SIG_POS[2] + 3.55 * SIG_SCALE[2]
    hw = 0.16 * SIG_SCALE[0]; hh = 0.15 * SIG_SCALE[2]
    # wound so the geometric normal faces the camera: area emitters are one-sided
    quad("SignalRedLens", [(lx-hw, ly, lz-hh), (lx+hw, ly, lz-hh), (lx+hw, ly, lz+hh), (lx-hw, ly, lz+hh)], m_red)
else:
    print("[signal] WARNING no signal.blend, falling back to primitives")
    cylinder("SignalPole", (5.0, 0.90, 0.15), 0.075, 5.45, m_pole)
    box("SignalHousing", (5.0, 0.82, 4.50), (0.36, 0.24, 0.85), m_housing)
    quad("SignalRedLens",
         [(4.86, 0.695, 4.36), (5.14, 0.695, 4.36), (5.14, 0.695, 4.62), (4.86, 0.695, 4.62)], m_red)

# Real tree geometry, appended from the saved library and instanced. Copies share mesh
# data in Blender; the exporter writes each instance out in world space, which is what
# the renderer wants since it has no instancing.
ntri = 0; ti = 0; nsrc = 0
for lib_name, specs in TREE_SETS:
    tree_blend = os.path.join(ASSETS, lib_name)
    if not os.path.isfile(tree_blend):
        print("[trees] WARNING no tree library at", tree_blend); continue
    src_objs = load_lib(tree_blend); nsrc += len(src_objs)
    for spec in specs:
        tx, ty, rz, sc = spec[:4]
        lean = spec[4] if len(spec) > 4 else 0.0
        for so in src_objs:
            ob = so.copy()
            base = (so.name.replace("Melia_azedarach_HD_", "").replace("Melia ", "").replace("Oak_", "")
                        .replace(" ", "_").replace(".", "_"))
            ob.name = "Tree%d_%s" % (ti, base)
            bpy.context.collection.objects.link(ob)
            # lean is applied after the z-rotation so it always tilts toward +x
            place(ob, mathutils.Matrix.Translation((tx, ty, 0.0))
                      @ mathutils.Matrix.Rotation(lean, 4, 'Y')
                      @ mathutils.Euler((0.0, 0.0, rz), 'XYZ').to_matrix().to_4x4()
                      @ mathutils.Matrix.Scale(sc, 4))
            ntri += len(so.data.polygons)
        ti += 1
if ti:
    # Leaves are thin enough to transmit: without this the backlit canopy goes black,
    # since Disney on its own is reflection-only. Read by the exporter as
    # <float name="translucency">.
    nleaf = 0
    for m in bpy.data.materials:
        if "leaf" in m.name.lower() or "leavs" in m.name.lower():
            m["nori_translucency"] = LEAF_TRANSLUCENCY
            nleaf += 1
    print("[trees] %d instances, %d source meshes, %d faces total, %d leaf materials at translucency %.2f"
          % (ti, nsrc, ntri, nleaf, LEAF_TRANSLUCENCY))

# sun: Blender lamp is preview only (the exporter ignores lamps). The Nori render uses
# the geometric emissive disk from tools/scene_build/sun_disk.py, driven by the same angles.
el, az = math.radians(SUN_ELEV), math.radians(SUN_AZIM)
sd = (math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el))
sl = bpy.data.lights.new("Sun", type='SUN'); sl.energy = 4.0; sl.angle = math.radians(0.53)
so = bpy.data.objects.new("Sun", sl); bpy.context.collection.objects.link(so)
so.location = (sd[0]*60, sd[1]*60, sd[2]*60)
so.rotation_mode = 'QUATERNION'
so.rotation_quaternion = mathutils.Vector(sd).to_track_quat('Z', 'Y')

cd = bpy.data.cameras.new("Camera")
cd.sensor_fit = 'HORIZONTAL'
cd.angle = math.radians(CAM_FOV_H)
co = bpy.data.objects.new("Camera", cd); bpy.context.collection.objects.link(co)
co.location = CAM_LOC
d = mathutils.Vector((CAM_TGT[0]-CAM_LOC[0], CAM_TGT[1]-CAM_LOC[1], CAM_TGT[2]-CAM_LOC[2]))
co.rotation_mode = 'QUATERNION'
co.rotation_quaternion = d.to_track_quat('-Z', 'Y')
bpy.context.scene.camera = co
bpy.context.scene.render.resolution_x = RES_W
bpy.context.scene.render.resolution_y = RES_H

# the exporter reads matrix_world directly, which Blender only refreshes on evaluation
bpy.context.view_layer.update()
print("[build] objects:", len(bpy.data.objects), "bars:", n)
print("[build] sun dir:", ["%.4f" % c for c in sd])
