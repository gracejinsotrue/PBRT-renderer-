# tools/scene_build/pool_store_build.py -- full deterministic build of scenes/pool_store
#
# Rebuilds everything except the water (tools/scene_build/pool_water_build.py) and the slides
# (tools/scene_build/slide_build.py), which own their own geometry. Run inside Blender:
#   exec(open(r"C:\Users\gjin3\Desktop\nori-26sp\tools\scene_build\pool_store_build.py").read(), {})
#
# Blender Z-up, corridor along +Y, camera at the origin. All dimensions are
# fitted from the reference by back-projection; see /areas/pool-store-scene.md.

import bpy, math, os
import numpy as np

REPO = r"C:\Users\gjin3\Desktop\nori-26sp"
TEX  = os.path.join(REPO, "scenes", "pool_store", "textures")
sc   = bpy.context.scene

# ---------------------------------------------------------------- dimensions
WL, WR, HC = 1.964, 1.586, 2.85        # wall faces, ceiling
Y0, D      = -3.0, 26.0                # near limit, back wall
LWEND, COLW = 7.00, 0.63               # left wall ends / column
BAYX       = -6.20
PX0, PX1, PY0, PY1 = -1.85, 0.75, -3.0, 20.0
ZW, ZB     = -0.06, -0.41              # water surface, pool bottom
OZ0, OZ1   = 0.62, 2.38                # cooler opening height
CO1, CO2   = (1.30, 2.42), (3.05, 5.20)
CDEP       = 0.64                      # cooler recess depth
# ceiling penetrations
TEAL_HOLE  = (-0.40, 0.50, 5.35, 6.20)   # x0,x1,y0,y1
ORNG_HOLE  = ( 0.56, WR,   5.60, 7.40)   # wide enough for the angled tube
HDR_Z      = 2.64                        # header box underside

# ---------------------------------------------------------------- materials
def _clear(nt):
    for n in list(nt.nodes):
        if n.type != 'OUTPUT_MATERIAL':
            nt.nodes.remove(n)

def mat_tex(name, png, npng, rough, spec=0.5):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True; nt = m.node_tree; _clear(nt)
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(b.outputs[0], nt.nodes['Material Output'].inputs[0])
    t = nt.nodes.new('ShaderNodeTexImage')
    t.image = bpy.data.images.load(os.path.join(TEX, png), check_existing=True)
    nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    if npng:
        tn = nt.nodes.new('ShaderNodeTexImage')
        tn.image = bpy.data.images.load(os.path.join(TEX, npng), check_existing=True)
        tn.image.colorspace_settings.name = 'Non-Color'
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.inputs['Strength'].default_value = 1.0
        nt.links.new(tn.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], b.inputs['Normal'])
    b.inputs['Roughness'].default_value = rough
    b.inputs['Specular IOR Level' if 'Specular IOR Level' in b.inputs else 'Specular'].default_value = spec
    return m

def mat_col(name, rgb, rough=0.5, metallic=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True; nt = m.node_tree; _clear(nt)
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(b.outputs[0], nt.nodes['Material Output'].inputs[0])
    b.inputs['Base Color'].default_value = (*rgb, 1.0)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metallic
    return m

def mat_em(name, rgb, strength):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True; nt = m.node_tree; _clear(nt)
    e = nt.nodes.new('ShaderNodeEmission')
    e.inputs[0].default_value = (*rgb, 1.0); e.inputs[1].default_value = strength
    nt.links.new(e.outputs[0], nt.nodes['Material Output'].inputs[0])
    return m

def mat_glass(name, rgb, ior=1.52):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True; nt = m.node_tree; _clear(nt)
    g = nt.nodes.new('ShaderNodeBsdfGlass')
    g.inputs['Color'].default_value = (*rgb, 1.0)
    g.inputs['Roughness'].default_value = 0.0
    g.inputs['IOR'].default_value = ior
    nt.links.new(g.outputs[0], nt.nodes['Material Output'].inputs[0])
    return m

K = 1.0/8.3
M = {}
M['tile']    = mat_tex("MatWallPink",   "tile_wall_pink.png",  "tile_wall_pink_n.png",  0.13)
M['brick']   = mat_tex("MatWallBrick",  "tile_brick_pink.png", "tile_brick_pink_n.png", 0.11)
M['pool']    = mat_tex("MatPoolTile",   "tile_pool.png",       "tile_pool_n.png",       0.18)
M['deck']    = mat_tex("MatDeck",       "tile_deck.png",       "tile_deck_n.png",       0.30)
M['ceil']    = mat_tex("MatCeiling",    "ceiling_panel.png",   "ceiling_panel_n.png",   0.30)
M['prod']    = mat_tex("MatProducts",   "products.png",        None,                    0.55)
M['frame']   = mat_col("MatFrame",     (0.86,0.80,0.86), 0.22)
M['carcass'] = mat_col("MatCarcass",   (0.70,0.73,0.78), 0.45)
M['metal']   = mat_col("MatMetal",     (0.55,0.56,0.60), 0.34, metallic=1.0)
M['dark']    = mat_col("MatDark",      (0.06,0.06,0.07), 0.60)
M['gond']    = mat_col("MatGondola",   (0.80,0.13,0.13), 0.42)
M['signbody']= mat_col("MatSignBody",  (0.72,0.73,0.76), 0.35)
M['orange']  = mat_col("MatSlideOrg",  (0.88,0.46,0.34), 0.13)
M['teal']    = mat_col("MatSlideTeal", (0.44,0.79,0.84), 0.13)
M['glass']   = mat_glass("MatGlass",   (0.88,0.95,0.93))
M['water']   = mat_glass("MatWater",   (0.82,0.95,0.94), 1.333)
M['emCeil']  = mat_em("MatEmCeiling",  (1.00,0.70,0.80),  9.0*K*2.71)
M['emCool2'] = mat_em("MatEmCeilCool", (0.92,0.92,1.00),  9.0*K*1.02)
M['emBay']   = mat_em("MatEmBay",      (1.00,0.60,0.48), 16.0*K*0.22)
M['emCooler']= mat_em("MatEmCooler",   (0.96,0.98,1.00),  7.0*K*2.60)
M['emSignW'] = mat_em("MatEmSignW",    (1.00,0.93,0.95), 12.0*K*0.30)
M['emSignT'] = mat_em("MatEmSignT",    (0.18,0.80,0.72), 12.0*K*0.34)
M['emSignC'] = mat_em("MatEmSignC",    (1.00,0.42,0.36), 12.0*K*0.34)
M['emPink']  = mat_em("MatEmNeonPink", (1.00,0.34,0.60), 28.0*K*1.45)
M['emCyan']  = mat_em("MatEmNeonCyan", (0.25,0.85,1.00), 22.0*K*3.10)

# ---------------------------------------------------------------- geometry helpers
def qz(x0,x1,y0,y1,z,up):
    return [(x0,y0,z),(x1,y0,z),(x1,y1,z),(x0,y1,z)] if up else [(x0,y0,z),(x0,y1,z),(x1,y1,z),(x1,y0,z)]
def qx(x,y0,y1,z0,z1,plus):
    return [(x,y0,z0),(x,y1,z0),(x,y1,z1),(x,y0,z1)] if plus else [(x,y1,z0),(x,y0,z0),(x,y0,z1),(x,y1,z1)]
def qy(y,x0,x1,z0,z1,plus):
    return [(x1,y,z0),(x0,y,z0),(x0,y,z1),(x1,y,z1)] if plus else [(x0,y,z0),(x1,y,z0),(x1,y,z1),(x0,y,z1)]
def box(x0,x1,y0,y1,z0,z1):
    return [qz(x0,x1,y0,y1,z1,True), qz(x0,x1,y0,y1,z0,False),
            qx(x1,y0,y1,z0,z1,True), qx(x0,y0,y1,z0,z1,False),
            qy(y1,x0,x1,z0,z1,True), qy(y0,x0,x1,z0,z1,False)]
def zplane_holes(x0,x1,y0,y1,z,up,holes):
    """Horizontal plane with rectangular holes, split into y-bands."""
    out=[]; edges=sorted(set([y0,y1]+[v for h in holes for v in (h[2],h[3])]))
    edges=[e for e in edges if y0-1e-9<=e<=y1+1e-9]
    for a,b in zip(edges[:-1],edges[1:]):
        if b-a<1e-6: continue
        mid=(a+b)/2
        cut=[h for h in holes if h[2]<mid<h[3]]
        if not cut: out.append(qz(x0,x1,a,b,z,up)); continue
        xs=sorted(set([x0,x1]+[v for h in cut for v in (h[0],h[1])]))
        xs=[e for e in xs if x0-1e-9<=e<=x1+1e-9]
        for c,d in zip(xs[:-1],xs[1:]):
            if d-c<1e-6: continue
            mx=(c+d)/2
            if any(h[0]<mx<h[1] for h in cut): continue
            out.append(qz(c,d,a,b,z,up))
    return out
def acc(qs):
    v,f=[],[]
    for q in qs:
        n=len(v); v+=list(q); f.append(tuple(range(n,n+len(q))))
    return v,f
def mkobj(name, qs, mat, uv_span=None, smooth=False):
    v,f = acc(qs)
    me = bpy.data.meshes.new(name); me.from_pydata(v,[],f); me.update(); me.validate()
    if smooth:
        for p in me.polygons: p.use_smooth = True
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    ob.data.materials.append(mat)
    if uv_span: set_planar_uv(ob, uv_span)
    return ob
def set_planar_uv(ob, span):
    me = ob.data
    if not me.uv_layers: me.uv_layers.new(name="UVMap")
    uvl = me.uv_layers.active.data
    for poly in me.polygons:
        n = poly.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            if   ax == 2: u, v = co.x, co.y
            elif ax == 0: u, v = co.y, co.z
            else:         u, v = co.x, co.z
            uvl[li].uv = (u/span, v/span)

# ---------------------------------------------------------------- wipe old build
KEEP = {"Camera", "Water", "SlideOrange", "SlideTeal", "SlideLegs"}
for ob in list(sc.objects):
    if ob.name not in KEEP:
        bpy.data.objects.remove(ob, do_unlink=True)

# ---------------------------------------------------------------- shell
mkobj("Ceiling", zplane_holes(BAYX, WR, Y0, D+0.2, HC, False, [TEAL_HOLE, ORNG_HOLE]),
      M['ceil'], uv_span=2.44)

# left wall with the two cooler openings cut out, plus back / bay walls / column
qs  = [qx(-WL, Y0, LWEND, 0, OZ0, True), qx(-WL, Y0, LWEND, OZ1, HC, True),
       qx(-WL, Y0, CO1[0], OZ0, OZ1, True), qx(-WL, CO1[1], CO2[0], OZ0, OZ1, True),
       qx(-WL, CO2[1], LWEND, OZ0, OZ1, True)]
qs += [qy(D, BAYX, WR, 0, HC, False), qx(BAYX, LWEND, D, 0, HC, True)]
qs += box(-WL, -WL+COLW, LWEND, LWEND+COLW, 0.0, HC)
mkobj("TileWalls", qs, M['tile'], uv_span=0.5)
mkobj("WallRight", [qx(WR, Y0, D, 0, HC, False)], M['brick'], uv_span=0.4)

# deck (three strips around the pool) and pool basin
mkobj("Deck", [qz(BAYX, PX0, Y0, D, 0.0, True), qz(PX1, WR, Y0, D, 0.0, True),
               qz(PX0, PX1, PY1, D, 0.0, True)], M['deck'], uv_span=0.5)
mkobj("PoolShell", [qz(PX0,PX1,PY0,PY1,ZB,True), qx(PX0,PY0,PY1,ZB,0,True),
                    qx(PX1,PY0,PY1,ZB,0,False), qy(PY1,PX0,PX1,ZB,0,False)],
      M['pool'], uv_span=0.5)

# ---------------------------------------------------------------- ceiling penetrations
tx0,tx1,ty0,ty1 = TEAL_HOLE
mkobj("CeilRecessTeal", [qz(tx0,tx1,ty0,ty1,HC+0.45,False),
                         qx(tx1,ty0,ty1,HC,HC+0.45,False), qx(tx0,ty0,ty1,HC,HC+0.45,True),
                         qy(ty1,tx0,tx1,HC,HC+0.45,False), qy(ty0,tx0,tx1,HC,HC+0.45,True)],
      M['dark'])
ox0,ox1,oy0,oy1 = ORNG_HOLE
hx0,hx1,hy0,hy1 = ox0-0.10, WR, oy0-0.14, oy1+0.14
qs  = zplane_holes(hx0,hx1,hy0,hy1,HDR_Z,False,[(ox0,ox1,oy0,oy1)])
qs += [qy(hy1,hx0,hx1,HDR_Z,HC,True), qy(hy0,hx0,hx1,HDR_Z,HC,False),
       qx(hx0,hy0,hy1,HDR_Z,HC,False)]
mkobj("SlideHeader", qs, M['frame'])

# ---------------------------------------------------------------- coolers
frames, carcass, glass, shelves = [], [], [], []
FW, FP = 0.135, 0.10          # frame width, projection proud of the wall
for (a,b) in (CO1, CO2):
    xf = -WL + FP              # frame stands proud, into the room
    # frame ring: four bars around the opening, projecting into the room
    for (za,zb,ya,yb) in ((OZ1, OZ1+FW, a-FW, b+FW), (OZ0-FW, OZ0, a-FW, b+FW)):
        frames += box(-WL, xf, ya, yb, za, zb)
    for (ya,yb) in ((a-FW, a), (b, b+FW)):
        frames += box(-WL, xf, ya, yb, OZ0, OZ1)
    # vertical mullions splitting the opening into cooler doors
    ndoor = max(1, int(round((b-a)/0.95)))
    for k in range(1, ndoor):
        yk = a + (b-a)*k/ndoor
        frames += box(-WL, xf, yk-0.032, yk+0.032, OZ0, OZ1)
    for k in range(ndoor):
        ya = a + (b-a)*k/ndoor; yb = a + (b-a)*(k+1)/ndoor
        frames += box(-WL, xf-0.02, ya+0.05, yb-0.05, 1.46, 1.51)   # door pull rail
    # sill / curb under the opening
    frames += box(-WL, -WL+0.16, a-FW, b+FW, OZ0-0.16, OZ0-FW)
    # recess carcass
    carcass += [qz(-WL-CDEP,-WL,a,b,OZ1,False), qz(-WL-CDEP,-WL,a,b,OZ0,True),
                qy(a,-WL-CDEP,-WL,OZ0,OZ1,True), qy(b,-WL-CDEP,-WL,OZ0,OZ1,False)]
    # glazing, set just inside the opening
    xb = -WL+0.115; xt = -WL+0.012
    glass += [[(xb,a,OZ0),(xb,b,OZ0),(xt,b,OZ1),(xt,a,OZ1)]]
    # shelves
    for z in (0.80, 1.22, 1.64, 2.06):
        shelves += box(-WL-CDEP+0.03, -WL-0.17, a+0.03, b-0.03, z-0.013, z)
mkobj("CoolerFrames", frames, M['frame'], uv_span=0.5)
mkobj("CoolerCarcass", carcass, M['carcass'])
mkobj("CoolerGlass", glass, M['glass'])
mkobj("CoolerShelves", shelves, M['metal'])
# lit back panels
mkobj("EmCooler", [qz(-WL-CDEP+0.04, -WL-0.10, a+0.04, b-0.04, OZ1-0.045, False) for (a,b) in (CO1,CO2)], M['emCooler'])
mkobj("CoolerKick", [q for (a,b) in (CO1,CO2) for q in box(-WL-CDEP, -WL, a, b, OZ0-0.02, OZ0+0.06)], M['dark'])

# ---------------------------------------------------------------- product boxes
rng = np.random.default_rng(7)
PV, PF, PUV = [], [], []
CGRID = 8
def add_prod(x0,x1,y0,y1,z0,z1):
    cell = int(rng.integers(0, CGRID*CGRID))
    cu, cv = cell % CGRID, cell // CGRID
    s = 1.0/CGRID; u0, v0 = cu*s, cv*s
    uvq = [(u0+0.02*s, v0+0.02*s), (u0+0.98*s, v0+0.02*s), (u0+0.98*s, v0+0.98*s), (u0+0.02*s, v0+0.98*s)]
    for q in box(x0,x1,y0,y1,z0,z1):
        n = len(PV); PV.extend(q); PF.append(tuple(range(n, n+4))); PUV.extend(uvq)
# stock the coolers
for (a,b) in (CO1, CO2):
    for z in (0.80, 1.22, 1.64, 2.06):
        y = a + 0.06
        while y < b - 0.14:
            w = float(rng.uniform(0.10, 0.17)); h = float(rng.uniform(0.26, 0.34))
            d = float(rng.uniform(0.18, 0.28))
            add_prod(-2.22-d, -2.22, y, y+w, z, z+h)
            y += w + float(rng.uniform(0.01, 0.03))
# stock the back-bay gondola
GX0, GX1 = -2.70, -1.80
for gy0, gy1 in ((8.60, 11.40), (12.20, 15.00)):
    for z in (0.42, 0.86, 1.30, 1.74):
        for side, xb in ((1, GX1),):
            y = gy0 + 0.05
            while y < gy1 - 0.16:
                w = float(rng.uniform(0.10, 0.18)); h = float(rng.uniform(0.20, 0.36))
                d = float(rng.uniform(0.14, 0.24))
                if side > 0: add_prod(xb-d, xb, y, y+w, z, z+h)
                else:        add_prod(xb, xb+d, y, y+w, z, z+h)
                y += w + float(rng.uniform(0.01, 0.03))
me = bpy.data.meshes.new("Products"); me.from_pydata(PV, [], PF); me.update(); me.validate()
uvl = me.uv_layers.new(name="UVMap").data
for i, poly in enumerate(me.polygons):
    for k, li in enumerate(poly.loop_indices):
        uvl[li].uv = PUV[i*4+k]
obp = bpy.data.objects.new("Products", me); sc.collection.objects.link(obp)
obp.data.materials.append(M['prod'])

# gondola frame
gq = []
for gy0, gy1 in ((8.60, 11.40), (12.20, 15.00)):
    gq += box(GX0, GX1, gy0, gy1, 0.0, 0.30)
    for z in (0.42, 0.86, 1.30, 1.74):
        gq += box(GX0, GX1, gy0, gy1, z-0.04, z)
    gq += box(GX0+0.60, GX1-0.60, gy0, gy1, 0.0, 2.05)
mkobj("Gondola", gq, M['gond'])

# ---------------------------------------------------------------- storefront doors
dq, dg = [], []
DZ = 2.35; DY = D - 0.02
for dx0, dx1 in ((-1.55, -0.35), (-0.25, 0.95)):
    dq += box(dx0-0.06, dx0, DY-0.06, DY, 0.0, DZ)
    dq += box(dx1, dx1+0.06, DY-0.06, DY, 0.0, DZ)
    dq += box(dx0, dx1, DY-0.06, DY, DZ-0.06, DZ)
    dq += box(dx0, dx1, DY-0.06, DY, 0.95, 1.01)
    dg += [qy(DY-0.03, dx0, dx1, 0.02, DZ-0.06, False)]
mkobj("DoorFrames", dq, M['frame'])
mkobj("DoorGlass", dg, M['glass'])

# ---------------------------------------------------------------- blade sign
SY0, SY1 = 3.86, 3.98
SX0, SX1, SZ0, SZ1 = -1.46, -0.36, 1.45, 2.45
mkobj("SignBody", [qz(SX0,SX1,SY0,SY1,SZ1,True), qz(SX0,SX1,SY0,SY1,SZ0,False),
                   qx(SX0,SY0,SY1,SZ0,SZ1,False), qx(SX1,SY0,SY1,SZ0,SZ1,True)]
                  + box(-WL, SX0, SY0+0.03, SY1-0.03, 1.86, 2.08), M['signbody'])
mkobj("SignFieldW", [qy(SY0, SX0, SX1, SZ0, SZ1, False), qy(SY1, SX0, SX1, SZ0, SZ1, True)], M['emSignW'])
# invented store mark: a teal wave band and a coral disc
band = []
for i in range(24):
    t0, t1 = i/24.0, (i+1)/24.0
    xa, xb = SX0+0.07 + t0*(SX1-SX0-0.14), SX0+0.07 + t1*(SX1-SX0-0.14)
    za = 1.80 + 0.075*math.sin(t0*math.pi*2.0); zb = 1.80 + 0.075*math.sin(t1*math.pi*2.0)
    band.append([(xa,SY0-0.004,za-0.055),(xb,SY0-0.004,zb-0.055),(xb,SY0-0.004,zb+0.055),(xa,SY0-0.004,za+0.055)])
mkobj("SignMarkT", band, M['emSignT'])
disc = []
cx, cz, r = SX0+0.24, 2.16, 0.105
for i in range(20):
    a0, a1 = 2*math.pi*i/20, 2*math.pi*(i+1)/20
    disc.append([(cx,SY0-0.004,cz), (cx+r*math.cos(a0),SY0-0.004,cz+r*math.sin(a0)),
                 (cx+r*math.cos(a1),SY0-0.004,cz+r*math.sin(a1))])
mkobj("SignMarkC", disc, M['emSignC'])

# ---------------------------------------------------------------- ceiling lights
qs_pink, qs_cool = [], []
for x0, x1 in ((-1.62,-1.06), (-0.28,0.28)):
    for y in (0.0,2.9,5.8,8.7,11.6,14.5,17.4,20.3):
        qs_pink.append(qz(x0,x1,y,y+2.3,HC-0.005,False))
for y in (0.0,2.9,8.7,11.6,14.5,17.4,20.3):
    qs_cool.append(qz(0.86,1.42,y,y+2.3,HC-0.005,False))
mkobj("EmCeiling", qs_pink, M['emCeil'])
mkobj("EmCeilRight", qs_cool, M['emCool2'])
mkobj("EmBay", [qz(x0,x1,y,y+2.0,HC-0.005,False)
                for x0,x1 in ((-5.30,-4.95),(-3.70,-3.35),(-2.30,-1.95))
                for y in (8.5,11.5,14.5,17.5)], M['emBay'])

# ---------------------------------------------------------------- neon tubes
def tube(name, p0, p1, r, mat, K=10):
    a = np.array(p0,float); b = np.array(p1,float); d = b-a; L = np.linalg.norm(d); d/=L
    up = np.array([0,0,1.0])
    if abs(np.dot(up,d))>0.9: up = np.array([1.0,0,0])
    u = np.cross(d,up); u/=np.linalg.norm(u); v = np.cross(d,u)
    qs=[]
    for i in range(K):
        t0,t1 = 2*math.pi*i/K, 2*math.pi*(i+1)/K
        p00=a+r*(math.cos(t0)*u+math.sin(t0)*v); p01=a+r*(math.cos(t1)*u+math.sin(t1)*v)
        qs.append([tuple(p00),tuple(p01),tuple(p01+d*L),tuple(p00+d*L)])
    return mkobj(name, qs, mat, smooth=True)
tube("NeonPink", (WR-0.10, 7.60, 2.34), (WR-0.10, 9.90, 2.34), 0.030, M['emPink'])
tube("NeonCyan", (WR-0.10, 12.00, 2.36), (WR-0.10, 14.30, 2.36), 0.026, M['emCyan'])

# ---------------------------------------------------------------- camera + render settings
cam = bpy.data.objects.get("Camera")
if cam is None:
    cd = bpy.data.cameras.new("Camera"); cam = bpy.data.objects.new("Camera", cd)
    sc.collection.objects.link(cam)
cam.data.lens = 24.0; cam.data.sensor_width = 36.0; cam.data.sensor_fit = 'AUTO'
cam.location = (0.0, 0.0, 1.362)
cam.rotation_euler = (math.radians(87.479), 0.0, math.radians(12.322))
sc.camera = cam
sc.render.resolution_x, sc.render.resolution_y = 1000, 1233

bpy.ops.wm.save_mainfile()
print("built:", sorted(o.name for o in sc.objects))
print("products:", len(PF)//6, "boxes")
