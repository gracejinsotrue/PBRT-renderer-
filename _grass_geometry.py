# Scatter real grass-blade geometry onto an already-exported Nori ground mesh.
#
# Reads scenes/<scene>/meshes/Ground.obj (Nori Y-up world space, per-corner
# triangles as written by _blender_to_nori.py) and writes grass_a.obj/grass_b.obj:
# tapered 2-triangle blades standing on the ground surface.
#
# Density falls off as 1/d^2 from the camera so screen-space blade density stays
# roughly constant and distant grass stays cheap (the ground texture covers the
# far field). Blades are split into two files by mow band, leaning in opposite
# directions - that is what actually produces mower stripes in real turf.
#
#   GRASS_D0=12000 GRASS_DREF=5 GRASS_MAXDIST=45 python3 _grass_geometry.py
import os, math, numpy as np, xml.etree.ElementTree as ET

SCENE = os.environ.get("GRASS_SCENE", os.path.expanduser("~/mnt/nori-26sp/scenes/liminal_bed"))
OUTDIR= os.environ.get("GRASS_OUTDIR", os.path.expanduser("~"))
D0    = float(os.environ.get("GRASS_D0", "12000"))    # blades/m^2 at/below DREF
DREF  = float(os.environ.get("GRASS_DREF", "5.0"))
MAXD  = float(os.environ.get("GRASS_MAXDIST", "45"))
HMEAN = float(os.environ.get("GRASS_H", "0.045"))     # mown lawn blade height (m)
CAP   = int(float(os.environ.get("GRASS_CAP", "1200000")))
BAND  = float(os.environ.get("GRASS_BAND", str(8.0/11.0)))   # mow band width, matches the texture
SEED  = int(os.environ.get("GRASS_SEED", "7"))
rng = np.random.default_rng(SEED)

# ---- camera from scene.xml -------------------------------------------------
root = ET.parse(os.path.join(SCENE,"scene.xml")).getroot()
la = root.find("camera").find(".//lookat")
O = np.array([float(x) for x in la.get("origin").split(",")], np.float64)
T = np.array([float(x) for x in la.get("target").split(",")], np.float64)
fov = float(root.find("camera").find("./float[@name='fov']").get("value"))
D = T - O; D /= np.linalg.norm(D)
print("[cam] origin",O,"dir",np.round(D,3),"hfov",fov)

# ---- ground triangles (per-corner => tri i = verts 3i..3i+2) ---------------
V=[]
with open(os.path.join(SCENE,"meshes","Ground.obj")) as f:
    for ln in f:
        if ln[0]=='v' and ln[1]==' ':
            p=ln.split(); V.append((float(p[1]),float(p[2]),float(p[3])))
V=np.asarray(V,np.float64)
assert len(V)%3==0, "expected per-corner triangles"
tri=V.reshape(-1,3,3)
print("[ground] %d triangles"%len(tri))

A,B,C = tri[:,0],tri[:,1],tri[:,2]
cen=(A+B+C)/3.0
cross=np.cross(B-A,C-A)
area=0.5*np.linalg.norm(cross,axis=1)
nrm=cross/np.maximum(np.linalg.norm(cross,axis=1,keepdims=True),1e-12)
nrm*=np.sign(nrm[:,1:2]+1e-12)                       # point up

rel=cen-O; dist=np.linalg.norm(rel,axis=1)
# horizontal angle to the view direction
hv=np.stack([rel[:,0],rel[:,2]],1); hv/=np.maximum(np.linalg.norm(hv,axis=1,keepdims=True),1e-12)
hd=np.array([D[0],D[2]]); hd/=np.linalg.norm(hd)
cosang=hv@hd
keep=(dist<MAXD)&((cosang>math.cos(math.radians(fov*0.5+22.0)))|(dist<6.0))
print("[select] %d of %d triangles in view"%(keep.sum(),len(tri)))

dens = D0*np.minimum(1.0,(DREF/np.maximum(dist,1e-3))**2)
counts=np.zeros(len(tri),np.int64)
exp = dens[keep]*area[keep]
scale=1.0
tot=exp.sum()
if tot>CAP:
    scale=CAP/tot; print("[cap] scaling density by %.3f to respect cap"%scale)
counts[keep]=rng.poisson(exp*scale)
N=int(counts.sum()); print("[blades] %d"%N)

idx=np.repeat(np.arange(len(tri)),counts)
# uniform barycentric sample
u=rng.random(N); v=rng.random(N)
m=u+v>1.0; u[m]=1-u[m]; v[m]=1-v[m]
P = A[idx] + (B[idx]-A[idx])*u[:,None] + (C[idx]-A[idx])*v[:,None]
Nrm = nrm[idx]

# ---- curved blades: 2 segments (6 verts, 4 tris) so they ARCH instead of
# standing as rigid needles, which is what read as "choppy" ------------------
HVAR = float(os.environ.get("GRASS_HVAR", "0.10"))   # mown turf = even canopy
BEND = float(os.environ.get("GRASS_BEND", "0.55"))

h = np.clip(rng.normal(HMEAN, HMEAN*HVAR, N), HMEAN*0.80, HMEAN*1.25)
band = (np.floor(P[:,0]/BAND).astype(np.int64) % 2)
bandsign = np.where(band==0, 1.0, -1.0)

# lean direction: biased along the mow band, with per-blade spread
a = rng.random(N)*2*math.pi
randd = np.stack([np.cos(a), np.zeros(N), np.sin(a)], 1)
bandd = np.stack([np.zeros(N), np.zeros(N), bandsign], 1)
leanU = 0.60*bandd + 0.70*randd
leanU[:,1] = 0.0
leanU /= np.maximum(np.linalg.norm(leanU,axis=1,keepdims=True),1e-9)

up = Nrm
side = np.cross(up, leanU)
side /= np.maximum(np.linalg.norm(side,axis=1,keepdims=True),1e-9)

bend = (BEND*rng.uniform(0.55,1.45,N))[:,None]
wb = (0.0035*rng.uniform(0.7,1.3,N))[:,None]
wt = (0.0010*rng.uniform(0.7,1.3,N))[:,None]
base = P - up*0.004
hh = h[:,None]

def cross_section(t):
    rise  = hh*(t - 0.30*t*t)        # decelerating rise
    horiz = hh*bend*(t*t)            # accelerating sideways => arch
    c = base + up*rise + leanU*horiz
    w = (wb*(1.0-t) + wt*t)*0.5
    return c - side*w, c + side*w

L0,R0 = cross_section(0.0)
L1,R1 = cross_section(0.5)
L2,R2 = cross_section(1.0)

def write_obj(path, mask):
    n=int(mask.sum())
    if n==0: return 0,0
    q=np.empty((n*6,3),np.float32)
    for k,arr in enumerate((L0,R0,L1,R1,L2,R2)):
        q[k::6]=arr[mask]
    with open(path,"w") as f:
        f.write("# curved grass blades (%d)\n"%n)
        CH=400000
        for s0 in range(0,len(q),CH):
            c=q[s0:s0+CH]
            f.write("\n".join(np.char.add(np.char.add(
                np.char.mod("v %.4f",c[:,0]), np.char.mod(" %.4f",c[:,1])),
                np.char.mod(" %.4f",c[:,2])))+"\n")
        b=np.arange(n,dtype=np.int64)*6+1
        for s0 in range(0,n,CH):
            bb=b[s0:s0+CH]
            def tri(i,j,k):
                return np.char.add(np.char.add(np.char.mod("f %d",bb+i),
                       np.char.mod(" %d",bb+j)), np.char.mod(" %d",bb+k))
            t1,t2,t3,t4 = tri(0,1,3), tri(0,3,2), tri(2,3,5), tri(2,5,4)
            inter=np.empty(len(bb)*4,dtype=object)
            inter[0::4]=t1; inter[1::4]=t2; inter[2::4]=t3; inter[3::4]=t4
            f.write("\n".join(inter)+"\n")
    return n, n*4

os.makedirs(OUTDIR,exist_ok=True)
for name,mask in (("grass_a",band==0),("grass_b",band==1)):
    p=os.path.join(OUTDIR,name+".obj")
    nb,nt=write_obj(p,mask)
    print("[out] %s: %d blades, %d tris, %.1f MB"%(p,nb,nt,os.path.getsize(p)/1e6))
