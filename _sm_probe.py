# Is a candidate camera position actually in open air?
# Renders are ~5 min, so check first. Scans only the meshes whose bbox comes near
# the point, and reports the closest vertex of each -- foliage flagged separately,
# since leaves are what buried the camera in every failed framing so far.
#   python _sm_probe.py 22.0 1.7 5.0 [radius]
import json, os, sys, numpy as np

BASE = "scenes/san_miguel/meshes"
P = np.array([float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])], np.float32)
R = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
LEAF = {"materialt.obj", "materialu.obj", "materialv.obj", "materials.obj"}

bb = json.load(open("scenes/san_miguel/_scratch/bbox.json"))
near = [k for k, v in bb.items()
        if all(v["min"][i] - R <= P[i] <= v["max"][i] + R for i in range(3))]

hits = []
for fn in near:
    vs = []
    for line in open(os.path.join(BASE, fn), errors="ignore"):
        if line[0] == "v" and line[1] == " ":
            a = line.split()
            vs.append((float(a[1]), float(a[2]), float(a[3])))
    if not vs:
        continue
    v = np.array(vs, np.float32)
    d = np.linalg.norm(v - P, axis=1)
    if d.min() <= R:
        hits.append((float(d.min()), fn, int((d <= R).sum())))

hits.sort()
print(f"probe ({P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f})  radius {R} m   -- {len(near)} meshes to scan")
if not hits:
    print("  CLEAR: no geometry within radius")
for d, fn, n in hits[:12]:
    tag = "FOLIAGE" if fn in LEAF else "       "
    print(f"  {tag} {fn:22s} nearest {d:5.2f} m  ({n} verts inside radius)")
