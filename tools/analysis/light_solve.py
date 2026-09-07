# tools/analysis/light_solve.py -- solve emitter gains against reference patches.
#
# Radiance is linear in each emitter's radiance, so render one pass per light
# group with only that group on, then fit non-negative gains so the summed
# image matches the reference's measured patches after the display transform.
# Run inside Blender (it needs to launch the Windows exe):
#   exec(open(r"C:\Users\gjin3\Desktop\nori-26sp\tools\analysis\light_solve.py").read(), {})
# Writes scenes/pool_store/_ls/basis_<k>.exr and _ls/DONE.

import os, re, subprocess, shutil

REPO = r"C:\Users\gjin3\Desktop\nori-26sp"
SCN  = os.path.join(REPO, "scenes", "pool_store")
OUT  = os.path.join(SCN, "_ls")
os.makedirs(OUT, exist_ok=True)

GROUPS = [
    ("ceiling",  ["EmCeiling.obj"]),
    ("ceilrght", ["EmCeilRight.obj"]),
    ("bay",      ["EmBay.obj"]),
    ("cooler",   ["EmCooler.obj"]),
    ("sign",     ["SignFieldW.obj", "SignMarkT.obj", "SignMarkC.obj"]),
    ("neon",     ["NeonPink.obj", "NeonCyan.obj"]),
]

src = open(os.path.join(SCN, "scene.xml")).read()
blocks = re.findall(r"<mesh type=\"obj\">.*?</mesh>", src, re.S)

def variant(keep):
    out = src
    for b in blocks:
        if "<emitter" not in b:
            continue
        fn = re.search(r'filename" value="meshes/([^"]+)"', b).group(1)
        if fn in keep:
            continue
        nb = re.sub(r'(<color name="radiance" value=")[^"]+(")', r'\g<1>0 0 0\g<2>', b)
        out = out.replace(b, nb)
    out = out.replace('name="width" value="1000"', 'name="width" value="500"')
    out = out.replace('name="height" value="1233"', 'name="height" value="617"')
    out = re.sub(r'(<integer name="sampleCount" value=")\d+(")', r'\g<1>64\g<2>', out)
    return out

marker = os.path.join(OUT, "DONE")
if os.path.exists(marker):
    os.remove(marker)
log = open(os.path.join(REPO, "_lightsolve.log"), "w")

def run():
    for i, (name, keep) in enumerate(GROUPS):
        xml = os.path.join(SCN, "_ls_scene.xml")
        open(xml, "w").write(variant(keep))
        for f in os.listdir(SCN):
            if f.startswith("snapshot_64"):
                try: os.remove(os.path.join(SCN, f))
                except Exception: pass
        subprocess.run([os.path.join(REPO, r"build\Release\nori-dxr.exe"),
                        r"scenes\pool_store\_ls_scene.xml", "--headless"],
                       cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                       creationflags=0x08000000)
        srcf = os.path.join(SCN, "snapshot_64.exr")
        if os.path.exists(srcf):
            shutil.copyfile(srcf, os.path.join(OUT, "basis_%d_%s.exr" % (i, name)))
            log.write("wrote basis %d %s\n" % (i, name)); log.flush()
    open(marker, "w").write("ok")

import threading
threading.Thread(target=run, daemon=False).start()
print("light solve started, %d groups" % len(GROUPS))
