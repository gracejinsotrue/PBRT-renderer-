# Direct Wavefront OBJ+MTL -> Nori scene converter (no Blender).
# Streams a large OBJ, splits by material into per-material OBJs (authored
# normals + UVs, no dedup for low memory), maps each MTL material to a disney
# BSDF with albedo/normal textures, and writes scene.xml with a bbox camera.
import os, re, math, traceback

HOME   = os.path.expanduser("~")
SRC    = os.path.join(HOME, "mnt", "San_Miguel")
OBJ    = os.path.join(SRC, "san-miguel.obj")
MTL    = os.path.join(SRC, "san-miguel.mtl")
OUT    = os.path.join(HOME, "mnt", "nori-26sp", "scenes", "san_miguel")
MESH   = os.path.join(OUT, "meshes")
STATUS = os.path.join(HOME, "sanmiguel.status")

def log(*a): print(*a, flush=True)
def san(n): return re.sub(r'[^A-Za-z0-9_.-]', '_', n)

def parse_mtl(path):
    mats = {}; cur = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            t = line.split()
            if not t: continue
            k = t[0].lower()
            if k == 'newmtl':
                cur = line.split(None, 1)[1].strip(); mats[cur] = {}
            elif cur is None:
                continue
            elif k == 'kd' and len(t) >= 4: mats[cur]['Kd'] = (t[1], t[2], t[3])
            elif k == 'ns' and len(t) >= 2:
                try: mats[cur]['Ns'] = float(t[1])
                except: pass
            elif k == 'map_kd': mats[cur]['map_Kd'] = t[-1]
            elif k in ('map_bump', 'bump'): mats[cur]['map_bump'] = t[-1]
    return mats

def bsdf_xml(name, mtl):
    info  = mtl.get(name, {})
    kd    = info.get('Kd', ('0.8', '0.8', '0.8'))
    ns    = info.get('Ns', 50.0)
    rough = max(0.02, min(1.0, math.sqrt(2.0 / (ns + 2.0))))
    L = ['\t\t<bsdf type="disney">',
         '\t\t\t<color name="baseColor" value="%s %s %s"/>' % (kd[0], kd[1], kd[2]),
         '\t\t\t<float name="roughness" value="%.4f"/>' % rough,
         '\t\t\t<float name="metallic" value="0.0"/>']
    tex = info.get('map_Kd')
    if tex:
        L.append('\t\t\t<string name="albedoTexture" value="textures/%s"/>' % os.path.basename(tex.replace('\\', '/')))
    bump = info.get('map_bump')
    if bump:
        L.append('\t\t\t<string name="normalTexture" value="textures/%s"/>' % os.path.basename(bump.replace('\\', '/')))
    L.append('\t\t</bsdf>')
    return '\n'.join(L)

def main():
    os.makedirs(MESH, exist_ok=True)
    mtl = parse_mtl(MTL); log("materials in mtl:", len(mtl))
    V = []; VT = []; VN = []
    fh = {}; cnt = {}; mode = {}; order = []
    bmin = [1e30] * 3; bmax = [-1e30] * 3
    cur = None; nline = 0

    def ensure(m):
        if m not in fh:
            fh[m] = open(os.path.join(MESH, san(m) + ".obj"), 'w')
            fh[m].write("# material %s\n" % m); cnt[m] = 0; order.append(m)

    with open(OBJ, 'r', errors='replace') as f:
        for line in f:
            nline += 1
            if nline % 4000000 == 0:
                log("  %dM lines  V=%d VT=%d VN=%d mats=%d" % (nline // 1000000, len(V), len(VT), len(VN), len(order)))
            c = line[0]
            if c == 'v':
                c1 = line[1]
                if c1 == ' ':
                    s = line[2:].strip(); V.append(s); p = s.split()
                    try:
                        x = float(p[0]); y = float(p[1]); z = float(p[2])
                        if x < bmin[0]: bmin[0] = x
                        if y < bmin[1]: bmin[1] = y
                        if z < bmin[2]: bmin[2] = z
                        if x > bmax[0]: bmax[0] = x
                        if y > bmax[1]: bmax[1] = y
                        if z > bmax[2]: bmax[2] = z
                    except: pass
                elif c1 == 't': VT.append(line[3:].strip())
                elif c1 == 'n': VN.append(line[3:].strip())
            elif c == 'u' and line.startswith('usemtl'):
                pr = line.split(None, 1); cur = pr[1].strip() if len(pr) > 1 else '_default'; ensure(cur)
            elif c == 'f' and line[1] == ' ':
                if cur is None: cur = '_default'; ensure(cur)
                h = fh[cur]; toks = line[2:].split()
                if cur not in mode:
                    p0 = toks[0].split('/')
                    if len(p0) >= 3 and p0[2] != '':
                        mode[cur] = 'vtn' if (len(p0) >= 2 and p0[1] != '') else 'vn'
                    elif len(p0) >= 2 and p0[1] != '':
                        mode[cur] = 'vt'
                    else:
                        mode[cur] = 'v'
                m = mode[cur]
                nv = len(V); nvt = len(VT); nvn = len(VN)
                buf = []; corners = []
                for tk in toks:
                    p = (tk + '//').split('/')
                    vi = int(p[0])
                    if vi < 0: vi = nv + vi + 1
                    buf.append("v " + V[vi - 1] + "\n")
                    if m in ('vtn', 'vt') and p[1]:
                        ti = int(p[1]);  ti = (nvt + ti + 1) if ti < 0 else ti
                        buf.append("vt " + (VT[ti - 1] if 1 <= ti <= nvt else "0 0") + "\n")
                    if m in ('vtn', 'vn') and p[2]:
                        ni = int(p[2]);  ni = (nvn + ni + 1) if ni < 0 else ni
                        buf.append("vn " + (VN[ni - 1] if 1 <= ni <= nvn else "0 0 1") + "\n")
                    corners.append(1)
                k = len(corners); base = cnt[cur]
                li = [base + 1 + i for i in range(k)]; cnt[cur] = base + k
                fl = []
                for i in range(1, k - 1):
                    a, b, cc = li[0], li[i], li[i + 1]
                    if m == 'vtn': fl.append("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (a, a, a, b, b, b, cc, cc, cc))
                    elif m == 'vn': fl.append("f %d//%d %d//%d %d//%d\n" % (a, a, b, b, cc, cc))
                    elif m == 'vt': fl.append("f %d/%d %d/%d %d/%d\n" % (a, a, b, b, cc, cc))
                    else: fl.append("f %d %d %d\n" % (a, b, cc))
                h.write("".join(buf)); h.write("".join(fl))

    log("parse done  V=%d VT=%d VN=%d materials=%d" % (len(V), len(VT), len(VN), len(order)))
    for h in fh.values(): h.close()

    blocks = []; written = 0
    for mi, mat in enumerate(order):
        if cnt.get(mat, 0) == 0: continue
        fn = san(mat) + ".obj"
        blocks.append('\t<mesh type="obj">\n\t\t<string name="filename" value="meshes/%s"/>\n%s\n\t</mesh>' % (fn, bsdf_xml(mat, mtl)))
        written += 1

    cx = (bmin[0] + bmax[0]) / 2; cy = (bmin[1] + bmax[1]) / 2; cz = (bmin[2] + bmax[2]) / 2
    sx = bmax[0] - bmin[0]; sy = bmax[1] - bmin[1]; sz = bmax[2] - bmin[2]
    eye_y = bmin[1] + min(1.6, sy * 0.4)
    if sx >= sz:
        ox = bmax[0] - 0.02 * sx; oz = cz; tx = bmin[0]; tz = cz
    else:
        oz = bmax[2] - 0.02 * sz; ox = cx; tz = bmin[2]; tx = cx
    cam = ('\t<camera type="perspective">\n'
           '\t\t<float name="fov" value="55"/>\n'
           '\t\t<transform name="toWorld">\n'
           '\t\t\t<lookat target="%.4f, %.4f, %.4f" origin="%.4f, %.4f, %.4f" up="0, 1, 0"/>\n'
           '\t\t</transform>\n'
           '\t\t<integer name="width" value="1024"/>\n'
           '\t\t<integer name="height" value="1024"/>\n'
           '\t</camera>' % (tx, eye_y, tz, ox, eye_y, oz))
    header = ("<?xml version='1.0' encoding='utf-8'?>\n\n<scene>\n"
              '\t<string name="envmap" value="white.hdr"/>\n'
              '\t<float name="envmapScale" value="1.0"/>\n'
              '\t<float name="evCompensation" value="0.0"/>\n\n')
    sampler = '\t<sampler type="independent">\n\t\t<integer name="sampleCount" value="16"/>\n\t</sampler>'
    xml = header + cam + "\n\n" + sampler + "\n\n" + "\n\n".join(blocks) + "\n</scene>\n"
    with open(os.path.join(OUT, "scene.xml"), 'w') as sf:
        sf.write(xml)
    log("scene.xml: %d bytes, %d meshes" % (len(xml), written))
    open(STATUS, 'w').write("DONE meshes=%d V=%d bbox_min=%s bbox_max=%s\n" % (written, len(V), bmin, bmax))

try:
    main()
except Exception as e:
    open(STATUS, 'w').write("ERROR: %s\n%s\n" % (e, traceback.format_exc()))
    raise
