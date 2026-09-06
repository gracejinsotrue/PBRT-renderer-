# Blender -> Nori exporter
# Per-object triangulated OBJs + a Nori scene.xml with disney BSDFs from each
# Principled BSDF, camera from the active Blender camera. Run inside Blender:
#   g = {"NORI_OUT": "smoke_test", "NORI_SAMPLES": 64}
#   exec(open(r"C:\Users\gjin3\Desktop\nori-26sp\_blender_to_nori.py").read(), g)
#
# Optional globals (backward-compatible defaults):
#   NORI_ENV / NORI_ENVSCALE   env map filename + scale (default white.hdr / 0.4)
#   NORI_AREALIGHT             add synthetic overhead light (default True)
#   NORI_W / NORI_H            render dims (default 800x800); fov written as HORIZONTAL fov
#   NORI_YUP                   convert Blender Z-up world -> Nori Y-up (Rx-90). Default False.
#                              Needed for sky-HDRI scenes: the DXR envmap+camera are Y-up.
#   Glass BSDF -> <bsdf type="dielectric"> (intIOR from IOR, tintColor from Color)
#   NORI_NORMALS               write vn from Blender corner normals (default True) so shade_smooth/
#                              shade_flat/sharp edges survive export. False = old behaviour.
#   NORI_SKIP                  iterable of object names to NOT export as meshes (e.g. a volume domain)
import bpy, os, math, shutil, mathutils

_YUP = False
def _tp(x, y, z):
    # Blender Z-up -> Nori Y-up: (x, y, z) -> (x, z, -y)   [proper rotation Rx(-90)]
    return (x, z, -y) if _YUP else (x, y, z)

def _f(x):   return "%.6f" % x
def _col(c): return "%.6f %.6f %.6f" % (c[0], c[1], c[2])

def principled(mat):
    if not mat or not mat.use_nodes: return None
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED': return n
    return None

def inp(node, name, default=None):
    s = node.inputs.get(name)
    if s is None: return default
    try:    return s.default_value
    except: return default

def _img_through(sock):
    if not sock or not sock.is_linked: return None
    fn = sock.links[0].from_node
    if fn.type == 'TEX_IMAGE': return fn.image
    for i in fn.inputs:
        if i.is_linked and i.links[0].from_node.type == 'TEX_IMAGE':
            return i.links[0].from_node.image
    return None

def base_color_image(mat):
    n = principled(mat); return _img_through(n.inputs.get('Base Color')) if n else None
def normal_image(mat):
    n = principled(mat); return _img_through(n.inputs.get('Normal')) if n else None

def export_image(img, texdir):
    if img is None: return None
    os.makedirs(texdir, exist_ok=True)
    src = bpy.path.abspath(img.filepath_raw) if img.filepath_raw else ""
    base = bpy.path.clean_name(os.path.splitext(os.path.basename(src or img.name))[0]) or "tex"
    fn = base + ".png"; dst = os.path.join(texdir, fn)
    try:
        if src and os.path.isfile(src) and os.path.abspath(src) == os.path.abspath(dst): return fn
        if src.lower().endswith(".png") and os.path.isfile(src):
            shutil.copyfile(src, dst); return fn
        old = img.filepath_raw; img.file_format = 'PNG'
        img.filepath_raw = dst; img.save(); img.filepath_raw = old; return fn
    except Exception as e:
        print("tex export warn", img.name, e); return None

def write_obj(obj, path, depsgraph, with_uv, with_normals=True):
    ev = obj.evaluated_get(depsgraph); me = ev.to_mesh(); mw = obj.matrix_world
    me.calc_loop_triangles()
    uvl = me.uv_layers.active.data if (with_uv and me.uv_layers.active) else None
    cn = None; nmat = None
    if with_normals:
        try:
            nmat = mw.inverted_safe().transposed().to_3x3(); cn = me.corner_normals
        except Exception as e:
            print("normal export warn", obj.name, e); cn = None
    vnl, vnmap = [], {}
    def _vni(li):
        v = nmat @ mathutils.Vector(cn[li].vector)
        L = v.length
        if L > 1e-12: v = v / L
        k = (round(v.x, 4), round(v.y, 4), round(v.z, 4))
        i = vnmap.get(k)
        if i is None:
            t = _tp(k[0], k[1], k[2])
            vnl.append("vn %s %s %s\n" % (_f(t[0]), _f(t[1]), _f(t[2])))
            i = len(vnl); vnmap[k] = i
        return i
    with open(path, "w") as f:
        f.write("# from Blender object '%s'\n" % obj.name)
        if uvl is None:
            vs, fs = [], []
            for v in me.vertices:
                c = mw @ v.co; c = _tp(c.x, c.y, c.z)
                vs.append("v %s %s %s\n" % (_f(c[0]), _f(c[1]), _f(c[2])))
            for tri in me.loop_triangles:
                a, b, cc = (i + 1 for i in tri.vertices)
                if cn is None:
                    fs.append("f %d %d %d\n" % (a, b, cc))
                else:
                    n0, n1, n2 = (_vni(li) for li in tri.loops)
                    fs.append("f %d//%d %d//%d %d//%d\n" % (a, n0, b, n1, cc, n2))
            f.write("".join(vs)); f.write("".join(vnl)); f.write("".join(fs))
            nv, nt = len(me.vertices), len(me.loop_triangles)
        else:
            vs, vts, fs, idx, nt = [], [], [], 1, 0
            for tri in me.loop_triangles:
                fi, ni = [], []
                for li in tri.loops:
                    vi = me.loops[li].vertex_index
                    c = mw @ me.vertices[vi].co; c = _tp(c.x, c.y, c.z); uv = uvl[li].uv
                    vs.append("v %s %s %s\n" % (_f(c[0]), _f(c[1]), _f(c[2])))
                    vts.append("vt %s %s\n" % (_f(uv.x), _f(uv.y)))
                    fi.append(idx); idx += 1
                    if cn is not None: ni.append(_vni(li))
                if cn is None:
                    fs.append("f %d/%d %d/%d %d/%d\n" % (fi[0], fi[0], fi[1], fi[1], fi[2], fi[2]))
                else:
                    fs.append("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (fi[0], fi[0], ni[0], fi[1], fi[1], ni[1], fi[2], fi[2], ni[2]))
                nt += 1
            f.write("".join(vs)); f.write("".join(vts)); f.write("".join(vnl)); f.write("".join(fs)); nv = idx - 1
    ev.to_mesh_clear(); return nv, nt

def disney_xml(mat, texdir):
    n = principled(mat)
    if n is None:
        base = mat.diffuse_color if mat else (0.8, 0.8, 0.8, 1.0)
        return ('\t\t<bsdf type="diffuse">\n\t\t\t<color name="albedo" value="%s"/>\n\t\t</bsdf>' % _col(base))
    bc=inp(n,'Base Color',(0.8,0.8,0.8,1.0)); rough=inp(n,'Roughness',0.5); metal=inp(n,'Metallic',0.0)
    spec=inp(n,'Specular IOR Level',inp(n,'Specular',0.5)); sheen=inp(n,'Sheen Weight',inp(n,'Sheen',0.0))
    coat=inp(n,'Coat Weight',inp(n,'Clearcoat',0.0)); aniso=inp(n,'Anisotropic',0.0)
    sss=inp(n,'Subsurface Weight',inp(n,'Subsurface',0.0))
    L=['\t\t<bsdf type="disney">',
       '\t\t\t<color name="baseColor" value="%s"/>'%_col(bc),
       '\t\t\t<float name="roughness" value="%s"/>'%_f(rough),
       '\t\t\t<float name="metallic" value="%s"/>'%_f(metal),
       '\t\t\t<float name="specular" value="%s"/>'%_f(spec),
       '\t\t\t<float name="sheen" value="%s"/>'%_f(sheen),
       '\t\t\t<float name="clearcoat" value="%s"/>'%_f(coat),
       '\t\t\t<float name="anisotropic" value="%s"/>'%_f(aniso),
       '\t\t\t<float name="subsurface" value="%s"/>'%_f(sss)]
    afn = export_image(base_color_image(mat), texdir)
    if afn:
        L.append('\t\t\t<string name="albedoTexture" value="textures/%s"/>' % afn)
        nfn = export_image(normal_image(mat), texdir)
        if nfn: L.append('\t\t\t<string name="normalTexture" value="textures/%s"/>' % nfn)
    L.append('\t\t</bsdf>'); return "\n".join(L)

def emission_of(mat):
    n = principled(mat)
    if n is not None:
        col = inp(n,'Emission Color',inp(n,'Emission',(0,0,0,1))); strg = inp(n,'Emission Strength',0.0)
        if col is not None and strg is not None:
            r=(col[0]*strg,col[1]*strg,col[2]*strg)
            if max(r)>1e-6: return r
    if mat and mat.use_nodes:
        for e in mat.node_tree.nodes:
            if e.type == 'EMISSION':
                col=inp(e,'Color',(0,0,0,1)); strg=inp(e,'Strength',1.0)
                if col is None or strg is None: continue
                r=(col[0]*strg,col[1]*strg,col[2]*strg)
                if max(r)>1e-6: return r
    return None

def glass_of(mat):
    if not mat or not mat.use_nodes: return None
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_GLASS': return n
    return None

def dielectric_xml(n):
    ior = inp(n, 'IOR', 1.45)
    col = inp(n, 'Color', (1.0, 1.0, 1.0, 1.0))
    return ('\t\t<bsdf type="dielectric">\n'
            '\t\t\t<float name="intIOR" value="%s"/>\n'
            '\t\t\t<color name="tintColor" value="%s"/>\n'
            '\t\t</bsdf>' % (_f(ior), _col(col)))

def horizontal_fov_deg(cam, W, H):
    a=cam.data.angle; fit=cam.data.sensor_fit
    if fit=='VERTICAL' or (fit=='AUTO' and H>W):
        a=2.0*math.atan(math.tan(a/2.0)*(float(W)/float(H)))
    return math.degrees(a)

def camera_xml(scene, W, H):
    cam=scene.camera; mw=cam.matrix_world; q=mw.to_quaternion(); o=mw.translation
    fwd=q@mathutils.Vector((0,0,-1)); up=q@mathutils.Vector((0,1,0)); t=o+fwd
    O=_tp(o.x,o.y,o.z); T=_tp(t.x,t.y,t.z); U=_tp(up.x,up.y,up.z)
    fov=horizontal_fov_deg(cam,W,H)
    return ('\t<camera type="perspective">\n\t\t<float name="fov" value="%s"/>\n'
            '\t\t<transform name="toWorld">\n'
            '\t\t\t<lookat target="%s, %s, %s" origin="%s, %s, %s" up="%s, %s, %s"/>\n'
            '\t\t</transform>\n\t\t<integer name="width" value="%d"/>\n'
            '\t\t<integer name="height" value="%d"/>\n\t</camera>' % (_f(fov),
              _f(T[0]),_f(T[1]),_f(T[2]),_f(O[0]),_f(O[1]),_f(O[2]),_f(U[0]),_f(U[1]),_f(U[2]),W,H))

def export_scene(repo, out_name, samples=64, W=800, H=800,
                 env="white.hdr", envscale=0.4, arealight=True, yup=False, skip=None,
                 medium_xml="", normals=True):
    global _YUP; _YUP = yup
    skip = set(skip or [])
    scene=bpy.context.scene; dg=bpy.context.evaluated_depsgraph_get()
    out_dir=os.path.join(repo,"scenes",out_name); mesh_dir=os.path.join(out_dir,"meshes"); tex_dir=os.path.join(out_dir,"textures")
    os.makedirs(mesh_dir, exist_ok=True)
    if env=="white.hdr":
        try: shutil.copyfile(os.path.join(repo,"scenes","final_scene_real","white.hdr"), os.path.join(out_dir,"white.hdr"))
        except Exception as e: print("hdr copy warn:", e)
    mesh_objs=[o for o in scene.objects if o.type=='MESH' and o.visible_get() and o.name not in skip]
    stats,blocks,allco=[],[],[]
    for o in mesh_objs:
        mat=o.active_material; with_uv=base_color_image(mat) is not None
        fn=o.name.replace(" ","_")+".obj"
        nv,nt=write_obj(o,os.path.join(mesh_dir,fn),dg,with_uv,normals); stats.append((o.name,nv,nt))
        for corner in o.bound_box:
            c=o.matrix_world@mathutils.Vector(corner); allco.append(_tp(c.x,c.y,c.z))
        emis=emission_of(mat)
        _gl=glass_of(mat)
        if emis:   body=('\t\t<emitter type="area">\n\t\t\t<color name="radiance" value="%s"/>\n\t\t</emitter>'%_col(emis))
        elif _gl:  body=dielectric_xml(_gl)
        else:      body=disney_xml(mat,tex_dir)
        blocks.append('\t<mesh type="obj">\n\t\t<string name="filename" value="meshes/%s"/>\n%s\n\t</mesh>'%(fn,body))
    if arealight and allco:
        xs=[c[0] for c in allco]; ys=[c[1] for c in allco]; zs=[c[2] for c in allco]
        cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2; span=max(max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs),1.0)
        h=span*0.6; zL=max(zs)+span*0.8
        with open(os.path.join(mesh_dir,"_arealight.obj"),"w") as f:
            f.write("v %s %s %s\n"%(_f(cx-h),_f(cy-h),_f(zL))); f.write("v %s %s %s\n"%(_f(cx+h),_f(cy-h),_f(zL)))
            f.write("v %s %s %s\n"%(_f(cx+h),_f(cy+h),_f(zL))); f.write("v %s %s %s\n"%(_f(cx-h),_f(cy+h),_f(zL)))
            f.write("f 1 3 2\nf 1 4 3\n")
        blocks.append('\t<mesh type="obj">\n\t\t<string name="filename" value="meshes/_arealight.obj"/>\n\t\t<emitter type="area">\n\t\t\t<color name="radiance" value="5 5 5"/>\n\t\t</emitter>\n\t</mesh>')
        stats.append(("_arealight",4,2))
    cam_xml=camera_xml(scene,W,H)
    sampler='\t<sampler type="independent">\n\t\t<integer name="sampleCount" value="%d"/>\n\t</sampler>'%samples
    header=("<?xml version='1.0' encoding='utf-8'?>\n\n<scene>\n"
            '\t<string name="envmap" value="%s"/>\n\t<float name="envmapScale" value="%s"/>\n'
            '\t<float name="evCompensation" value="0.0"/>\n\n'%(env,_f(envscale)))
    if medium_xml:
        header = header + medium_xml.rstrip("\n") + "\n\n"

    xml=header+cam_xml+"\n\n"+sampler+"\n\n"+"\n\n".join(blocks)+"\n</scene>\n"
    with open(os.path.join(out_dir,"scene.xml"),"w") as f: f.write(xml)
    return {"out_dir":out_dir,"objects":stats,"yup":yup,"scene_xml_bytes":len(xml)}

_out=globals().get("NORI_OUT","smoke_test"); _smp=globals().get("NORI_SAMPLES",64)
_repo=globals().get("NORI_REPO",r"C:\Users\gjin3\Desktop\nori-26sp")
_env=globals().get("NORI_ENV","white.hdr"); _escl=globals().get("NORI_ENVSCALE",0.4)
_area=globals().get("NORI_AREALIGHT",True); _W=globals().get("NORI_W",800); _H=globals().get("NORI_H",800)
_yup=globals().get("NORI_YUP",False); _skip=globals().get("NORI_SKIP",None)
_med=globals().get("NORI_MEDIUM_XML","")
_nrm=globals().get("NORI_NORMALS",True)
print("EXPORT_RESULT:", export_scene(_repo,_out,_smp,_W,_H,_env,_escl,_area,_yup,_skip,_med,_nrm))
