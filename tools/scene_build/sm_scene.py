# Regenerate a San Miguel Nori scene.xml. Camera/exposure/output configurable
# via env vars: SM_OUT, SM_EYE="x,y,z", SM_TGT="x,y,z", SM_EV, SM_ENVSCALE, SM_FOV.
import os, re, math, glob
BASE=os.path.join(os.path.expanduser("~"),"mnt","nori-26sp","scenes","san_miguel")
MTL=os.path.join(os.path.expanduser("~"),"mnt","San_Miguel","san-miguel.mtl")
ENVMAP=os.environ.get("SM_ENV","sky.hdr")
ENVSCALE=float(os.environ.get("SM_ENVSCALE","1.0"))
EV=float(os.environ.get("SM_EV","0.0"))
FOV=float(os.environ.get("SM_FOV","55"))
RW=int(os.environ.get("SM_W","1280")); RH=int(os.environ.get("SM_H","1280"))
ROT=float(os.environ.get("SM_ROT","0"))
OUT=os.environ.get("SM_OUT","scene.xml")
def vec(s,dflt): 
    try: return [float(x) for x in s.split(",")]
    except: return dflt
EYE=vec(os.environ.get("SM_EYE","11.5,1.5,-1.5"),[11.5,1.5,-1.5])
TGT=vec(os.environ.get("SM_TGT","20.7,1.4,7.7"),[20.7,1.4,7.7])
def san(n): return re.sub(r'[^A-Za-z0-9_.-]','_',n)
def png_has_alpha(p):
    try:
        h=open(p,'rb').read(29); return len(h)>=26 and h[:8]==b'\x89PNG\r\n\x1a\n' and h[25] in (4,6)
    except: return False
def parse_mtl(path):
    mats={};order=[];cur=None
    for line in open(path,'r',errors='replace'):
        t=line.split()
        if not t: continue
        k=t[0].lower()
        if k=='newmtl': cur=line.split(None,1)[1].strip(); mats[cur]={}; order.append(cur)
        elif cur is None: continue
        elif k=='kd' and len(t)>=4: mats[cur]['Kd']=(t[1],t[2],t[3])
        elif k=='ns' and len(t)>=2:
            try: mats[cur]['Ns']=float(t[1])
            except: pass
        elif k=='map_kd': mats[cur]['map_Kd']=t[-1]
        elif k in ('map_bump','bump'): mats[cur]['map_bump']=t[-1]
        elif k=='d' and len(t)>=2:
            try: mats[cur]['d']=float(t[1])
            except: pass
    return mats,order
METAL_KW=['forja','metal','hierro','fierro','iron','reja','farol','lampar','lantern','laton','bronce','cobre','herreria']
FABRIC_KW=['tela','cojin','cloth','manta','tapiz','mantel']
def is_fabric(name,info):
    ss=(name+' '+info.get('map_Kd','')).lower()
    return any(k in ss for k in FABRIC_KW)

def is_metal(name,info):
    s=(name+' '+info.get('map_Kd','')).lower()
    return any(k in s for k in METAL_KW)

def bsdf(name,info,aset):
    if info.get('d',1.0) < 0.999: return '\t\t<bsdf type="dielectric"/>'
    kd=info.get('Kd',('0.8','0.8','0.8')); ns=info.get('Ns',50.0)
    rough=max(0.02,min(1.0,math.sqrt(2.0/(ns+2.0))))
    met = is_metal(name,info)
    if met: rough=0.35
    fab=is_fabric(name,info)
    if fab: rough=0.9
    L=['\t\t<bsdf type="disney">','\t\t\t<color name="baseColor" value="%s %s %s"/>'%(kd[0],kd[1],kd[2]),
       '\t\t\t<float name="roughness" value="%.4f"/>'%rough,'\t\t\t<float name="metallic" value="%.1f"/>'%(1.0 if met else 0.0)]
    if fab: L.append('\t\t\t<float name="sheen" value="0.6"/>')
    tex=info.get('map_Kd')
    if tex:
        bn=os.path.basename(tex.replace('\\','/')); L.append('\t\t\t<string name="albedoTexture" value="textures/%s"/>'%bn)
        if bn in aset: L.append('\t\t\t<string name="alphaTexture" value="textures/%s"/>'%bn)
    bump=info.get('map_bump')
    if bump: L.append('\t\t\t<string name="normalTexture" value="textures/%s"/>'%os.path.basename(bump.replace('\\','/')))
    L.append('\t\t</bsdf>'); return '\n'.join(L)
mtl,order=parse_mtl(MTL)
aset={os.path.basename(p) for p in glob.glob(BASE+"/textures/*.png") if png_has_alpha(p)}
stems={fn[:-4] for fn in os.listdir(BASE+"/meshes") if fn.endswith('.obj')}
blocks=[];used=set()
for name in order:
    st=san(name)
    if st in stems:
        used.add(st); blocks.append('\t<mesh type="obj">\n\t\t<string name="filename" value="meshes/%s.obj"/>\n%s\n\t</mesh>'%(st,bsdf(name,mtl[name],aset)))
for st in sorted(stems-used):
    blocks.append('\t<mesh type="obj">\n\t\t<string name="filename" value="meshes/%s.obj"/>\n%s\n\t</mesh>'%(st,bsdf(st,{},aset)))
cam=('\t<camera type="perspective">\n\t\t<float name="fov" value="%.1f"/>\n\t\t<transform name="toWorld">\n'
     '\t\t\t<lookat target="%.3f, %.3f, %.3f" origin="%.3f, %.3f, %.3f" up="0, 1, 0"/>\n'
     '\t\t</transform>\n\t\t<integer name="width" value="%d"/>\n\t\t<integer name="height" value="%d"/>\n\t</camera>'
     %(FOV,TGT[0],TGT[1],TGT[2],EYE[0],EYE[1],EYE[2],RW,RH))
header=("<?xml version='1.0' encoding='utf-8'?>\n\n<scene>\n"
        '\t<string name="envmap" value="%s"/>\n\t<float name="envmapScale" value="%.3f"/>\n'
        '\t<float name="evCompensation" value="%.3f"/>\n\t<float name="envmapRotation" value="%.4f"/>\n\n'%(ENVMAP,ENVSCALE,EV,ROT))
sampler='\t<sampler type="independent">\n\t\t<integer name="sampleCount" value="64"/>\n\t</sampler>'
xml=header+cam+"\n\n"+sampler+"\n\n"+"\n\n".join(blocks)+"\n</scene>\n"
open(os.path.join(BASE,OUT),'w').write(xml)
print("%s: eye=(%.1f,%.1f,%.1f) look=(%.1f,%.1f,%.1f) EV=%.1f  %d meshes %d bytes"%(OUT,EYE[0],EYE[1],EYE[2],TGT[0],TGT[1],TGT[2],EV,len(blocks),len(xml)))
