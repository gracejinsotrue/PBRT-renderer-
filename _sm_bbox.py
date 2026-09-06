import os, json, numpy as np
BASE="scenes/san_miguel/meshes"
out={}
for fn in sorted(os.listdir(BASE)):
    if not fn.endswith(".obj"): continue
    mn=[1e9]*3; mx=[-1e9]*3; n=0
    with open(os.path.join(BASE,fn),'r',errors='ignore') as f:
        for line in f:
            if line[0]=='v' and line[1]==' ':
                a=line.split()
                x,y,z=float(a[1]),float(a[2]),float(a[3])
                if x<mn[0]:mn[0]=x
                if y<mn[1]:mn[1]=y
                if z<mn[2]:mn[2]=z
                if x>mx[0]:mx[0]=x
                if y>mx[1]:mx[1]=y
                if z>mx[2]:mx[2]=z
                n+=1
    out[fn]={"n":n,"min":mn,"max":mx}
json.dump(out,open("scenes/san_miguel/_scratch/bbox.json","w"))
print("done",len(out))
