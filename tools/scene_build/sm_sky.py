# Rebuild scenes/san_miguel/sky_custom.hdr: a blue sky dome + a painted warm sun.
# Edit the params, then run:  python tools/scene_build/sm_sky.py   (needs: pip install opencv-python numpy)
import cv2, numpy as np, os, math
BASE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scenes", "san_miguel")

# ---- SUN / SKY PARAMETERS ----
ELEV     = 66.0          # sun height, degrees. HIGHER -> shadows fall on the GROUND; lower -> on the walls
AZIM_U   = 0.20          # sun azimuth 0..1 in the image (you can also spin it live via envmapRotation in scene.xml)
SIGMA    = 1.20          # sun size in pixels. SMALLER -> sharper shadow edges
SUN_RGB  = (641000.0, 449000.0, 288000.0)  # sun colour+intensity (R,G,B). Bigger -> brighter/more contrast
SKY_MULT = 1.7           # sky/ambient brightness (fills the shadows so they aren't black)
SKY_BLUE = 1.50          # extra blue tint on the sky dome (bluer shadows)
OUT_NAME = "sky_custom.hdr"  # set SUN_RGB=(0,0,0) + a new name to build a sun-less dome
                             # for use with the emissive sun disk (tools/scene_build/sm_sun_geo.py)
# ------------------------------

img = cv2.imread(os.path.join(BASE,"sky.hdr"), cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR).astype(np.float32)
H,W,_ = img.shape
b,g,r = img[...,0],img[...,1],img[...,2]
L = 0.2126*r+0.7152*g+0.0722*b
img *= np.minimum(1.0, np.percentile(L,99.0)/np.maximum(L,1e-6))[...,None]  # remove baked sun
img *= SKY_MULT
img[...,0] *= SKY_BLUE
theta = math.radians(90-ELEV); v = theta/math.pi
su = int(AZIM_U*W); sv = int(v*H)
yy,xx = np.mgrid[0:H,0:W].astype(np.float32)
dx = np.minimum(np.abs(xx-su), W-np.abs(xx-su)); dy = yy-sv
gauss = np.exp(-(dx*dx+dy*dy)/(2*SIGMA*SIGMA))
sun_bgr = (SUN_RGB[2], SUN_RGB[1], SUN_RGB[0])
for c in range(3): img[...,c] += gauss*sun_bgr[c]
cv2.imwrite(os.path.join(BASE,OUT_NAME), np.clip(img,0,None))
print("%s rebuilt: ELEV=%g AZIM_U=%g SIGMA=%g SUN=%s" % (OUT_NAME,ELEV,AZIM_U,SIGMA,SUN_RGB))
