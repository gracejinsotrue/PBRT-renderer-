# Rebuild scenes/liminal_bed/sky.hdr from the original hard-sun version.
#   SUN_SCALE   dims the sun disk (contrast between lit and shaded fabric)
#   BACK_BOOST  brightens ONLY the sky hemisphere behind the camera, which fills
#               camera-facing shaded surfaces and is never seen in frame
#   ZEN_BOOST   brightens the zenith cap, also outside the camera's view
import numpy as np, math, cv2, sys
SRC = sys.argv[1]; DST = sys.argv[2]
SUN_SCALE = float(sys.argv[3]); ZEN_FRAC = float(sys.argv[4])
BACK_RGB  = np.array([float(x) for x in sys.argv[5].split(',')], np.float32)  # R,G,B fill gains, unseen hemisphere
VIS_RGB   = np.array([float(x) for x in sys.argv[6].split(',')], np.float32)  # R,G,B gains on the wedge the camera sees
SUN_RGB   = np.array([float(x) for x in sys.argv[7].split(',')], np.float32) if len(sys.argv) > 7 else np.ones(3, np.float32)  # sun tint
SUN_ELEV  = float(sys.argv[8]) if len(sys.argv) > 8 else None   # move the sun to this elevation (deg), same azimuth
SUN_SIG   = float(sys.argv[9]) if len(sys.argv) > 9 else 3.0    # sun disk sigma in texels (bigger = softer shadow)
SUN_AZ    = float(sys.argv[10]) if len(sys.argv) > 10 else None # Blender azimuth (deg, atan2(y,x)); Nori u = (-az/360) mod 1
HAZE      = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0  # neutral horizon haze (linear radiance added at the horizon, ramps from 25 deg up)
U_CAM = 0.883          # equirect u the camera looks toward (Nori atan2(d.z,d.x) convention)
VIS_HALF = 0.105       # half-width in u of the wedge the camera actually sees
img = cv2.imread(SRC, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_ANYCOLOR).astype(np.float32)
H,W,_ = img.shape
lum = img.sum(axis=2); sv,su = np.unravel_index(np.argmax(lum), lum.shape)
yy,xx = np.mgrid[0:H,0:W].astype(np.float32)
dx = np.minimum(np.abs(xx-su), W-np.abs(xx-su)); dy = yy-sv
G = np.exp(-(dx*dx+dy*dy)/(2*3.0*3.0)).astype(np.float32)
r = np.sqrt(dx*dx+dy*dy); ann=(r>12)&(r<20)
dome = np.array([np.median(img[...,c][ann]) for c in range(3)], np.float32)
amp  = img[sv,su].astype(np.float32) - dome
sun  = G[...,None]*amp[None,None,:]
sky  = img - sun

u = (xx+0.5)/W; v = (yy+0.5)/H
du = np.abs(u - U_CAM); du = np.minimum(du, 1.0-du)          # wrapped distance from view centre
w  = np.clip((du - VIS_HALF) / 0.060, 0.0, 1.0)              # 0 in view, 1 just outside it
w  = w*w*(3.0-2.0*w)
zen = np.clip((0.20 - v)/0.20, 0.0, 1.0); zen = zen*zen*(3.0-2.0*zen)
# per-channel: a deep blue dome makes an unnaturally cyan fill, so the unseen
# hemisphere is boosted warmer than it looks, which neutralises the shadows.
zen_rgb = 1.0 + (BACK_RGB - 1.0) * ZEN_FRAC
gain = (VIS_RGB[None,None,::-1] * (1.0 - w)[...,None]
        + BACK_RGB[None,None,::-1] * w[...,None])
gain = gain * (1.0 + (zen_rgb[None,None,::-1] - 1.0) * (zen * (1.0 - w))[...,None])
if SUN_ELEV is not None:
    sv2 = (90.0 - SUN_ELEV) / 180.0 * H
    dy2 = yy - sv2
    su2 = su if SUN_AZ is None else ((-SUN_AZ / 360.0) % 1.0) * W
    dx2 = np.minimum(np.abs(xx-su2), W-np.abs(xx-su2))
    G2 = np.exp(-(dx2*dx2+dy2*dy2)/(2*SUN_SIG*SUN_SIG)).astype(np.float32)
    G2 *= G.sum() / max(G2.sum(), 1e-6)          # same total sun energy whatever the disk size
    sun = G2[...,None]*amp[None,None,:]
    print('sun moved to elevation %.1f deg (v=%.3f) u=%.4f, sigma %.1f' % (SUN_ELEV, sv2/H, su2/W, SUN_SIG))
out = sky*gain
if HAZE > 0:
    elev = 90.0 - v*180.0
    ramp = np.clip((22.0 - elev)/22.0, 0.0, 1.0)**1.1
    out = out + (HAZE*ramp)[...,None]*np.array([0.95,1.05,1.0], np.float32)[None,None,:]   # BGR order: slightly more G, a bit less B
# the terrain crest sits below eye level, so the camera sees the HDR just below its horizon:
# continue the sky there instead of the dark ground hemisphere of the source panorama
hrow = out[int(0.5*H)-1]
out[int(0.5*H):] = hrow[None,:,:]
out = out + sun*SUN_SCALE*SUN_RGB[None,None,::-1]
cv2.imwrite(DST, np.clip(out,0,None).astype(np.float32))
print("sun x%.2f | visible wedge gain %s | fill gain %s | zenith frac %.2f"
      % (SUN_SCALE, gain[H//3, int(U_CAM*W)][::-1].round(2), gain[H//3, int(((U_CAM+0.5)%1.0)*W)][::-1].round(2), ZEN_FRAC))
