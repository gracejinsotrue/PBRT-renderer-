# Render just the sky of an envmap through the floating_causeway camera, so a sky
# can be judged before anything is exported or path traced.
#
# Nothing here is a substitute for a render: it shows the envmap only, no water, no
# geometry. But the sky is a third of this frame and it appears again mirrored in
# the water, so getting it right first is worth a 2 second preview instead of a
# 16 minute render.
#
# Tonemap matches shaders/Resolve.hlsl exactly: Narkowicz ACES then gamma 2.2, on
# raw linear radiance. Exposure is solved here rather than guessed, against the
# sky patch measured off the reference.
#
#   python tools/analysis/sky_preview.py <sky.hdr> <out.png> [bearing_deg]
import numpy as np, cv2, sys, math

SKY = sys.argv[1]
OUT = sys.argv[2]
BEARING = float(sys.argv[3]) if len(sys.argv) > 3 else 197.6   # puts the sun 17.21 deg right of centre

W, H = 1920, 818
C = 1662.7
HOR = 517.9
REF_SRGB = np.array([0.381, 0.548, 0.682], np.float32)   # reference sky, x .05-.60 y .02-.30
LUM = np.array([0.0722, 0.7152, 0.2126], np.float32)     # BGR

env = cv2.imread(SKY, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
EH, EW, _ = env.shape

px, py = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
sx = (px - W / 2.0) / C
sy = (HOR - py) / C
B = math.radians(BEARING)
dx = math.cos(B) + sx * math.sin(B)
dy = math.sin(B) - sx * math.cos(B)
dz = sy
n = np.sqrt(dx * dx + dy * dy + dz * dz)
elev = np.degrees(np.arcsin(dz / n))
azim = np.degrees(np.arctan2(dy, dx)) % 360.0

u = (azim / 360.0 * EW - 0.5).astype(np.float32)
v = ((90.0 - elev) / 180.0 * EH - 0.5).astype(np.float32)
img = cv2.remap(env, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def aces(c):
    return np.clip((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0, 1)


def display(c, ev):
    return aces(c * (2.0 ** ev)) ** (1 / 2.2)


sky = img[int(0.02 * H):int(0.30 * H), int(0.05 * W):int(0.60 * W)]
best, bev = 1e9, 0.0
for ev in np.arange(-6, 8, 0.05):
    d = display(sky, ev).reshape(-1, 3).mean(0)
    e = float(((d[::-1] - REF_SRGB) ** 2).sum())
    if e < best:
        best, bev = e, float(ev)

out = display(img, bev)
cv2.imwrite(OUT, (out * 255).astype(np.uint8))
got = display(sky, bev).reshape(-1, 3).mean(0)[::-1]
print("wrote %s   solved evCompensation %.2f" % (OUT, bev))
print("sky patch sRGB  got %.3f/%.3f/%.3f   reference %.3f/%.3f/%.3f" % (got[0], got[1], got[2], *REF_SRGB))
print("suggested TINT multiplier to close the gap: %.3f %.3f %.3f"
      % tuple((REF_SRGB / np.maximum(got, 1e-6)) ** 2.2))
g = out @ LUM
print("frame stats: p10 %.3f p50 %.3f p90 %.3f   fraction >0.6: %.3f"
      % (np.percentile(g, 10), np.percentile(g, 50), np.percentile(g, 90), float((g > 0.6).mean())))
