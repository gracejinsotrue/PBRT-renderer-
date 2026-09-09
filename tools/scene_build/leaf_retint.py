# Brighten and warm the oak leaf albedo for the floating_causeway canopy.
#
# The reference canopy sits at L p50 0.232 with only 8% of pixels dark; two instances
# of the stock oak render 0.088 with 72% dark. Depth is not the cause (thinning to one
# instance overshoots to 0.423 and reads sky-blue) and neither is translucency
# (0.35 -> 0.70 measured DARKER, 0.088 -> 0.069, because the reflection lobe scales by
# 1-T). That leaves the albedo, which the crosswalk retint left at a deep summer green
# of (0.42,0.49,0.19) against this reference's lighter olive (0.312,0.323,0.215 in
# display, so R about equal to G rather than well below it).
#
# Scales in LINEAR and leaves alpha untouched, since the same PNG is the cutout mask.
#
#   python tools/scene_build/leaf_retint.py <src.png> <dst.png> [gain] [r_mul] [b_mul]
import numpy as np, cv2, sys

SRC, DST = sys.argv[1], sys.argv[2]
GAIN = float(sys.argv[3]) if len(sys.argv) > 3 else 1.80
RMUL = float(sys.argv[4]) if len(sys.argv) > 4 else 1.13
BMUL = float(sys.argv[5]) if len(sys.argv) > 5 else 1.55
# The lilac library's leaf atlas includes purple flower clusters, which have no place
# in this scene. Any pixel whose blue beats its green is a petal, so swap it for the
# atlas's own mean leaf chroma at the same luminance.
DEPURPLE = len(sys.argv) > 6 and sys.argv[6] == "depurple"

im = cv2.imread(SRC, cv2.IMREAD_UNCHANGED)
if im is None:
    raise SystemExit("cannot read " + SRC)
if im.shape[2] < 4:
    raise SystemExit("expected RGBA, the alpha channel is the leaf cutout")
if im.dtype == np.uint16:
    rgb = im[..., :3].astype(np.float32) / 65535.0
    peak = 65535.0
else:
    rgb = im[..., :3].astype(np.float32) / 255.0
    peak = 255.0
a = im[..., 3]

lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

if DEPURPLE:
    LU = np.array([0.0722, 0.7152, 0.2126], np.float32)   # BGR
    op0 = a > 200
    petal = (lin[..., 0] > lin[..., 1] * 1.02) & op0
    leafy = op0 & ~petal
    if leafy.any() and petal.any():
        chroma = lin[leafy].reshape(-1, 3).mean(0)
        chroma = chroma / max(float(np.dot(chroma, LU)), 1e-6)
        lum = (lin @ LU)[..., None]
        lin = np.where(petal[..., None], lum * chroma, lin)
    print("depurpled %.1f%% of opaque pixels" % (100.0 * petal.sum() / max(op0.sum(), 1)))
lin *= np.array([BMUL * GAIN, GAIN, RMUL * GAIN], np.float32)   # cv2 is BGR
lin = np.clip(lin, 0, 1)
out = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)

op = a > 200
before = rgb[op].reshape(-1, 3).mean(0)[::-1]
after = out[op].reshape(-1, 3).mean(0)[::-1]
cv2.imwrite(DST, np.dstack([(out * peak).astype(im.dtype), a]))
print("wrote %s" % DST)
print("opaque-pixel sRGB mean  %.3f %.3f %.3f  ->  %.3f %.3f %.3f" % (*before, *after))
print("reference canopy target %.3f %.3f %.3f (that is the rendered canopy, not the leaf itself)"
      % (0.312, 0.323, 0.215))
