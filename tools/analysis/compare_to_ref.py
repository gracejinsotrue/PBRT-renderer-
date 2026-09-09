# Compare a floating_causeway render against the reference, in the bands that
# matter, through the renderer's exact display transform.
#
# Resolve.hlsl is Narkowicz ACES then gamma 2.2 on the RAW linear radiance in the
# EXR, which has no exposure baked in. So multiply by 2^ev, never by a difference
# of ev values. Solving exposure here against the EXR takes a second instead of a
# sixteen minute re-render per guess.
#
#   python tools/analysis/compare_to_ref.py <render.exr> <reference.png> [ev]
import numpy as np, cv2, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import exrtool   # cv2 here has no EXR support and imageio misreads these files

EXR, REF = sys.argv[1], sys.argv[2]
EV_FIXED = float(sys.argv[3]) if len(sys.argv) > 3 else None

hdr = exrtool.read(EXR)[0].astype(np.float32)
ref = cv2.imread(REF, cv2.IMREAD_COLOR).astype(np.float32)[..., ::-1] / 255.0
H, W, _ = hdr.shape
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def aces(c):
    return np.clip((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0, 1)


def disp(c, ev):
    return aces(c * (2.0 ** ev)) ** (1 / 2.2)


def patch(img, y0, y1, x0, x1):
    h, w, _ = img.shape
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


# Anchors chosen to be unclipped in both: sky away from the sun, and near water.
ANCH = [(0.02, 0.30, 0.05, 0.60), (0.86, 0.99, 0.06, 0.42)]
if EV_FIXED is None:
    best, ev = 1e18, 0.0
    for e in np.arange(-8, 8, 0.05):
        err = 0.0
        for a in ANCH:
            err += float(((disp(patch(hdr, *a), e).reshape(-1, 3).mean(0)
                           - patch(ref, *a).reshape(-1, 3).mean(0)) ** 2).sum())
        if err < best:
            best, ev = err, float(e)
else:
    ev = EV_FIXED
out = disp(hdr, ev)
print("evCompensation %.2f%s" % (ev, "" if EV_FIXED is None else " (fixed)"))
print()

REG = [("sky upper", 0.02, 0.30, 0.05, 0.60),
       ("ridge", 0.585, 0.625, 0.05, 0.55),
       ("causeway", 0.650, 0.670, 0.10, 0.55),
       ("water bright band", 0.690, 0.730, 0.06, 0.62),
       ("water mid", 0.75, 0.82, 0.06, 0.42),
       ("water near", 0.86, 0.99, 0.06, 0.42)]
print("%-19s %-22s %-22s" % ("region", "render sRGB", "reference sRGB"))
for nm, y0, y1, x0, x1 in REG:
    a = disp(patch(hdr, y0, y1, x0, x1), ev).reshape(-1, 3).mean(0)
    b = patch(ref, y0, y1, x0, x1).reshape(-1, 3).mean(0)
    d = float(np.abs(a - b).mean())
    print("%-19s %6.3f %6.3f %6.3f   %6.3f %6.3f %6.3f   dE %.3f%s"
          % (nm, *a, *b, d, "   <<<" if d > 0.09 else ""))

print()
print("water row profile: p50 / p99 / frac>0.75, x in [0.06,0.62]")
print("%-8s %-24s %-24s" % ("y", "render", "reference"))
for yf in [0.645, 0.670, 0.695, 0.720, 0.745, 0.795, 0.845, 0.895, 0.945, 0.995]:
    r = (patch(out, yf, min(yf + 0.02, 1.0), 0.06, 0.62) @ LUM)
    q = (patch(ref, yf, min(yf + 0.02, 1.0), 0.06, 0.62) @ LUM)
    print("%-8.3f %6.3f %6.3f %7.3f    %6.3f %6.3f %7.3f"
          % (yf, np.percentile(r, 50), np.percentile(r, 99), (r > 0.75).mean(),
             np.percentile(q, 50), np.percentile(q, 99), (q > 0.75).mean()))

g, gr = out @ LUM, ref @ LUM
print()
print("whole frame  render: p10 %.3f p50 %.3f p90 %.3f  frac>0.6 %.3f"
      % (np.percentile(g, 10), np.percentile(g, 50), np.percentile(g, 90), (g > 0.6).mean()))
print("          reference: p10 %.3f p50 %.3f p90 %.3f  frac>0.6 %.3f"
      % (np.percentile(gr, 10), np.percentile(gr, 50), np.percentile(gr, 90), (gr > 0.6).mean()))
