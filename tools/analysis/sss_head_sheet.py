# Builds the 2x2 contact sheet for the pbrt head SSS-vs-Disney comparison.
#
# Rows are the two lighting setups, columns the two skin models, so reading
# across a row isolates the material and reading down a column isolates the
# light. Run after tools/exporters/pbrt_head_to_nori.py has rendered scenes/head.
#
# usage: python tools/analysis/sss_head_sheet.py [--raw]     (--raw uses the un-denoised PNGs)
import os, sys, cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root: tools/<group>/ -> ..
SRC = os.path.join(REPO, "scenes", "head")
OUT = os.path.join(REPO, "images", "sss_head_pbrt_grid.png")
SUFFIX = "" if '--raw' in sys.argv else "_denoised"

CELLS = [("head_sky_disney", "Disney BRDF"), ("head_sky_sss", "random-walk BSSRDF"),
         ("head_dark_disney", "Disney BRDF"), ("head_dark_sss", "random-walk BSSRDF")]
ROWS = ["sky HDRI", "dark room, one area light"]

PAD, LABEL, TITLE = 12, 34, 30
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load(name):
    p = os.path.join(SRC, "out_%s%s.png" % (name, SUFFIX))
    if not os.path.exists(p):
        p = os.path.join(SRC, "out_%s.png" % name)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit("missing render: %s" % p)
    return im


ims = [load(n) for n, _ in CELLS]
h, w = ims[0].shape[:2]
scale = 620.0 / w
w, h = int(w * scale), int(h * scale)
ims = [cv2.resize(i, (w, h), interpolation=cv2.INTER_AREA) for i in ims]

W = PAD + 2 * (w + PAD)
H = TITLE + 2 * (LABEL + h + PAD) + PAD
sheet = np.full((H, W, 3), 22, np.uint8)

cv2.putText(sheet, "pbrt-v4 head, same geometry / albedo map / camera / 1024 spp",
            (PAD, 20), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

for k, (im, (_, col)) in enumerate(zip(ims, CELLS)):
    r, c = divmod(k, 2)
    y = TITLE + r * (LABEL + h + PAD)
    x = PAD + c * (w + PAD)
    cv2.putText(sheet, "%s  -  %s" % (ROWS[r], col), (x, y + 22),
                FONT, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    sheet[y + LABEL:y + LABEL + h, x:x + w] = im

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, sheet)
print("wrote %s  (%dx%d)" % (OUT, W, H))
