# Placeholder canopy cutout mask for the crosswalk scene.
#
# The renderer's alpha test is stochastic and commits with probability a*a
# (RayQueryTrace.hlsli), so the mask is kept near-binary: a soft ramp would read
# ~30% more transparent than it looks. Alpha lives in the PNG's 4th channel,
# which is what AlphaTexture samples.
#
#   python tools/scene_build/canopy_mask.py <out.png>
import numpy as np, cv2, sys, os

OUT = sys.argv[1] if len(sys.argv) > 1 else "canopy_mask.png"
RES = 4096
SPAN_X, SPAN_Y = 48.0, 39.0          # metres covered by the plane, must match crosswalk_build.py
X0, Y0 = -30.0, -7.0
# A tree-lined street, not one isolated tree: the near plane tree plus two more down
# the road, so the canopy reads continuous where it fills the top of frame.
CROWNS = [(-6.5, 2.5, 16.0), (-11.0, 24.0, 15.0), (6.0, 31.0, 14.0)]   # x, y, radius
CORE_COV = 0.86                      # opaque fraction under the dense part of a crown

rng = np.random.default_rng(7)

def vnoise(cells_y, cells_x):
    g = rng.random((cells_y + 1, cells_x + 1)).astype(np.float32)
    return cv2.resize(g, (RES, RES), interpolation=cv2.INTER_CUBIC)

def fbm(feat_m, octaves=3):
    out = np.zeros((RES, RES), np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        f = feat_m / (2 ** o)
        out += amp * vnoise(max(2, int(SPAN_Y / f)), max(2, int(SPAN_X / f)))
        tot += amp; amp *= 0.5
    return out / tot

# leaf density: branch masses at ~2.5 m, clumps at ~0.8 m, leaf detail at ~0.22 m.
# The fine octave is what actually sets dapple blob size on the road.
dens = 0.30 * fbm(2.5) + 0.34 * fbm(0.8) + 0.36 * fbm(0.22)

# rank-transform to a uniform distribution so a coverage target is exact
flat = dens.ravel()
order = np.argsort(flat)
u = np.empty_like(flat)
u[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
dens_u = u.reshape(RES, RES)

# crown footprints, with an irregular edge so no boundary reads as a circle
yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32)
wx = X0 + (xx + 0.5) / RES * SPAN_X
wy = Y0 + (yy + 0.5) / RES * SPAN_Y
wobble = 1.0 + 0.26 * (fbm(6.0) - 0.5) * 2.0
foot = np.zeros((RES, RES), np.float32)
for cx, cy, rad in CROWNS:
    r = np.sqrt((wx - cx) ** 2 + (wy - cy) ** 2) / rad * wobble
    f = np.clip((1.12 - r) / 0.42, 0.0, 1.0)
    foot = np.maximum(foot, f * f * (3.0 - 2.0 * f))

cov = CORE_COV * foot
edge = 0.020
alpha = np.clip((dens_u - (1.0 - cov - edge)) / edge, 0.0, 1.0).astype(np.float32)
alpha[cov <= 0.002] = 0.0

leaf = np.zeros((RES, RES, 3), np.float32)
tint = 0.85 + 0.30 * fbm(0.5)
leaf[..., 2] = 0.085 * tint   # R
leaf[..., 1] = 0.150 * tint   # G
leaf[..., 0] = 0.042 * tint   # B

rgba = np.dstack([np.clip(leaf, 0, 1) * 255.0, alpha * 255.0]).astype(np.uint8)
# The quad's uv.y runs 0 at Y0 to 1 at Y0+SPAN_Y, and DXRApp_Scene.cpp writes
# v_gpu = 1 - v_obj, so uv.y = 1 samples image ROW 0. Row 0 must therefore hold the
# LARGEST world y, the opposite of the mgrid order used above.
rgba = rgba[::-1]
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
cv2.imwrite(OUT, rgba, [cv2.IMWRITE_PNG_COMPRESSION, 6])

op = float((alpha > 0.5).mean())
core = alpha[foot > 0.9]
print("wrote %s  %dx%d" % (OUT, RES, RES))
print("opaque fraction overall %.3f, core %.3f" % (op, float((core > 0.5).mean()) if core.size else -1))
print("effective (a^2) core transmittance %.3f" % (1.0 - float((core ** 2).mean()) if core.size else -1))
