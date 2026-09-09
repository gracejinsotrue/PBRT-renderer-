# Sky envmap for the floating_causeway scene.
#
# This camera only sees elevations 0 to 17.3 deg, and measured off the reference
# the altocumulus puffs there are 0.21 deg across at 0.36 coverage. No Poly Haven
# sky has that. evening_road_01_puresky is the closest source available: its sun
# sits at 10 deg so its cloud shading is consistent with our 12.85 deg sun, but its
# puffs are 0.44 deg at 0.13 coverage.
#
# One operation fixes both. A cloud deck at altitude Hc is met by a ray of
# elevation e at horizontal distance Hc/tan(e), so resampling the source through a
# deck at a different altitude is just the radial remap
#     tan(e_src) = K * tan(e_out)
# whose Jacobian is cos^2(e_out) / (K cos^2(e_src)), and cos^2 is ~1 this low, so
# features shrink by 1/K. Coverage is scale invariant under that remap, so it is
# raised for free by the same remap: it draws from the source's 4-40 deg band,
# where cloud covers 0.345 of the solid angle rather than the 0.130 sitting in the
# source's own 0-20 deg band. So one layer carries it and K sets the puff size.
#
# LAYERS stays a list because extra layers each take a rotation and a planar offset
# in deck coordinates, which reads a different patch of real cloud rather than
# repeating one. Adding a second layer goes overcast at this ramp, so keep it at one
# unless the ramp is raised to match.
#
# No sun is painted in. Direct light is the geometric disk from sun_disk.py, which
# can hold a true angular diameter; the source's own disk is knocked down here so
# the two are not double counted. Keep envmapScale at 1.0 and expose with
# evCompensation, or envmapScale silently moves the sun-to-sky ratio.
#
# Everything stays in cv2's native BGR and the work runs in row strips, because
# the device VM has 3 GB and one 8k float32 buffer is already 400 MB. Only the
# upper hemisphere is computed; below the horizon the envmap is visible only where
# the water mesh runs out, so the horizon row is copied down (a puresky's dark
# lower hemisphere reads as a navy band, which cost a day on liminal_bed).
#
#   python tools/scene_build/causeway_sky.py <src.hdr> <out.hdr> [out_width]
import numpy as np, cv2, sys, os, math

SRC = sys.argv[1]
OUT = sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
H = W // 2

# (K, rotation deg, deck offset x, deck offset y), offsets in units of deck altitude
# Swept from the command line while choosing a source; see tools/analysis/sky_preview.py.
# SKY_K = 0 leaves this empty and the procedural cumulus below supplies the clouds;
# any other value restores the deck remap of the source HDRI's own cloud field.
_K = float(os.environ.get("SKY_K", "0"))
LAYERS = [(_K, 0.0, 0.0, 0.0)] if _K > 0 else []

CLOUD_LO = float(os.environ.get("SKY_LO", "1.25"))
CLOUD_HI = float(os.environ.get("SKY_HI", "1.75"))
TINT = tuple(float(x) for x in os.environ.get("SKY_TINT", "0.425,0.913,1.797").split(","))  # RGB
# Source clouds are evening-warm; the reference's are near-neutral and faintly cool
# (linear ratios 0.913/1.0/1.077 measured off it). Replace cloud chroma but keep the
# luminance, which is where all the cloud structure lives.
CLOUD_RGB = (0.925, 1.013, 1.091)
CLOUD_NEUTRAL = float(os.environ.get("SKY_CN", "0.8"))
SUN_KILL_DEG = 2.5                 # knock down the disk only, keep the haze glow

# LAYERED ridges, not one band. A single profile filled with one tone measured
# horizontal variation 0.016 and tonal range 0.206 against the reference's 0.177 and
# 0.780: a flat grey stripe. Real distance reads as overlapping ranges each at its
# own aerial-perspective tone, so paint far-to-near and let the darker near layers
# overlay the paler far ones.
#
# This stays in the envmap rather than becoming geometry on purpose. At 3-8 km the
# brightness is scattered air, not surface reflectance: a sky-lit diffuse ridge with
# albedo 0.6 renders about 0.09 in display against the reference's 0.6, so geometry
# would need an emissive fudge anyway, and painting keeps the water reflection right.
#
# (top elevation deg, tone as a fraction of the local sky, seed, along-ridge swing)
# Tones are ABOVE 1.0 because the reference ridge is brighter than the deep sky
# above it: hazy air near the horizon out-scatters the terrain entirely. The first
# layered attempt used 0.95/0.78/0.58 and, once the depth darkening stacked on top,
# bottomed out at 0.34 and rendered as a dark wall. Tops also come down to the
# reference's measured 0 to 2.05 deg band.
# Each layer carries its OWN base falloff. A single shared falloff either left the
# whole band brighter than the sky (no silhouette) or darkened all of it into a wall.
# The reference runs 0.50 at the ridge top down to 0.17 at the waterline, so the far
# peaks stay pale and only the near base goes dark.
# (top elevation deg, tone, seed, along-ridge swing, base falloff)
RIDGE_LAYERS = [(2.05, 1.50, 3, 0.06, 0.15),
                (1.45, 1.30, 11, 0.10, 0.25),
                (0.85, 0.88, 23, 0.13, 0.66)]
RIDGE_RGB = (0.930, 1.000, 1.054)

# RAYLEIGH DEEPENING WITH ELEVATION. TINT is a single constant, so the sky carried one
# chroma from horizon to zenith: measured B/R ran 1.28 / 1.50 / 1.63 / 1.67 at 4, 8.5,
# 13.5 and 19 deg against the reference's 1.35 / 1.77 / 2.43 / 2.71, and its luminance
# ratio ours-to-reference ran 1.02 / 0.93 / 0.77 / 0.72 over the same bands. A real sky
# deepens and darkens toward the zenith and a constant tint cannot do either, which is
# what left ours looking pale and flat next to the reference.
#
# Chroma is applied luminance-preserving (B up and R down by sqrt of the ratio, then
# rescaled back to the original luminance) so the two ramps stay independent, and both
# run BEFORE the irradiance normalisation so total sky illumination is unchanged and
# only its distribution moves.
# (elevation deg, B/R multiplier), interpolated
ZENITH_CHROMA = [(0.0, 1.00), (4.0, 1.06), (8.5, 1.18), (13.5, 1.49),
                 (19.0, 1.62), (30.0, 1.70)]
# (elevation deg, radiance multiplier)
ZENITH_LUM = [(0.0, 1.03), (4.0, 1.03), (8.5, 0.86), (13.5, 0.62),
              (19.0, 0.52), (30.0, 0.48)]

HAZE = float(os.environ.get("SKY_HAZE", "0.50"))
HAZE_EL = 22.0
# Total sky irradiance. 1.0 left the rendered sky at display luminance 0.79/0.73/0.67/0.62
# across the 2-6, 6-11, 11-16 and 16-22 deg bands against the reference's 0.68/0.59/0.47/0.44,
# and since ACES desaturates as it brightens, that excess brightness was most of why the
# sky looked pale: at 0.50 the same bands land at 0.63/0.56/0.49/0.43 with saturation
# 0.26/0.43/0.53/0.61 against the reference's 0.26/0.44/0.58/0.64.
#
# Halving the sky would darken everything it lights, so the difference is moved into the
# sun disk instead: sky 0.50 + aureole 0.235 + sun 10.265 keeps the old total of 11.0.
# A brighter sun against a deeper sky is just a clearer, lower-aerosol day.
SKY_E = float(os.environ.get("SKY_E", "0.50"))
SUN_ELEV, SUN_AZIM = 12.85, 72.79   # must match causeway_build.py and sun_disk.py

LUM = np.array([0.0722, 0.7152, 0.2126], np.float32)      # BGR
TINT_B = np.array(TINT[::-1], np.float32)
RIDGE_RGB_B = np.array(RIDGE_RGB[::-1], np.float32)
RIDGE_RGB_B /= float(np.dot(RIDGE_RGB_B, LUM))
TINT_LUM = float(np.dot(TINT_B, LUM))
CLOUD_RGB_B = np.array(CLOUD_RGB[::-1], np.float32)
CLOUD_RGB_B /= float(np.dot(CLOUD_RGB_B, LUM))

src = cv2.imread(SRC, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
if src is None:
    raise SystemExit("cannot read " + SRC)
src = np.ascontiguousarray(src.astype(np.float32))
SH, SW, _ = src.shape
print("source %dx%d  output %dx%d" % (SW, SH, W, H))

sl = src @ LUM
sj, si = np.unravel_index(int(np.nanargmax(sl)), sl.shape)
print("source sun at elev %.1f azim %.1f" % (90.0 - (sj + 0.5) / SH * 180.0, (si + 0.5) / SW * 360.0))

rad = max(int(SUN_KILL_DEG * SW / 360.0), 2)
y0, y1 = max(sj - 3 * rad, 0), min(sj + 3 * rad + 1, SH)
xs = (np.arange(-3 * rad, 3 * rad + 1) + si) % SW
patch = src[y0:y1][:, xs].copy()
pl = patch @ LUM
yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
r2 = (yy - (sj - y0)) ** 2 + (xx - 3 * rad) ** 2
ring = (r2 > (2.0 * rad) ** 2) & (r2 < (3.0 * rad) ** 2)
surround = float(np.median(pl[ring])) if ring.any() else float(np.median(pl))
scale = np.ones_like(pl)
over = (r2 <= (2.0 * rad) ** 2) & (pl > surround)
scale[over] = surround / pl[over]
feather = np.clip((np.sqrt(r2) - rad) / float(rad), 0, 1)
scale = scale * (1 - feather) + feather
src[y0:y1, xs] = patch * scale[..., None]
del patch, pl, yy, xx, r2, ring, scale, feather, sl

# Clear-sky estimate by a hole-filling blur that ignores cloud pixels, so the large
# scale azimuthal structure (bright toward the sun, blue away) survives.
DS = max(SW // 1024, 1)
small = cv2.resize(src, (SW // DS, SH // DS), interpolation=cv2.INTER_AREA)
sml = small @ LUM
w = (sml < cv2.blur(sml, (81, 81)) * 1.10).astype(np.float32)
k = (max(small.shape[1] // 12, 3) | 1,) * 2
clear_s = cv2.blur(cv2.blur(small * w[..., None], k) / np.maximum(cv2.blur(w, k), 1e-4)[..., None], k)
clear_sl = np.ascontiguousarray(clear_s @ LUM)
del small, sml, w
horizon_ref = clear_s[int(0.5 * clear_s.shape[0]) - 1].mean(0)


def sample(img, u, v):
    return cv2.remap(img, (u * img.shape[1] - 0.5).astype(np.float32),
                     (v * img.shape[0] - 0.5).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


uu = ((np.arange(W, dtype=np.float32) + 0.5) / W)


def ridge_profile(top, seed, bands):
    rng = np.random.default_rng(seed)
    q = np.zeros(W, np.float32)
    for period, amp in bands:
        q += amp * np.sin(2 * math.pi * uu * (360.0 / period) + rng.random() * 2 * math.pi)
    q = (q - q.min()) / max(float(q.max() - q.min()), 1e-6)
    return top * (0.45 + 0.55 * q), q


PROFILES = []
for i, (top, tone, seed, swing, fall) in enumerate(RIDGE_LAYERS):
    # each range gets its own frequency mix so they do not read as one shape scaled
    bands = [(360.0, 0.30 + 0.05 * i), (150.0 - 30 * i, 0.26), (61.0 - 12 * i, 0.19),
             (23.0 - 4 * i, 0.12), (9.0 - 1.5 * i, 0.07)]
    prof, q = ridge_profile(top, seed, bands)
    PROFILES.append((prof, tone, q, swing, fall))

# PROCEDURAL CUMULUS, OFF BY DEFAULT (SKY_CUM_COVER=0). It was built because the
# deck remap could not produce discrete puffs near the horizon, and it does hit the
# reference's coverage, size, aspect and cloud/sky contrast per elevation band. It
# still lost: statistics matched and it read as an even grey stipple, like camouflage
# rather than weather, because thresholded noise has no organic clustering and no
# internal shading. The real fix was a better SOURCE - farm_field_puresky carries
# 0.260 coverage in the band a K=2.2 remap draws from against evening_road's 0.177,
# with 120 discrete blobs, so the remap has proper cumulus to work with and the
# shapes come out organic for free. Kept here in case a scene ever needs a sky with
# no usable source. The original note follows.
#
# The deck remap cannot produce this: tan(e_src)=K*tan(e_out)
# compresses elevation by ~1/K, so round source puffs flatten and then MERGE, and
# measured against the reference per elevation band the remap gave 6 blobs at aspect
# 14.0 near the horizon where the reference has 39 at aspect 3.2, with coverage
# 0.052 against 0.186 in the 2-5 deg band. Lowering K reduces the flattening but
# draws from the source's less cloudy low band, so no K gives both.
#
# Generated in ANGULAR space instead, fitted to the reference's own numbers: median
# feature 0.62 deg, coverage 0.27, aspect 3.1 at the horizon easing to 1.5 at 20 deg,
# cloud radiance 1.9x the local clear sky with near-neutral chroma (measured
# 1.75-2.03x across bands). Aspect is integrated into the vertical coordinate,
# W(e) = a1*e + (a0-a1)*e^2/(2*TOP), so the squash varies smoothly with elevation.
#
# THREE THINGS THE FIRST VERSION GOT WRONG, all of which matched the statistics and
# still looked like a texture rather than a sky:
#   - a small base cell put the threshold crossing on the noise lattice itself, so
#     the puffs came out as axis-aligned rectangles. The base cell is now LARGE
#     (3 deg) and the puffs are its peaks, which is also how cumulus actually sit.
#   - with one frequency band every puff was the same size and evenly spread. A very
#     low frequency field now modulates the threshold, so there are cloudy stretches
#     and clear ones and the blob sizes spread out on their own.
#   - the lattice is axis-aligned to azimuth and elevation, which reads as a grid, so
#     the sample coordinates are domain-warped before lookup.
CUM_SIZE = float(os.environ.get("SKY_CUM_SIZE", "2.6"))     # deg, base noise cell
CUM_COVER = float(os.environ.get("SKY_CUM_COVER", "1.0"))   # scales the profile below
# COVERAGE IS ALMOST ENTIRELY AN ELEVATION EFFECT and a flat profile misses it badly.
# Measured on the reference against a GLOBAL threshold (a per-band percentile, which
# is what the earlier per-band table used, normalises this trend away and is why it
# went unnoticed): coverage runs 0.70 at 0.3-2 deg, 0.71 at 2-5.4, 0.41 at 5.4-9.6,
# 0.14 at 9.6-13.6 and 0.007 above 13.6, i.e. cloud packed along the horizon and
# clear blue overhead. A flat 0.38 put cloud across the top where the reference has
# essentially none, and left the horizon half as full as it should be.
CUM_PROFILE_E = [0.0, 3.0, 6.0, 8.0, 11.0, 14.0, 18.0, 24.0]
CUM_PROFILE_C = [0.74, 0.72, 0.55, 0.41, 0.20, 0.06, 0.012, 0.0]
CUM_ASPECT = (1.5, 3.1)          # (at CUM_TOP, at the horizon)
CUM_TOP = 22.0
CUM_FADE = 31.0
CUM_RATIO = float(os.environ.get("SKY_CUM_RATIO", "2.55"))  # cloud / clear-sky radiance
CUM_RGB = (0.985, 1.000, 1.020)
CUM_OCT = 6
# Amplitude falloff per octave. 0.5 is the textbook value and it gave a field whose
# blob sizes spanned only p10 0.13 to p90 0.76 deg against the reference's 0.11 to
# 0.99 with a p99 of 6.51, i.e. no large clumps at all. A shallower falloff keeps
# low frequency energy in the field, so thresholding it yields specks and masses
# together instead of one preferred size.
CUM_FALLOFF = 0.68
# Two patch scales, not one: the big one opens genuine clear stretches, the small
# one breaks the field up inside them.
CUM_PATCH = (0.30, 0.13)
CUM_PATCH_CELL = (30.0, 11.0)
CUM_WARP = 0.55                  # cells of domain warp
CUM_SEED = 17

CUM_RGB_B = np.array(CUM_RGB[::-1], np.float32)
CUM_RGB_B /= float(np.dot(CUM_RGB_B, LUM))


def _lattice(cell, nv_deg, seed):
    per = max(int(round(360.0 / cell)), 8)
    nv = max(int(nv_deg / cell) + 3, 4)
    return np.random.default_rng(seed).random((nv, per), dtype=np.float32)


def _vnoise(lat, u, v):
    nvv, nuu = lat.shape
    v = np.clip(v, 0.0, nvv - 1.0001)
    i0 = np.floor(u).astype(np.int32)
    j0 = np.floor(v).astype(np.int32)
    fu = u - i0
    fv = v - j0
    fu = fu * fu * (3.0 - 2.0 * fu)
    fv = fv * fv * (3.0 - 2.0 * fv)
    i0 = np.mod(i0, nuu)
    i1 = np.mod(i0 + 1, nuu)
    a = lat[j0, i0] * (1 - fu) + lat[j0, i1] * fu
    b = lat[j0 + 1, i0] * (1 - fu) + lat[j0 + 1, i1] * fu
    return a * (1 - fv) + b * fv


_cum_wmax = (CUM_ASPECT[1] * CUM_FADE
             + (CUM_ASPECT[0] - CUM_ASPECT[1]) * CUM_FADE ** 2 / (2.0 * CUM_TOP))
_cum_oct = [_lattice(CUM_SIZE / (1 << o), _cum_wmax, CUM_SEED + o) for o in range(CUM_OCT)]
_cum_patch = [_lattice(c, _cum_wmax, CUM_SEED + 40 + i) for i, c in enumerate(CUM_PATCH_CELL)]
_cum_wx = _lattice(CUM_SIZE * 2.5, _cum_wmax, CUM_SEED + 50)
_cum_wy = _lattice(CUM_SIZE * 2.5, _cum_wmax, CUM_SEED + 60)


def cum_field(az_deg, e_deg):
    """(fbm, threshold) in angular space; both already aspect-warped."""
    a0, a1 = CUM_ASPECT
    e = np.clip(e_deg, 0.0, CUM_FADE)
    wv = a1 * e + (a0 - a1) * e * e / (2.0 * CUM_TOP)

    cw = CUM_SIZE * 2.5
    wu = _vnoise(_cum_wx, az_deg / cw, wv / cw) - 0.5
    wvv = _vnoise(_cum_wy, az_deg / cw, wv / cw) - 0.5
    au = az_deg + wu * CUM_WARP * CUM_SIZE * 2.0
    av = wv + wvv * CUM_WARP * CUM_SIZE * 2.0

    fbm = np.zeros_like(az_deg)
    norm = 0.0
    for o, lat in enumerate(_cum_oct):
        cell = CUM_SIZE / (1 << o)
        amp = CUM_FALLOFF ** o
        fbm += amp * _vnoise(lat, au / cell, av / cell)
        norm += amp
    fbm /= norm

    patch = np.zeros_like(az_deg)
    for lat, cell, amp in zip(_cum_patch, CUM_PATCH_CELL, CUM_PATCH):
        patch += amp * (_vnoise(lat, az_deg / cell, wv / cell) - 0.5)
    return fbm, patch


def cum_cover(e):
    return CUM_COVER * np.interp(e, CUM_PROFILE_E, CUM_PROFILE_C)


out = np.zeros((H, W, 3), np.float32)
STRIP = 128
HALF = H // 2
for r0 in range(0, HALF, STRIP):
    r1 = min(r0 + STRIP, HALF)
    v = ((np.arange(r0, r1, dtype=np.float32) + 0.5) / H)[:, None]
    elev = (0.5 - v) * math.pi
    phi = (uu * 2.0 * math.pi)[None, :]
    U = np.ascontiguousarray(np.broadcast_to(uu[None, :], (r1 - r0, W)).astype(np.float32))
    V = np.ascontiguousarray(np.broadcast_to(v, (r1 - r0, W)).astype(np.float32))

    acc = sample(clear_s, U, V)
    basel = sample(clear_sl, U, V)
    A = np.zeros((r1 - r0, W, 1), np.float32)

    te = np.tan(np.clip(elev, 1e-4, math.pi / 2 - 1e-4))
    rr = 1.0 / te
    cph, sph = np.cos(phi), np.sin(phi)
    for (K, psi, dx, dy) in LAYERS:
        c, s = math.cos(math.radians(psi)), math.sin(math.radians(psi))
        Xp = rr * (cph * c - sph * s) + dx
        Yp = rr * (cph * s + sph * c) + dy
        rp = np.maximum(np.hypot(Xp, Yp), 1e-6)
        vs = (0.5 - np.arctan(K / rp) / math.pi).astype(np.float32)
        us = (np.arctan2(Yp, Xp) / (2.0 * math.pi) % 1.0).astype(np.float32)
        lay = sample(src, us, vs)
        layc = sample(clear_sl, us, vs)
        ll = lay @ LUM
        lay = lay * (1.0 - CLOUD_NEUTRAL) + ll[..., None] * CLOUD_RGB_B * CLOUD_NEUTRAL
        a = np.clip((ll - CLOUD_LO * layc) /
                    np.maximum((CLOUD_HI - CLOUD_LO) * layc, 1e-6), 0, 1)[..., None]
        acc = acc * (1 - a) + lay * (basel / np.maximum(layc, 1e-6))[..., None] * a
        A = A * (1 - a) + a

    ed_ = np.degrees(elev)
    zc = np.interp(ed_, [z[0] for z in ZENITH_CHROMA], [z[1] for z in ZENITH_CHROMA])
    zl = np.interp(ed_, [z[0] for z in ZENITH_LUM], [z[1] for z in ZENITH_LUM])
    zc = np.broadcast_to(zc, acc.shape[:2])[..., None].astype(np.float32)
    zl = np.broadcast_to(zl, acc.shape[:2])[..., None].astype(np.float32)
    root = np.sqrt(zc)
    chroma = np.concatenate([root, np.ones_like(root), 1.0 / root], axis=2)
    lum0 = (acc @ LUM)[..., None]
    tinted = acc * chroma
    tinted *= lum0 / np.maximum((tinted @ LUM)[..., None], 1e-9)
    acc = tinted * zl

    if CUM_COVER > 0.0:
        azd = np.degrees(phi) + np.zeros_like(U)
        f, patch = cum_field(azd, ed_ + np.zeros_like(U))
        cov = cum_cover(ed_)
        # Row threshold from the target coverage, then shifted by the patch field so
        # coverage holds on average while individual stretches go cloudy or clear.
        t = np.empty((f.shape[0], 1), np.float32)
        for k in range(f.shape[0]):
            ck = float(np.ravel(cov)[k])
            t[k, 0] = np.quantile(f[k], 1.0 - ck) if ck > 1e-4 else 2.0
        # A wide ramp plus a separate core term. With one narrow ramp every puff came
        # out a flat mid-grey slab with a hard rim, which is what made the field read
        # as camouflage: real cumulus are near-white where they are deep and fade to a
        # thin translucent fringe, so opacity and brightness must not share one curve.
        edge = np.maximum(f.std(axis=1, keepdims=True) * 1.45, 1e-4)
        x = (f - (t + patch)) / edge
        ac = np.clip(x, 0.0, 1.0)
        ac = (ac * ac * (3.0 - 2.0 * ac))[..., None]
        core = np.clip((x - 0.55) / 1.35, 0.0, 1.0)[..., None]
        col = basel[..., None] * (CUM_RATIO * (0.58 + 0.55 * core)) * CUM_RGB_B
        acc = acc * (1 - ac) + col * ac
        A = A * (1 - ac) + ac

    # Tint the sky toward the reference's deeper blue but leave the cloud neutral.
    # Tinting everything turns the puffs blue too and they stop reading as cloud.
    acc *= TINT_B * (1 - A) + TINT_LUM * A

    ed = ed_
    acc += (HAZE * np.clip((HAZE_EL - ed) / HAZE_EL, 0, 1) ** 1.1)[..., None] * horizon_ref[None, None, :]

    for prof, tone, q, swing, fall in PROFILES:
        m = np.clip((prof[None, :] - ed) / max(prof.max() * 0.035, 1e-4), 0, 1)[..., None]
        if float(m.max()) <= 0:
            continue
        depth = np.clip(1.0 - ed / np.maximum(prof[None, :], 1e-6), 0, 1)[..., None]
        # each range darkens toward its own base, and its tone drifts along the ridge
        f = tone * (1.0 - fall * depth) * (1.0 + swing * (2.0 * q - 1.0))[None, :, None]
        acc = acc * (1 - m) + (acc @ LUM)[..., None] * RIDGE_RGB_B * f * m

    out[r0:r1] = acc

del src, clear_s, clear_sl
out[HALF:] = out[HALF - 1][None, :, :]

th = (np.arange(H, dtype=np.float32) + 0.5) * math.pi / H
E = 0.0
for r0 in range(0, HALF, STRIP):
    r1 = min(r0 + STRIP, HALF)
    E += float(((out[r0:r1] @ LUM) * (np.cos(th[r0:r1]) * np.sin(th[r0:r1]))[:, None]).sum())
E *= (math.pi / H) * (2 * math.pi / W)
out *= SKY_E / E

# SOLAR AUREOLE. Killing the source HDRI's disk also killed the forward-scattered
# glow around it, and that glow is most of why the reference's sun looks like a sun
# rather than a dot. Measured off the reference on SKY-ONLY pixels (leaves masked
# out by hue) at increasing angular distance from the sun, its sky runs 0.957 display
# at 2 deg, 0.894 at 4, 0.875 at 6.5, 0.871 at 10 and 0.753 at 25; ours ran 0.514,
# 0.525, 0.569, 0.667, 0.722, i.e. DARKER near the sun than away from it, which is
# backwards. With no aureole the only thing that can brighten the region is bloom,
# and bloom washes over the leaves instead of shining between them, so the sun reads
# as a pasted-on disk. The table below is the measured shortfall converted back
# through the display transform, display -> ACES^-1 -> radiance at ev -1.95.
#
# It is added AFTER the irradiance normalisation and its own irradiance is reported,
# because this glow is sunlight scattered out of the direct beam: subtract E_aureole
# from the disk's E_TARGET in sun_disk.py to keep total lighting unchanged.
# Two exponentials, not the measured table interpolated. Piecewise-linear interp
# put visible concentric contour rings across the whole left half of the sky: the
# gradient is smooth enough that the derivative jump at each knot shows as a Mach
# band. This is a least-squares fit to the same measured points (rms 0.36 radiance)
# and it is C-infinity, so there is nothing to band, and the tail dies on its own
# (0.12 at 40 deg, 0.03 at 60) instead of needing a cutoff.
AUREOLE = ((float(os.environ.get("SKY_AU1", "4.6587")), float(os.environ.get("SKY_AUS1", "3.50"))),
           (float(os.environ.get("SKY_AU2", "3.0411")), float(os.environ.get("SKY_AUS2", "12.50"))))
AUREOLE_RGB = (1.0, 0.98, 0.94)

sun_el = math.radians(SUN_ELEV)
sun_az = math.radians(-SUN_AZIM)
sd = np.array([math.cos(sun_el) * math.cos(sun_az), math.sin(sun_el),
               math.cos(sun_el) * math.sin(sun_az)], np.float32)
a_rgb = np.array(AUREOLE_RGB[::-1], np.float32)

E_aur = 0.0
for r0 in range(0, H, STRIP):
    r1 = min(r0 + STRIP, H)
    v = (np.arange(r0, r1, dtype=np.float32) + 0.5) / H
    u = (np.arange(W, dtype=np.float32) + 0.5) / W
    polar = v * math.pi
    y = np.cos(polar)[:, None]
    sp = np.sin(polar)[:, None]
    phi = (u * 2.0 * math.pi)[None, :]
    cosang = sp * np.cos(phi) * sd[0] + y * sd[1] + sp * np.sin(phi) * sd[2]
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    g = sum(amp * np.exp(-ang / sc) for amp, sc in AUREOLE).astype(np.float32)
    out[r0:r1] += g[..., None] * a_rgb
    E_aur += float((g * np.maximum(y, 0.0) * sp).sum())
E_aur *= (math.pi / H) * (2 * math.pi / W) * float(np.dot(a_rgb, LUM))
print("aureole added, its irradiance %.4f (subtract from the sun disk E_TARGET)" % E_aur)

# RGBE DITHER. cv2 writes .hdr as Radiance RGBE: an 8-bit mantissa per channel
# with the EXPONENT SHARED across RGB, so whichever channel is smallest gets the
# coarsest relative step. In this blue sky the exponent follows blue, which leaves
# red at 1.2% per step against blue's 0.43%, and once the aureole put a smooth
# radial gradient into red it staircased into concentric arcs 11 deg wide across
# the whole left half of the sky. Measured on the written file: 3.906e-3 absolute
# step, flat runs of 11 consecutive 1-deg samples in red.
#
# Half a step of uniform noise per texel breaks the staircase. The envmap is
# minified roughly 4x into frame, so the noise averages away and only the banding
# goes. stbi_loadf cannot read EXR, so a float format is not an option here.
rng = np.random.default_rng(11)
for r0 in range(0, H, STRIP):
    r1 = min(r0 + STRIP, H)
    strip = out[r0:r1]
    step = np.exp2(np.ceil(np.log2(np.maximum(strip.max(2, keepdims=True), 1e-20)))) / 256.0
    strip += (rng.random(strip.shape, dtype=np.float32) - 0.5) * step
    np.maximum(strip, 0.0, out=strip)

os.makedirs(os.path.dirname(os.path.abspath(OUT)) or ".", exist_ok=True)
cv2.imwrite(OUT, out)
print("wrote %s  %dx%d  irradiance was %.4f, now %.4f" % (OUT, W, H, E, SKY_E))

b0 = int((0.5 - 20 / 180.0) * H)
band = out[b0:HALF] @ LUM
mxc = out[b0:HALF].max(2); mnc = out[b0:HALF].min(2)
sat = np.where(mxc > 1e-9, (mxc - mnc) / np.maximum(mxc, 1e-9), 0)
cl = (band > np.percentile(band, 60) * 1.45) & (sat < 0.22)
sth = np.sin(th[b0:HALF])[:, None]
cov = float((cl * sth).sum() / (np.ones_like(cl) * sth).sum())
n, _, stats, _ = cv2.connectedComponentsWithStats(cl.astype(np.uint8), connectivity=8)
hh = [stats[i][3] * (180.0 / H) for i in range(1, n) if stats[i][4] >= 9]
print("check: coverage %.3f in the visible band (reference 0.36), median puff %.2f deg (reference 0.21), %d blobs"
      % (cov, float(np.median(hh)) if hh else 0.0, len(hh)))
