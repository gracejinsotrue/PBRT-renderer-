# Displaced water surface for floating_causeway, as a screen-space PROJECTED GRID.
#
# The dielectric BSDF is ideal-smooth with no roughness and no texture inputs, so
# every ripple has to be real geometry, across 8.9 m to the horizon. A world-space
# grid cannot do that: viewed at grazing incidence a horizontal plane needs lateral
# spacing d/C for one pixel but along-view spacing d^2/(C*h), so along-view cost
# grows as d^2 and a uniform grid is millions of wasted quads in the far field while
# still too coarse near the horizon.
#
# So build the grid uniform in SCREEN space, project each vertex through the camera
# onto the water plane, and displace it there. Cost then scales with the water's
# image area rather than its world area, and the far field is nearly free: 70 m out
# to 20 km costs about 15 rows.
#
# Along-view spacing takes the MINIMUM of screen-uniform and a ripple-resolving cap,
# because at this camera the screen-uniform spacing is already 94 mm at the nearest
# visible water, coarser than the waves need.
#
# Wave amplitudes taper twice: the short wavelengths fade out over 55-70 m where the
# grid can no longer carry them, and everything fades by 300 m so the far field is a
# clean mirror instead of aliased noise. That flat zone starts 13 px below the
# horizon, and the reference's water is glassy there anyway.
#
# Writes to a VM-local path by default: the mount takes writes at only 4.4 MB/s, so
# generate locally and copy in a separate step.
#
#   python tools/scene_build/causeway_water.py <scene.xml> <out.obj>
import numpy as np, sys, os, re, math, time

SCENE, OUT = sys.argv[1], sys.argv[2]

DFINE = float(os.environ.get("WATER_DFINE", "70.0"))    # end of the fine along-view zone
DD = float(os.environ.get("WATER_DD", "0.045"))         # along-view cap, m
PX = float(os.environ.get("WATER_PX", "2.5"))           # screen spacing, px
DMAX = float(os.environ.get("WATER_DMAX", "20000.0"))
GAIN = float(os.environ.get("WAVE_GAIN", "1.0"))        # scales every amplitude, so slope scales linearly
MARGIN = 48.0                                           # px of grid beyond the frame edges

# (wavelength m, amplitude m, direction deg from the view-to-sun bearing)
WAVES = [(2.40, 0.0090, 8.0),
         (0.90, 0.0045, -12.0),
         (0.35, 0.0022, 5.0),
         (0.20, 0.0010, -20.0)]
NOISE = [(0.55, 0.0018), (0.22, 0.0007)]
SHORT_TAPER = (55.0, 70.0)      # short wavelengths fade out across this band
# Pushed out from (150, 300) once the ripple band read as a thin strip under the
# causeway: the 2.40 m and 0.90 m components are resolved by the grid far past
# D_fine, so the far taper was throwing away texture the mesh could carry.
FAR_TAPER = (900.0, 2500.0)

# A FETCH PROFILE, measured rather than assumed. A single uniform amplitude made the
# near water sit flat at p50 0.558 all the way to the bottom of frame while the
# reference falls to 0.347: at grazing incidence the ripples catch the bright
# near-horizon sky from every azimuth and wash out the Fresnel falloff that darkens
# calm water toward the viewer.
#
# The near value was 0.18, which is glassy, and combined with the old far taper it
# confined all visible texture to roughly 30-150 m, i.e. a strip under the causeway.
# 0.55 near keeps some Fresnel falloff while carrying ripple to the bottom of frame;
# watch the foreground brightness when changing it, since that is the one thing this
# profile trades against.
# (distance m, amplitude multiplier), linearly interpolated
FETCH = [(0.0, 0.55), (15.0, 0.62), (40.0, 1.00), (200.0, 1.00), (2000.0, 0.85)]
SUN_H = np.array([0.29592, -0.95521])   # sun bearing projected on the water, Nori (x, z)

txt = open(SCENE, encoding="utf-8").read()


def grab(pat):
    m = re.search(pat, txt)
    if not m:
        raise SystemExit("cannot find %s in %s" % (pat, SCENE))
    return m


org = [float(v) for v in grab(r'origin="([^"]+)"').group(1).split(",")]
tgt = [float(v) for v in grab(r'target="([^"]+)"').group(1).split(",")]
fov = float(grab(r'name="fov" value="([^"]+)"').group(1))
W = int(grab(r'name="width" value="(\d+)"').group(1))
H = int(grab(r'name="height" value="(\d+)"').group(1))

h = org[1]
d = np.array(tgt) - np.array(org)
d /= np.linalg.norm(d)
pitch = math.asin(d[1])
C = (W / 2.0) / math.tan(math.radians(fov) / 2.0)
HOR = H / 2.0 + C * math.tan(pitch)
print("camera h %.3f m  pitch %.3f deg  C %.1f px/rad  horizon row %.1f of %d" %
      (h, math.degrees(pitch), C, HOR, H))

F = np.array([0.0, math.sin(pitch), -math.cos(pitch)])
U = np.array([0.0, math.cos(pitch), math.sin(pitch)])
R = np.array([1.0, 0.0, 0.0])

# Rows are chosen by distance, then converted to a screen row: at the centre column
# a point d away sits C*h/d pixels below the horizon.
rows = []
dd = 8.0
while dd < DFINE:
    rows.append(dd)
    dd += DD
py_fine = HOR + C * h / DFINE
py = py_fine
py_min = HOR + C * h / DMAX
far = []
while py > py_min:
    far.append(C * h / (py - HOR))
    py -= PX
dist = np.array(rows + far)
py_rows = HOR + C * h / dist
py_rows = np.concatenate([[H + MARGIN], py_rows])
py_rows = np.sort(py_rows)[::-1]
nrow = len(py_rows)

px_cols = np.arange(-MARGIN, W + MARGIN + 1e-9, PX)
ncol = len(px_cols)
print("grid %d rows x %d cols = %.2f M quads = %.2f M tris"
      % (nrow, ncol, nrow * ncol / 1e6, 2 * (nrow - 1) * (ncol - 1) / 1e6))

sx = (px_cols - W / 2.0) / C
sy = (H / 2.0 - py_rows) / C
SX, SY = np.meshgrid(sx, sy, indexing="xy")

dirs_y = F[1] + SY * U[1]
dirs_y = np.minimum(dirs_y, -1e-6)
t = -h / dirs_y
X = t * (SX * R[0])
Z = t * (F[2] + SY * U[2])

r = np.sqrt(X * X + Z * Z)
short = np.clip((SHORT_TAPER[1] - r) / (SHORT_TAPER[1] - SHORT_TAPER[0]), 0.0, 1.0)
farf = np.clip((FAR_TAPER[1] - r) / (FAR_TAPER[1] - FAR_TAPER[0]), 0.0, 1.0)
farf = farf * np.interp(r, [f[0] for f in FETCH], [f[1] for f in FETCH])

rng = np.random.default_rng(11)
Y = np.zeros_like(X)
for lam, amp, ang in WAVES:
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    kx = SUN_H[0] * c - SUN_H[1] * s
    kz = SUN_H[0] * s + SUN_H[1] * c
    k = 2.0 * math.pi / lam
    w = amp * GAIN * np.sin(k * (X * kx + Z * kz) + rng.random() * 2 * math.pi)
    Y += w * (short if lam < 0.5 else 1.0) * farf


def vnoise(px_, pz_, cell, seed, N=512):
    g = np.random.default_rng(seed).random((N, N))
    u = px_ / cell
    v = pz_ / cell
    i0 = np.floor(u).astype(np.int64)
    j0 = np.floor(v).astype(np.int64)
    fu = u - i0
    fv = v - j0
    su = fu * fu * (3 - 2 * fu)
    sv = fv * fv * (3 - 2 * fv)
    i0 %= N
    j0 %= N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    a = g[i0, j0] * (1 - su) + g[i1, j0] * su
    b = g[i0, j1] * (1 - su) + g[i1, j1] * su
    return (a * (1 - sv) + b * sv) * 2.0 - 1.0


for cell, amp in NOISE:
    Y += amp * GAIN * vnoise(X, Z, cell, int(cell * 1000)) * short * farf

# rms slope, the number to tune against: pool_store ran 5.2 deg for a chlorinated
# pool, this glassy lake wants about half that
gy_z, gy_x = np.gradient(Y, axis=(0, 1))
gx_z, gx_x = np.gradient(X, axis=(0, 1))
gz_z, gz_x = np.gradient(Z, axis=(0, 1))
near = r < 60.0
with np.errstate(divide="ignore", invalid="ignore"):
    sl_x = np.where(np.abs(gx_x) > 1e-9, gy_x / gx_x, 0.0)
    sl_z = np.where(np.abs(gz_z) > 1e-9, gy_z / gz_z, 0.0)
slf = np.sqrt(np.nan_to_num(sl_x) ** 2 + np.nan_to_num(sl_z) ** 2)
for lbl, lo, hi in [("near  <20 m", 0.0, 20.0), ("mid 30-60 m", 30.0, 60.0), ("far 80-150 m", 80.0, 150.0)]:
    m = (r >= lo) & (r < hi)
    if m.sum() < 100:
        continue
    q = slf[m]
    print("  rms slope %-13s %.2f deg   p99 %.2f deg" %
          (lbl, math.degrees(np.sqrt((q ** 2).mean())), math.degrees(np.percentile(q, 99))))
sl = slf[near]
print("amplitude p2p %.1f mm over the near field" % (1000 * (Y[near].max() - Y[near].min())))

V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
idx = np.arange(nrow * ncol).reshape(nrow, ncol)
a = idx[:-1, :-1].ravel()
b = idx[:-1, 1:].ravel()
c = idx[1:, 1:].ravel()
dd_ = idx[1:, :-1].ravel()
# wound so the geometric normal faces up, which is what the dielectric expects
Fc = np.concatenate([np.stack([a, dd_, c], 1), np.stack([a, c, b], 1)]) + 1

t0 = time.time()
os.makedirs(os.path.dirname(os.path.abspath(OUT)) or ".", exist_ok=True)
with open(OUT, "wb") as f:
    f.write(b"# projected-grid water, %d verts %d tris\n" % (len(V), len(Fc)))
    np.savetxt(f, V, fmt="v %.5f %.5f %.5f")
    np.savetxt(f, Fc, fmt="f %d %d %d")
print("wrote %s  %.1f MB in %.1f s" % (OUT, os.path.getsize(OUT) / 1e6, time.time() - t0))
print()
print("XML block to inject:")
print('\t<mesh type="obj">\n\t\t<string name="filename" value="meshes/water.obj"/>\n'
      '\t\t<bsdf type="dielectric">\n\t\t\t<float name="intIOR" value="1.333"/>\n'
      '\t\t\t<color name="tintColor" value="0.35 0.55 0.62"/>\n\t\t</bsdf>\n\t</mesh>')
