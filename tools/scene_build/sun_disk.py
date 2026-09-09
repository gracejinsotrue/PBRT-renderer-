# Emissive disk standing in for the sun. Generalised from sm_sun_geo.py.
#
# A disk, not a sphere: area lights are ONE-SIDED (Emitter.hlsli rejects a sample when
# dot(emitterNormal, -wi) <= 0) and that normal comes from triangle WINDING, not from vn.
# Half a sphere's samples would face away and be thrown out.
#
# Angles are given in BLENDER convention (Z-up, azimuth = atan2(y,x)) and converted here,
# so one pair of numbers drives both the Blender preview lamp and the Nori render.
# Under NORI_YUP the mapping is (x,y,z)->(x,z,-y), hence nori_azimuth = -blender_azimuth.
#
#   python tools/scene_build/sun_disk.py <out.obj> <elev> <azim> <E_target> [ang_diam] [dist]
import math, os, sys

OUT      = sys.argv[1]
ELEV     = float(sys.argv[2])
AZIM_B   = float(sys.argv[3])          # Blender azimuth, degrees
E_TARGET = float(sys.argv[4])          # irradiance on a HORIZONTAL surface, renderer units
ANG_DIAM = float(sys.argv[5]) if len(sys.argv) > 5 else 0.53   # degrees; the real sun
DIST     = float(sys.argv[6]) if len(sys.argv) > 6 else 300.0  # metres
TINT     = (1.0, 0.965, 0.895)         # slight warmth, midday sun
CENTER   = (0.0, 1.2, 0.0)             # Nori-space point the disk aims at
SEGMENTS = 64

el = math.radians(ELEV)
az = math.radians(-AZIM_B)             # Blender -> Nori azimuth
d  = (math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az))   # scene -> sun
r  = DIST * math.tan(math.radians(ANG_DIAM) / 2.0)
P  = tuple(CENTER[i] + d[i] * DIST for i in range(3))

omega = math.pi * r * r / (DIST * DIST)          # solid angle, small-angle
L = E_TARGET / (omega * math.sin(el))            # radiance needed for E_TARGET on horizontal

n = tuple(-d[i] for i in range(3))
a = (0.0, 1.0, 0.0) if abs(n[1]) < 0.9 else (1.0, 0.0, 0.0)
u = (a[1]*n[2]-a[2]*n[1], a[2]*n[0]-a[0]*n[2], a[0]*n[1]-a[1]*n[0])
m = math.sqrt(sum(c*c for c in u)); u = tuple(c/m for c in u)
w = (n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2], n[0]*u[1]-n[1]*u[0])

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w") as f:
    f.write("# sun disk: elev %.2f azimB %.2f angdiam %.3f dist %.1f\n" % (ELEV, AZIM_B, ANG_DIAM, DIST))
    f.write("v %.6f %.6f %.6f\n" % P)
    for i in range(SEGMENTS):
        t = 2 * math.pi * i / SEGMENTS
        p = tuple(P[k] + r * (math.cos(t) * u[k] + math.sin(t) * w[k]) for k in range(3))
        f.write("v %.6f %.6f %.6f\n" % p)
    # wind so the geometric normal faces the scene
    for i in range(SEGMENTS):
        f.write("f 1 %d %d\n" % (2 + i, 2 + (i + 1) % SEGMENTS))

rad = tuple(L * c for c in TINT)
print("sun dir (nori) %.5f %.5f %.5f   radius %.4f m   omega %.3e sr" % (d[0], d[1], d[2], r, omega))
print("radiance %.1f  (E_horizontal target %.3f)" % (L, E_TARGET))
print("---- XML ----")
print('\t<mesh type="obj">\n\t\t<string name="filename" value="meshes/%s"/>\n'
      '\t\t<emitter type="area">\n\t\t\t<color name="radiance" value="%.4f %.4f %.4f"/>\n'
      '\t\t</emitter>\n\t</mesh>' % (os.path.basename(OUT), rad[0], rad[1], rad[2]))
