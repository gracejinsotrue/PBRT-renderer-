# Build scenes/san_miguel/meshes/sun_disk.obj: an emissive disk standing in for the sun.
#
# Why a disk and not a sphere: the renderer's area lights are ONE-SIDED (Emitter.hlsli
# rejects the sample when dot(emitterNormal, -wi) <= 0, and that normal comes from the
# triangle WINDING, not from the .obj's vn). On a sphere half the sampled points face away
# and are thrown out - double the variance for nothing. A disk aimed at the courtyard has
# every sample facing the right way.
#
# Run:  python _sm_sun_geo.py     then paste the printed <mesh> block into scene.xml.
import math, os

# ---- SUN GEOMETRY PARAMETERS ----
ELEV      = 70.0      # degrees above the horizon (match _sm_sky.py if the sky keeps a sun)
AZIM      = 1.9966    # world azimuth in RADIANS = 2*pi*AZIM_U - envmapRotation
DIST      = 250.0     # metres from the courtyard. Bigger = more parallel light + flatter
                      # falloff across the scene, but blows up the scene bounds.
ANG_DIAM  = 0.53      # angular diameter in degrees. The real sun is 0.53. Smaller = harder
                      # shadow edges. This is the knob the HDR could not give you: a 2048-wide
                      # equirect cannot paint a sun sharper than ~0.18 deg.
E_TARGET  = 15.5      # target irradiance on a HORIZONTAL surface, in renderer units.
                      # 15.5 == what the painted sun currently delivers (see _sm_sunratio.py).
TINT      = (1.0, 0.7636, 0.5273)   # colour ratio of the current painted sun
CENTER    = (15.0, 1.0, 6.0)        # courtyard centre, what the disk aims at
SEGMENTS  = 64
# ---------------------------------

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes", "san_miguel")
OUT  = os.path.join(BASE, "meshes", "sun_disk.obj")

el = math.radians(ELEV)
d  = (math.cos(el) * math.cos(AZIM), math.sin(el), math.cos(el) * math.sin(AZIM))  # scene -> sun
r  = DIST * math.tan(math.radians(ANG_DIAM) / 2.0)
P  = tuple(CENTER[i] + d[i] * DIST for i in range(3))
n  = tuple(-d[i] for i in range(3))   # disk faces the courtyard

# orthonormal basis with u x v == n, so the fan winding emits toward the scene
up = (0.0, 0.0, 1.0) if abs(n[1]) > 0.9 else (0.0, 1.0, 0.0)
u  = (n[1] * up[2] - n[2] * up[1], n[2] * up[0] - n[0] * up[2], n[0] * up[1] - n[1] * up[0])
ul = math.sqrt(sum(c * c for c in u)); u = tuple(c / ul for c in u)
v  = (n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0])

# projected solid angle of the disk seen from the courtyard, then the cos at a flat floor
omega    = math.pi * r * r / (DIST * DIST)
radiance = E_TARGET / (omega * math.sin(el))
rgb      = tuple(radiance * c for c in TINT)

lines = ["# sun disk: elev %.1f deg, azim %.4f rad, dist %.0f m, angular diameter %.3f deg"
         % (ELEV, AZIM, DIST, ANG_DIAM)]
for i in range(SEGMENTS):
    a = 2.0 * math.pi * i / SEGMENTS
    p = tuple(P[k] + r * (math.cos(a) * u[k] + math.sin(a) * v[k]) for k in range(3))
    lines.append("v %.6f %.6f %.6f" % p)
lines.append("v %.6f %.6f %.6f" % P)               # centre is the last vertex
lines.append("vn %.6f %.6f %.6f" % n)
c = SEGMENTS + 1
for i in range(SEGMENTS):
    a, b = i + 1, (i + 1) % SEGMENTS + 1
    lines.append("f %d//1 %d//1 %d//1" % (c, a, b))  # centre, A_i, A_i+1  -> normal = u x v = n
open(OUT, "w").write("\n".join(lines) + "\n")

print("wrote %s" % OUT)
print("  disk centre   (%.1f, %.1f, %.1f), radius %.3f m, faces (%.3f, %.3f, %.3f)" % (P + (r,) + n))
print("  solid angle   %.3e sr   (real sun ~6.8e-5)" % omega)
print("  radiance for E_horiz=%.1f:  %.0f %.0f %.0f" % ((E_TARGET,) + rgb))
print("""
<mesh type="obj">
\t<string name="filename" value="meshes/sun_disk.obj"/>
\t<emitter type="area">
\t\t<color name="radiance" value="%.0f %.0f %.0f"/>
\t</emitter>
</mesh>""" % rgb)
