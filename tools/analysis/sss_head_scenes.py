# Generates the head2 subsurface comparison set.
#
# Every scene below shares one camera, one sampler and the same non-skin
# meshes (brows, lashes, eyeballs, tear film, lenses). The ONLY thing that
# varies is the material on Head.obj and which HDRI lights it, so any
# difference you see between the images is the skin model, not the setup.
#
#   sss_cmp_disney   Disney BRDF with subsurface=0.5 (the cheap wrap-diffuse
#                    approximation) - what head.xml already used
#   sss_cmp_walk     the real random-walk BSSRDF, same textures, same light
#   sss_r0005/2/5    radius sweep on the random walk: opaque -> skin -> wax
#   sss_backlit      random walk lit from directly behind by the low sunset
#                    HDRI, so light transilluminates the ears and nose
#
# usage: python tools/analysis/sss_head_scenes.py [--preview]
import os, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root: tools/<group>/ -> ..
OUT = os.path.join(REPO, "scenes", "final_scenes", "head2")
PREVIEW = '--preview' in sys.argv

# ---------------------------------------------------------------- camera

# Head.obj spans y 0.057..0.397 and faces +z. Aim at the eyeline, a little
# off-axis so the nose casts and the light wraps across one cheek.
# Framed on the face rather than the bust: Head.obj is cut open at the chest,
# and an unclosed boundary lets the random walk escape, so that slab glows
# waxy white and reads as a bug rather than as translucency. Cropping above it
# also gets the HDRI's grass out of the frame.
TARGET = (0.0, 0.285, 0.055)
DIST, AZ, EL = 0.62, 15.0, 6.0
FOV = 26.0
W, H = (450, 550) if PREVIEW else (900, 1100)
SPP = 48 if PREVIEW else 1024


def origin():
    a, e = math.radians(AZ), math.radians(EL)
    return (TARGET[0] + DIST * math.sin(a) * math.cos(e),
            TARGET[1] + DIST * math.sin(e),
            TARGET[2] + DIST * math.cos(a) * math.cos(e))


# ---------------------------------------------------------------- lighting

# Both HDRIs put the sun at azimuth 216 deg. EvalEnvmap looks up
# atan2(d.z, d.x) + envmapRotation, so to place the sun at a chosen world
# azimuth we rotate by (216 - wanted). Radians - the value is added to phi
# directly, with no degree conversion anywhere in the pipeline.
def rotation_for(sun_az_in_hdri, wanted_world_az):
    return math.radians(sun_az_in_hdri - wanted_world_az)


# key from front-left-above: the camera looks down -z, so +z is the face side
KEY = dict(envmap="textures/sunset.hdr",
           rotation=rotation_for(216.1, 115.0),
           scale=1.0, ev=0.0)

# Dead behind the head is world azimuth 255 for this camera, not 270 - the
# camera sits off-axis. sunset1's sun is at 3.7 deg elevation, so it rakes
# horizontally through the ear instead of lighting the top of the skull.
# Negative EV keeps the sky from blowing out so the lit ear is the brightest
# thing in frame.
BACK = dict(envmap="textures/sunset1.hdr",
            rotation=rotation_for(216.2, 255.0),
            scale=1.0, ev=-0.7)


# ---------------------------------------------------------------- materials

FACE_TEX = """\t\t\t<string name="albedoTexture"    value="textures/Face/Face_Albedo.jpg"/>
\t\t\t<string name="normalTexture"    value="textures/Face/Face_Normal.jpg"/>
\t\t\t<string name="roughnessTexture" value="textures/Face/Face_Roughness.jpg"/>"""


def skin_disney():
    return """\t\t<bsdf type="disney">
\t\t\t<color name="baseColor" value="0.75 0.57 0.48"/>
%s
\t\t\t<float name="roughness"      value="0.35"/>
\t\t\t<float name="metallic"       value="0.0"/>
\t\t\t<float name="specular"       value="0.35"/>
\t\t\t<float name="sheen"          value="0.05"/>
\t\t\t<float name="sheenTint"      value="1.0"/>
\t\t\t<float name="subsurface"     value="0.5"/>
\t\t\t<float name="clearcoat"      value="0.05"/>
\t\t\t<float name="clearcoatGloss" value="0.8"/>
\t\t</bsdf>""" % FACE_TEX


# The random walk takes its medium albedo from the BLURRED albedo texture
# (Shaders.hlsl: SubsurfaceParams(aBlur * sssTint, ...)), and Face_Albedo.jpg
# averages a linear (0.476, 0.219, 0.122). Fed in raw that gives per-channel
# single-scattering albedos of (0.90, 0.64, 0.45), and since a walk bounces
# many times inside the head, the blue channel is annihilated and skin comes
# out dark and muddy. `tint` scales that medium albedo; these values lift it
# to (0.75, 0.45, 0.32) -> alpha (0.985, 0.884, 0.778): red-dominant and
# translucent, without tipping over into the pale wax you get near alpha 1.
#
# `albedo` is set to the texture's own mean because the shader computes
# sssDetail = clamp(albedoConstant / aBlur, 0.5, 2); matching them keeps that
# factor near 1 so it does not double-tint the exit radiance.
SSS_MEDIUM_ALBEDO = (0.476, 0.219, 0.122)
SSS_TINT = (1.5756, 2.0548, 2.6230)


def skin_walk(radius):
    return """\t\t<bsdf type="subsurface">
\t\t\t<color name="albedo" value="%g %g %g"/>
\t\t\t<color name="tint"   value="%.4f %.4f %.4f"/>
%s
\t\t\t<string name="specularTexture"  value="textures/Face/Face_Specular.jpg"/>
\t\t\t<float name="radius"    value="%g"/>
\t\t\t<float name="intIOR"    value="1.33"/>
\t\t\t<float name="extIOR"    value="1.0"/>
\t\t\t<float name="g"         value="0.0"/>
\t\t\t<float name="roughness" value="0.30"/>
\t\t\t<float name="specular"  value="0.35"/>
\t\t</bsdf>""" % (SSS_MEDIUM_ALBEDO + SSS_TINT + (FACE_TEX, radius))


# Everything that is not skin. Lifted from head.xml so the two comparison
# images differ in exactly one material.
REST = """
\t<!-- brows -->
\t<mesh type="obj">
\t\t<string name="filename" value="meshes/Brows.obj"/>
\t\t<bsdf type="disney">
\t\t\t<color name="baseColor" value="0.08 0.05 0.03"/>
\t\t\t<float name="roughness" value="0.7"/>
\t\t\t<float name="specular"  value="0.15"/>
\t\t\t<float name="sheen"     value="0.3"/>
\t\t\t<float name="sheenTint" value="1.0"/>
\t\t</bsdf>
\t</mesh>

\t<!-- lashes -->
\t<mesh type="obj">
\t\t<string name="filename" value="meshes/Lashes.obj"/>
\t\t<bsdf type="disney">
\t\t\t<color name="baseColor" value="0.03 0.02 0.02"/>
\t\t\t<float name="roughness" value="0.5"/>
\t\t\t<float name="specular"  value="0.2"/>
\t\t\t<float name="sheen"     value="0.2"/>
\t\t\t<float name="sheenTint" value="1.0"/>
\t\t</bsdf>
\t</mesh>
"""

for side in ("Left", "Right"):
    REST += """
\t<!-- eyeball %s -->
\t<mesh type="obj">
\t\t<string name="filename" value="meshes/Realtime Eyeball %s.obj"/>
\t\t<bsdf type="disney">
\t\t\t<color name="baseColor" value="0.9 0.9 0.9"/>
\t\t\t<string name="albedoTexture"    value="textures/Eyes/Eyes_Balls_Diffuse.jpg"/>
\t\t\t<string name="normalTexture"    value="textures/Eyes/Eyes_Balls_Normals.jpg"/>
\t\t\t<string name="roughnessTexture" value="textures/Eyes/Eyes_Balls_Roughness.jpg"/>
\t\t\t<float name="roughness"      value="0.3"/>
\t\t\t<float name="specular"       value="0.5"/>
\t\t\t<float name="subsurface"     value="0.3"/>
\t\t\t<float name="clearcoat"      value="0.1"/>
\t\t\t<float name="clearcoatGloss" value="1.0"/>
\t\t</bsdf>
\t</mesh>
""" % (side.lower(), side)

REST += """
\t<!-- Tear film only. head.xml also wires "Lens Left/Right.obj" as dielectric,
\t     but those are NOT corneal bulges: each is 1666 verts over the same bbox
\t     as its eyeball, i.e. a duplicate shell lying exactly on the eyeball
\t     surface. A smooth dielectric coincident with the surface it covers traps
\t     rays between the two, and the eyes render solid black. Dropping them is
\t     what makes the sclera and iris visible. -->
\t<mesh type="obj">
\t\t<string name="filename" value="meshes/Eye Wet.obj"/>
\t\t<bsdf type="dielectric"/>
\t</mesh>
"""


# ---------------------------------------------------------------- assembly

def scene(note, skin, light):
    o = origin()
    return """<?xml version='1.0' encoding='utf-8'?>

<!-- %s

     Generated by tools/analysis/sss_head_scenes.py - edit that, not this file. Camera,
     sampler and every non-skin material are identical across the whole set,
     so differences between renders are the skin model alone.

     Head2 asset is untracked (scenes/final_scenes/ is gitignored); only the
     resulting images under images/ are committed. -->
<scene>
\t<string name="envmap" value="%s"/>
\t<float name="envmapScale" value="%g"/>
\t<float name="envmapRotation" value="%.6f"/>
\t<float name="evCompensation" value="%g"/>

\t<camera type="perspective">
\t\t<float name="fov" value="%g"/>
\t\t<transform name="toWorld">
\t\t\t<lookat target="%g, %g, %g" origin="%.5f, %.5f, %.5f" up="0, 1, 0"/>
\t\t</transform>
\t\t<integer name="width" value="%d"/>
\t\t<integer name="height" value="%d"/>
\t</camera>

\t<sampler type="independent">
\t\t<integer name="sampleCount" value="%d"/>
\t</sampler>

\t<!-- SKIN -->
\t<mesh type="obj">
\t\t<string name="filename" value="meshes/Head.obj"/>
%s
\t</mesh>
%s</scene>
""" % (note, light['envmap'], light['scale'], light['rotation'], light['ev'],
       FOV, TARGET[0], TARGET[1], TARGET[2], o[0], o[1], o[2], W, H, SPP,
       skin, REST)


SCENES = {
    'sss_cmp_disney': ("Disney BRDF with subsurface=0.5 - a wrap-diffuse "
                       "approximation, not real transport.", skin_disney(), KEY),
    'sss_cmp_walk':   ("Real random-walk BSSRDF at radius 0.008. Same textures "
                       "and light as sss_cmp_disney.", skin_walk(0.008), KEY),
    'sss_r0002':      ("Random walk, radius 0.002 - short mean free path, "
                       "reads nearly opaque.", skin_walk(0.002), KEY),
    'sss_r0008':      ("Random walk, radius 0.008 - skin. The head is 0.34 "
                       "units tall, so this is roughly 8 mm of mean free "
                       "path.", skin_walk(0.008), KEY),
    'sss_r003':       ("Random walk, radius 0.03 - long mean free path, waxy.",
                       skin_walk(0.03), KEY),
    'sss_backlit':    ("Random walk, radius 0.008, lit from directly behind by "
                       "the low sunset HDRI so light comes through the ears.",
                       skin_walk(0.008), BACK),
}

for name, (note, skin, light) in SCENES.items():
    p = os.path.join(OUT, name + '.xml')
    open(p, 'w').write(scene(note, skin, light))
    print('wrote', p)

print('\ncamera origin %.4f %.4f %.4f -> target %s, fov %g, %dx%d @ %d spp'
      % (origin() + (TARGET, FOV, W, H, SPP)))
print('key  envmapRotation %.4f rad (sun to world azimuth 115)' % KEY['rotation'])
print('back envmapRotation %.4f rad (sun to world azimuth 255)' % BACK['rotation'])
