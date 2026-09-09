# scenes/floating_causeway: build plan

Reference: wide lake shot, stone causeway crossing the frame, mackerel cloud
field, backlit tree entering top-right, shrub mass bottom-right, hazy ridge on
the horizon, sparkle band on the water just under the causeway.

Deliberate deviations from the reference, agreed up front:
- The six walking figures and the squirrel are OUT.
- The causeway FLOATS about 0.9 m (3 ft) clear of the water.
- The lens starburst is faked in post and is explicitly low priority. General
  scene read is what matters.

Read `SCENE_FROM_REFERENCE.md` first for pipeline/CLI, `context.md` for renderer
internals. This file is only the scene-specific numbers.

---

## 1. Camera fit (measured, not guessed)

Measured off the reference (1921x819) by luminance profiling, mean over x in
[0.05, 0.45] which is clear of the tree and the bush:

| feature | image row | y fraction |
|---|---|---|
| hazy sky above the ridge | 450-483 | 0.549-0.590 |
| ridge band (falls 0.50 to 0.17 linear) | 484-518 | 0.591-0.633 |
| **horizon (ridge base / waterline)** | **518.5** | **0.6331** |
| bright water beyond the causeway | 519-531 | 0.634-0.648 |
| causeway top edge | 534 | 0.6520 |
| causeway waterline | 552 | 0.6740 |
| causeway reflection fading out | 553-580 | 0.675-0.708 |

Roll fit on the causeway top edge came out 0.177 deg. Treat as zero.

**The fov-independent part.** Depression angles are ratios of pixel offsets, so
these two relations hold whatever focal length is chosen:

    tan(a_top)/tan(a_base) = 14/33.5 = 0.4619   =>  z_causeway = 0.538 * h_cam
    tallest figure head sits 112 px above horizon =>  z_figure  = 3.79 * h_cam

With a 1.70 m adult that gives h_cam = 0.45 m and z_causeway = 0.24 m. So the
reference camera is essentially at water level and the causeway is a very low
kerb. That is why it reads as a razor-thin line.

**Why we cannot keep that.** The causeway top crosses the horizon when
z_top = h_cam. At h_cam = 0.45 m the maximum possible float is 0.45 m, and past
that the slab occludes the ridge and the shot is ruined. Floating it a few feet
therefore requires raising the camera.

**Chosen camera.** Keep the reference's horizon fraction and causeway silhouette
position exactly, raise the camera to natural eye height, and solve for the
distance that keeps the slab face at the reference's 14 px:

| parameter | value |
|---|---|
| render | 1920 x 818 (aspect 2.347) |
| fov (Nori horizontal) | 60 deg -> fov_v 27.62 deg |
| C = (H/2)/tan(fov_v/2) | 1662.7 px/rad |
| horizon | 63.31% down frame = row 517.9 |
| camera pitch | 3.749 deg down |
| Blender rotation_euler | (86.251, 0, 0) deg, looking down +Y |
| camera location | (0, 0, 1.60) with water plane at z = 0 |
| causeway distance | 41.6 m |
| causeway underside | 0.90 m above water |
| slab thickness | 0.35 m, so top at 1.25 m |

Resulting screen geometry: slab top row 532.5, slab underside row 546.5 (a 14 px
face, matching the reference), then a 36 px gap of open water and sky, then the
water directly beneath the slab at row 582.5. The gap is the whole point of the
float and it is comfortably readable.

**fov is a choice, not a measurement.** Same conclusion as pool_store. The
angles above are fixed by the image; picking fov fixes C, and picking h_cam then
fixes every distance. If you change fov, re-derive the distance table, and
remember the water grid is built from the camera so it must be regenerated too.

---

## 2. Sun

Sun centroid measured at row 139, col 1476.

| parameter | value |
|---|---|
| elevation | 12.85 deg |
| azimuth, right of view | 17.21 deg |
| Blender azimuth (view along +Y) | 72.79 deg |
| `sun_disk.py` args | SUN_ELEV 12.85, SUN_AZIM 72.79 |

The perfect-mirror image of the sun lands at row 898, which is off the bottom of
an 818-row frame. So **every bright thing in the visible water is glitter off
tilted facets, not the mirror sun.** That is the single fact the water design
hangs on.

**Make the sun disk FAT here: 1.5 to 2.0 deg angular diameter, not 0.53 deg.**
Reasoning, and it is the opposite of the crosswalk call:
- Glitter off a specular dielectric comes only from BSDF-sampled rays that
  happen to land on the disk. NEE does not help on a delta BSDF. A 0.53 deg disk
  is 6.7e-5 sr, so glitter variance is brutal.
- Widening the disk to 2 deg is a 14x larger target and cuts that variance hard.
- Nothing in this scene needs a crisp penumbra. The only cast shadow is the
  causeway's, thrown 3.9 m onto water at 41.6 m; a 2 deg sun gives 13.6 cm of
  penumbra there, which is 5 px. Invisible.

Keep total flux constant when resizing, per the san_miguel lesson: radiance
scales as 1/theta^2. Re-run `sun_disk.py` and paste the new radiance into the
injected SUN_XML any time the angles or the size change.

---

## 3. Sky HDRI

The cloud field is roughly a third of the frame and it appears a second time
mirrored in the water, so the HDRI carries double weight. Pick on cloud
character above all else.

Wanted: mackerel / altocumulus rows, bright hazy horizon, low sun. Fetch through
the Blender Poly Haven addon; direct `dl.polyhaven.org` is 403 from both the
device VM and the cloud container.

Then run `tools/scene_build/causeway_sky.py` (to write, model it on
`sky_build.py` and `crosswalk_sky.py`):
1. Strip the HDRI's own sun by capping luminance at about p99, so the geometric
   disk is not double counted.
2. **Paint the ridge into the HDR** rather than modelling it. The reference ridge
   is nearly featureless (sRGB mean 0.608/0.654/0.689, p50 0.42 linear) and pure
   aerial perspective. Painting it means it is perfectly flat, correctly hazed,
   free of geometry, and it reflects in the water for free at the correct mirror
   angle. Band from the horizon up to about 2.4 deg elevation with a soft
   irregular top edge.
3. Haze ramp toward the horizon, as in liminal_bed.
4. Normalise sky horizontal irradiance to 1.0 and set exposure with
   `evCompensation`. Pin `envmapScale` at 1.0 so the sun-to-sky ratio does not
   drift.

Sun-to-sky ratio: do NOT reuse crosswalk's 28.6:1. That was a clear day with the
sun at 62 deg. A 13 deg sun runs through several times the air mass and this
reference is visibly hazy, so start nearer 10:1 and solve it offline against the
EXR the way crosswalk did rather than re-rendering per guess.

Geometry fallback if a painted ridge does not sit right: a displaced strip at
3-8 km. `homogeneous` IS registered and falls back to the scene bbox when bounds
are omitted, so a global atmosphere is plausible, but I have not verified the
DXR path treats it as an exterior medium. Do not spend time there unless the
painted ridge fails.

---

## 4. The water

This is the long pole. Everything else in the scene is a day's work; this is
where the iterations go.

### 4.1 The constraint

`dielectric` (src/cpu/dielectric.cpp) is ideal-smooth with no roughness and no
texture inputs. There is no slope-variance-to-roughness handoff to hide behind,
so every ripple must be real geometry, and the mesh must span 8.9 m to about
2.7 km of water.

### 4.2 Use a projected grid, not a world grid

At grazing incidence the screen footprint of a horizontal plane is wildly
anisotropic. For a patch at distance d:

    lateral, 1 px:     dx = d / C
    along-view, 1 px:  dd = d^2 / (C * h)

Along-view spacing grows as d^2. A uniform world grid is therefore hopeless: it
is millions of wasted quads in the far field and still too coarse near the
horizon. Build the grid in SCREEN space instead, uniform in image (x, y) over
the water region, project each vertex through the camera onto z = 0, then
displace vertically by the wave field at the resulting world position. Cost then
scales with the water's IMAGE area, not its world area.

Distance mapping for the chosen camera, y_px measured below the horizon:

    d = C * h / y_px = 2660.3 / y_px

| y_px below horizon | distance |
|---|---|
| 1 | 2660 m |
| 10 | 266 m |
| 40 | 66.5 m |
| 64 (causeway waterline) | 41.6 m |
| 150 | 17.7 m |
| 300.1 (bottom of frame) | 8.86 m |

### 4.3 The schedule

Screen density 2.5 px per quad laterally. Along-view, take the MINIMUM of the
screen-uniform spacing and a 45 mm ripple-resolving cap, out to a cutoff
D_fine = 70 m, then let it relax back to screen-uniform.

The cap binds everywhere in frame: screen-uniform along-view spacing at the
nearest visible water (8.86 m) is already 94 mm, coarser than the ripples need.

| segment | spacing | rows |
|---|---|---|
| 8.86 m -> 70 m | 45 mm fixed | 1359 |
| 70 m -> 2660 m | screen-uniform, 2.5 px | 15 |
| **total rows** | | **1374** |
| columns | 1920 / 2.5 | 768 |

**1.06 M quads = 2.11 M tris, about 83 MB of OBJ.** The far field costing only
15 rows is the whole payoff of the projected grid: 2.6 km of water for 11k
quads.

Why D_fine = 70 m is enough: the reference's sparkle band sits at 38 to 53 m
(section 7), and 70 m covers it with margin. Beyond that the water is a smooth
Fresnel mirror, which is exactly what the reference shows.

### 4.4 Wave spectrum

Calm lake, light air. Target rms slope about 2.7 deg with p99 near 8 deg, which
is roughly half of pool_store's 5.2 / 12.2 because this water is glassy.
Directions are relative to the view-to-sun bearing, since slope toward the sun
is what glitters.

| lambda (m) | amplitude (mm) | direction (deg) |
|---|---|---|
| 2.40 | 9.0 | +8 |
| 0.90 | 4.5 | -12 |
| 0.35 | 2.2 | +5 |
| 0.20 | 1.0 | -20 |
| value noise 0.55 | 1.8 | isotropic |
| value noise 0.22 | 0.7 | isotropic |

That sums to rms slope 2.73 deg. The 2.40 m swell carries the gentle reflection
wobble in the near field; the 0.20 to 0.35 m terms carry the sparkle.

Facet tilt needed for glitter at distance d is `(12.85 - atan(h/d)) / 2` in
degrees, which is 5.2 deg at 38 m and 5.6 deg at 53 m. A 0.30 m wave at 4.5 mm
reaches that, and 45 mm spacing resolves it at 6.7 samples per wavelength.

**Taper the short wavelengths out, do not cut them.** Terms shorter than about
4x the local along-view spacing will alias. Ramp the 0.20 and 0.35 m amplitudes
to zero over d in [55, 70] m so the far field is pure long swell and there is no
visible seam at D_fine.

### 4.5 Two diagnostics worth memorising

- The sparkle band's screen POSITION is set by sun elevation and camera height
  only. Wave amplitude sets its width and density, never its position. Band in
  the wrong place means the sun elevation is wrong; band in the right place but
  too weak means scale all amplitudes together (slope is linear in amplitude).
- The near water darkening toward the bottom of frame is Fresnel falling off as
  incidence steepens, and it is automatic. Do not chase it with materials.

### 4.6 Material and what is underneath

A dielectric with nothing behind it refracts into the envmap's lower hemisphere
and you get the liminal_bed navy band back. So:

| element | setting |
|---|---|
| water BSDF | `dielectric`, intIOR 1.33, tintColor about (0.35, 0.55, 0.62) |
| lake bed | flat plane 3.5 m down, `disney` baseColor about (0.02, 0.03, 0.03), roughness 0.9 |

Reference near water measures sRGB (0.397, 0.506, 0.591), p50 0.21 linear, which
is mostly sky reflected at about 80 deg incidence plus near-black transmission.

### 4.7 Shading normals: the risk to watch

Write the water OBJ with SHARED vertices and no `vn`, so obj.cpp's angle-weighted
fallback smooths it. Smooth normals are correct here and give better glitter than
flat facets, because faceting quantises the slope distribution.

The risk: interpolated shading normals on a specular surface at 88 deg incidence
can point below the geometric surface and produce dark speckle. If the far water
shows black pepper, that is the cause, not sampling noise. Fix by flat-shading
beyond D_fine, not by adding samples.

### 4.8 Generate it outside Blender

1.06 M quads must not go through Blender or the exporter. Follow the
`grass_geometry.py` pattern exactly: `tools/scene_build/causeway_water.py` reads
the camera from `scene.xml`, builds the grid in numpy, writes the OBJ, and its
`<mesh>` block is injected via `NORI_MEDIUM_XML` alongside the sun disk block so
it survives re-export.

Write to VM-local `$HOME` first, then copy to the mount in a size-checked
resumable loop. The mount writes at 4.4 MB/s, so 83 MB is about 20 s and needs
the 180 s call budget.

**Any camera move invalidates the water mesh.** Regenerate it.

---

## 5. Foliage

**Canopy, top-right.** Use `assets/crosswalk/tree_oak.blend`, the PLAIN one, not
`tree_oak_fine.blend`. Crosswalk needed 6-8 cm leaves in the foreground; here the
canopy leaves read at 15-40 px, so the stock 12 cm cards at about 8 m are right
and it saves 200k faces per instance. One instance, trunk near the right frame
edge with a branch entering the frame, canopy filling the top-right corner.

Set `translucency` 0.35 via the `nori_translucency` custom property. That lobe
was added for exactly this backlit read. Remember it is not a brightness knob:
raising it darkens the canopy because the reflection lobe scales by (1 - T).

The sun sitting visibly in the canopy gaps is automatic from the alpha cutouts.
The alpha test lives in `shaders/RayQueryTrace.hlsli` and reads `.a`, already
fixed.

**Shrub mass, bottom-right.** Big soft mass of small backlit leaves. First
candidate is the Sketchfab "Lilac bush pack"
(`10312697ec994fc99355cb94f1963a2e`, CC-BY), already known good from
spirited_away. 3 to 5 instances at mixed scale and rotation so nothing tiles.
Same translucency treatment.

Asset gotchas that have each cost a cycle before: the Sketchfab importer deletes
its temp texture folder immediately, so `img.save()` the packed datablocks out to
real paths before anything else; and never set `root.scale` on a Sketchfab import
because it overwrites the importer's size normalisation.

BlenderKit had no `api_key` set as of liminal_bed. If you want it, log in first.

---

## 6. Causeway

Trivial geometry and the easiest thing in the scene. Floating it is easier than
the reference because there is no waterline contact and no wet-stone transition
to fake.

| parameter | value |
|---|---|
| distance | 41.6 m |
| underside | 0.90 m above water |
| slab thickness | 0.35 m |
| top width | 2.0 m |
| lateral extent | x -60 to +60 m |
| yaw | about 6 deg so the right end recedes slightly |
| blocks | about 1 m each, roughly 40 px on screen, joints visible |

The yaw is loosely fitted. The measured top-edge slope of 5.9 px across the frame
is within the noise of my edge detection, so tune it by eye; the reference reads
as within a few degrees of perpendicular.

Free bonus from the float: the sun at 12.85 deg throws a 3.9 m shadow from the
slab toward the camera, landing on the water just below it. That shadow strip is
the strongest possible cue that the thing is airborne, and it costs nothing.

---

## 7. Verification targets, measured off the reference

Compare row bands on the linear EXR through her exact display model:
`display = ACES_narkowicz(L * 2^ev)^(1/2.2)`. Not the sRGB piecewise curve.

### Water rows (x in [0.06, 0.62], display units)

| y frac | distance at h=1.60 | p50 | p99 | frac > 0.75 |
|---|---|---|---|---|
| 0.645 | 280 m | 0.237 | 0.599 | 0.030 |
| 0.670 | 90 m (causeway reflection) | 0.055 | 0.280 | 0.000 |
| 0.695 | 52.6 m | 0.414 | 0.650 | **0.124** |
| 0.720 | 37.7 m | 0.436 | 0.641 | **0.061** |
| 0.745 | 29.0 m | 0.384 | 0.557 | 0.007 |
| 0.795 | 20.2 m | 0.298 | 0.508 | 0.003 |
| 0.845 | 15.2 m | 0.217 | 0.505 | 0.001 |
| 0.895 | 12.6 m | 0.152 | 0.417 | 0.000 |
| 0.945 | 10.3 m | 0.129 | 0.386 | 0.000 |
| 0.995 | 9.1 m | 0.104 | 0.346 | 0.000 |

The sparkle band is rows 560 to 600 and essentially nothing below row 620. That
is the primary water check.

### Region means (sRGB) and linear percentiles

| region | sRGB mean | L p50 |
|---|---|---|
| sky upper | 0.381 / 0.548 / 0.682 | 0.239 |
| ridge | 0.608 / 0.654 / 0.689 | 0.425 |
| water near | 0.397 / 0.506 / 0.591 | 0.208 |
| canopy top-right | 0.336 / 0.346 / 0.239 | 0.052 |
| shrub bottom-right | 0.281 / 0.286 / 0.157 | 0.038 |
| causeway | 0.392 / 0.417 / 0.432 | 0.105 |

Frame fraction above L = 0.6 is 0.137. Bright and high-key, but nothing like the
blown-out crosswalk road, so exposure should land in one or two sweeps.

### Water texture, high-pass std at sigma 0.7 / 1.5 / 3.0 px, 8-bit

| band | values |
|---|---|
| near, y 0.86-0.99 | 2.86 / 6.83 / 10.54 |
| mid, y 0.75-0.82 | 5.33 / 10.47 / 13.54 |
| sparkle, y 0.655-0.70 | 6.79 / 12.02 / 14.76 |

Caveat: these include reflected cloud structure, not only ripple, so treat them
as an upper bound rather than a ripple target.

---

## 8. Budget and render settings

| element | tris |
|---|---|
| water | 2.11 M |
| oak, 1 instance | 1.37 M |
| shrub mass, 3-5 instances | 0.8 M |
| causeway | 0.02 M |
| lake bed, sun disk | negligible |
| **total** | **about 4.3 M** |

Comfortably under crosswalk's proven 6.2 M on the RTX A3000 12 GB.

Export globals: `NORI_YUP` True, `NORI_NORMALS` True, `NORI_AREALIGHT` False,
`NORI_ENV` sky.hdr, `NORI_ENVSCALE` 1.0, `NORI_W` 1920, `NORI_H` 818,
`NORI_MEDIUM_XML` carrying the sun disk block AND the water mesh block.

The exporter always writes `evCompensation` 0.0 and never writes bloom, so `sed`
both into `scene.xml` after every export. Bloom is what gives the sun its halo;
start from crosswalk's 0.10 / threshold 3.0 / knee 0.8.

Render:

    .\build\Release\nori-dxr.exe .\scenes\floating_causeway\scene.xml --headless --denoise --png

Launch it from Blender's Python with `subprocess.Popen`, `cwd` at the repo root,
and poll the log. Scaling from crosswalk (6.2 M tris, 1.08 M px, 256 spp, 11 min)
this is about 16 min on pixel count alone, and the specular water will push it
past that.

**Thermal.** The laptop hard-crashes under sustained GPU load and has corrupted a
.blend that way before. Do the composition passes at 64 spp with `--denoise`, and
only go to 512+ once the sparkle band and exposure are settled. Save incrementally
to new `.blend` filenames.

---

## 9. Build order

Cheap things that define the composition first, the risky thing after.

1. **Sky.** Pick the HDRI, strip its sun, paint the ridge, haze ramp, normalise.
   Nothing else can be judged until the horizon reads.
2. **Blockout.** Causeway, camera, lake bed, flat water plane, no ripples, no
   foliage. Export, render 64 spp. Check the horizon at 63.31%, the slab top at
   row 532, the 36 px float gap, and solve exposure offline.
3. **Water.** `causeway_water.py`, inject, render. Check the sparkle band lands
   at rows 560-600. This is where the iterations live.
4. **Foliage.** Oak first, then the shrub mass. Check the canopy p50 against
   0.052.
5. **Match pass** against the section 7 tables.
6. **Starburst.** Last, and minimal.

## 10. Scripts to write

| path | purpose |
|---|---|
| `tools/scene_build/causeway_build.py` | whole Blender scene, parametric, constants at top |
| `tools/scene_build/causeway_sky.py` | HDR: strip sun, paint ridge, haze, normalise |
| `tools/scene_build/causeway_water.py` | numpy projected-grid water OBJ, reads camera from scene.xml |
| `tools/post/starburst.py` | flare fake plus final PNG through the ACES model |
| reuse `tools/scene_build/sun_disk.py` | SUN_ELEV 12.85, SUN_AZIM 72.79, disk 1.5-2.0 deg |

Write patched files via a temp path plus rename. `crosswalk_build.py` was
truncated to 0 bytes once because `io.open(..., "w")` truncates before raising on
a bad `newline=` argument.

## 11. Starburst, minimal

Not renderable. The six-point star is aperture diffraction and a pinhole path
tracer cannot make it; bloom only gives the round halo.

Cheapest acceptable fake, about 20 lines in `tools/post/starburst.py`: threshold
the sun pixels in the linear EXR, convolve with a 6-armed streak kernel, add it
back BEFORE the ACES step so it tonemaps like real flare, then write the PNG
through the exact display model. Adding it pre-tonemap is what keeps it from
looking pasted on.

## 12. Accepted compromises

- No Poly Haven HDRI will match the reference's exact cloud field. Pick on
  character and accept it. It shows twice, in the sky and mirrored in the water.
- The far water beyond 70 m is a smooth mirror with no discrete sparkle. The
  reference's own sparkle stops well short of there, so this should not show.
- The ridge is painted into the HDR, so it has no parallax. At 3 km plus, there
  is none to have.
- Raising the camera from the reference's 0.45 m to 1.60 m was forced by the
  float. Nothing in frame gives the scale away once the figures are gone, so the
  image reads the same.
