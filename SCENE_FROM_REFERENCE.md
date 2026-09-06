# Recreating a render from a reference image with nori-dxr: fresh-session context

Read this top to bottom before touching anything. It is the handoff for the workflow
"Grace gives a reference photo, Claude rebuilds it as a Blender scene, exports it to her
nori-dxr path tracer, and iterates against the reference until it matches". It was written
after the `scenes/liminal_bed` project (a bed hanging from a cloud over a lawn), which is the
worked example throughout. `context.md` next to this file is the renderer-internals handoff
(SSS, perf); this file is the scene-building one.

Owner: Grace (gsj33, Cornell CS, systems + graphics). Terse, targeted feedback, usually one or
two observations per round ("the blanket looks bubbly", "shadow at a different angle"). She
judges the DXR render against the reference, not the Blender preview. No em dashes in prose.

---

## 1. The setup: what is connected and where things run

### 1.1 Machines and paths
- Her laptop: Windows, Blender 5.1.1, RTX A3000 (two GPUs, thermally limited). The repo is
  `C:\Users\gjin3\Desktop\nori-26sp` (the `.blend` files live on the Desktop, e.g.
  `C:\Users\gjin3\Desktop\liminal.blend`). The repo is usually open in VS Code on the laptop.
- The session runs in a cloud Linux container and is linked to the laptop through the Claude
  desktop app. The repo folder is connected and mounted in a small Linux VM on her machine at
  `~/mnt/nori-26sp` (reachable with `device_bash`). Three places code can run:

| Where | Tool | Use it for |
|---|---|---|
| Blender's own Python (Windows) | `blender__execute_blender_code` | everything in Blender, AND launching Windows exes (`nori-dxr.exe`, `dxc.exe`) with `subprocess` |
| Linux VM on her laptop | `device_bash` | numpy/cv2 scripts over the repo, file ops on the mount, polling logs, md5sum |
| Cloud container | `Bash` | heavy numpy/cv2 on staged copies, image comparisons, anything that needs pip |

- Moving files: `device_stage_files` copies laptop -> cloud (`/mnt/user-data/uploads/nori-26sp/...`),
  `device_commit_files` copies cloud -> laptop (stagedPath must be under `/mnt/user-data/outputs/`,
  20 MB per file limit, `force: true` to overwrite).
- VS Code: no VS Code tool surfaced in the session that wrote this. Edit files through the mount
  (`device_bash`, `sed -i`, python read-modify-write) or through Blender's Python; she sees the
  changes live in VS Code. If a VS Code / IDE tool is present in your session, prefer it for editing.

### 1.2 Blender MCP
- Addon "Blender MCP" v1.5 (protocol 4; server expects 5, works via fallbacks; update with
  `uvx blender-mcp install-addon` then restart Blender). Server on port 9876. Known trap: a second
  copy of the addon in `scripts/addons` (addon.py + blender_mcp.py) makes both bind 9876 and the
  handshake fails (WinError 10054/10053). Keep exactly one enabled, restart Blender, click
  "Connect to Claude".
- Tools: `execute_blender_code` (arbitrary bpy, this is 95% of the work), `get_scene_info`,
  `get_object_info`, `get_viewport_screenshot`, plus the asset integrations:
  - PolyHaven: enabled. `search_polyhaven_assets` / `download_polyhaven_asset` for HDRIs, textures,
    models. Use this for any HDRI: direct downloads from dl.polyhaven.org are blocked (403) from both
    the cloud and the device VM.
  - Sketchfab: enabled, logged in as gracejin. `search_sketchfab_models` / `download_sketchfab_model`
    (downloadable models only; quality varies, check the preview and triangle count first).
  - BlenderKit: installed as a Blender extension (`bl_ext.user_default.blenderkit`, importable from
    bpy) but has NO API key set, so nothing can be downloaded until she logs in inside Blender.
  - Hyper3D Rodin: disabled (checkbox in the BlenderMCP N-panel). Hunyuan3D / Poly Pizza: present as
    tools, status unverified.
  - Every MCP tool takes `user_prompt`: pass Grace's latest request verbatim.
- Timeouts: an MCP call returns after ~60 s but Blender keeps running the code. For renders,
  exports and subprocess launches, write a marker/log file and poll it from `device_bash`.
  `device_bash` calls cap at 180 s; background processes started there die when the call returns.
  Children of Blender's Python (`subprocess.Popen`) survive.

### 1.3 File-system gotchas (each one cost time)
- `rm` is blocked on the mount. `mv` into a `_to_delete/` or `_intermediate/` folder instead.
- Writing to the mount from the VM is ~4.4 MB/s; write big outputs to `$HOME` in the VM first,
  then `cp` to the mount in a separate call with `timeout_ms: 180000` (148 MB takes ~25 s).
- `device_commit_files` served a STALE copy when the same `stagedPath` was committed twice with new
  content. Give every new build a fresh filename under `/mnt/user-data/outputs/` and `md5sum` both
  sides.
- Files over 20 MB cannot be committed; edit those in place (a 26 MB PNG texture was rescaled inside
  Blender through `image.pixels` and saved with `image.save()`).
- Reading EXRs in the cloud: `OPENCV_IO_ENABLE_OPENEXR=1` before importing cv2.
- Blender saves nothing on its own. `bpy.ops.wm.save_mainfile()` after every builder run.

---

## 2. The renderer: nori-dxr

DirectX 12 / DXR megakernel path tracer (HLSL) grown out of Cornell CS5630's Nori. Progressive
accumulator (1 spp per frame), NEE + MIS, Disney/microfacet/dielectric/mirror/diffuse/hair BSDFs,
random-walk SSS, heterogeneous volumes, IBL, textures + normal maps, thin-lens DoF, OIDN denoiser,
Owen-Sobol sampler. `context.md` and `README.md` cover the internals.

### 2.1 Running it (from Blender's Python, no need to ask Grace)
```python
import subprocess, os
repo = r"C:\Users\gjin3\Desktop\nori-26sp"
log = open(os.path.join(repo, "_dxr_render.log"), "w")
p = subprocess.Popen([os.path.join(repo, r"build\Release\nori-dxr.exe"),
                      r"scenes\<name>\scene.xml", "--headless", "--denoise", "--png"],
                     cwd=repo, stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000)
```
- `cwd` must be the repo root (`build/Release/Shaders.cso` is found relative to it).
- Poll: `cat ~/mnt/nori-26sp/_dxr_render.log | tr '\r' '\n' | tail` from `device_bash`.
- Headless renders to the sampler's `sampleCount` and writes into the scene folder:
  `snapshot_N.exr` (linear radiance, no bloom), `snapshot_N_albedo.exr`, `snapshot_N_normal.exr`,
  `snapshot_N.png` (display image: exposure, bloom, ACES, gamma), and with `--denoise`
  `snapshot_N_denoised.exr` + `snapshot_N_denoised.png`. OIDN runs on CUDA, ~0.2 s.
- liminal_bed timing: ~80 s loading (two 148 MB grass OBJs), 256 spp at 1000x1250 in 83 s
  (3.1 spp/s, cloud volume + 5.6M triangles), so a 1024 spp render is about 7 minutes total.
- Windowed mode (no `--headless`) is interactive: P snapshot, O EXR, N denoise, M auto-denoise.
- Shaders are prebuilt. After editing HLSL recompile from Blender's Python the same way:
  `"C:\VulkanSDK\1.4.341.1\Bin\dxc.exe" -T lib_6_5 -Fo build\Release\Shaders.cso shaders\Shaders.hlsl`
  (`-D MAX_BOUNCES=n` overrides the default 32). Primary rays go through the inline RayQuery path
  (`shaders/RayQueryTrace.hlsli`), not the any-hit shaders; fix alpha/visibility logic there.

### 2.2 Scene format (Nori XML), annotated
```xml
<scene>
  <string name="envmap" value="sky.hdr"/>          <!-- equirect HDR, path relative to scene.xml -->
  <float name="envmapScale" value="1.0"/>          <!-- IBL multiplier -->
  <float name="evCompensation" value="0.0"/>       <!-- display exposure, stops (pow(2, ev)) -->
  <float name="envmapRotation" value="0.0"/>       <!-- radians, added to phi: spins sun+sky by one number -->
  <medium type="heterogeneous">                    <!-- one medium per scene, dense VOL1 grid -->
    <string name="volume" value="volumes/cloud.vol"/>
    <float name="sigmaA" value="0"/> <float name="sigmaS" value="110"/> <float name="g" value="0.877"/>
    <point name="boundsMin" value="-1.888, 2.700, -2.124"/> <point name="boundsMax" value="1.888, 4.600, 2.124"/>
  </medium>
  <camera type="perspective">
    <float name="fov" value="41.508"/>             <!-- HORIZONTAL fov in degrees; aspect from width/height -->
    <transform name="toWorld"><lookat target="..." origin="..." up="..."/></transform>
    <integer name="width" value="1000"/> <integer name="height" value="1250"/>
  </camera>
  <sampler type="independent"><integer name="sampleCount" value="256"/></sampler>
  <mesh type="obj">
    <string name="filename" value="meshes/Duvet.obj"/>
    <bsdf type="disney">                           <!-- maps 1:1 to Blender's Principled BSDF -->
      <color name="baseColor" value="0.54 0.60 0.72"/>
      <float name="roughness" value="0.95"/> <float name="metallic" value="0"/> <float name="specular" value="0.15"/>
      <float name="sheen" value="0.55"/> <float name="clearcoat" value="0"/> <float name="anisotropic" value="0"/>
      <float name="subsurface" value="0"/>
      <string name="albedoTexture" value="textures/duvet_albedo.png"/>   <!-- REPLACES baseColor, read as sRGB -->
      <string name="normalTexture" value="textures/duvet_normal.png"/>   <!-- tangent space, UNORM, full strength -->
    </bsdf>
  </mesh>
  <mesh type="obj"><string name="filename" value="meshes/lamp.obj"/>
    <emitter type="area"><color name="radiance" value="5 5 5"/></emitter></mesh>
</scene>
```
Conventions that bite:
- World is Y-UP (envmap and camera). Blender is Z-up; the exporter converts with `NORI_YUP`
  ((x, y, z) -> (x, z, -y)). Anything you compute by hand for scene.xml (bounds, lookat) needs the
  same swap.
- Envmap lookup: u = atan2(d.z, d.x) / 2pi, v = acos(d.y) / pi. Against Blender's equirect this
  is exactly 180 degrees off in azimuth, so set the World Mapping node Rotation Z = pi in Blender
  or every preview has the sun on the wrong side. For a sun at Blender azimuth az (deg, atan2(y,x))
  and elevation el: u = (-az/360) mod 1, v = (90 - el)/180.
- Display transform (`shaders/Resolve.hlsl`): exposure, bloom, ACES (Hill 2016 fit), gamma 2.2.
  Any preview you judge must go through the same curve (section 4.3).
- No `<integrator>` needed (the GPU build ignores it). No .mtl parsing: materials are inline XML.
- Other BSDFs: `dielectric` (glass), `mirror`, `microfacet`, `diffuse`, `subsurface`, `hair_bsdf`.
  Disney extras: specularTint, sheenTint, clearcoatGloss, roughness/metallic/specular/subsurface/
  alpha textures (`alphaTexture` reads the PNG alpha channel for cutouts).
- Volumes: VOL1 dense grid, one medium per scene, `boundsMin/Max` place it. Density scales with
  sigmaS; with MAX_BOUNCES 32 a thick pure-scattering cloud goes grey (sigmaS 110 was the ceiling
  for a 4 m cloud, 140 went grey). Writers: `_disney_cloud_to_vol.py` (NanoVDB -> VOL1, mean-pooled
  downsample), `_pbrt_bunnycloud_to_nori.py`. Pick the grid resolution from the cloud's PIXEL size
  in the final frame (aim for >= 2 voxels per pixel), not from file size.
- OBJ loader computes angle-weighted vertex normals if `vn` is absent; the exporter writes none.

---

## 3. The exporter and the helper scripts (repo root, all `_`-prefixed)

### 3.1 `_blender_to_nori.py`: Blender scene -> `scenes/<name>/`
Run inside Blender with globals, never as a file argument:
```python
scn = bpy.context.scene; scn.camera = bpy.data.objects["Camera"]
xml = open(r"C:\Users\gjin3\Desktop\nori-26sp\scenes\<name>\scene.xml").read()   # if it exists
MED = xml[xml.index("<medium"):xml.index("<camera")].rstrip()                     # keep hand-written blocks
g = {"NORI_OUT": "<name>", "NORI_SAMPLES": 256, "NORI_ENV": "sky.hdr", "NORI_ENVSCALE": 1.0,
     "NORI_AREALIGHT": False, "NORI_W": 1000, "NORI_H": 1250, "NORI_YUP": True,
     "NORI_SKIP": {"CloudVolume", "_dcam", "_dtgt"}, "NORI_MEDIUM_XML": MED}
exec(open(r"C:\Users\gjin3\Desktop\nori-26sp\_blender_to_nori.py").read(), g)
```
What it does: one triangulated world-space OBJ per visible mesh object (modifiers applied via
the evaluated depsgraph), Principled BSDF -> disney, Emission -> area emitter, active camera ->
lookat + true horizontal fov for any aspect, image textures copied to `textures/` (albedo and
normal, only when the Base Color socket has an Image node; per-corner `v/vt` OBJ, which triples
the file size), `NORI_MEDIUM_XML` injected verbatim (this is how the cloud medium and the grass
meshes survive re-exports; the exporter knows nothing about either).
What it does not do: lamps, the Normal Map node's Strength (bake it into the PNG and keep Blender
at 1.0 so Cycles predicts DXR), roughness/metallic maps, vertex normals, instancing. Duvet-scale
meshes (490k tris with UVs) export at ~100 MB; the whole liminal_bed export is 17 s.

### 3.2 Helper index
- `_sky_build.py <src.hdr> <dst.hdr> SUN_SCALE ZEN_FRAC BACK_R,G,B VIS_R,G,B SUN_R,G,B SUN_ELEV SUN_SIG SUN_AZ HAZE`
  Rebuilds a sky HDR from a source panorama: removes the source sun (Gaussian fit), applies
  per-channel gains to the wedge the camera sees (`U_CAM`, `VIS_HALF` constants at the top) versus
  the unseen hemisphere (the fill light), re-paints a Gaussian sun at any azimuth/elevation with the
  same total energy, adds a horizon haze ramp and continues the sky below the horizon (a terrain
  crest below eye level otherwise shows the panorama's dark ground band). Runs in the cloud in a
  second; commit the result under a new filename.
- `_grass_geometry.py` (device VM): scatters ~1.2M curved blade quads onto the exported
  `meshes/Ground.obj`, view-culled and 1/d^2 density from the camera in scene.xml, split into two
  OBJs by mow band with opposite lean. Env knobs `GRASS_SCENE GRASS_OUTDIR GRASS_D0 GRASS_DREF
  GRASS_MAXDIST GRASS_H GRASS_HVAR GRASS_BEND GRASS_CAP GRASS_BAND GRASS_SEED`. Re-run after ANY
  ground or camera change: `cd ~/mnt/nori-26sp && GRASS_OUTDIR=$HOME python3 _grass_geometry.py`
  (~25 s), then cp `grass_a.obj`/`grass_b.obj` into `meshes/` one per call.
- `_duvet_build.py`, `_pillow_build.py`: procedural bedding as thick heightfield sheets (see the
  memory notes for the design rules; the short version is in section 6).
- `_disney_cloud_to_vol.py`: NanoVDB -> VOL1. `_pbrt_*_to_nori.py`: pbrt-v4 scene converters.
  `_obj_mtl_to_nori.py` / `_sm_scene.py`: OBJ+MTL -> Nori without Blender (San Miguel), with
  keyword detectors for metal/fabric/glass materials. `_exrtool.py`: EXR utilities.
- Scratch-only helpers that were not saved in the repo and are trivial to rewrite: the ACES
  tonemap snippet (section 4.3), a fabric albedo/normal generator, patch measurers.

---

## 4. The playbook: from reference image to matching render

The method that converged is "measure, fit, verify" with numbers, never eyeballing. Every round
ends with a side-by-side (reference left, ours right, same height) plus a close-up crop of the
thing that changed, sent with `SendUserFile`.

### 4.1 Read the reference quantitatively (before opening Blender)
Save the attachment as a PNG in the cloud scratchpad and measure with PIL/numpy:
- Frame: aspect ratio, where the horizon crosses the left and right edges (as % of height), its
  slope, where the subject sits (top/bottom/left/right as % of frame).
- Light: shadow direction and length relative to the subject (the tip of the shadow, the band it
  makes at a frame edge), the lit/shaded split on the subject, sky gradient top -> horizon and
  left -> right (the sun side is paler).
- Colour: 8x8 patch means in sRGB for sky top, sky at horizon, subject lit, subject shaded, ground
  lit, ground shadow, any secondary object. Convert to linear when comparing ratios.
- Texture: high-pass std of a flat lit patch at sigma 0.7 / 1.5 / 3 px tells you how much fine
  variation the material carries (the reference bedding had 7 to 9x more than our first attempt).
- Note what the reference is: many "photos" in this aesthetic are 3D composites, so shadow, cloud
  lighting and horizon may not be mutually consistent. Match what is visible, not physics.

### 4.2 Block out the scene in Blender
- Camera from the numbers: distance and height above the subject from the subject's size in frame,
  tilt from the horizon height (eye level is where a flat ground's horizon lands; a horizon below
  that means the terrain falls away, above it means it rises), lens from the subject's width. Put
  the reference's horizon and subject at the same % of the frame. Verify by projecting object
  vertices with `bpy_extras.object_utils.world_to_camera_view` (normalized, y up) or with numpy:
  camera-space `u = -x/z, v = -y/z`, then `X = u*f/sens*(ry/rx) + 0.5`, `Y = v*f/sens + 0.5` for a
  portrait frame with sensor fit AUTO (the 36 mm sensor spans the larger image dimension).
- Ground: a plane is never enough. Build the terrain in numpy and write vertices back through the
  inverse `matrix_world`. For a horizon that slopes, rotate about the view axis; for a horizon below
  eye level, rotate the terrain about the screen-x axis through the subject so it descends away
  from the camera; keep the ground height under the subject fixed and check it by ray casting
  (`ground.ray_cast` in object space). The horizon line is the max projected Y per column of the
  ground vertices, no render needed.
- Sky: a PolyHaven pure-sky HDRI through the addon, then rebuilt with `_sky_build.py`. Set the
  World Mapping Rotation Z = pi. Remember the dome is both the visible background and the fill
  light: a deep blue sky gives cyan shadows, which is why the script boosts the unseen hemisphere
  separately (warmer than it looks) and tints the visible wedge to the reference's sky patches.
- Subject: placeholders first (boxes, cylinders) so the camera and light can be fitted before any
  modelling effort.
- Fit the sun from the reference's shadow, do not guess: sample vertices of the subject,
  `ground.ray_cast` each along -sun_dir onto the terrain for a grid of (azimuth, elevation),
  project the hits, and compare the shadow's extreme point and its band at a frame edge with the
  reference's. This also tells you the subject's height above ground; but the visual cue Grace
  reacts to is the gap between the object and its shadow on screen, so confirm with her before
  trusting a fitted height.

### 4.3 The preview loop (Cycles, but only through Nori's tonemap)
Blender's Standard view clips highlights; DXR applies ACES. Render Cycles to OPEN_EXR
(`resolution_percentage 50`, 48 samples, ~10 s on her GPU), stage it, and tonemap in numpy:
```python
def aces(c): return np.clip((c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14), 0, 1)**(1/2.2)
```
Then build the side-by-side and measure the same patches as in 4.1. What Cycles cannot tell you:
the cloud (the Blender object is a Principled Volume stand-in and renders far darker than the DXR
medium), blade grass (Cycles shades the lawn texture, DXR shades the blades), normal-map strength
(DXR always uses full strength), volume noise. For those, run the real thing (section 2.1).
When a patch is off, compute the per-channel linear ratio reference/ours and apply exactly that to
the material or the sky gain; two rounds of measured ratios beat ten rounds of eyeballing.

### 4.4 Materials and textures: work out the pixel footprint first
`ComputeTexLOD = log2(uvFootprint * texRes)`; at 6 to 9 m from a 1250 px tall frame one pixel
covers ~7 mm of surface, so anything finer than ~2 cm in a texture is mip-averaged away before it
is sampled, and box-filtered normal maps flatten faster than albedo. Consequences: fabric crumple
goes into the MESH (displacement at 2 to 5 cm wavelengths); albedo grain is authored 4x stronger
than looks sane in the map (rendered std ~1/4 of authored); UVs must be arc-length over the
surface (planar UVs on steep folds stretch grain into streaks that read as fur); textures replace
baseColor and are read as sRGB. Roughness 0.9 to 0.95 + sheen 0.5 for cloth; specular above ~0.2
reads as plastic on fabric.

### 4.5 Assets: what worked and what did not
- PolyHaven (via the addon) for HDRIs and tileable textures; Sketchfab for scanned props. Check
  the licence (CC-BY needs credit in the README).
- Photogrammetry props carry baked lighting and wrong forms; recolour by each object's own
  `active_material` (imports create `.001` copies), never by material name.
- Procedural numpy meshes beat both cloth simulation and metaballs for soft goods. Cloth pressure
  on an open sheet is net thrust (it launched the duvet 9 m); pressure on a closed mesh converges on
  a vacuum bag; metaballs have no boundary so they read as pool noodles.
- Blender scale traps: `primitive_cube_add(size=1)` has half-extent 0.5, so `scale = full extent`;
  `transform_apply(scale=True)` ALSO applies location and rotation by default; setting `root.scale`
  on a Sketchfab import overwrites the importer's normalisation factor; scaling a child scales about
  its own origin. `ob.bound_box` lags one mesh replacement, read vertices with `foreach_get`.
- Node type strings: the Principled Volume node is `'PRINCIPLED_VOLUME'`; a bare `next()` over the
  wrong string surfaces as an opaque "Communication error with Blender".

### 4.6 Export, extras, render
1. Save the .blend. Export with the recipe in 3.1 (re-slice the hand-written `<medium ... <camera`
   block first). Check `grep -c "<mesh" scene.xml` and that every `filename` exists.
2. Regenerate grass if the ground or camera moved; copy both OBJs.
3. Hand-edit scene.xml for things the exporter does not own: medium bounds, grass blade colours,
   `evCompensation`, `sampleCount`.
4. Render headless with `--denoise --png` from Blender's Python, poll the log, stage
   `snapshot_N_denoised.png`, compare against the reference, send the side-by-side.
5. Write what changed to the memory file for the scene (numbers, not adjectives).

---

## 5. Worked example: `scenes/liminal_bed` (final state)
Reference: an anotherworldcore composite, pale blue bed on four strings under a cumulus over a
mown hillside, portrait 4:5-ish. Final scene: 12 meshes, 5.57M triangles, 1000x1250, 256 spp.
- Camera (Blender): location (-4.88, -4.74, 2.00), rotation (87.4 deg, 0, -46.8 deg), lens 38 mm.
  Derived from the reference's horizon (44% down at the left edge, 52% at the right) and bed (62%).
- Terrain: 436 m mesh, ridge scaled 0.55, tilted 3.3 deg about the view axis (horizon slope 0.12),
  rotated 3.8 deg about screen-x so it descends away (horizon 43.5% -> 52.6%), ground z -0.02
  under the bed so the plinth (z 0.60 to 0.81) floats ~0.6 m.
- Sun: azimuth -30, elevation 31, fitted from the reference's shadow; sky
  `_sky_build.py src dst 1.05 1.0 3.3,2.9,1.7 0.55,0.80,0.84 1.0,0.93,0.80 31 6.0 -30 0.036`.
- Cloud: Disney cloud NanoVDB -> `volumes/liminal_cloud.vol`, sigmaS 110, g 0.877, bounds
  y 2.7 to 4.6; strings run 1 m into it.
- Bedding: `_duvet_build.py` (thick sheet, cusped fold profiles, box quilting, hem roll, spill over
  the plinth edges, arc-length UVs, 1024 px fabric maps per 0.40 m) and `_pillow_build.py` (three
  superellipse pillows leaning back to front).
- Grass: `_grass_geometry.py`, blades 0.185/0.270/0.092 and 0.148/0.224/0.078, lawn albedo
  darkened in Blender by linear (0.85, 0.60, 0.62).
- Measured match at the end: sky top (62,151,211) vs (63,150,200); lit duvet (229,236,241) vs
  (228,231,237); shaded duvet (111,181,210) vs (67,150,186) (still lighter than theirs).
Full history and every failed attempt: memory file `/areas/liminal-bed-scene.md`.

---

## 6. Lessons in one line each (the expensive ones)
- Judge only through Nori's ACES curve; a Blender Standard-view preview lies about highlights.
- The envmap azimuth is 180 degrees off between Blender and Nori; Mapping Rotation Z = pi.
- A finite ground shows the panorama below the horizon; either a crest occludes the edge or the
  HDR continues the sky below its horizon (`_sky_build.py` does the latter).
- Sun direction comes from the reference's shadow by ray-cast fitting; blue shadows come from a
  dim, blue fill with a warm sun, not from more fill.
- Compute the texture LOD before authoring detail; put detail finer than ~2 cm into geometry.
- Arc-length UVs on folded cloth; planar UVs make fur.
- Fabric reads as fabric from a hem you can trace, sharp creases between broad arcs, thickness,
  even height along its length (any local pile-up reads as something hidden underneath), and
  nothing periodic reaching the silhouette.
- Pillows lean on each other; a stacked pile parallel to the bed reads as stacked too cleanly.
- Grass reads as grass from blade geometry with opposite lean per mow band; normal maps on a
  plane only ever read as a bumpy surface.
- One MCP call is 60 s; write a marker file and poll. Fresh filename per commit. No `rm`.
- Save the .blend after every builder run; the scene lives only in Blender's session otherwise.

---

## 7. Fresh-session checklist
1. `get_addon_status`, `get_polyhaven_status`, `get_sketchfab_status`; `get_device_info` for the
   connected folder; `ls ~/mnt/nori-26sp` from `device_bash`.
2. Read `/areas/blender-nori-pipeline.md` and the memory file of any scene being continued.
3. Save the reference image in the scratchpad and take the measurements in 4.1.
4. New scene folder: `scenes/<name>/` with `meshes/`, `textures/`, `volumes/` as needed; a new
   `.blend` on the Desktop; the World with the HDRI and Rotation Z = pi.
5. Camera + terrain + placeholders, verify by projection, fit the sun, first ACES side-by-side.
6. Build the subject, iterate with measured patches, one targeted change per round.
7. Export, extras, headless DXR render with `--denoise --png`, side-by-side, memory note.
