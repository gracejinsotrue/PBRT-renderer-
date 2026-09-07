# tools

Supporting scripts for the renderer. None of these are needed to build or run
`nori-dxr` — they produce the scenes it renders and the measurements behind the
numbers in the README.

Most are run with plain CPython; the ones marked **(Blender)** must be executed
from inside Blender's Python console, since they use `bpy` for geometry.

Paths are derived from the repository root, so these can be invoked from
anywhere — `python tools/exporters/pbrt_bmw_to_nori.py` and
`./tools/validation/verify.sh` both work regardless of the working directory.

## `exporters/` — third-party formats into Nori scenes

| Script | What it does |
|---|---|
| `blender_to_nori.py` | **(Blender)** Exports the active scene: one triangulated OBJ per object plus a `scene.xml`, mapping each Principled BSDF to a Nori `disney` BSDF and the active camera to a `perspective` camera. This built the final CS5630 scene. |
| `obj_mtl_to_nori.py` | Wavefront OBJ+MTL straight to Nori, no Blender. Streams large OBJs, splits by material, and maps MTL entries to Disney BSDFs with albedo/normal textures. Used for San Miguel. |
| `pbrt_bmw_to_nori.py` | pbrt-v4 `bmw-m6` → `scenes/bmw_m6/`. Binary-LE plymesh geometry plus an equal-area octahedral envmap. |
| `pbrt_sportscar_to_nori.py` | pbrt-v4 `sportscar` → `scenes/sportscar/`. |
| `pbrt_head_to_nori.py` | pbrt-v4 `head` → `scenes/head/`, and the subsurface-vs-Disney comparison set used in the README. |
| `pbrt_sssdragon_to_nori.py` | pbrt-v4 `sssdragon` → `scenes/sssdragon/`. A backlit translucent dragon makes subsurface scattering obvious where a face keeps it subtle. |
| `pbrt_bunnycloud_to_nori.py` | pbrt-v4 `bunny-cloud` → `scenes/bunny_cloud/`. Reads the NanoVDB sparse tree directly out of the container, since NanoVDB has no Python bindings on PyPI. |
| `disney_cloud_to_vol.py` | Disney's `wdas_cloud` NanoVDB grid → the renderer's dense `VOL1` format, reusing the NanoVDB reader above. |
| `rgl_measured_to_disney.py` | Fits a Disney BSDF stand-in to each measured BSDF in an RGL material directory. |

## `scene_build/` — procedural geometry and lighting

| Script | What it does |
|---|---|
| `duvet_build.py` | **(Blender)** Procedural duvet for `scenes/liminal_bed`, built as a thick heightfield sheet. |
| `duvet_cloth.py` | **(Blender)** Drapes slack cloth over the duvet stuffing so it creases like fabric instead of reading as soap bubbles. |
| `pillow_build.py` | **(Blender)** Pillow stack for the same scene: thick sheets joined round a hem ring, so the corners survive. |
| `pool_store_build.py` | **(Blender)** Deterministic full rebuild of `scenes/pool_store`, minus the water and slides. |
| `pool_water_build.py` | **(Blender)** Displaced water surface, wave direction matched to the reference photograph. |
| `slide_build.py` | **(Blender)** Water-slide flumes: a cross-section swept along a Catmull-Rom path with parallel-transport frames, handling the closed-tube → open-flume transition. |
| `grass_geometry.py` | Scatters real grass-blade geometry onto an already-exported ground mesh. |
| `sky_build.py` | Rebuilds `scenes/liminal_bed/sky.hdr` with a dimmable sun disk and a back-hemisphere fill that lights camera-facing surfaces without appearing in frame. |
| `make_restir_scene.py` | Builds a deliberately hard many-light Cornell box for the ReSTIR-DI A/B: 256 lights across only 8 emitter meshes, grouped into spatially interleaved radiance tiers spanning a 128× range. |
| `sm_scene.py` | Regenerates a San Miguel `scene.xml`. Camera, exposure and output are set through `SM_*` environment variables. |
| `sm_sky.py` | Rebuilds `scenes/san_miguel/sky_custom.hdr`: a blue sky dome with a painted warm sun. |
| `sm_sun_geo.py` | Builds an emissive sun disk mesh. A disk rather than a sphere because the renderer's area lights are one-sided. |

## `analysis/` — measurement and diagnostics

| Script | What it does |
|---|---|
| `sss_furnace.py` | White-furnace test for the random-walk BSSRDF: does it conserve energy against the albedo it was handed? Written to explain why the pbrt head rendered darker and redder under subsurface than under Disney. |
| `sss_head_scenes.py` | Generates the head subsurface-vs-Disney comparison scenes. |
| `sss_head_sheet.py` | Assembles those renders into the 2×2 contact sheet in the README — rows are lighting setups, columns are skin models, so a row isolates the material and a column isolates the light. |
| `light_solve.py` | Solves emitter gains against reference patches. Radiance is linear in each emitter's radiance, so this renders one pass per light group and fits non-negative gains to the target. |
| `sm_probe.py` | Checks whether a candidate San Miguel camera position is actually in open air before committing to a five-minute render. Flags foliage separately, since leaves buried the camera in every early framing. |
| `sm_bbox.py` | Dumps per-mesh bounding boxes for San Miguel, feeding `sm_probe.py`. |
| `sweep_bounces.ps1` | Real-time feasibility probe: sweeps `MAX_BOUNCES` across the coverage scenes and reports pure `DispatchRays` GPU time (min-of-N headless `--profile` timestamps), which is the number a frame budget is actually spent against. Real-time builds run 1–2 bounces, not the offline default of 32. |
| `sm_sunratio.py` | Measures the painted sun's integrated irradiance against the sky dome's. Integrated energy is what matters, not peak value: a huge sun colour over a sub-pixel radius contributes almost nothing. |

## `validation/` — correctness harnesses

| Script | What it does |
|---|---|
| `verify.sh` | Golden-image regression test. Compiles the shader library with `dxc`, renders four scenes headless (Cornell box, mixed materials, hair, heterogeneous volume) and compares each output EXR against a known-good SHA-256. This is what makes an optimisation safe to claim: a change that is supposed to be render-identical has to come out bit-identical here. Run `./tools/validation/verify.sh` for all four, or pass one name. |
| `analyze_furnace.py` | White-furnace check: reads a rendered EXR and asserts mean luminance ≈ 1.0. A furnace scene is lit so that a perfectly energy-conserving BSDF must return exactly the light it receives, so any deviation is lost or invented energy. |
| `make_white_furnace.py` | Generates the constant-white RGBE environment map the furnace scenes are lit by. |
| `validate_volumetrics.py` | Quantitative tests for the volumetric transport in `shaders/Volume.hlsl`. Each scene is built so its correct pixel value is known analytically (Beer-Lambert transmittance, single-scattering albedo), then the render is checked against that closed form rather than against another render. |

## `exrtool.py`

Minimal reader for the scanline float EXRs the renderer writes, plus error
metrics (`info`, `rmse`, `curve` for relMSE-vs-reference). Deliberately
dependency-light — stdlib `zlib` plus numpy — because installing OpenEXR on
Windows is more trouble than parsing the subset actually emitted.

## `exr_to_png.py`

Tonemaps a rendered EXR to PNG using the same Hill 2016 ACES approximation the
renderer's `Shaders.hlsl` applies, so offline conversions match the viewport.
