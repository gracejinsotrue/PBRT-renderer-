# Disney BRDF validation

Goal: nori-dxr's Disney implementation matches Mitsuba 3's `principled` BSDF
**pixel-for-pixel** in linear scene-referred HDR.

The earlier visual mismatch (washed-out blue vs deep blue) was almost certainly
display-pipeline: nori-dxr's window applies Reinhard + gamma 2.2 + EV
compensation, Mitsuba's `.exr` is raw linear. This harness compares the linear
HDR outputs directly, so tonemapping is out of the loop.

## One-time setup

```powershell
pip install mitsuba opencv-python numpy
python render_mitsuba_ref.py           # writes ref_*.exr (7 files)
```

White-furnace lighting reuses `../hair_furnace_test/textures/white_furnace.hdr`.
Sunset lighting reuses `textures/sunset.hdr`. Mitsuba loads the same `sphere.obj`
nori-dxr uses, so geometry is identical on both sides.

## Step-by-step: visual scene (sunset)

From `validation_tests/disney/`:

```powershell
# 1. render ours (auto-saves snapshot_1024.exr after 1024 spp, then exits)
..\..\dxr\build\Release\nori-dxr.exe cmp_metallic_blue.xml

# 2. compare linear HDR — the truth
python ..\compare_exr.py snapshot_1024.exr ref_metallic_blue.exr --out diff_blue
```

Repeat for `cmp_rough_red_clearcoat.xml` and `cmp_sheen_fabric.xml`.

`compare_exr.py` prints per-channel mean & ratio, linear RMSE, relMSE,
tonemapped PSNR, and writes four PNGs into `--out`:
- `ours_srgb.png`, `ref_srgb.png` — same Reinhard+gamma applied to both
- `abs_diff.png` — |ours − ref|
- `false_color.png` — mid-gray = match, red = ours brighter, blue = ref brighter

## Step-by-step: lobe-isolating furnace tests

```powershell
..\..\dxr\build\Release\nori-dxr.exe furnace_diffuse.xml
python analyze_furnace.py snapshot_512.exr ref_furnace_diffuse.exr

..\..\dxr\build\Release\nori-dxr.exe furnace_metallic_smooth.xml
python analyze_furnace.py snapshot_512.exr ref_furnace_metallic_smooth.exr

..\..\dxr\build\Release\nori-dxr.exe furnace_metallic_rough.xml
python analyze_furnace.py snapshot_512.exr ref_furnace_metallic_rough.exr

..\..\dxr\build\Release\nori-dxr.exe furnace_clearcoat.xml
python analyze_furnace.py snapshot_512.exr ref_furnace_clearcoat.exr
```

`analyze_furnace.py` masks the sphere with an analytic ray-sphere intersect
matching the camera and prints `ours / ref` per channel.

| Scene                     | Isolates                                | A mismatch here points at                  |
| ------------------------- | --------------------------------------- | ------------------------------------------ |
| `furnace_diffuse`         | Burley diffuse                          | `FD90`, Schlick weights, π normalisation   |
| `furnace_metallic_smooth` | GGX at α≈0.01, F0 = baseColor           | `DisneyGGX_D`, Smith G                     |
| `furnace_metallic_rough`  | GGX at α=0.25 (single-scatter loss)     | Should deviate from 1 — match Mitsuba's deviation |
| `furnace_clearcoat`       | GTR1 lobe in isolation                  | `DisneyGTR1_D`, fixed-α G, 0.25 scale       |

Ratio interpretation:
- **0.98–1.02** PASS — BRDF math agrees with Mitsuba
- **0.95–1.05** CLOSE — likely a normalisation constant
- **outside ±5%** FAIL — real divergence; look at the corresponding eval
  function in [Shaders.hlsl:622+](../../dxr/shaders/Shaders.hlsl#L622)

## Tip: save EXR at any time

You don't have to wait for `sampleCount`. While nori-dxr is running, press **O**
to dump the current accumulator as `snapshot_<sampleCount>.exr`.

## Files

| File                   | Purpose                                       |
| ---------------------- | --------------------------------------------- |
| `cmp_*.xml`            | nori-dxr scenes mirroring Mitsuba sunset tests |
| `furnace_*.xml`        | Single-lobe energy tests, white envmap         |
| `render_mitsuba_ref.py`| Renders all 7 `ref_*.exr` references           |
| `analyze_furnace.py`   | Sphere-masked mean + ratio for furnace         |
| `../compare_exr.py`    | Linear-HDR diff for any two EXRs               |
