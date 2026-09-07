# validation_tests

Scenes and scripts for checking that the renderer is *correct*, not just that it
produces a plausible picture. Two kinds of test live here.

**Analytic tests** compare a render against a closed-form answer. A white-furnace
scene is lit so that a perfectly energy-conserving BSDF must return exactly the
light it receives, so mean luminance has to come out at 1.0 and any deviation is
energy the implementation lost or invented. Beer-Lambert transmittance through a
homogeneous medium is likewise known exactly. These need no reference renderer.

**Cross-validation tests** render the same scene in nori-dxr and in Mitsuba 3 and
compare the linear HDR outputs directly, never the tonemapped ones: nori-dxr's
viewport applies Reinhard, gamma and EV compensation, so comparing what is on
screen measures the display pipeline rather than the BSDF. Metrics are linear
RMSE (primary), relMSE, and tonemapped PSNR as a sanity check.

Measured results for every feature are written up per-feature in the
[final report](../final_report/report.html), under each feature's *Quantitative
Validation* heading. This file is the index of how to reproduce them.

## Contents

| Directory | Validates | Against |
|---|---|---|
| `disney/` | The five Disney BRDF lobes, individually and combined | Mitsuba 3 `principled`, plus white-furnace energy checks per lobe. See [disney/README.md](disney/README.md) |
| `dof/` | Thin-lens depth of field: circle-of-confusion growth and aperture sampling | Mitsuba 3, swept over aperture radius. See [dof/README_DOF_VALIDATION.md](dof/README_DOF_VALIDATION.md) |
| `ibl_test/` | Envmap evaluation, CDF construction, solid-angle weighting and MIS | Mitsuba 3, on diffuse, mirror and furnace spheres |
| `normalmap_test/` | Tangent-frame reconstruction and tangent-space normal decode | Mitsuba 3, textured cube with and without the map |
| `texture_test/` | UV interpolation, sRGB decode and mip selection | Mitsuba 3, textured vs flat sphere |
| `hair_furnace_test/` | Whether the Chiang BCSDF conserves energy across all four lobes | Analytic (white furnace) |
| `hair_visual_test/` | Hair appearance under a directional envmap | Visual |
| `volumetric_tests/` | Beer-Lambert transmittance, single-scattering albedo, Henyey-Greenstein anisotropy at g = 0, 0.3, 0.8 | Analytic, plus a Mitsuba visual comparison |
| `meshes/` | Shared geometry (`wWavy.hair`) used by several tests | |

## Running them

Build first, from the repository root:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The analytic tests are driven by the harnesses in [`tools/validation/`](../tools/validation/):

```powershell
python tools/validation/make_white_furnace.py      # generates the furnace envmap
python tools/validation/validate_volumetrics.py    # Beer-Lambert + albedo + phase
build\Release\nori-dxr.exe validation_tests\hair_furnace_test\white_furnace_test.xml --headless
python tools/validation/analyze_furnace.py validation_tests\hair_furnace_test\snapshot_N.exr
```

The cross-validation tests need Mitsuba 3, and each directory carries a
`render_mitsuba_ref.py` that writes the `ref_*.exr` files its comparison expects:

```powershell
pip install mitsuba opencv-python numpy
cd validation_tests\ibl_test
python render_mitsuba_ref.py
..\..\build\Release\nori-dxr.exe sphere_ibl.xml --headless
python ..\compare_exr.py snapshot_512.exr ref_ibl.exr
```

`compare_exr.py` at this directory's root reports linear RMSE, relMSE and
tonemapped PSNR for any two EXRs. For quick metrics without Mitsuba installed,
[`tools/exrtool.py`](../tools/exrtool.py) does RMSE and relMSE-vs-reference
curves with nothing but numpy.

## Note on the regression harness

These tests answer "is the maths right". The separate question, "did this change
alter the image", is answered by [`tools/validation/verify.sh`](../tools/validation/verify.sh),
which renders four scenes and compares SHA-256 hashes against known-good values.
Run that before and after any change that is meant to be render-identical.
