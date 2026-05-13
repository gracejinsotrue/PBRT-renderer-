# DOF Camera Quantitative Validation

This directory contains infrastructure for **quantitative validation** of the nori-dxr depth-of-field camera implementation against **Mitsuba 3** (reference renderer).

## Overview

The validation pipeline renders the same scene in both renderers and compares outputs with:
- **Linear RMSE** — root mean square error in linear color space (primary metric)
- **relMSE** — variance-normalized MSE (Mitsuba-style, secondary metric)
- **Tonemapped PSNR** — sanity check (tertiary metric)
- **Visual diffs** — pixel-by-pixel difference maps

## Setup

### 1. Build nori-dxr

```bash
cd /path/to/nori-26sp
cmake -S dxr -B dxr/build
cmake --build dxr/build --config Release
```

This produces `dxr/build/Release/nori-dxr.exe` (or Debug variant).

### 2. Install Mitsuba 3

```bash
pip install mitsuba
```

Verify installation:
```bash
python -c "import mitsuba; print(mitsuba.__version__)"
```

### 3. Install OpenCV (for EXR support)

```bash
pip install opencv-python
```

## Running Validation

### Quick Start

From the repo root:

```bash
python validation_tests/validate_dof.py
```

This:
1. Renders `dof_test_scene.xml` with nori-dxr (`--headless` mode)
2. Renders the same scene with Mitsuba 3
3. Compares outputs and generates diff images

### Custom DXR Executable

```bash
python validation_tests/validate_dof.py --dxr-exe dxr/build/Debug/nori-dxr.exe
```

### Skip Mitsuba (DXR only)

```bash
python validation_tests/validate_dof.py --skip-dxr
```

### Mitsuba Only

```bash
python validation_tests/validate_dof.py --no-mitsuba
```

## Output

After successful validation:

```
validation_tests/
├── dof_dxr_out.exr              (DXR linear output)
├── dof_mitsuba_out.exr          (Mitsuba linear output)
└── dof_diff/
    ├── ours_srgb.png            (DXR tonemapped)
    ├── ref_srgb.png             (Mitsuba tonemapped)
    ├── abs_diff.png             (absolute difference)
    └── false_color.png          (signed diff: red=ours brighter, blue=ref)
```

Console output includes:
```
Linear RMSE     : 0.00123         (lower is better)
relMSE          : 0.00456         (lower is better)
Tonemap PSNR    : 42.15 dB        (>30 dB is good)
```

## Test Scene

**File:** `dof_test_scene.xml`

- **Geometry:** Blender default cube (red diffuse)
- **Camera:** Perspective, 50° FOV, positioned at (0, 1, 3)
- **DOF parameters:**
  - `lensRadius` = 0.1 (aperture radius)
  - `focalDistance` = 4.0 (distance from camera to focus plane)
  - Focus plane passes through cube center (0, 1, 0)
- **Light:** Area light plane above scene (radiance 10)
- **Samples:** 256 per pixel
- **Resolution:** 512×512

## Expected Results

When DOF is correctly implemented:

- **Linear RMSE < 0.02** — outputs should match closely
- **relMSE < 0.05** — variance-normalized error acceptable
- **Visual test:** false_color.png should be mid-gray with only minor noise artifacts

Larger errors may indicate:
- Incorrect lens sample generation (wrong DOF shape)
- Camera ray direction errors
- Focus distance not applied correctly
- Sample distribution differences

## Customizing the Test Scene

Edit `dof_test_scene.xml` to test different configurations:

```xml
<!-- Stronger DOF (more blur) -->
<float name="lensRadius" value="0.2"/>

<!-- Weaker DOF (less blur) -->
<float name="lensRadius" value="0.05"/>

<!-- Focus nearer to camera -->
<float name="focalDistance" value="2.5"/>

<!-- More samples for lower noise -->
<integer name="sampleCount" value="512"/>
```

Then re-run `validate_dof.py`.

## Files

| File | Purpose |
|------|---------|
| `dof_test_scene.xml` | Nori scene for DOF validation |
| `blender_default_cube.obj` | Cube geometry |
| `light_plane.obj` | Simple area light |
| `validate_dof.py` | Main validation harness (you are here) |
| `mitsuba_dof_render.py` | Mitsuba renderer script |
| `compare_exr.py` | EXR comparison & metrics |

## Troubleshooting

### "nori-dxr executable not found"
Build the project: `cmake -S dxr -B dxr/build && cmake --build dxr/build --config Release`

### "No snapshot found"
- Ensure the scene has a `<sampler>` with `sampleCount` set
- Check headless mode output for errors
- Verify `snapshot_*.exr` files exist in `validation_tests/`

### Mitsuba errors
- Verify: `python -c "import mitsuba as mi; print(mi.__version__)"`
- Try reinstalling: `pip install --upgrade mitsuba`
- Check Python version (Mitsuba 3 requires Python 3.8+)

### OpenCV EXR support
```bash
export OPENCV_IO_ENABLE_OPENEXR=1
python validation_tests/compare_exr.py <ours.exr> <ref.exr>
```

## Citation

When reporting validation results, reference both:
- **nori-dxr:** GPU ray tracer (DirectX 12 DXR)
- **Mitsuba 3:** Reference implementation (https://mitsuba-renderer.org/)

Example: "Validated against Mitsuba 3 with 256 samples/pixel, Linear RMSE 0.0145"
