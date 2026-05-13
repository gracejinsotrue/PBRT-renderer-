#!/usr/bin/env python3
"""
Validate nori-dxr DOF camera against Mitsuba 3.

This script:
1. Renders the DOF test scene in nori-dxr (headless mode)
2. Renders the same scene in Mitsuba 3
3. Compares outputs with quantitative metrics (RMSE, relMSE)
4. Generates visual diff images

Run from validation_tests/dof/:
    python validate_dof.py

Requirements:
    - Compiled nori-dxr executable
    - Mitsuba 3 Python package
    - OpenCV with EXR support (for compare_exr.py)
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description, cwd):
    """Run a command and check for success."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed (exit code {result.returncode})")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate nori-dxr DOF camera against Mitsuba 3'
    )
    parser.add_argument(
        '--dxr-exe',
        default='../../dxr/build/Release/nori-dxr.exe',
        help='Path to nori-dxr executable (relative to this script)'
    )
    parser.add_argument(
        '--no-mitsuba',
        action='store_true',
        help='Skip Mitsuba render (only render with nori-dxr)'
    )
    parser.add_argument(
        '--skip-dxr',
        action='store_true',
        help='Skip nori-dxr render (only render with Mitsuba)'
    )

    args = parser.parse_args()

    # All paths relative to THIS script's directory (validation_tests/dof/)
    script_dir = Path(__file__).parent.resolve()
    dxr_exe = (script_dir / args.dxr_exe).resolve()

    print(f"[INFO] Working directory: {script_dir}")
    print(f"[INFO] DXR executable: {dxr_exe}")

    if not dxr_exe.exists() and not args.skip_dxr:
        print(f"[ERROR] nori-dxr executable not found: {dxr_exe}", file=sys.stderr)
        print(f"        Build with: cmake -S dxr -B dxr/build && cmake --build dxr/build --config Release", file=sys.stderr)
        return 1

    scene_xml = script_dir / 'dof_test_scene.xml'
    if not scene_xml.exists():
        print(f"[ERROR] Test scene not found: {scene_xml}", file=sys.stderr)
        return 1

    dxr_out = script_dir / 'dof_dxr_out.exr'
    mitsuba_out = script_dir / 'dof_mitsuba_out.exr'
    diff_dir = script_dir / 'dof_diff'

    # Render with nori-dxr (headless)
    if not args.skip_dxr:
        print("\n[Stage 1/3] Rendering with nori-dxr (DXR GPU)...")
        cmd = [str(dxr_exe), str(scene_xml), '--headless']
        if not run_command(cmd, "nori-dxr headless render", script_dir):
            return 1

        # Check if snapshot was saved
        # The snapshot is saved next to the scene XML, named snapshot_<samples>.exr
        import glob
        snapshots = sorted(glob.glob(str(script_dir / 'snapshot_*.exr')))
        if snapshots:
            latest_snapshot = snapshots[-1]
            print(f"[SUCCESS] DXR rendered: {latest_snapshot}")
            # Copy to standard output name
            import shutil
            shutil.copy(latest_snapshot, dxr_out)
            print(f"[INFO] Copied to: {dxr_out}")
        else:
            print(f"[ERROR] No snapshot found in {script_dir}")
            return 1

    # Render with Mitsuba
    if not args.no_mitsuba:
        print("\n[Stage 2/3] Rendering with Mitsuba 3...")
        mitsuba_script = script_dir / 'mitsuba_dof_render.py'
        cmd = [sys.executable, str(mitsuba_script), str(mitsuba_out), str(script_dir)]
        if not run_command(cmd, "Mitsuba 3 render", script_dir):
            return 1

    # Compare
    if not args.skip_dxr and not args.no_mitsuba:
        print("\n[Stage 3/3] Comparing renders...")
        diff_dir.mkdir(exist_ok=True)

        compare_script = script_dir / 'compare_exr.py'
        cmd = [
            sys.executable,
            str(compare_script),
            str(dxr_out),
            str(mitsuba_out),
            '--out', str(diff_dir),
            '--exposure', '0'
        ]
        if not run_command(cmd, "EXR comparison", script_dir):
            return 1

        print(f"\n[SUCCESS] Validation complete!")
        print(f"  DXR output:      {dxr_out}")
        print(f"  Mitsuba output:  {mitsuba_out}")
        print(f"  Diff images:     {diff_dir}/")
        print(f"\nMetrics:")
        print(f"  - ours_srgb.png       (DXR tonemapped)")
        print(f"  - ref_srgb.png        (Mitsuba tonemapped)")
        print(f"  - abs_diff.png        (absolute difference)")
        print(f"  - false_color.png     (signed difference, gray=match)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
