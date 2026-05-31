"""
export_vol_from_blender.py — Run inside Blender's Scripting tab.

Select the volume object, then run this script. It reads the volume's
density grid, applies the object's world transform to get correct
scene-space bounds, and writes a .vol file that matches your OBJ exports.

Change OUTPUT_PATH to wherever you want the .vol file.
Change Y_UP to False if your renderer uses Z-up.
"""

import bpy, struct, numpy as np, os

# ── CONFIG ──────────────────────────────────────────────────────────
OUTPUT_PATH = r"C:\Users\gjin3\Desktop\nori-26sp\scenes\final_scene_real\clouds\volumes\cloud.vol"
Y_UP = True   # convert Blender Z-up → Y-up (matches OBJ export default)
# ────────────────────────────────────────────────────────────────────

obj = bpy.context.active_object
if obj is None or obj.type != 'VOLUME':
    raise RuntimeError("Select a Volume object first!")

# Get the object's world matrix (includes location, rotation, scale)
world_mat = obj.matrix_world

# Access the volume data
vol = obj.data
grids = vol.grids
grids.load()  # ensure grids are loaded

# Find the density grid
grid = None
for g in grids:
    if g.name == 'density':
        grid = g
        break
if grid is None:
    raise RuntimeError("No 'density' grid found in volume")

# Get grid resolution and transform
# The grid has its own transform (index → object local space)
# Combined with the object's world matrix, we get index → world space

# We need to iterate through the volume using depsgraph evaluation
depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = obj.evaluated_get(depsgraph)

# Get bounding box in world space from the object
bbox_corners = [world_mat @ bpy.mathutils.Vector(c) for c in obj.bound_box]
ws_min = [min(c[i] for c in bbox_corners) for i in range(3)]
ws_max = [max(c[i] for c in bbox_corners) for i in range(3)]

print(f"World-space bbox: ({ws_min[0]:.4f}, {ws_min[1]:.4f}, {ws_min[2]:.4f}) - "
      f"({ws_max[0]:.4f}, {ws_max[1]:.4f}, {ws_max[2]:.4f})")

# Determine voxel resolution from the grid's active bounding box
# We'll sample the volume on a regular grid covering the world-space bbox
# Resolution: use the grid's native voxel count if available, else default
try:
    channels = grid.channels
    res_info = f"{channels} channels"
except:
    res_info = "unknown"

# Estimate resolution from the VDB file's metadata
# We'll use the world-space extent and the grid's voxel size
import mathutils

# Try to get resolution from grid domain bounds
# Blender doesn't expose VDB voxel iteration directly, so we sample
# the volume shader at regular points using the depsgraph

# Determine resolution: aim for ~1 voxel per Blender unit * density
# Use the original VDB's resolution if we can detect it
extent = [ws_max[i] - ws_min[i] for i in range(3)]
max_extent = max(extent)

# Default resolution scaling — adjust RES_SCALE for quality vs file size
RES_SCALE = 80  # voxels per unit of max extent
res = [max(4, int(extent[i] / max_extent * RES_SCALE + 0.5)) for i in range(3)]

print(f"Sampling at resolution: {res[0]} x {res[1]} x {res[2]}")
print(f"Extent: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}")

# Sample the volume density at each voxel center
# We use Blender's internal evaluation via the object's evaluated data
data = np.zeros((res[0], res[1], res[2]), dtype=np.float32)

# For sampling, we need to convert world positions back to object local space
# and query the volume there
world_mat_inv = world_mat.inverted()

# Use bpy.data to sample — we'll iterate and use the grid accessor
# Actually, the most reliable way is to use OpenVDB directly if available
try:
    import openvdb
    HAS_OPENVDB = True
except ImportError:
    try:
        import pyopenvdb as openvdb
        HAS_OPENVDB = True
    except ImportError:
        HAS_OPENVDB = False

if HAS_OPENVDB:
    # Read the VDB file directly — most reliable
    filepath = bpy.path.abspath(vol.filepath)
    print(f"Reading VDB from: {filepath}")
    vdb_grids = openvdb.readAll(filepath)
    if isinstance(vdb_grids, (list, tuple)) and len(vdb_grids) == 2:
        if isinstance(vdb_grids[0], list):
            vdb_grid_list = vdb_grids[0]
        else:
            vdb_grid_list = list(vdb_grids)
    else:
        vdb_grid_list = list(vdb_grids)

    vdb_grid = None
    for g in vdb_grid_list:
        if hasattr(g, 'name') and g.name.lower() == 'density':
            vdb_grid = g
            break
    if vdb_grid is None and vdb_grid_list:
        vdb_grid = vdb_grid_list[0]

    acc = vdb_grid.getConstAccessor()

    for iz in range(res[2]):
        for iy in range(res[1]):
            for ix in range(res[0]):
                # World-space position of this voxel center
                wx = ws_min[0] + (ix + 0.5) * extent[0] / res[0]
                wy = ws_min[1] + (iy + 0.5) * extent[1] / res[1]
                wz = ws_min[2] + (iz + 0.5) * extent[2] / res[2]

                # Transform world → object local → VDB index space
                local_pos = world_mat_inv @ mathutils.Vector((wx, wy, wz))
                idx = vdb_grid.transform.worldToIndex(
                    (float(local_pos.x), float(local_pos.y), float(local_pos.z)))

                # Trilinear would be better but nearest-neighbor works
                ii, ij, ik = int(round(idx[0])), int(round(idx[1])), int(round(idx[2]))
                try:
                    val = acc.probeValue((ii, ij, ik))
                    if isinstance(val, tuple):
                        val = val[0]
                    data[ix, iy, iz] = max(0.0, float(val))
                except:
                    pass

        if (iz + 1) % 10 == 0:
            print(f"  Sampled slice {iz+1}/{res[2]}")
else:
    raise RuntimeError(
        "OpenVDB Python module not found. Install pyopenvdb or run "
        "this in a Blender build that includes openvdb Python bindings.")

# Normalize to [0, 1]
mx = data.max()
if mx > 0:
    data /= mx
    print(f"Normalized density by max={mx:.4f}")

# Apply coordinate system conversion
if Y_UP:
    # Blender: X=right, Y=forward, Z=up
    # Renderer: X=right, Y=up, Z=forward
    # Swap Y<->Z in both data and bounds
    data = np.ascontiguousarray(data.transpose(0, 2, 1))
    bbox_min = (ws_min[0], ws_min[2], ws_min[1])
    bbox_max = (ws_max[0], ws_max[2], ws_max[1])
    # Ensure min < max
    bbox_min = tuple(min(a, b) for a, b in zip(bbox_min, bbox_max))
    bbox_max = tuple(max(a, b) for a, b in zip(bbox_min, bbox_max))
    # Recalculate after swap
    bbox_min = (min(ws_min[0], ws_max[0]), min(ws_min[2], ws_max[2]), min(ws_min[1], ws_max[1]))
    bbox_max = (max(ws_min[0], ws_max[0]), max(ws_min[2], ws_max[2]), max(ws_min[1], ws_max[1]))
else:
    bbox_min = tuple(ws_min)
    bbox_max = tuple(ws_max)

# Write .vol file with correct axis ordering for LoadVolume
W, H, D = data.shape
vol_data = np.ascontiguousarray(data.transpose(2, 1, 0)).astype(np.float32)

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
with open(OUTPUT_PATH, 'wb') as f:
    f.write(b'VOL1')
    f.write(struct.pack('<III', W, H, D))
    f.write(struct.pack('<fff', *bbox_min))
    f.write(struct.pack('<fff', *bbox_max))
    f.write(vol_data.tobytes())

mb = (W * H * D * 4) / (1024 * 1024)
print(f"\nWrote {OUTPUT_PATH}")
print(f"  Resolution: {W}x{H}x{D} ({mb:.1f} MB)")
print(f"  bbox=({bbox_min[0]:.4f},{bbox_min[1]:.4f},{bbox_min[2]:.4f})"
      f"-({bbox_max[0]:.4f},{bbox_max[1]:.4f},{bbox_max[2]:.4f})")
print(f"  density range=[{data.min():.4f}, {data.max():.4f}]")
print(f"\nXML values:")
print(f'    <point name="boundsMin" value="{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}"/>')
print(f'    <point name="boundsMax" value="{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}"/>')