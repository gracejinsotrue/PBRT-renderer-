#!/usr/bin/env python3
"""
vdb_to_vol.py - Convert OpenVDB density grids to dense .vol format

Usage:
    python3 vdb_to_vol.py input.vdb output.vol [--grid density] [--pad 2]
    python3 vdb_to_vol.py --test-sphere output.vol [--res 64]
"""
import struct, argparse, numpy as np, os

def write_vol(path, data, bbox_min, bbox_max):
    W, H, D = data.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'VOL1')
        f.write(struct.pack('<III', W, H, D))
        f.write(struct.pack('<fff', *bbox_min))
        f.write(struct.pack('<fff', *bbox_max))
        f.write(data.astype(np.float32).tobytes())
    mb = (W*H*D*4)/(1024*1024)
    print(f"Wrote {path}: {W}x{H}x{D} ({mb:.1f} MB)")
    print(f"  bbox=({bbox_min[0]:.4f},{bbox_min[1]:.4f},{bbox_min[2]:.4f})-({bbox_max[0]:.4f},{bbox_max[1]:.4f},{bbox_max[2]:.4f})")
    print(f"  density range=[{data.min():.4f}, {data.max():.4f}]")

def generate_test_sphere(res):
    data = np.zeros((res,res,res), dtype=np.float32)
    center = res/2.0; radius = res*0.4
    for z in range(res):
        for y in range(res):
            for x in range(res):
                d = np.sqrt((x-center)**2+(y-center)**2+(z-center)**2)
                data[x,y,z] = max(0.0, 1.0-(d/radius)**2)
    return data

def convert_vdb(vdb_path, grid_name, pad):
    import pyopenvdb as vdb
    grids_data = vdb.readAll(vdb_path)
    if isinstance(grids_data,(list,tuple)) and len(grids_data)==2:
        if isinstance(grids_data[0],list) and isinstance(grids_data[1],dict):
            grids = grids_data[0]
        else: grids = list(grids_data)
    else: grids = list(grids_data)

    grid = None
    for g in grids:
        if hasattr(g,'name') and g.name.lower()==grid_name.lower():
            grid = g; break
    if grid is None and grids:
        grid = grids[0]
        print(f"Grid '{grid_name}' not found, using '{grid.name}'")
    if grid is None: raise ValueError("No grids found")

    print(f"Grid: '{grid.name}', type: {type(grid).__name__}")
    bbox = grid.evalActiveVoxelBoundingBox()
    bmin_idx, bmax_idx = bbox[0], bbox[1]
    print(f"  Active bbox: {bmin_idx} - {bmax_idx}")
    print(f"  Voxel size: {grid.transform.voxelSize()}")

    bp = (bmin_idx[0]-pad, bmin_idx[1]-pad, bmin_idx[2]-pad)
    ep = (bmax_idx[0]+pad, bmax_idx[1]+pad, bmax_idx[2]+pad)
    W,H,D = ep[0]-bp[0]+1, ep[1]-bp[1]+1, ep[2]-bp[2]+1
    print(f"  Dense grid (pad={pad}): {W}x{H}x{D}")

    acc = grid.getConstAccessor()
    data = np.zeros((W,H,D), dtype=np.float32)
    for z in range(D):
        for y in range(H):
            for x in range(W):
                try:
                    val = acc.probeValue((bp[0]+x, bp[1]+y, bp[2]+z))
                    if isinstance(val, tuple): val = val[0]
                    data[x,y,z] = max(0.0, float(val))
                except: pass

    mx = data.max()
    if mx > 0:
        data /= mx
        print(f"  Normalized by max={mx:.4f}")

    bmin_ws = grid.transform.indexToWorld(bp)
    bmax_ws = grid.transform.indexToWorld((ep[0]+1, ep[1]+1, ep[2]+1))
    return data, tuple(bmin_ws), tuple(bmax_ws)

def main():
    p = argparse.ArgumentParser(description="Convert VDB to .vol")
    p.add_argument('input', nargs='?', help='Input .vdb')
    p.add_argument('output', help='Output .vol')
    p.add_argument('--grid', default='density')
    p.add_argument('--pad', type=int, default=2)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--test-sphere', action='store_true')
    p.add_argument('--res', type=int, default=64)
    p.add_argument('--bbox-min', type=float, nargs=3, default=[-0.5,0.2,-0.5])
    p.add_argument('--bbox-max', type=float, nargs=3, default=[0.5,1.2,0.5])
    args = p.parse_args()

    if args.test_sphere:
        print(f"Generating test sphere res={args.res}...")
        data = generate_test_sphere(args.res)
        write_vol(args.output, data, tuple(args.bbox_min), tuple(args.bbox_max))
    else:
        if not args.input: p.error("Input .vdb required (or --test-sphere)")
        print(f"Converting {args.input}...")
        data, bmin, bmax = convert_vdb(args.input, args.grid, args.pad)
        if args.scale != 1.0:
            bmin = tuple(v*args.scale for v in bmin)
            bmax = tuple(v*args.scale for v in bmax)
        write_vol(args.output, data, bmin, bmax)
        print(f'\nXML values:')
        print(f'    <point name="boundsMin" value="{bmin[0]:.4f}, {bmin[1]:.4f}, {bmin[2]:.4f}"/>')
        print(f'    <point name="boundsMax" value="{bmax[0]:.4f}, {bmax[1]:.4f}, {bmax[2]:.4f}"/>')

if __name__ == '__main__': main()