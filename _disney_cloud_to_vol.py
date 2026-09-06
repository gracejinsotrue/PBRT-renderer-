# Disney cloud (wdas_cloud_quarter.nvdb, from mmp/pbrt-v4-scenes/disney-cloud)
# -> our dense VOL1 grid.
#
# Reuses the NanoVDB v32 FloatGrid byte layout from _pbrt_bunnycloud_to_nori.py,
# but mean-pools each leaf into a DOWNSAMPLED grid so the dense array stays small
# (native 498x338x613 = 0.41 GB; /2 = 249x169x307 = 52 MB).
#
# The nvdb grid is natively Y-up: its index dims 498:338:613 match the pbrt
# scene's box Scale 206.544 140.4 254.592, so y (the smallest) is vertical.
# That matches our Y-up Nori scene directly, no axis swap needed.
#
#   DCLOUD_SRC=... DCLOUD_OUT=... DCLOUD_DOWN=2 python3 _disney_cloud_to_vol.py
import struct, zlib, os
import numpy as np

LEAF_STRIDE = 2144
LEAF_VALUES = 96
GRID_DATA   = 672

def read_nvdb_leaves(path):
    raw = open(path,'rb').read()
    magic,ver,gridCount,codec = struct.unpack_from('<QIHH', raw, 0)
    if magic != 0x304244566F6E614E: raise SystemExit("not a NanoVDB file")
    m=16
    gridSize,fileSize,nameKey,voxelCount = struct.unpack_from('<QQQQ', raw, m)
    gridType,gridClass = struct.unpack_from('<II', raw, m+32)
    ib = struct.unpack_from('<6i', raw, m+88)
    nameSize, = struct.unpack_from('<I', raw, m+136)
    if gridType != 1: raise SystemExit("expected float grid")
    print("[nvdb] v%d.%d.%d codec %d, %d active voxels"
          % ((ver>>21)&0x7FF,(ver>>10)&0x7FF,ver&0x3FF, codec, voxelCount))
    blob_off = 16+176+nameSize
    if codec == 1:
        n, = struct.unpack_from('<Q', raw, blob_off)
        blob = zlib.decompress(raw[blob_off+8: blob_off+8+n], bufsize=gridSize)
    elif codec == 0:
        blob = raw[blob_off: blob_off+gridSize]
    else: raise SystemExit("unsupported codec %d" % codec)
    nodeOffset = struct.unpack_from('<4q', blob, GRID_DATA)
    nodeCount  = struct.unpack_from('<3I', blob, GRID_DATA+32)
    leafOff, leafCount = GRID_DATA+nodeOffset[0], nodeCount[0]
    print("[nvdb] %d leaves" % leafCount)
    leaves = np.frombuffer(blob, dtype=np.uint8, count=leafCount*LEAF_STRIDE,
                           offset=leafOff).reshape(leafCount, LEAF_STRIDE)
    bbmin   = leaves[:,0:12].copy().view(np.int32).reshape(leafCount,3)
    origins = (bbmin//8)*8
    values  = leaves[:, LEAF_VALUES:LEAF_VALUES+2048].copy().view(np.float32)
    masks   = leaves[:,16:80]
    active  = np.unpackbits(masks, axis=1, bitorder='little').astype(bool)
    values  = np.where(active, values, 0.0).reshape(leafCount,8,8,8)
    return values, origins, np.array(ib[:3]), [ib[3+i]-ib[i]+1 for i in range(3)]

def write_vol(path, data, bbox_min, bbox_max):
    W,H,D = data.shape
    out = np.ascontiguousarray(data.transpose(2,1,0)).astype(np.float32)
    with open(path,'wb') as f:
        f.write(b'VOL1'); f.write(struct.pack('<III', W,H,D))
        f.write(struct.pack('<fff', *bbox_min)); f.write(struct.pack('<fff', *bbox_max))
        f.write(out.tobytes())

SRC  = os.environ.get("DCLOUD_SRC",  os.path.expanduser("~/wdas_cloud_quarter.nvdb"))
OUT  = os.environ.get("DCLOUD_OUT",  os.path.expanduser("~/liminal_cloud.vol"))
DOWN = int(os.environ.get("DCLOUD_DOWN","2"))

values, origins, ibmin, dims = read_nvdb_leaves(SRC)
print("[nvdb] native dims", dims)
# leaf-local mean pool: an 8-cube becomes (8/DOWN)^3 coarse voxels
assert 8 % DOWN == 0, "DOWN must divide 8"
k = 8//DOWN
pooled = values.reshape(-1, k,DOWN, k,DOWN, k,DOWN).mean(axis=(2,4,6))   # (n,k,k,k)
cdims = [int(np.ceil(d/DOWN)) for d in dims]
grid  = np.zeros(cdims, dtype=np.float32)
base  = (origins - ibmin)//DOWN
for i in range(pooled.shape[0]):
    p  = base[i]
    d0 = np.maximum(p,0); d1 = np.minimum(p+k, cdims)
    if (d1<=d0).any(): continue
    s0, s1 = d0-p, d1-p
    np.maximum(grid[d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]],
               pooled[i][s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]],
               out=grid[d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]])
dmax = float(grid.max()); dmean = float(grid[grid>0].mean())
print("[grid] %s  raw max %.4f  mean(nonzero) %.4f  fill %.1f%%"
      % (cdims, dmax, dmean, 100*(grid>0.01).mean()))
grid /= dmax                                   # normalize to [0,1]
write_vol(OUT, grid, (-1.5,3.1,-1.6), (1.5,4.7,1.6))
print("[vol] %s  %s  %.1f MB  (bbox in header is informational; scene.xml bounds win)"
      % (OUT, cdims, os.path.getsize(OUT)/1e6))
print("[hint] pbrt uses sigma_a 0, sigma_s 1, scale 4, g 0.877 over a 206.5-unit box")
