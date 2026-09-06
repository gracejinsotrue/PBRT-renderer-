# Converts mmp/pbrt-v4-scenes/bunny-cloud into scenes/bunny_cloud/.
#
# The interesting part is bunny_cloud.nvdb. NanoVDB is a sparse VDB tree, and
# there are no Python bindings for it on PyPI, so this reads the container
# directly. It does NOT walk the tree: NanoVDB stores every leaf node
# contiguously at a known offset with a fixed stride, and each leaf carries its
# own origin, so the whole dense grid can be reassembled by striding over the
# leaf array and splatting 8x8x8 blocks. Root and internal nodes are skipped
# entirely - they only carry tiles, and this grid has 2 (both background).
#
# Everything else is coordinate bookkeeping:
#   - the pbrt scene is Z-up, our GPU camera is Y-up only (it rebuilds its
#     basis from yaw/pitch and drops roll), so the world is rotated by Rx(-90)
#   - the medium carries its own `Rotate 180 0 0 1` / `Rotate 90 1 0 0`, which
#     composed with the Z-up fix collapses to a 180 deg turn about Y, i.e. a
#     flip on x and z of the dense array
#   - the sky is a 2048^2 equal-area octahedral map and has to be resampled to
#     equirectangular, same as the BMW envmap
#
# usage: python _pbrt_bunnycloud_to_nori.py [--downsample N] [--spp N]
import os, sys, math, struct, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scenes", "bunny_cloud")
NVDB = os.path.join(OUT, "volumes", "bunny_cloud.nvdb")
VOL = os.path.join(OUT, "volumes", "bunny_cloud.vol")
SKY_EXR = os.path.join(OUT, "textures", "sky.exr")
SKY_HDR = os.path.join(OUT, "textures", "sky.hdr")

sys.path.insert(0, HERE)
from _pbrt_bmw_to_nori import (equal_area_sphere_to_square, lookat_matrix,
                               coated_diffuse_albedo)


def _arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


DOWNSAMPLE = _arg('--downsample', 1)
SPP = _arg('--spp', 512)
W_IMG, H_IMG = 1920, 1080

# ------------------------------------------------------------------ NanoVDB

# Byte layouts for NanoVDB v32 FloatGrid. Sizes verified against this file:
# 16 (FileHeader) + 176 (FileMetaData) + nameSize + fileSize == file size.
LEAF_STRIDE = 2144   # bboxMin 12 + bboxDif 3 + flags 1 + valueMask 64
LEAF_VALUES = 96     # + min/max/avg/std 16, then 512 floats
GRID_DATA = 672      # GridData, then TreeData


def read_nvdb(path):
    """Returns (dense[x,y,z] float32, indexBBoxMin, voxelSize)."""
    raw = open(path, 'rb').read()
    magic, ver, gridCount, codec = struct.unpack_from('<QIHH', raw, 0)
    if magic != 0x304244566F6E614E:
        raise SystemExit("not a NanoVDB file: %s" % path)
    print("[nvdb] version %d.%d.%d, %d grid(s), codec %d"
          % ((ver >> 21) & 0x7FF, (ver >> 10) & 0x7FF, ver & 0x3FF, gridCount, codec))

    m = 16
    gridSize, fileSize, nameKey, voxelCount = struct.unpack_from('<QQQQ', raw, m)
    gridType, gridClass = struct.unpack_from('<II', raw, m + 32)
    ib = struct.unpack_from('<6i', raw, m + 88)
    voxelSize = struct.unpack_from('<3d', raw, m + 112)
    nameSize, = struct.unpack_from('<I', raw, m + 136)
    if gridType != 1:
        raise SystemExit("expected a float grid (gridType 1), got %d" % gridType)
    print("[nvdb] grid %r  %d active voxels  voxelSize %g"
          % (raw[192:192 + nameSize].rstrip(b'\0').decode(), voxelCount, voxelSize[0]))

    blob_off = 16 + 176 + nameSize
    if codec == 1:                       # ZIP: uint64 length, then a zlib stream
        n, = struct.unpack_from('<Q', raw, blob_off)
        blob = zlib.decompress(raw[blob_off + 8: blob_off + 8 + n], bufsize=gridSize)
    elif codec == 0:
        blob = raw[blob_off: blob_off + gridSize]
    else:
        raise SystemExit("unsupported codec %d (only NONE/ZIP)" % codec)
    if len(blob) != gridSize:
        raise SystemExit("blob is %d bytes, metadata says %d" % (len(blob), gridSize))

    # TreeData sits right after GridData. mNodeOffset[0] and mNodeCount[0] are
    # the leaf level; offsets are relative to the start of the grid.
    # mNodeOffset is relative to the start of TreeData, not the grid. The check
    # is exact: GRID_DATA + offset + leafCount*stride == gridSize.
    nodeOffset = struct.unpack_from('<4q', blob, GRID_DATA)
    nodeCount = struct.unpack_from('<3I', blob, GRID_DATA + 32)
    leafOff, leafCount = GRID_DATA + nodeOffset[0], nodeCount[0]
    print("[nvdb] %d leaves at offset %d (stride %d -> %.1f MB)"
          % (leafCount, leafOff, LEAF_STRIDE, leafCount * LEAF_STRIDE / 1e6))
    if leafOff + leafCount * LEAF_STRIDE > gridSize:
        raise SystemExit("leaf array overruns the grid; stride assumption is wrong")

    leaves = np.frombuffer(blob, dtype=np.uint8, count=leafCount * LEAF_STRIDE,
                           offset=leafOff).reshape(leafCount, LEAF_STRIDE)
    # mBBoxMin is the leaf's TIGHT active bbox, not its origin - updateBBox()
    # shrinks it - so floor to the enclosing 8-cube to recover the origin.
    bbmin = leaves[:, 0:12].copy().view(np.int32).reshape(leafCount, 3)
    origins = (bbmin // 8) * 8
    values = leaves[:, LEAF_VALUES:LEAF_VALUES + 2048].copy().view(np.float32)
    masks = leaves[:, 16:80]

    if ((bbmin < np.array(ib[:3])) | (bbmin > np.array(ib[3:]))).any():
        raise SystemExit("leaf bboxes fall outside the metadata index bbox")

    # Zero out inactive voxels. NanoVDB indexes a leaf as (x<<6)|(y<<3)|z and
    # its mask bit n is word[n>>6] & (1<<(n&63)), which for a little-endian
    # byte view is exactly unpackbits(bitorder='little') at position n.
    active = np.unpackbits(masks, axis=1, bitorder='little').astype(bool)
    values = np.where(active, values, 0.0).reshape(leafCount, 8, 8, 8)

    # A leaf's 8-cube can hang off the edge of the grid's tight index bbox, so
    # every splat is clipped. Only inactive (zero) voxels are ever dropped.
    dims = [ib[3 + i] - ib[i] + 1 for i in range(3)]
    dense = np.zeros(dims, dtype=np.float32)
    base = origins - np.array(ib[:3], dtype=np.int32)
    clipped = 0
    for i in range(leafCount):
        p = base[i]
        d0 = np.maximum(p, 0)
        d1 = np.minimum(p + 8, dims)
        if (d1 <= d0).any():
            continue
        s0, s1 = d0 - p, d1 - p
        if (s0 != 0).any() or (s1 != 8).any():
            clipped += 1
        dense[d0[0]:d1[0], d0[1]:d1[1], d0[2]:d1[2]] = \
            values[i][s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
    if clipped:
        print("[nvdb] %d leaves clipped at the grid boundary" % clipped)

    got = int((dense > 0).sum())
    print("[nvdb] dense %s, %d nonzero (metadata says %d active)"
          % (dims, got, voxelCount))
    return dense, np.array(ib[:3]), voxelSize[0]


def write_vol(path, data, bbox_min, bbox_max):
    """VOL1 + W,H,D + bbox + dense floats, x-fastest (see scripts/vdb_to_vol.py)."""
    W, H, D = data.shape
    out = np.ascontiguousarray(data.transpose(2, 1, 0)).astype(np.float32)
    with open(path, 'wb') as f:
        f.write(b'VOL1')
        f.write(struct.pack('<III', W, H, D))
        f.write(struct.pack('<fff', *bbox_min))
        f.write(struct.pack('<fff', *bbox_max))
        f.write(out.tobytes())
    print("[vol]  %s: %dx%dx%d (%.1f MB), density max %.4f"
          % (path, W, H, D, W * H * D * 4 / 1048576.0, data.max()))


# ------------------------------------------------------------------- envmap

def convert_sky(exr_path, hdr_path, width=4096, height=2048):
    """Equal-area octahedral -> equirectangular. The light carries
    `Rotate 10 1 0 0` in the pbrt Z-up world; composed with the Rx(-90) that
    takes that world to ours, light-to-world is Rx(-80)."""
    import cv2
    try:
        import OpenEXR, Imath
    except ImportError:
        raise SystemExit(
            "This step needs the OpenEXR python bindings, missing for %s.\n"
            "OpenCV 5 has no EXR decoder, so there is no fallback." % sys.executable)
    exr = OpenEXR.InputFile(exr_path)
    dw = exr.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    img = np.stack([np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(H, W)
                    for c in ('B', 'G', 'R')], axis=-1)

    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    phi = (u * 2.0 * math.pi)[None, :]
    theta = (v * math.pi)[:, None]
    ones = np.ones((height, width))
    d = np.stack([np.sin(theta) * np.cos(phi) * ones,
                  np.cos(theta) * ones,
                  np.sin(theta) * np.sin(phi) * ones], axis=-1)

    t = math.radians(-80.0)
    c, s = math.cos(t), math.sin(t)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    dl = d.reshape(-1, 3) @ R

    su, sv = equal_area_sphere_to_square(dl[:, 0], dl[:, 1], dl[:, 2])
    px = np.clip((su * W).astype(np.int32), 0, W - 1)
    py = np.clip((sv * H).astype(np.int32), 0, H - 1)
    out = img[py, px].reshape(height, width, 3).astype(np.float32)
    cv2.imwrite(hdr_path, out)
    print("[sky]  %s -> %s (%dx%d)" % (exr_path, hdr_path, width, height))


# --------------------------------------------------------------------- main

def main():
    dense, ibmin, vs = read_nvdb(NVDB)

    # Grid index space -> pbrt medium space is just scale-by-voxelSize.
    med_min = ibmin * vs
    med_max = (ibmin + np.array(dense.shape)) * vs

    # medium-to-world is Rz(180)Rx(90); world Z-up -> our Y-up is Rx(-90).
    # The composite is Ry(180): x -> -x, z -> -z, y unchanged. That is a flip
    # of the dense array on axes 0 and 2 rather than a transpose.
    dense = dense[::-1, :, ::-1].copy()
    lo = np.array([-med_max[0], med_min[1], -med_max[2]])
    hi = np.array([-med_min[0], med_max[1], -med_min[2]])
    print("[xform] world bounds %s .. %s" % (np.round(lo, 3), np.round(hi, 3)))

    if DOWNSAMPLE > 1:
        n = DOWNSAMPLE
        sh = [(d // n) * n for d in dense.shape]
        dense = dense[:sh[0], :sh[1], :sh[2]].reshape(
            sh[0] // n, n, sh[1] // n, n, sh[2] // n, n).mean(axis=(1, 3, 5))
        print("[xform] downsampled by %d -> %s" % (n, list(dense.shape)))

    # Our shader wants density in [0,1] and folds the scale into sigma, so
    # normalise and push the peak density into the coefficients instead.
    dmax = float(dense.max())
    dense = (dense / dmax).astype(np.float32)
    sigmaA, sigmaS = 0.5 * dmax, 10.0 * dmax
    print("[xform] density peak %.4f -> sigmaA %.4f sigmaS %.4f" % (dmax, sigmaA, sigmaS))

    write_vol(VOL, dense, lo, hi)
    convert_sky(SKY_EXR, SKY_HDR)

    # Ground: pbrt has a DISK of radius 1000 in the z=0 plane (Z-up), i.e. a
    # horizontal disk at y=0 for us. It has to be a disk and not a quad: the
    # camera looks 15.4 deg down with a 12.5 deg half-angle, so the top of the
    # frame grazes ~1004 units out. A quad of half-size 1000 reaches 1414 at
    # its corners and would cover the sky in the upper corners, which is
    # exactly where the reference shows blue.
    r, seg = 1000.0, 256
    with open(os.path.join(OUT, "meshes", "ground.obj"), 'w') as f:
        f.write("# ground disk radius %g at y=0 (pbrt: Shape \"disk\")\n" % r)
        # pbrt does `Translate 0 -50 0` then a disk in the z=0 plane of a Z-up
        # world, which lands as a disk at y=0 centred on (0,0,50) for us.
        cz = 50.0
        f.write("v 0 0 %g\n" % cz)
        for i in range(seg):
            a = 2.0 * math.pi * i / seg
            f.write("v %g 0 %g\n" % (r * math.cos(a), cz + r * math.sin(a)))
        f.write("vn 0 1 0\n")
        for i in range(seg):
            f.write("f 1//1 %d//1 %d//1\n" % (2 + i, 2 + (i + 1) % seg))

    # Z-up -> Y-up on the camera: (x,y,z) -> (x,z,-y).
    eye = (0.0, 50.0, -120.0)
    tgt = (7.0, 17.0, 0.0)
    # pbrt's fov is on the SHORT axis; ours is horizontal.
    fov = math.degrees(2.0 * math.atan(math.tan(math.radians(25.0) / 2.0)
                                       * W_IMG / float(H_IMG)))
    ground = coated_diffuse_albedo((0.4, 0.45, 0.35))

    xml = """<?xml version='1.0' encoding='utf-8'?>

<!-- Bunny cloud, converted from mmp/pbrt-v4-scenes/bunny-cloud by
     _pbrt_bunnycloud_to_nori.py. The density grid is the original
     bunny_cloud.nvdb decoded to our dense .vol; the source scene is Z-up and
     has been rotated into Y-up because our GPU camera is Y-up only. -->
<scene>
\t<string name="envmap" value="textures/sky.hdr"/>
\t<float name="envmapScale" value="4.0"/>
\t<float name="envmapRotation" value="0.0"/>

\t<camera type="perspective">
\t\t<float name="fov" value="%.4f"/>
\t\t<transform name="toWorld">
\t\t\t<matrix value="%s"/>
\t\t</transform>
\t\t<integer name="width" value="%d"/>
\t\t<integer name="height" value="%d"/>
\t</camera>

\t<sampler type="independent">
\t\t<integer name="sampleCount" value="%d"/>
\t</sampler>

\t<!-- sigma_s 10 / sigma_a 0.5 from the pbrt scene, scaled by the grid's peak
\t     density because our shader takes density normalised to [0,1] and
\t     computes sigma_t(x) = (sigmaA + sigmaS) * density(x). g=0: the pbrt
\t     scene does not set one, and pbrt's default phase is isotropic. -->
\t<medium type="heterogeneous">
\t\t<string name="volume" value="volumes/bunny_cloud.vol"/>
\t\t<float name="sigmaA" value="%.4f"/>
\t\t<float name="sigmaS" value="%.4f"/>
\t\t<float name="g" value="0.0"/>
\t\t<point name="boundsMin" value="%.4f, %.4f, %.4f"/>
\t\t<point name="boundsMax" value="%.4f, %.4f, %.4f"/>
\t</medium>

\t<!-- pbrt coateddiffuse with roughness 0: a SMOOTH dielectric coat over a
\t     diffuse base. Keeping it smooth matters here. The camera grazes the
\t     ground at ~3 deg at the top of frame, where Fresnel approaches 1, so the
\t     far ground mirrors the sky and reads blue - that reflection is most of
\t     the upper half of the reference image, and a rough ground loses it
\t     entirely. Base colour is pre-multiplied by the coupled-diffuse factor,
\t     same correction as the BMW scene. -->
\t<mesh type="obj">
\t\t<string name="filename" value="meshes/ground.obj"/>
\t\t<bsdf type="disney">
\t\t\t<color name="baseColor" value="%.5f %.5f %.5f"/>
\t\t\t<float name="roughness" value="0.02"/>
\t\t\t<float name="metallic" value="0.0"/>
\t\t\t<float name="specular" value="0.5"/>
\t\t</bsdf>
\t</mesh>

</scene>
""" % (fov, lookat_matrix(eye, tgt, (0, 1, 0)), W_IMG, H_IMG, SPP,
       sigmaA, sigmaS, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2],
       ground[0], ground[1], ground[2])

    p = os.path.join(OUT, "scene.xml")
    open(p, 'w').write(xml)
    print("[xml]  %s (fov %.3f, %d spp)" % (p, fov, SPP))


if __name__ == '__main__':
    os.makedirs(os.path.join(OUT, "meshes"), exist_ok=True)
    main()
