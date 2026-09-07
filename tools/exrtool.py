#!/usr/bin/env python3
"""Minimal reader for the scanline float EXRs this renderer writes, plus error metrics.

Deliberately dependency-light (stdlib zlib + numpy) because installing OpenEXR on Windows
is more trouble than parsing the subset we actually emit: scanline, FLOAT channels,
NONE/ZIP/ZIPS compression, single part.

  python tools/exrtool.py info  a.exr
  python tools/exrtool.py rmse  a.exr b.exr
  python tools/exrtool.py curve ref.exr img1.exr img2.exr ...   # relMSE vs ref, one row each
"""
import struct, sys, zlib
import numpy as np

PIXTYPE = {0: ("UINT", 4), 1: ("HALF", 2), 2: ("FLOAT", 4)}


def _read_header(d):
    magic, ver = struct.unpack("<II", d[:8])
    if magic != 0x01312F76:
        raise ValueError("not an EXR file")
    if ver & 0x200:
        raise ValueError("tiled EXR not supported")
    p, attrs = 8, {}
    while True:
        e = d.index(b"\0", p); name = d[p:e].decode(); p = e + 1
        if not name:
            break
        e = d.index(b"\0", p); typ = d[p:e].decode(); p = e + 1
        (sz,) = struct.unpack("<i", d[p:p + 4]); p += 4
        attrs[name] = (typ, d[p:p + sz]); p += sz
    return attrs, p


def _channels(val):
    out, q = [], 0
    while q < len(val) and val[q] != 0:
        e = val.index(b"\0", q); nm = val[q:e].decode(); q = e + 1
        (pt,) = struct.unpack("<i", val[q:q + 4]); q += 16  # +xSampling/ySampling/pLinear
        out.append((nm, pt))
    return out  # already alphabetical, which is the on-disk order


def _unpredict(buf):
    """Undo EXR's ZIP delta + byte-deinterleave (ImfZipCompressor, in reverse)."""
    a = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
    a = np.cumsum(np.concatenate(([a[0]], a[1:] - 128)), dtype=np.int32).astype(np.uint8)
    half = (a.size + 1) // 2
    out = np.empty(a.size, dtype=np.uint8)
    out[0::2] = a[:half][: out[0::2].size]
    out[1::2] = a[half:][: out[1::2].size]
    return out.tobytes()


def read(path):
    """-> (HxWx3 float32 in R,G,B order, dict of channel name -> plane)."""
    d = open(path, "rb").read()
    attrs, p = _read_header(d)
    chans = _channels(attrs["channels"][1])
    x0, y0, x1, y1 = struct.unpack("<4i", attrs["dataWindow"][1])
    W, H = x1 - x0 + 1, y1 - y0 + 1
    comp = attrs["compression"][1][0]
    rows = {0: 1, 1: 1, 2: 1, 3: 16}.get(comp)
    if rows is None:
        raise ValueError(f"unsupported compression id {comp}")
    for nm, pt in chans:
        if PIXTYPE[pt][0] != "FLOAT":
            raise ValueError(f"channel {nm} is {PIXTYPE[pt][0]}, only FLOAT supported")

    nblocks = (H + rows - 1) // rows
    offsets = struct.unpack(f"<{nblocks}Q", d[p:p + 8 * nblocks])
    planes = {nm: np.empty((H, W), dtype=np.float32) for nm, _ in chans}
    rowbytes = W * 4 * len(chans)

    for off in offsets:
        y, sz = struct.unpack("<ii", d[off:off + 8])
        raw = d[off + 8: off + 8 + sz]
        n = min(rows, H - (y - y0))
        expect = n * rowbytes
        block = raw if (comp == 0 or sz >= expect) else _unpredict(zlib.decompress(raw))
        for i in range(n):
            base = i * rowbytes
            for j, (nm, _) in enumerate(chans):
                s = base + j * W * 4
                planes[nm][y - y0 + i] = np.frombuffer(block[s:s + W * 4], dtype="<f4")

    rgb = np.stack([planes.get(c, np.zeros((H, W), np.float32)) for c in ("R", "G", "B")], -1)
    return rgb, planes


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def relmse(img, ref, eps=1e-2):
    """Relative MSE, the standard rendering metric: squared error normalised by reference
    intensity so bright regions don't dominate the average."""
    return float(np.mean((img - ref) ** 2 / (ref ** 2 + eps)))


def main(argv):
    if len(argv) < 3:
        print(__doc__); return 1
    cmd = argv[1]
    if cmd == "info":
        img, planes = read(argv[2])
        print(f"{argv[2]}: {img.shape[1]}x{img.shape[0]} channels={sorted(planes)}")
        print(f"  min={img.min():.6g} max={img.max():.6g} mean={img.mean():.6g} "
              f"nan={int(np.isnan(img).sum())} inf={int(np.isinf(img).sum())}")
    elif cmd == "rmse":
        a, _ = read(argv[2]); b, _ = read(argv[3])
        if a.shape != b.shape:
            print(f"shape mismatch {a.shape} vs {b.shape}"); return 1
        print(f"rawRMSE {rmse(a, b):.8g}\nrelMSE  {relmse(a, b):.8g}")
    elif cmd == "curve":
        ref, _ = read(argv[2])
        print(f"{'image':<44} {'rawRMSE':>12} {'relMSE':>12}")
        for path in argv[3:]:
            img, _ = read(path)
            print(f"{path:<44} {rmse(img, ref):>12.6g} {relmse(img, ref):>12.6g}")
    else:
        print(__doc__); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
