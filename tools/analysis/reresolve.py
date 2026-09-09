"""Re-run the resolve pass on a saved EXR with an adjustable bloom falloff.

Mirrors shaders/Resolve.hlsl and shaders/Bloom.hlsl exactly, with one extra
parameter the shader does not have yet: a per-level weight applied to each
coarse mip as it is accumulated back up the chain. The shipped chain adds
every level at full weight, so the coarsest mip -- one texel per 64x64 render
pixels at 1080p -- contributes as much energy as mip 0 and a bright emitter
spreads a wide flat halo that clips to white. A falloff below 1 keeps the core
and thins the tail.

Validated against snapshot_64_denoised.png: falloff 1.0 reproduces the
renderer clipped-white fraction to 0.0069 vs 0.0067.

    python tools/analysis/reresolve.py in.exr out.png --ev -1.95 \
        --intensity 0.45 --falloff 0.55
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import exrtool

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def prefilter(c, threshold, knee):
    h, w = c.shape[0] // 2, c.shape[1] // 2
    blocks = c[: h * 2, : w * 2].reshape(h, 2, w, 2, 3)
    weight = 1.0 / (1.0 + blocks @ LUMA)
    avg = (blocks * weight[..., None]).sum((1, 3)) / weight.sum((1, 3))[..., None]

    bright = avg.max(2)
    soft = np.clip(bright - threshold + knee, 0.0, 2.0 * knee)
    soft = soft * soft / (4.0 * knee + 1e-6)
    contrib = np.maximum(soft, bright - threshold) / np.maximum(bright, 1e-6)
    return avg * contrib[..., None]


def downsample(a):
    h, w = a.shape[0] // 2, a.shape[1] // 2
    return a[: h * 2, : w * 2].reshape(h, 2, w, 2, 3).mean((1, 3))


def tent3x3(a):
    k = np.array([0.25, 0.5, 0.25], np.float32)
    pad = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    a = sum(k[i] * pad[i : i + a.shape[0], 1:-1] for i in range(3))
    pad = np.pad(a, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return sum(k[i] * pad[:, i : i + a.shape[1]] for i in range(3))


def upsample(a, shape):
    h, w = shape
    y = (np.arange(h) + 0.5) / 2.0 - 0.5
    x = (np.arange(w) + 0.5) / 2.0 - 0.5
    y0 = np.clip(np.floor(y).astype(int), 0, a.shape[0] - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, a.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, a.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, a.shape[1] - 1)
    fy = (y - np.floor(y)).astype(np.float32)[:, None, None]
    fx = (x - np.floor(x)).astype(np.float32)[None, :, None]
    top = a[np.ix_(y0, x0)] * (1 - fx) + a[np.ix_(y0, x1)] * fx
    bot = a[np.ix_(y1, x0)] * (1 - fx) + a[np.ix_(y1, x1)] * fx
    return top * (1 - fy) + bot * fy


def mip_levels(width, height):
    lo = min(width, height)
    n = 1
    while (lo >> (n + 1)) >= 8 and n < 6:
        n += 1
    return n


def bloom(c, threshold, knee, falloff):
    chain = [prefilter(c, threshold, knee)]
    for _ in range(mip_levels(c.shape[1], c.shape[0]) - 1):
        chain.append(downsample(chain[-1]))
    for i in range(len(chain) - 1, 0, -1):
        chain[i - 1] += upsample(tent3x3(chain[i]), chain[i - 1].shape[:2]) * falloff
    return chain[0]


def aces(c):
    return np.clip((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0)


def resolve(hdr, ev, intensity, threshold, knee, falloff):
    c = hdr * (2.0 ** ev)
    if intensity > 0.0:
        c = c + upsample(bloom(c, threshold, knee, falloff), c.shape[:2]) * intensity
    return aces(c) ** (1.0 / 2.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exr")
    p.add_argument("png")
    p.add_argument("--ev", type=float, default=0.0)
    p.add_argument("--intensity", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--knee", type=float, default=0.5)
    p.add_argument("--falloff", type=float, default=1.0)
    p.add_argument("--dither", type=float, default=1.0)
    a = p.parse_args()

    hdr = exrtool.read(a.exr)[0].astype(np.float32)
    out = resolve(hdr, a.ev, a.intensity, a.threshold, a.knee, a.falloff)

    # Triangular dither before quantisation. A denoised render has no grain, so a
    # smooth sky gradient lands on 8-bit contour arcs about one level deep, which
    # the eye picks up as Mach bands; the solar aureole made them visible because it
    # steepened the gradient. One level of TPDF noise breaks the contours and is
    # under the noise floor of any real photograph.
    q = out * 255.0
    if a.dither > 0.0:
        rng = np.random.default_rng(7)
        q = q + a.dither * (rng.random(q.shape, np.float32) - rng.random(q.shape, np.float32))
    q = np.clip(q + 0.5, 0.0, 255.0).astype(np.uint8)

    import cv2

    cv2.imwrite(a.png, q[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6])

    clipped = (out.min(2) > 0.95).mean()
    print("%s  %dx%d  clipped-white %.4f" % (a.png, out.shape[1], out.shape[0], clipped))


if __name__ == "__main__":
    main()
