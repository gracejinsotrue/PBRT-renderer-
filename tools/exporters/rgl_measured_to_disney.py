"""Derive a Disney stand-in for every RGL measured BSDF in a directory."""
import struct, sys, glob, os, numpy as np

NP = {1: np.uint8, 2: np.int8, 3: np.uint16, 4: np.int16, 5: np.uint32,
      6: np.int32, 7: np.uint64, 8: np.int64, 9: np.float16, 10: np.float32,
      11: np.float64}


def load(path):
    d = open(path, 'rb').read()
    assert d[:12] == b'tensor_file\0'
    nf = struct.unpack('<I', d[14:18])[0]
    off, fields = 18, {}
    for _ in range(nf):
        nl = struct.unpack('<H', d[off:off + 2])[0]; off += 2
        name = d[off:off + nl].decode(); off += nl
        ndim = struct.unpack('<H', d[off:off + 2])[0]; off += 2
        dtype = d[off]; off += 1
        doff = struct.unpack('<Q', d[off:off + 8])[0]; off += 8
        shape = struct.unpack('<%dQ' % ndim, d[off:off + 8 * ndim]); off += 8 * ndim
        n = int(np.prod(shape)) if ndim else 1
        fields[name] = np.frombuffer(d, NP[dtype], n, doff).reshape(shape)
    return fields


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689, 1.8758, 0.0415],
                       [0.0557, -0.2040, 1.0570]])


def spec_to_rgb(lam, spec):
    m = (lam >= 380) & (lam <= 780)
    l, s = lam[m], spec[m]
    xb = (1.056 * _g(l, 599.8, 37.9, 31.0) + 0.362 * _g(l, 442.0, 16.0, 26.7)
          - 0.065 * _g(l, 501.1, 20.4, 26.2))
    yb = 0.821 * _g(l, 568.8, 46.9, 40.5) + 0.286 * _g(l, 530.9, 16.3, 31.1)
    zb = 1.217 * _g(l, 437.0, 11.8, 36.0) + 0.681 * _g(l, 459.0, 26.0, 13.8)
    I = np.trapezoid
    xyz = np.array([I(s * xb, l), I(s * yb, l), I(s * zb, l)]) / I(yb, l)
    return np.maximum(XYZ_TO_RGB @ xyz, 0.0)


# The XYZ->sRGB matrix is D65-referenced, but integrating a flat spectrum
# through the matching functions gives illuminant E. Without adapting, a
# spectrally flat sample comes out magenta. White-balance by dividing through
# the RGB of a flat spectrum, which sends equal-energy to neutral grey.
_WL = np.linspace(380.0, 780.0, 401)
WB = spec_to_rgb(_WL, np.ones_like(_WL))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


for path in sorted(glob.glob(os.path.join(sys.argv[1], '*.bsdf'))):
    F = load(path)
    stem = os.path.basename(path).replace('.bsdf', '')
    desc = bytes(F['description']).decode('utf8', 'replace')
    lam = np.asarray(F['wavelengths'], np.float64)
    sp = np.asarray(F['spectra'], np.float64)[0]
    cols = np.array([spec_to_rgb(lam, sp[i].reshape(sp.shape[1], -1).mean(1)) / WB
                     for i in range(sp.shape[0])])
    drift = np.degrees(np.arccos(np.clip(unit(cols[0]) @ unit(cols[-2]), -1, 1)))
    a = float(np.sqrt(1.0 / (np.pi * F['ndf'].max())))     # GGX D(0)=1/(pi a^2)
    rough = float(np.sqrt(a))                              # our alpha = rough^2
    print('%s' % stem)
    print('   %s' % desc[:86])
    print('   rgb @0deg   %6.4f %6.4f %6.4f     @grazing  %6.4f %6.4f %6.4f'
          % (tuple(cols[0]) + tuple(cols[-2])))
    print('   hue drift %5.1f deg     alpha %.5f -> disney roughness %.3f'
          % (drift, a, rough))
    print()
