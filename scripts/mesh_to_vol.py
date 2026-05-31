#!/usr/bin/env python3
"""
mesh_to_vol.py — Voxelize a watertight OBJ mesh into a .vol density grid.

Computes a signed distance field using the generalized winding number for
inside/outside classification.  Density is 1.0 deep inside the mesh and
falls off smoothly near the surface, with optional FBM noise modulation
for cloud-like detail.

Usage:
    python3 mesh_to_vol.py cloud_guide.obj cloud.vol [--res 100] [--padding 0.1]

The output .vol uses the same coordinate space as the OBJ.
"""
import struct, argparse, os
import numpy as np

# ── OBJ loader (vertices + faces only) ──────────────────────────────

def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                idx = []
                for p in parts[1:]:
                    idx.append(int(p.split('/')[0]) - 1)
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)


# ── Solid angle of a triangle as seen from a point ──────────────────

def solid_angles_batch(points, v0, v1, v2):
    """
    Compute the signed solid angle subtended by triangle (v0,v1,v2) as
    seen from each query point.  Uses the Van Oosterom & Strackee formula:

        tan(Ω/2) = (a · (b × c)) / (abc + c(a·b) + a(b·c) + b(a·c))

    where a, b, c are unit vectors from the query point to each vertex
    and a, b, c (scalars) are their magnitudes.

    Returns array of solid angles in [-2π, 2π] (one per query point).
    """
    N = len(points)
    omegas = np.zeros(N, dtype=np.float64)
    chunk = 100000

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        p = points[start:end]

        a_vec = v0 - p  # (M, 3)
        b_vec = v1 - p
        c_vec = v2 - p

        a_len = np.linalg.norm(a_vec, axis=1, keepdims=True)
        b_len = np.linalg.norm(b_vec, axis=1, keepdims=True)
        c_len = np.linalg.norm(c_vec, axis=1, keepdims=True)

        # Skip degenerate (point on vertex)
        degen = ((a_len < 1e-10) | (b_len < 1e-10) | (c_len < 1e-10)).ravel()

        a_hat = a_vec / np.maximum(a_len, 1e-10)
        b_hat = b_vec / np.maximum(b_len, 1e-10)
        c_hat = c_vec / np.maximum(c_len, 1e-10)

        # Numerator: a · (b × c)
        bc_cross = np.cross(b_hat, c_hat)
        numer = np.sum(a_hat * bc_cross, axis=1)

        # Denominator
        ab = np.sum(a_hat * b_hat, axis=1)
        bc = np.sum(b_hat * c_hat, axis=1)
        ac = np.sum(a_hat * c_hat, axis=1)
        a_s = a_len.ravel()
        b_s = b_len.ravel()
        c_s = c_len.ravel()

        # Normalize lengths out (they cancel, but we need them for stability
        # in the original formula — here we already use unit vectors)
        denom = 1.0 + ab + bc + ac

        omega = 2.0 * np.arctan2(numer, denom)
        omega[degen] = 0.0
        omegas[start:end] = omega

    return omegas


def winding_number(points, verts, faces):
    """
    Generalized winding number: sum of solid angles / 4π.
    Returns ~1.0 for points inside a closed mesh, ~0.0 for outside.
    """
    N = len(points)
    wn = np.zeros(N, dtype=np.float64)
    num_faces = len(faces)
    report_interval = max(1, num_faces // 10)

    for fi in range(num_faces):
        v0 = verts[faces[fi, 0]]
        v1 = verts[faces[fi, 1]]
        v2 = verts[faces[fi, 2]]
        wn += solid_angles_batch(points, v0, v1, v2)

        if (fi + 1) % report_interval == 0:
            print(f"    winding number: {fi + 1}/{num_faces} triangles")

    return wn / (4.0 * np.pi)


# ── Unsigned distance from point to triangle ────────────────────────

def point_triangle_dist_batch(points, v0, v1, v2):
    """Unsigned distance from each point to triangle (v0,v1,v2)."""
    edge0 = v1 - v0
    edge1 = v2 - v0

    N = len(points)
    dists = np.full(N, np.inf)
    chunk = 50000

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        p = points[start:end]
        v0p = p - v0

        d00 = np.dot(edge0, edge0)
        d01 = np.dot(edge0, edge1)
        d11 = np.dot(edge1, edge1)
        d20 = v0p @ edge0
        d21 = v0p @ edge1

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            t = np.clip(d20 / max(d00, 1e-12), 0, 1)
            closest = v0 + np.outer(t, edge0)
            dists[start:end] = np.linalg.norm(p - closest, axis=1)
            continue

        s = (d11 * d20 - d01 * d21) / denom
        t = (d00 * d21 - d01 * d20) / denom

        inside = (s >= 0) & (t >= 0) & (s + t <= 1)
        closest_inside = v0 + np.outer(s, edge0) + np.outer(t, edge1)

        # Edge v0-v1
        s01 = np.clip(d20 / max(d00, 1e-12), 0, 1)
        c01 = v0 + np.outer(s01, edge0)
        d_01 = np.linalg.norm(p - c01, axis=1)

        # Edge v0-v2
        t02 = np.clip(d21 / max(d11, 1e-12), 0, 1)
        c02 = v0 + np.outer(t02, edge1)
        d_02 = np.linalg.norm(p - c02, axis=1)

        # Edge v1-v2
        edge12 = v2 - v1
        d12 = np.dot(edge12, edge12)
        v1p = p - v1
        t12 = np.clip((v1p @ edge12) / max(d12, 1e-12), 0, 1)
        c12 = v1 + np.outer(t12, edge12)
        d_12 = np.linalg.norm(p - c12, axis=1)

        d_outside = np.minimum(d_01, np.minimum(d_02, d_12))
        d_inside_tri = np.linalg.norm(p - closest_inside, axis=1)

        dists[start:end] = np.where(inside, d_inside_tri, d_outside)

    return dists


# ── Simple 3D value noise ───────────────────────────────────────────

def _hash3(ix, iy, iz):
    n = ix * 374761393 + iy * 668265263 + iz * 1274126177
    n = ((n ^ (n >> 13)) * 1103515245 + 12345) & 0x7FFFFFFF
    return n / 0x7FFFFFFF


def value_noise_3d(x, y, z):
    ix, iy, iz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    fx, fy, fz = x - ix, y - iy, z - iz
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    fz = fz * fz * (3 - 2 * fz)

    c000 = _hash3(ix, iy, iz)
    c100 = _hash3(ix + 1, iy, iz)
    c010 = _hash3(ix, iy + 1, iz)
    c110 = _hash3(ix + 1, iy + 1, iz)
    c001 = _hash3(ix, iy, iz + 1)
    c101 = _hash3(ix + 1, iy, iz + 1)
    c011 = _hash3(ix, iy + 1, iz + 1)
    c111 = _hash3(ix + 1, iy + 1, iz + 1)

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def fbm_noise(x, y, z, octaves=4, lacunarity=2.0, gain=0.5):
    val = 0.0
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        val += amp * value_noise_3d(x * freq, y * freq, z * freq)
        amp *= gain
        freq *= lacunarity
    return val


# ── Voxelization ────────────────────────────────────────────────────

def voxelize_mesh(verts, faces, res, padding_frac=0.1,
                  falloff_width=None, noise_scale=3.0, noise_strength=0.3):
    """
    Create a dense density grid from a watertight mesh.

    1. Compute unsigned distance to nearest triangle for every voxel.
    2. Compute generalized winding number to classify inside/outside.
    3. Sign the distance: negative inside, positive outside.
    4. Convert signed distance → density with smooth falloff at boundary.
    5. Optionally modulate with FBM noise.
    """
    # Bounding box with padding
    mesh_min = verts.min(axis=0)
    mesh_max = verts.max(axis=0)
    extent = mesh_max - mesh_min
    pad = extent * padding_frac
    bbox_min = mesh_min - pad
    bbox_max = mesh_max + pad
    extent_padded = bbox_max - bbox_min

    max_ext = extent_padded.max()
    dims = np.maximum((extent_padded / max_ext * res + 0.5).astype(int), 4)
    W, H, D = int(dims[0]), int(dims[1]), int(dims[2])

    print(f"  Grid: {W}x{H}x{D} = {W * H * D} voxels")
    print(f"  Padded bbox: ({bbox_min[0]:.4f},{bbox_min[1]:.4f},{bbox_min[2]:.4f}) - "
          f"({bbox_max[0]:.4f},{bbox_max[1]:.4f},{bbox_max[2]:.4f})")

    if falloff_width is None:
        falloff_width = min(extent) * 0.3
    print(f"  Falloff width: {falloff_width:.4f}")

    # Grid points
    xs = np.linspace(bbox_min[0], bbox_max[0], W)
    ys = np.linspace(bbox_min[1], bbox_max[1], H)
    zs = np.linspace(bbox_min[2], bbox_max[2], D)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # ── Step 1: unsigned distance to nearest triangle ───────────────
    print(f"  Computing distances to {len(faces)} triangles...")
    min_dist = np.full(len(points), np.inf, dtype=np.float64)
    num_faces = len(faces)
    report_interval = max(1, num_faces // 10)

    for fi in range(num_faces):
        v0 = verts[faces[fi, 0]]
        v1 = verts[faces[fi, 1]]
        v2 = verts[faces[fi, 2]]

        # AABB culling
        tri_min = np.minimum(v0, np.minimum(v1, v2)) - falloff_width * 2
        tri_max = np.maximum(v0, np.maximum(v1, v2)) + falloff_width * 2
        if (tri_max < bbox_min).any() or (tri_min > bbox_max).any():
            continue

        d = point_triangle_dist_batch(points, v0, v1, v2)
        np.minimum(min_dist, d, out=min_dist)

        if (fi + 1) % report_interval == 0:
            print(f"    distance: {fi + 1}/{num_faces} triangles")

    # ── Step 2: winding number for inside/outside ───────────────────
    print(f"  Computing winding number...")
    wn = winding_number(points, verts, faces)

    # Inside = |winding number| > 0.5 (handles either face winding order)
    is_inside = np.abs(wn) > 0.5

    # ── Step 3: signed distance (negative inside) ───────────────────
    signed_dist = np.where(is_inside, -min_dist, min_dist)

    # ── Step 4: density from signed distance ────────────────────────
    # Points deep inside (signed_dist << -falloff_width) → density 1.0
    # Points near surface (signed_dist ~ 0) → smooth falloff
    # Points outside (signed_dist > 0) → density 0.0
    density = np.clip(-signed_dist / falloff_width, 0.0, 1.0)
    density = density.reshape((W, H, D))

    # ── Step 5: noise modulation ────────────────────────────────────
    if noise_strength > 0:
        print(f"  Adding FBM noise (scale={noise_scale}, strength={noise_strength})...")
        for ix in range(W):
            for iy in range(H):
                for iz in range(D):
                    if density[ix, iy, iz] > 0.001:
                        n = fbm_noise(xs[ix] * noise_scale,
                                      ys[iy] * noise_scale,
                                      zs[iz] * noise_scale, octaves=4)
                        density[ix, iy, iz] *= max(0.0, 1.0 + noise_strength * (n - 0.5))

    # Normalize to [0, 1]
    mx = density.max()
    if mx > 0:
        density /= mx

    return density.astype(np.float32), tuple(bbox_min), tuple(bbox_max)


# ── .vol writer ─────────────────────────────────────────────────────

def write_vol(path, data, bbox_min, bbox_max):
    W, H, D = data.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'VOL1')
        f.write(struct.pack('<III', W, H, D))
        f.write(struct.pack('<fff', *bbox_min))
        f.write(struct.pack('<fff', *bbox_max))
        f.write(data.astype(np.float32).tobytes())
    mb = (W * H * D * 4) / (1024 * 1024)
    print(f"\nWrote {path}: {W}x{H}x{D} ({mb:.1f} MB)")
    print(f"  bbox=({bbox_min[0]:.4f},{bbox_min[1]:.4f},{bbox_min[2]:.4f})"
          f"-({bbox_max[0]:.4f},{bbox_max[1]:.4f},{bbox_max[2]:.4f})")
    print(f"  density range=[{data.min():.4f}, {data.max():.4f}]")


def main():
    p = argparse.ArgumentParser(description="Voxelize OBJ mesh to .vol cloud")
    p.add_argument('input', help='Input .obj mesh (should be watertight)')
    p.add_argument('output', help='Output .vol file')
    p.add_argument('--res', type=int, default=100,
                   help='Max voxel resolution along longest axis (default: 100)')
    p.add_argument('--padding', type=float, default=0.1,
                   help='Padding fraction around mesh bbox (default: 0.1)')
    p.add_argument('--falloff', type=float, default=None,
                   help='Density falloff width in world units (default: auto)')
    p.add_argument('--noise-scale', type=float, default=3.0,
                   help='Noise frequency scale (default: 3.0)')
    p.add_argument('--noise-strength', type=float, default=0.3,
                   help='Noise modulation strength (default: 0.3, 0=off)')
    args = p.parse_args()

    print(f"Loading {args.input}...")
    verts, faces = load_obj(args.input)
    print(f"  {len(verts)} vertices, {len(faces)} triangles")

    print("Voxelizing...")
    data, bmin, bmax = voxelize_mesh(
        verts, faces, args.res,
        padding_frac=args.padding,
        falloff_width=args.falloff,
        noise_scale=args.noise_scale,
        noise_strength=args.noise_strength
    )

    write_vol(args.output, data, bmin, bmax)
    print(f'\nXML snippet:')
    print(f'    <point name="boundsMin" value="{bmin[0]:.4f}, {bmin[1]:.4f}, {bmin[2]:.4f}"/>')
    print(f'    <point name="boundsMax" value="{bmax[0]:.4f}, {bmax[1]:.4f}, {bmax[2]:.4f}"/>')


if __name__ == '__main__':
    main()