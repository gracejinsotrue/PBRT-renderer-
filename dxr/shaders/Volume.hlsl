// Volume.hlsl — Multi-instance participating media.
//
// All volume math takes a GPUVolume parameter so the same code works for the
// current "loop over a flat structured buffer" enumeration and a future
// procedural-AABB BLAS path. SampleVolumeFreeFlight and MultiVolumeTransmittance
// are the only two entry points the path tracer needs.

// ---------------------------------------------------------------------------
// Ray / AABB
// ---------------------------------------------------------------------------

bool RayAABBIntersect(float3 origin, float3 dir,
                      float3 bmin, float3 bmax,
                      out float tNear, out float tFar)
{
    float3 invDir = 1.0 / dir;
    float3 t0 = (bmin - origin) * invDir;
    float3 t1 = (bmax - origin) * invDir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    tNear = max(max(tmin.x, tmin.y), tmin.z);
    tFar = min(min(tmax.x, tmax.y), tmax.z);
    return tFar >= max(tNear, 0.0);
}

// ---------------------------------------------------------------------------
// Phase function — Henyey-Greenstein [PBRT4 Eq. 11.24-25]
// ---------------------------------------------------------------------------

float HenyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 / (4.0 * M_PI)) * (1.0 - g2) / (denom * sqrt(max(denom, 1e-8)));
}

float3 SampleHG(float3 wo, float g, inout RNG rng, out float pdf)
{
    float xi1 = NextFloat(rng);
    float xi2 = NextFloat(rng);

    float cosTheta;
    if (abs(g) < 1e-3)
    {
        cosTheta = 1.0 - 2.0 * xi1;
    }
    else
    {
        float sqr = (1.0 - g * g) / (1.0 + g - 2.0 * g * xi1);
        cosTheta = (1.0 + g * g - sqr * sqr) / (2.0 * g);
        cosTheta = clamp(cosTheta, -1.0, 1.0);
    }

    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * M_PI * xi2;

    float3 T, B;
    BuildONB(wo, T, B);

    float3 wi = sinTheta * cos(phi) * T + sinTheta * sin(phi) * B + cosTheta * wo;

    pdf = HenyeyGreenstein(cosTheta, g);
    return normalize(wi);
}

// ---------------------------------------------------------------------------
// Free-flight in a homogeneous medium: p(s) = σ_t exp(-σ_t s)
// ---------------------------------------------------------------------------

float SampleFreeFlightHomogeneous(float sigmaT, inout RNG rng)
{
    float xi = max(NextFloat(rng), 1e-10);
    return -log(xi) / sigmaT;
}

// ---------------------------------------------------------------------------
// Per-volume helpers
// ---------------------------------------------------------------------------

bool VolumeIsHeterogeneous(GPUVolume v)
{
    return (v.flags & VOLUME_FLAG_HETEROGENEOUS) != 0u;
}

// Per-channel extinction.
float3 VolumeSigmaT(GPUVolume v)
{
    return v.sigmaA + v.sigmaS;
}

// Scalar majorant used for free-flight stepping. Picking max across
// channels gives an upper bound on per-channel extinction, so distance
// sampling under this majorant is conservative for every channel.
float VolumeSigmaTMax(GPUVolume v)
{
    float3 s = VolumeSigmaT(v);
    return max(s.r, max(s.g, s.b));
}

// Sample the density field of a single volume at a world-space point.
// For homogeneous volumes, density is implicitly 1 (σ_t already absolute).
// For heterogeneous volumes, density ∈ [0,1] modulates σ_t = σ_max · density(x).
float SampleVolumeDensity(GPUVolume v, float3 worldPos)
{
    if (!VolumeIsHeterogeneous(v) || v.densityTexIndex == VOLUME_INVALID_TEX)
        return 1.0;

    float3 uvw = (worldPos - v.vMin) / max(v.vMax - v.vMin, float3(1e-6, 1e-6, 1e-6));
    return g_volumeDensities[v.densityTexIndex].SampleLevel(g_volumeSampler, uvw, 0);
}

// ---------------------------------------------------------------------------
// Tracked majorants — local μ + brick boundary distance
// ---------------------------------------------------------------------------
//
// The brick-max-density mip stores, per coarse voxel, the maximum dense
// density inside that voxel's BRICK_SIZE^3 region. So the local majorant
// μ_local within a brick is brick_max · σ_t_volume_max — a tighter bound
// than the global μ wherever density is sparse, which makes free-flight
// step further per iteration.
//
// Returns:
//   localMu     – valid majorant for σ_t throughout the current brick.
//   boundaryDist – distance along the ray (forward) to the brick boundary.
//
// Falls back to (globalMu, +∞) when no majorant texture is bound.

void SampleLocalMajorant(GPUVolume v, float3 pos, float3 dir,
                         out float localMu, out float boundaryDist)
{
    float globalMu = VolumeSigmaTMax(v);

    if (v.majorantTexIndex == VOLUME_INVALID_TEX)
    {
        localMu = globalMu;
        boundaryDist = 1e20;
        return;
    }

    // Brick grid resolution = majorant texture dimensions.
    Texture3D<float> mtex = g_volumeDensities[v.majorantTexIndex];
    uint mW, mH, mD, mLevels;
    mtex.GetDimensions(0, mW, mH, mD, mLevels);
    float3 mres = float3(mW, mH, mD);

    float3 size = max(v.vMax - v.vMin, float3(1e-6, 1e-6, 1e-6));
    float3 brickWorldSize = size / mres;
    float3 normCoord = (pos - v.vMin) / size;
    float3 brickFloat = normCoord * mres;
    int3 brickCoord = clamp(int3(brickFloat), int3(0, 0, 0),
                            int3(mres) - int3(1, 1, 1));

    float densityBrickMax = mtex.Load(int4(brickCoord, 0));
    localMu = densityBrickMax * globalMu;

    // Distance to the next brick boundary along the ray. For each axis,
    // pick the boundary in front of `pos` (depends on dir sign), then take
    // the minimum across axes.
    float3 distVec;
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        if (abs(dir[i]) < 1e-20)
        {
            distVec[i] = 1e20;
        }
        else
        {
            float nextEdge = float(brickCoord[i]) + (dir[i] > 0.0 ? 1.0 : 0.0);
            float worldEdge = v.vMin[i] + nextEdge * brickWorldSize[i];
            float d = (worldEdge - pos[i]) / dir[i];
            // pos may be exactly on a boundary; force a small forward step
            // to guarantee progress (prevents zero-distance loops).
            distVec[i] = max(d, 1e-5);
        }
    }
    boundaryDist = min(distVec.x, min(distVec.y, distVec.z));
}

// ---------------------------------------------------------------------------
// Volume segment enumeration along a ray
// ---------------------------------------------------------------------------
//
// Walks all g_volumes, intersects each AABB with [0, tMax], and returns the
// overlapping intervals sorted by tNear. Capped at MAX_VOLUME_SEGMENTS.
//
// Limitations:
//   - Volumes that overlap in space are walked sequentially without combined
//     null-collision majorants. This is correct for non-overlapping setups
//     and a stable approximation otherwise. A combined-majorant pass can
//     replace the per-segment loop later without changing call sites.

#define MAX_VOLUME_SEGMENTS 8

struct VolumeSegment
{
    uint volumeIndex;
    float tNear;
    float tFar;
};

uint EnumerateVolumeSegments(float3 origin, float3 dir, float tMax,
                             out VolumeSegment segments[MAX_VOLUME_SEGMENTS])
{
    uint count = 0;
    [loop]
    for (uint i = 0; i < volumeCount && count < MAX_VOLUME_SEGMENTS; i++)
    {
        GPUVolume v = g_volumes[i];
        float tN, tF;
        if (!RayAABBIntersect(origin, dir, v.vMin, v.vMax, tN, tF))
            continue;
        tN = max(tN, 0.0);
        tF = min(tF, tMax);
        if (tF <= tN)
            continue;

        // Insertion sort by tNear.
        uint pos = count;
        [loop]
        for (; pos > 0; pos--)
        {
            if (segments[pos - 1].tNear <= tN) break;
            segments[pos] = segments[pos - 1];
        }
        segments[pos].volumeIndex = i;
        segments[pos].tNear = tN;
        segments[pos].tFar = tF;
        count++;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Null-collision free-flight (replaces plain delta tracking)
// ---------------------------------------------------------------------------
//
// Picks a hero wavelength = the volume's max-σ_t channel, which makes the
// scalar real-collision probability collapse to density(x) (matching the
// majorant μ = max_c σ_t,c). At each step we either:
//
//   - Treat as real scatter (prob = density(x)). Per-channel weight gets
//     a factor σ_t,c(x) / (μ · p_e) = σ_t_volume,c / μ.
//   - Treat as null collision. Per-channel weight gets a factor
//     σ_n,c(x) / (μ · (1 - p_e)) = (μ - density·σ_t_volume,c) / (μ · (1 - density)).
//     This is the term that lets dim-extinction channels "catch up" so the
//     accumulated weight matches per-channel transmittance through the
//     segment, even when free-flight was sampled with the scalar majorant.
//
// Accumulates into `weight` (inout). Returns true with tScatter on real
// scatter; false (and no scatter) if the ray exits [tNear, tFar].
//
// Reference: Novák et al, "Monte Carlo methods for volumetric light
// transport"; PBRT v4 §15.1.

bool NullCollisionFreeFlight(GPUVolume v, float3 origin, float3 dir,
                             float tNear, float tFar,
                             inout RNG rng,
                             inout float3 weight,
                             out float tScatter)
{
    float globalMu = VolumeSigmaTMax(v);
    float3 sigmaT_v = VolumeSigmaT(v);
    float t = tNear;
    [loop]
    for (uint step = 0; step < 512; step++)
    {
        float3 pos = origin + t * dir;

        // Local majorant for the brick we're currently in. In sparse
        // regions this is much smaller than globalMu so the exponential
        // sample takes us far in one go.
        float localMu, boundaryDist;
        SampleLocalMajorant(v, pos, dir, localMu, boundaryDist);

        // Empty-brick fast path: no extinction here, just walk to the
        // brick boundary with no weight change.
        if (localMu < 1e-8)
        {
            t += boundaryDist;
            if (t >= tFar)
            {
                tScatter = tFar;
                return false;
            }
            continue;
        }

        // Sample step at local majorant.
        float xi = max(NextFloat(rng), 1e-10);
        float dt = -log(xi) / localMu;

        // If the step exits the brick first, advance to the boundary and
        // re-sample with the next brick's μ. No weight change here — the
        // sampled "no collision in this brick" outcome is already weighted
        // implicitly by exp(-localMu · dist) via the exponential pdf.
        if (dt > boundaryDist)
        {
            t += boundaryDist;
            if (t >= tFar)
            {
                tScatter = tFar;
                return false;
            }
            continue;
        }

        t += dt;
        if (t >= tFar)
        {
            tScatter = tFar;
            return false;
        }

        // Collision in current brick.
        float3 newPos = origin + t * dir;
        float density = SampleVolumeDensity(v, newPos);

        // pReal = σ_t,h(x) / localMu = density · σ_t_volume,h / localMu
        //       = density · globalMu  / localMu = density / density_brick_max.
        // Clamped so floating-point density >= density_brick_max forces a
        // real scatter (otherwise the null-weight denominator goes to zero).
        float pReal = saturate((density * globalMu) / max(localMu, 1e-20));
        if (pReal >= 1.0 - 1e-6 || NextFloat(rng) < pReal)
        {
            // Real scatter. Weight collapses to σ_t_volume,c / globalMu —
            // independent of localMu, so the local-μ sampling doesn't bias
            // per-channel weights at scatter sites.
            weight *= sigmaT_v / max(globalMu, 1e-20);
            tScatter = t;
            return true;
        }
        else
        {
            // Null collision. Per-channel weight (localMu - σ_t,c) / (localMu · (1 - pReal)).
            float3 sigmaN_c = localMu - density * sigmaT_v;
            float denom = max(localMu * (1.0 - pReal), 1e-20);
            weight *= sigmaN_c / denom;
        }
    }
    tScatter = tFar;
    return false;
}

// ---------------------------------------------------------------------------
// Ratio tracking (replaces deterministic ray-march transmittance)
// ---------------------------------------------------------------------------
//
// Stochastic, unbiased per-channel transmittance estimator. At each step
// we treat every collision as null and accumulate σ_n,c / μ. The expected
// product over a segment is exp(-∫σ_t,c ds), giving the correct per-channel
// transmittance. Variance is moderate; for the path-tracer's many-sample
// integration it's not a problem in practice.
//
// Same call signature as the old RayMarchTransmittance, except this needs
// an RNG (it's stochastic) and there's no fixed step count.

float3 RatioTracking(GPUVolume v, float3 origin, float3 dir,
                     float tNear, float tFar, inout RNG rng)
{
    float3 sigmaT_v = VolumeSigmaT(v);
    float3 tr = float3(1, 1, 1);
    float t = tNear;
    [loop]
    for (uint step = 0; step < 512; step++)
    {
        float3 pos = origin + t * dir;
        float localMu, boundaryDist;
        SampleLocalMajorant(v, pos, dir, localMu, boundaryDist);

        // Empty brick — transmittance through it is 1, just walk to exit.
        if (localMu < 1e-8)
        {
            t += boundaryDist;
            if (t >= tFar)
                return tr;
            continue;
        }

        float xi = max(NextFloat(rng), 1e-10);
        float dt = -log(xi) / localMu;

        // Step exits brick: no event sampled in this brick. The implicit
        // exp(-localMu · dist) weighting is correct for the hero channel,
        // and matches our null-collision treatment (no per-step weight on
        // exit — small per-channel bias for tracked majorants in exchange
        // for big perf wins on sparse volumes).
        if (dt > boundaryDist)
        {
            t += boundaryDist;
            if (t >= tFar)
                return tr;
            continue;
        }

        t += dt;
        if (t >= tFar)
            return tr;

        // Null collision: accumulate (localMu - σ_t,c) / localMu per channel.
        float3 newPos = origin + t * dir;
        float density = SampleVolumeDensity(v, newPos);
        float3 sigmaN_c = localMu - density * sigmaT_v;
        tr *= sigmaN_c / max(localMu, 1e-20);
        if (max(tr.r, max(tr.g, tr.b)) < 1e-6)
            return float3(0, 0, 0);
    }
    return tr;
}

// ---------------------------------------------------------------------------
// Multi-volume entry points (used by Shaders.hlsl)
// ---------------------------------------------------------------------------

// Sample a free-flight scatter event across all volumes the ray crosses.
// `weight` (inout, init to 1) accumulates the per-channel null-collision
// weights that correct for sampling at the scalar majorant — so:
//   - Pass-through case: `weight` ends as the per-channel transmittance
//     through every volume the ray crossed (caller multiplies into throughput).
//   - Real-scatter case: `weight` includes the σ_t_volume,c / μ factor at
//     the scatter site; the bounce loop combines it with σ_s,c / σ_t_volume,c
//     to get the per-channel single-scattering albedo.
bool SampleVolumeFreeFlight(float3 origin, float3 dir, float tMax,
                            inout RNG rng,
                            inout float3 weight,
                            out float tScatter, out uint scatterVolumeIndex)
{
    tScatter = tMax;
    scatterVolumeIndex = 0;

    if (volumeCount == 0)
        return false;

    VolumeSegment segments[MAX_VOLUME_SEGMENTS];
    uint count = EnumerateVolumeSegments(origin, dir, tMax, segments);
    if (count == 0)
        return false;

    [loop]
    for (uint k = 0; k < count; k++)
    {
        VolumeSegment seg = segments[k];
        GPUVolume v = g_volumes[seg.volumeIndex];
        float mu = VolumeSigmaTMax(v);
        if (mu < 1e-8)
            continue;

        // Unified null-collision path. Homogeneous volumes have density=1
        // everywhere, so the inner loop scatters on its first iteration
        // (real-collision prob = 1) — equivalent to the old direct
        // exponential sample, but with the σ_t_volume,c / μ weight
        // applied per channel.
        float tHit;
        bool scattered = NullCollisionFreeFlight(
            v, origin, dir, seg.tNear, seg.tFar, rng, weight, tHit);
        if (scattered)
        {
            tScatter = tHit;
            scatterVolumeIndex = seg.volumeIndex;
            return true;
        }
    }
    return false;
}

// Combined RGB transmittance through all volumes along [0, tMax]. Per
// segment: heterogeneous uses ratio tracking (stochastic, unbiased);
// homogeneous uses the exact analytic Beer-Lambert form (deterministic and
// cheaper — no benefit to sampling when σ_t is constant).
float3 MultiVolumeTransmittance(float3 origin, float3 dir, float tMax,
                                inout RNG rng)
{
    if (volumeCount == 0)
        return float3(1, 1, 1);

    VolumeSegment segments[MAX_VOLUME_SEGMENTS];
    uint count = EnumerateVolumeSegments(origin, dir, tMax, segments);
    if (count == 0)
        return float3(1, 1, 1);

    float3 tr = float3(1, 1, 1);
    [loop]
    for (uint k = 0; k < count; k++)
    {
        VolumeSegment seg = segments[k];
        GPUVolume v = g_volumes[seg.volumeIndex];
        float3 sigmaT = VolumeSigmaT(v);
        if (max(sigmaT.r, max(sigmaT.g, sigmaT.b)) < 1e-8)
            continue;

        float3 segTr;
        if (VolumeIsHeterogeneous(v))
        {
            segTr = RatioTracking(v, origin, dir, seg.tNear, seg.tFar, rng);
        }
        else
        {
            segTr = exp(-sigmaT * (seg.tFar - seg.tNear));
        }
        tr *= segTr;
        // Stop only when every channel is below the noise floor.
        if (max(tr.r, max(tr.g, tr.b)) < 1e-6)
            return float3(0, 0, 0);
    }
    return tr;
}
