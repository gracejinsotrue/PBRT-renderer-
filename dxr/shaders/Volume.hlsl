// Volume.hlsl — Participating media support: homogeneous and heterogeneous.
// Ray-AABB intersection, Henyey-Greenstein phase function, free-flight
// sampling (homogeneous + delta tracking), and transmittance estimation.
//
// Requires: Common.hlsli, RNG.hlsli, GeometryUtils.hlsli (BuildONB)

#ifndef VOLUME_HLSL
#define VOLUME_HLSL

// ============================================================================
// Ray-AABB intersection (slab method)
// ============================================================================

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

// ============================================================================
// Henyey-Greenstein phase function
// ============================================================================

// [PBRT4] Eq. 11.24.  Normalized so that integral over 4π of f_p dω = 1.
float HenyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 / (4.0 * M_PI)) * (1.0 - g2) / (denom * sqrt(max(denom, 1e-8)));
}

// Sample a direction from the HG phase function.
// [PBRT4] Eq. 11.25 inversion sampling.
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

// ============================================================================
// Free-flight sampling — homogeneous
// ============================================================================

// p(s) = σ_t exp(-σ_t s).  Inversion: s = -ln(ξ) / σ_t.
float SampleFreeFlightHomogeneous(float sigmaT, inout RNG rng)
{
    float xi = max(NextFloat(rng), 1e-10);
    return -log(xi) / sigmaT;
}

// ============================================================================
// Density field lookup (heterogeneous)
// ============================================================================

// When a volume file is loaded (volumeHasTexture = 1), sample the Texture3D
// by mapping world position into [0,1]^3 UVW.
float SampleDensity(float3 worldPos)
{
    float3 vMin = float3(volumeMinX, volumeMinY, volumeMinZ);
    float3 vMax = float3(volumeMaxX, volumeMaxY, volumeMaxZ);

    if (volumeHasTexture)
    {
        float3 uvw = (worldPos - vMin) / max(vMax - vMin, float3(1e-6, 1e-6, 1e-6));
        return g_volumeDensity.SampleLevel(g_volumeSampler, uvw, 0);
    }
    else
    {
        // Procedural sphere falloff (test mode)
        float3 center = (vMin + vMax) * 0.5;
        float3 extent = (vMax - vMin) * 0.5;
        float radius = min(extent.x, min(extent.y, extent.z)) * 0.9;
        float d = length(worldPos - center) / max(radius, 1e-6);
        return saturate(1.0 - d * d);
    }
}

// ============================================================================
// Delta tracking (heterogeneous free-flight sampling)
// ============================================================================

// Samples a free-flight distance in heterogeneous media using null-collision
// method.  σ_t(x) = σ_max * density(x), density ∈ [0,1].
//
// Returns true if a real scatter event occurred within [tNear, tFar].
bool DeltaTracking(float3 origin, float3 dir,
                   float tNear, float tFar, float sigmaMax,
                   inout RNG rng, out float tScatter)
{
    float t = tNear;
    while (true)
    {
        float xi = max(NextFloat(rng), 1e-10);
        t += -log(xi) / sigmaMax;

        if (t >= tFar)
        {
            tScatter = tFar;
            return false;
        }

        float3 pos = origin + t * dir;
        float density = SampleDensity(pos);
        if (NextFloat(rng) < density)
        {
            tScatter = t;
            return true;
        }
    }
}

// ============================================================================
// Transmittance estimation
// ============================================================================

// Ray-marched transmittance (deterministic, biased but zero-variance).
#define TRANSMITTANCE_STEPS 32

float RayMarchTransmittance(float3 origin, float3 dir,
                            float tNear, float tFar, float sigmaMax)
{
    float dt = (tFar - tNear) / (float)TRANSMITTANCE_STEPS;
    float tau = 0.0;

    for (int i = 0; i < TRANSMITTANCE_STEPS; i++)
    {
        float t = tNear + (i + 0.5) * dt; // midpoint rule
        float3 pos = origin + t * dir;
        float density = SampleDensity(pos);
        tau += density * sigmaMax * dt;
    }

    return exp(-tau);
}

// Transmittance through volume.
// Homogeneous: exact Beer-Lambert.  Heterogeneous: deterministic ray marching.
float3 VolumeTransmittance(float3 origin, float3 dir, float tMax,
                           inout RNG rng)
{
    if (!volumeEnabled)
        return float3(1, 1, 1);

    float sigmaT = volumeSigmaA + volumeSigmaS;
    if (sigmaT < 1e-8)
        return float3(1, 1, 1);

    float3 vMin = float3(volumeMinX, volumeMinY, volumeMinZ);
    float3 vMax = float3(volumeMaxX, volumeMaxY, volumeMaxZ);

    float tNear, tFar;
    if (!RayAABBIntersect(origin, dir, vMin, vMax, tNear, tFar))
        return float3(1, 1, 1);

    tNear = max(tNear, 0.0);
    tFar = min(tFar, tMax);
    if (tFar <= tNear)
        return float3(1, 1, 1);

    float tr;
    if (volumeHeterogeneous)
    {
        tr = RayMarchTransmittance(origin, dir, tNear, tFar, sigmaT);
    }
    else
    {
        float d = tFar - tNear;
        tr = exp(-sigmaT * d);
    }

    return float3(tr, tr, tr);
}

#endif // VOLUME_HLSL
