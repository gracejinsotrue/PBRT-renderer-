// Subsurface.hlsli
// Random-walk subsurface scattering method
//
// Couples a refractive dielectric boundary to an interior homogeneous
// scattering medium bounded by the mesh itself. The camera/bounce loop in
// Shaders.hlsl handles the ENTRY Fresnel event ; once refracted in, SubsurfaceWalk() does the
// interior random walk and the EXIT event, returning the world-space ray that
// leaves the surface so the main loop can continue path tracing from there.
//
// Requires: Common, RNG, GeometryUtils, Volume (SampleHG / free-flight)

#ifndef SUBSURFACE_HLSLI
#define SUBSURFACE_HLSLI

#include "Common.hlsli"
#include "RNG.hlsli"
#include "GeometryUtils.hlsli"
#include "Volume.hlsl"

// Interior walk step cap. With Russian roulette walks terminate long
// before this; it only bounds pathological total-internal-reflection loops.
#define MAX_SSS_STEPS 256

// Inputs: a surface/scatter color A and a mean free path d. Output: a SCALAR extinction
// sigmaT = 1/d, so interior free-flight distance sampling is unambiguous and
// unbiased and no spectral MIS needed plus the PER-CHANNEL single-scattering
// albedo alpha = sigmaS/sigmaT. All of the color ends up in alpha; channels
// with lower alpha are absorbed faster during the walk, tinting the exit.
//
void SubsurfaceParams(float3 A, float d, out float sigmaT, out float3 alpha)
{
    sigmaT = 1.0 / max(d, 1e-4);
    float3 A2 = A * A;
    float3 A3 = A2 * A;
    alpha = 1.0 - exp(-5.09406 * A + 2.61188 * A2 - 4.31805 * A3);
}

// Random-walk through the interior medium starting just inside the surface.
//
//   startPos  : point inside boundary
//   startDir  : refracted direction heading into the medium (unit)
//   sssMatID  : InstanceID/materialID of THIS object, interior rays must hit
//               its own boundary, not other geometry (closed-mesh assumption;
//               foreign hits terminate the walk conservatively)
//   sigmaT    : scalar extinction (1/d). 0 => no scattering (clear).
//   alpha     : per-channel single-scattering albedo (sigmaS/sigmaT). 0 in inc2.
//   g         : Henyey-Greenstein phase anisotropy
//   intIOR,extIOR : boundary indices of refraction
//
// Returns true if the ray exited the surface, filling
// exitPos/exitDir/exitN (outward normal) and throughputMul. Returns false if
// the walk was absorbed or hit the step cap
bool SubsurfaceWalk(
    float3 startPos, float3 startDir,
    uint sssMatID,
    float sigmaT, float3 alpha, float g,
    float intIOR, float extIOR,
    inout RNG rng,
    out float3 exitPos, out float3 exitDir, out float3 exitN,
    out float3 throughputMul)
{
    float3 pos = startPos;
    float3 dir = startDir;
    throughputMul = float3(1, 1, 1);
    exitPos = startPos;
    exitDir = startDir;
    exitN = startDir;

    for (int step = 0; step < MAX_SSS_STEPS; step++)
    {
        float d = (sigmaT > 0.0) ? SampleFreeFlightHomogeneous(sigmaT, rng) : 1e30;

        // trace from the current interior point toward the next surface.
        RayDesc r;
        r.Origin = pos;
        r.Direction = dir;
        r.TMin = 1e-4;
        r.TMax = d;
        HitPayload p;
        p.hit = 0;
        p.rngState = rng.state;
        TRACE_CLOSEST(g_scene, r, p);
        rng.state = p.rngState;

        if (p.hit != 0 && p.hitT < d)
        {
            // reached a surface before scattering.
            if (p.materialID != sssMatID)
            {
                throughputMul = float3(0, 0, 0);
                return false;
            }

            float3 bPos = pos + dir * p.hitT;
            float3 Nb = normalize(GetInterpolatedNormal(p.materialID, p.primitiveID,
                                                        float2(p.baryX, p.baryY)));
            if (dot(Nb, dir) < 0.0)
                Nb = -Nb; // outward normal,same side as the outgoing travel dir

            float cosI = dot(dir, Nb); // > 0: hitting the boundary from inside
            float Fr = FresnelDielectric(cosI, intIOR, extIOR);

            if (NextFloat(rng) < Fr)
            {
                // Total-internal reflection back into the medium.
                dir = normalize(reflect(dir, Nb));
                pos = bPos - Nb * 1e-4;
                continue;
            }

            // Transmit out. n = -Nb points against the incident dir.
            float3 outDir = refract(dir, -Nb, intIOR / extIOR);
            if (dot(outDir, outDir) < 1e-8)
            {

                dir = normalize(reflect(dir, Nb));
                pos = bPos - Nb * 1e-4;
                continue;
            }
            exitPos = bPos;
            exitDir = normalize(outDir);
            exitN = Nb;
            return true;
        }

        pos = pos + dir * d;
        throughputMul *= alpha; // per-channel single-scattering albedo
        float phasePdf;
        dir = SampleHG(dir, g, rng, phasePdf);
        if (step > 4)
        {
            float q = clamp(max(throughputMul.x, max(throughputMul.y, throughputMul.z)),
                            0.05, 0.95);
            if (NextFloat(rng) >= q)
            {
                throughputMul = float3(0, 0, 0);
                return false;
            }
            throughputMul /= q;
        }
    }
    throughputMul = float3(0, 0, 0);
    return false;
}

#endif // SUBSURFACE_HLSLI
