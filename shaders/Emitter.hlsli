// Emitter.hlsli
// Next-event estimation: area-light + envmap direct lighting with MIS, and
// the in-volume NEE variants.
//
// Requires: Common, RNG, GeometryUtils, Envmap, Volume, Material

#ifndef EMITTER_HLSLI
#define EMITTER_HLSLI

#include "Common.hlsli"
#include "RNG.hlsli"
#include "GeometryUtils.hlsli"
#include "Envmap.hlsli"
#include "Volume.hlsl"
#include "Material.hlsli"

uint CdfSample(uint cdfOffset, uint numEntries, float u)
{
    uint lo = 0;
    uint hi = numEntries - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        float val = asfloat(g_emitterCdf.Load((cdfOffset + mid) * 4));
        if (val <= u)
            lo = mid + 1;
        else
            hi = mid;
    }
    return max(0, int(lo) - 1);
}

struct EmitterSample
{
    float3 position;
    float3 normal;
    float3 radiance;
    float pdfArea;
    uint emitterID;
    bool valid;
};

EmitterSample SampleEmitter(inout RNG rng)
{
    EmitterSample es;
    es.valid = false;

    if (emitterCount == 0)
        return es;

    uint pick = min((uint)(NextFloat(rng) * float(emitterCount)), emitterCount - 1);

    uint count = 0;
    for (uint j = 0; j < meshCount; j++)
    {
        if (g_materials[j].isEmitter)
        {
            if (count == pick)
            {
                es.emitterID = j;
                es.valid = true;
                break;
            }
            count++;
        }
    }
    if (!es.valid)
        return es;

    GPUMaterial eMat = g_materials[es.emitterID];
    es.radiance = MatRadiance(eMat);
    uint numTris = eMat.indexCount / 3;
    float u = NextFloat(rng);
    uint triIdx = min(CdfSample(eMat.emitterCdfOffset, numTris + 1, u), numTris - 1);
    uint base = eMat.indexOffset + triIdx * 3;
    uint i0 = g_indices.Load((base + 0) * 4);
    uint i1 = g_indices.Load((base + 1) * 4);
    uint i2 = g_indices.Load((base + 2) * 4);
    float3 p0 = LoadFloat3(g_vertices, eMat.vertexOffset + i0);
    float3 p1 = LoadFloat3(g_vertices, eMat.vertexOffset + i1);
    float3 p2 = LoadFloat3(g_vertices, eMat.vertexOffset + i2);
    float r1 = NextFloat(rng);
    float r2 = NextFloat(rng);
    float sr1 = sqrt(r1);
    es.position = (1.0 - sr1) * p0 + sr1 * (1.0 - r2) * p1 + sr1 * r2 * p2;
    es.normal = normalize(cross(p1 - p0, p2 - p0));
    es.pdfArea = 1.0 / (eMat.surfaceArea * float(emitterCount));
    return es;
}

float EmitterPdfSolidAngle(GPUMaterial emitMat, float3 hitPos, float3 shadingPos, float3 emitNormal)
{
    float3 d = hitPos - shadingPos;
    float dist2 = dot(d, d);
    if (dist2 < 1e-12)
        return 0.0;
    float dist = sqrt(dist2);
    float cosL = abs(dot(emitNormal, -d / dist));
    if (cosL < 1e-8)
        return 0.0;

    return (1.0 / (emitMat.surfaceArea * float(emitterCount))) * dist2 / cosL;
}

float3 MISDirectIllumination(float3 hitPos, float3 N, float3 Ng, float3 T, float3 B,
                             float3 wi_local, GPUMaterial mat, float h, inout RNG rng)
{
    EmitterSample es = SampleEmitter(rng);
    if (!es.valid)
        return float3(0, 0, 0);
    float3 toLight = es.position - hitPos;
    float dist = length(toLight);
    float3 wi_world = toLight / dist;
    float cosTheta = dot(N, wi_world);
    float cosLight = dot(es.normal, -wi_world);
    bool isHair = (mat.type == 5);
    if ((!isHair && cosTheta <= 0.0) || cosLight <= 0.0)
        return float3(0, 0, 0);
    float absCosTheta = isHair ? abs(cosTheta) : cosTheta;

    // For hair, shadow rays may go to the back side of the fiber.
    bool isHairShadow = (mat.type == 5);
    float3 shadowNg = isHairShadow
                          ? (dot(wi_world, Ng) >= 0.0 ? Ng : -Ng)
                          : Ng;
    float3 shadowOrigin = OffsetRayOrigin(hitPos, shadowNg, shadowNg);
    RayDesc shadowRay;
    shadowRay.Origin = shadowOrigin;
    shadowRay.Direction = wi_world;
    shadowRay.TMin = 0.0;
    shadowRay.TMax = dist - 0.001;
    ShadowPayload shadow;
    shadow.shadowed = 1;
    shadow.transmission = float3(1, 1, 1);
    shadow.rngState = rng.state;
    TraceRay(g_scene, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 1, 0, 1, shadowRay, shadow);
    rng.state = shadow.rngState;
    if (shadow.shadowed)
        return float3(0, 0, 0);

    float3 wo_local = ToLocal(wi_world, T, B, N);
    float3 f = MaterialEval(wi_local, wo_local, mat, h);
    float pdfBsdf = MaterialPdf(wi_local, wo_local, mat, h);

    float pdfEms = es.pdfArea * dist * dist / cosLight;
    float w = BalanceHeuristic(pdfEms, pdfBsdf);
#if HAS_VOLUME
    float3 volTr = MultiVolumeTransmittance(shadowOrigin, wi_world, dist, rng);
#else
    float3 volTr = float3(1, 1, 1);
#endif
    return es.radiance * f * absCosTheta / max(pdfEms, 1e-20) * w * volTr * shadow.transmission;
}

// restir
#ifndef RIS_M
#define RIS_M 8
#endif

struct Reservoir
{
    EmitterSample y; // selected sample
    float wSum;      // running sum of RIS weights
    float pHat;      // target value of the selected sample
};

// unshadowed scalar target p̂ = lum(f·Le·cosθ)
// and the naive solid-angle pdf,
// and RGB color integrand!
bool EvalLightCandidate(EmitterSample es, float3 hitPos, float3 N, float3 wi_local,
                        GPUMaterial mat, float h, float3 T, float3 B,
                        out float3 wiw, out float dist,
                        out float3 integ, out float pHat, out float pSA)
{
    integ = float3(0, 0, 0);
    pHat = 0.0;
    pSA = 0.0;
    dist = 0.0;
    float3 toLight = es.position - hitPos;
    dist = length(toLight);
    if (dist < 1e-6)
    {
        wiw = float3(0, 0, 1);
        return false;
    }
    wiw = toLight / dist;

    float cosSurface = dot(N, wiw);
    float cosLight = dot(es.normal, -wiw);
    bool isHair = (mat.type == 5);
    if ((!isHair && cosSurface <= 0.0) || cosLight <= 1e-8)
        return false;
    float absCos = isHair ? abs(cosSurface) : cosSurface;

    float3 f = MaterialEval(wi_local, ToLocal(wiw, T, B, N), mat, h);
    integ = f * es.radiance * absCos;                  // integrand (no V)
    pHat = dot(integ, float3(0.2126, 0.7152, 0.0722)); // scalar target
    pSA = es.pdfArea * dist * dist / cosLight;         // naive solid-angle pdf
    return pHat > 0.0 && pSA > 0.0;
}

float3 RISDirectIllumination(float3 hitPos, float3 N, float3 Ng, float3 T, float3 B,
                             float3 wi_local, GPUMaterial mat, float h, inout RNG rng)
{
    Reservoir r;
    r.wSum = 0.0;
    r.pHat = 0.0;
    r.y.valid = false;

    // strean RIS_M unshadowed candidates. Each iteration is one proposal, so the denominator below is RIS_M
    [loop] for (uint i = 0; i < RIS_M; i++)
    {
        EmitterSample c = SampleEmitter(rng);
        if (!c.valid)
            continue;
        float3 wiwC;
        float distC, pHatC, pSAC;
        float3 integC;
        if (!EvalLightCandidate(c, hitPos, N, wi_local, mat, h, T, B,
                                wiwC, distC, integC, pHatC, pSAC))
            continue;
        float w = pHatC / pSAC; // RIS weight
        r.wSum += w;
        if (NextFloat(rng) < w / max(r.wSum, 1e-20))
        {
            r.y = c;
            r.pHat = pHatC;
        }
    }
    if (!r.y.valid || r.pHat <= 0.0)
        return float3(0, 0, 0);

    // unbiased contribution weight for the survivor.
    float W = (r.wSum / float(RIS_M)) / r.pHat;

    // re-derive the survivor's terms, then shadow-test ONLY it.
    float3 wiw;
    float dist, pHat, pSA;
    float3 integ;
    if (!EvalLightCandidate(r.y, hitPos, N, wi_local, mat, h, T, B,
                            wiw, dist, integ, pHat, pSA))
        return float3(0, 0, 0);

    bool isHair = (mat.type == 5);
    float3 shadowNg = isHair ? (dot(wiw, Ng) >= 0.0 ? Ng : -Ng) : Ng;
    float3 shadowOrigin = OffsetRayOrigin(hitPos, shadowNg, shadowNg);
    RayDesc shadowRay;
    shadowRay.Origin = shadowOrigin;
    shadowRay.Direction = wiw;
    shadowRay.TMin = 0.0;
    shadowRay.TMax = dist - 0.001;
    ShadowPayload shadow;
    shadow.shadowed = 1;
    shadow.transmission = float3(1, 1, 1);
    shadow.rngState = rng.state;
    TraceRay(g_scene, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 1, 0, 1, shadowRay, shadow);
    rng.state = shadow.rngState;
    if (shadow.shadowed)
        return float3(0, 0, 0);

    //  MIS vs BSDF sampling, where naive light pdf on BOTH sides keeps the weights a
    //    valid partition
    float pdfBsdf = MaterialPdf(wi_local, ToLocal(wiw, T, B, N), mat, h);
    float wMis = BalanceHeuristic(pSA, pdfBsdf);
#if HAS_VOLUME
    float3 volTr = MultiVolumeTransmittance(shadowOrigin, wiw, dist, rng);
#else
    float3 volTr = float3(1, 1, 1);
#endif
    return integ * W * wMis * volTr * shadow.transmission; // integ = f·Le·cosθ
}

// Envmap next-event estimation, which samples one direction from the envmap's importance distribution, shoots a shadow ray evaluates the BSDF in that direction, and
// MIS-weights against the BSDF pdf. Called from raygen for diffuse and microfacet hits, in addition to MISDirectIllumination and the two contributions are summed.
float3 EnvmapDirectIllumination(float3 hitPos, float3 N, float3 Ng, float3 T, float3 B,
                                float3 wi_local, GPUMaterial mat, float h, inout RNG rng)
{
    float u1 = NextFloat(rng);
    float u2 = NextFloat(rng);
    float3 wi_world, Lenv;
    float pdfEnv;
    SampleEnvmap(u1, u2, wi_world, Lenv, pdfEnv);
    if (pdfEnv <= 0.0)
        return float3(0, 0, 0);

    float cosTheta = dot(N, wi_world);
    bool isHair = (mat.type == 5);
    if (!isHair && cosTheta <= 0.0)
        return float3(0, 0, 0);
    float absCosTheta = isHair ? abs(cosTheta) : cosTheta;
    bool isHairEnvShadow = (mat.type == 5);
    float3 envShadowNg = isHairEnvShadow
                             ? (dot(wi_world, Ng) >= 0.0 ? Ng : -Ng)
                             : Ng;
    float3 shadowOrigin = OffsetRayOrigin(hitPos, envShadowNg, envShadowNg);
    RayDesc shadowRay;
    shadowRay.Origin = shadowOrigin;
    shadowRay.Direction = wi_world;
    shadowRay.TMin = 0.0;
    shadowRay.TMax = 1e20;
    ShadowPayload shadow;
    shadow.shadowed = 1;
    shadow.transmission = float3(1, 1, 1);
    shadow.rngState = rng.state;
    TraceRay(g_scene,
             RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 1, 0, 1, shadowRay, shadow);
    rng.state = shadow.rngState;
    if (shadow.shadowed)
        return float3(0, 0, 0);

    // BSDF evaluation at the envmap-sampled direction
    float3 wo_local = ToLocal(wi_world, T, B, N);
    float3 f = MaterialEval(wi_local, wo_local, mat, h);
    float pdfBsdf = MaterialPdf(wi_local, wo_local, mat, h);

    float w = BalanceHeuristic(pdfEnv, pdfBsdf);
#if HAS_VOLUME
    float3 volTr = MultiVolumeTransmittance(shadowOrigin, wi_world, 1e20, rng);
#else
    float3 volTr = float3(1, 1, 1);
#endif
    float3 contrib = Lenv * f * absCosTheta / max(pdfEnv, 1e-20) * w * volTr * shadow.transmission;
    float contribLum = dot(contrib, float3(0.2126, 0.7152, 0.0722));
    if (contribLum > kFireflyClamp)
        contrib *= kFireflyClamp / contribLum;
    return contrib;
}

// Volume NEE
//  at a medium scatter point, explicitly sample a light direction and evaluate the phase function.
// Unlike surface NEE, there is no cosine factor at the scatter point because volumes scatter isotropically w.r.t. geometry. the directional
// dependence is entirely in the phase function.
// MIS-weighted against the phase function sampling pdf.

float3 VolumeNEEAreaLight(float3 scatterPos, float3 wo, uint volumeIndex, inout RNG rng)
{

    GPUVolume vol = g_volumes[volumeIndex];
    float3 sigmaT = VolumeSigmaT(vol);
    float sigmaTmin = min(sigmaT.r, min(sigmaT.g, sigmaT.b));
    float phaseG = vol.phaseG;
    float3 dMin = scatterPos - vol.vMin;
    float3 dMax = vol.vMax - scatterPos;
    float minDist = min(min(dMin.x, dMin.y), min(dMin.z, min(dMax.x, min(dMax.y, dMax.z))));
    if (minDist * sigmaTmin > 8.0)
        return float3(0, 0, 0);

    EmitterSample es = SampleEmitter(rng);
    if (!es.valid)
        return float3(0, 0, 0);

    float3 toLight = es.position - scatterPos;
    float dist = length(toLight);
    float3 wi = toLight / dist;
    float cosLight = dot(es.normal, -wi);
    if (cosLight <= 0.0)
        return float3(0, 0, 0);

    // Shadow ray from scatter point
    RayDesc shadowRay;
    shadowRay.Origin = scatterPos;
    shadowRay.Direction = wi;
    shadowRay.TMin = 0.001;
    shadowRay.TMax = dist - 0.001;
    ShadowPayload shadow;
    shadow.shadowed = 1;
    shadow.transmission = float3(1, 1, 1);
    shadow.rngState = rng.state;
    TraceRay(g_scene, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 1, 0, 1, shadowRay, shadow);
    rng.state = shadow.rngState;
    if (shadow.shadowed)
        return float3(0, 0, 0);

    // Phase function value in the light direction
    // wo = ray travel direction (toward scatter point), wi = toward light
    float cosTheta = dot(wo, wi);
    float fp = HenyeyGreenstein(cosTheta, phaseG);

    // Emitter pdf in solid angle, phase function pdf = fp
    float pdfEms = es.pdfArea * dist * dist / cosLight;
    float pdfPhase = fp;
    float w = BalanceHeuristic(pdfEms, pdfPhase);

    // Transmittance along shadow ray through all volumes
    float3 volTr = MultiVolumeTransmittance(scatterPos, wi, dist, rng);

    // [Marschner] §5: estimator = σ_s · f_p · L / pdf_emitter, but σ_s is
    // already folded into the throughput (as σ_s/σ_t), so we just need f_p · L.
    // The σ_s/σ_t throughput weight accounts for the scattering coefficient.
    return es.radiance * fp / max(pdfEms, 1e-20) * w * volTr * shadow.transmission;
}

float3 VolumeNEEEnvmap(float3 scatterPos, float3 wo, uint volumeIndex, inout RNG rng)
{
    // envmap shadow rays must traverse the full remaining volume.
    // If the scatter point is deep inside every channel, transmittance will be negligible.
    GPUVolume vol = g_volumes[volumeIndex];
    float3 sigmaT = VolumeSigmaT(vol);
    float sigmaTmin = min(sigmaT.r, min(sigmaT.g, sigmaT.b));
    float phaseG = vol.phaseG;
    float3 dMin = scatterPos - vol.vMin;
    float3 dMax = vol.vMax - scatterPos;
    float minDist = min(min(dMin.x, dMin.y), min(dMin.z, min(dMax.x, min(dMax.y, dMax.z))));
    if (minDist * sigmaTmin > 8.0)
        return float3(0, 0, 0);

    float u1 = NextFloat(rng);
    float u2 = NextFloat(rng);
    float3 wi, Lenv;
    float pdfEnv;
    SampleEnvmap(u1, u2, wi, Lenv, pdfEnv);
    if (pdfEnv <= 0.0)
        return float3(0, 0, 0);
    RayDesc shadowRay;
    shadowRay.Origin = scatterPos;
    shadowRay.Direction = wi;
    shadowRay.TMin = 0.001;
    shadowRay.TMax = 1e20;
    ShadowPayload shadow;
    shadow.shadowed = 1;
    shadow.transmission = float3(1, 1, 1);
    shadow.rngState = rng.state;
    TraceRay(g_scene,
             RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 1, 0, 1, shadowRay, shadow);
    rng.state = shadow.rngState;
    if (shadow.shadowed)
        return float3(0, 0, 0);

    float cosTheta = dot(wo, wi);
    float fp = HenyeyGreenstein(cosTheta, phaseG);
    float pdfPhase = fp;
    float w = BalanceHeuristic(pdfEnv, pdfPhase);

    float3 volTr = MultiVolumeTransmittance(scatterPos, wi, 1e20, rng);
    return Lenv * fp / max(pdfEnv, 1e-20) * w * volTr * shadow.transmission;
}

#endif // EMITTER_HLSLI
