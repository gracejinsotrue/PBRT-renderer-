// nori-dxr Shaders.hlsl
//
// DXR entry-point translation unit. All shared declarations and the BSDF /
// emitter / volume math live in the modular .hlsli headers included below; this
// file is the only thing dxc compiles and contains just the shader entry points
// (RayGen, ClosestHit, Miss, PrimaryAnyHit, ShadowAnyHit, ShadowMiss).

#include "Common.hlsli"
#include "RNG.hlsli"
#include "GeometryUtils.hlsli"
#include "RayQueryTrace.hlsli"
#include "Microfacet.hlsli"
#include "Disney.hlsli"
#include "Hair.hlsli"
#include "Material.hlsli"
#include "Emitter.hlsli"
#include "Envmap.hlsli"
#include "Volume.hlsl"
#include "Subsurface.hlsli"
#include "PathTrace.hlsli"

#define ENVMAP_DEBUG_SAMPLER 0

// Ray Generation

[shader("raygeneration")] void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;

    RNG rng;
    float4 accumPrev;
    float2 momentsPrev;
    if (!BeginPixel(pixel, dims, rng, accumPrev, momentsPrev))
        return;

#if ENVMAP_DEBUG_SAMPLER
    {
        float u1 = NextFloat(rng);
        float u2 = NextFloat(rng);
        float3 dirE, radE;
        float pdfE;
        SampleEnvmap(u1, u2, dirE, radE, pdfE);
        float3 est = (pdfE > 1e-12) ? radE / pdfE : float3(0, 0, 0);

        if (any(isnan(est)) || any(isinf(est)))
            est = float3(0, 0, 0);

        float4 prev = (frameCount == 0) ? float4(0, 0, 0, 0) : g_accum[pixel];
        float4 accum = prev + float4(est, 1.0);
        g_accum[pixel] = accum;

        float3 averaged = accum.xyz / accum.w;
        averaged /= 10.0;
        averaged *= 0.5;
        // NOTE: CSResolve overwrites g_output every frame, so enabling this
        // debug mode also requires skipping the resolve dispatch.
        g_output[pixel] = float4(saturate(averaged), 1.0);
        return;
    }
#endif
    RayDesc ray = GeneratePrimaryRay(pixel, dims, rng);
    PathState P = TracePath(pixel, dims, ray, rng);
    WritePathResult(pixel, P, accumPrev, momentsPrev);
}


    [shader("closesthit")] void ClosestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // here we store only the minimal hit descriptor
    payload.hit = 1;
    payload.hitT = RayTCurrent();
    payload.materialID = InstanceID();
    payload.primitiveID = PrimitiveIndex();
    payload.baryX = attr.barycentrics.x;
    payload.baryY = attr.barycentrics.y;
}

[shader("miss")] void Miss(inout HitPayload payload)
{
    // Environment radiance is recomputed in RayGen via EvalEnvmap(ray.Direction)
    // on miss, so it no longer travels in the payload.
    payload.hit = 0;
}
    // Primary any-hit: hybrid cutoff + stochastic alpha test for the radiance ray.
    // Hard reject below 0.1, accept above 0.95, stochastic in between. The RNG
    // state is threaded through the payload so primary and shadow rays consume
    // the same per-path stream.
    [shader("anyhit")] void PrimaryAnyHit(inout HitPayload payload,
                                          in BuiltInTriangleIntersectionAttributes attr)
{
    uint instanceID = InstanceID();
    GPUMaterial mat = g_materials[instanceID];

    if (mat.alphaTexIndex == 0xFFFFFFFF)
        return;

    float2 aUV = GetInterpolatedUV(instanceID, PrimitiveIndex(), attr.barycentrics);
    float4 aTex = g_textures[NonUniformResourceIndex(mat.alphaTexIndex)].SampleLevel(g_sampler, aUV, 0);
    float a = aTex.a; // leaf cutout lives in the alpha channel

    if (a < 0.01)
    {
        IgnoreHit();
        return;
    }

    payload.rngState = PCGHash(payload.rngState);
    float xi = float(payload.rngState) / 4294967295.0;
    if (a * a < xi)
        IgnoreHit();
}

// Shadow any-hit: same hybrid alpha test as PrimaryAnyHit, plus Fresnel
// attenuation for dielectric/mirror surfaces. Opaque geometry never invokes
// this (stays D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE), so only instances
// marked FORCE_NON_OPAQUE trigger this shader.
[shader("anyhit")] void ShadowAnyHit(inout ShadowPayload payload,
                                     in BuiltInTriangleIntersectionAttributes attr)
{
    uint instanceID = InstanceID();
    GPUMaterial mat = g_materials[instanceID];

    if (mat.alphaTexIndex != 0xFFFFFFFF)
    {
        float2 aUV = GetInterpolatedUV(instanceID, PrimitiveIndex(), attr.barycentrics);
        float4 t = g_textures[NonUniformResourceIndex(mat.alphaTexIndex)].SampleLevel(g_sampler, aUV, 0);
        float a = t.a;

        if (a < 0.01)
        {
            IgnoreHit();
            return;
        }

        payload.rngState = PCGHash(payload.rngState);
        float xi = float(payload.rngState) / 4294967295.0;
        if (a * a < xi)
            IgnoreHit();
        return;
    }

    if (mat.type == 1 || mat.type == 2) // mirror or dielectric
    {
        float3 Ng = GetGeometricNormal(instanceID, PrimitiveIndex());
        float cosI = abs(dot(WorldRayDirection(), Ng));
        float Fr = FresnelDielectric(cosI, mat.extIOR, mat.intIOR);
        payload.transmission *= (1.0 - Fr);
        IgnoreHit();
    }
}
    [shader("miss")] void ShadowMiss(inout ShadowPayload payload)
{
    payload.shadowed = 0;
}