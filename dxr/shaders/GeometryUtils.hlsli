// GeometryUtils.hlsli — Vertex buffer accessors, coordinate frame helpers,
// sampling primitives, and Fresnel computation.  Pure math/geometry utilities
// shared across BSDFs, emitter sampling, and the integrator.
//
// Requires: Common.hlsli, RNG.hlsli (for NextFloat in sampling helpers)

#ifndef GEOMETRY_UTILS_HLSLI
#define GEOMETRY_UTILS_HLSLI

#include "Common.hlsli"
#include "RNG.hlsli"

// ============================================================================
// MIS
// ============================================================================

float BalanceHeuristic(float pdfA, float pdfB)
{
    float sum = pdfA + pdfB;
    if (sum < 1e-20)
        return 0.0;
    return pdfA / sum;
}

// ============================================================================
// Buffer accessors
// ============================================================================

float3 LoadFloat3(ByteAddressBuffer buf, uint elementIndex)
{
    return asfloat(buf.Load3(elementIndex * 12));
}

float2 LoadFloat2(ByteAddressBuffer buf, uint elementIndex)
{
    return asfloat(buf.Load2(elementIndex * 8));
}

// ============================================================================
// Per-vertex interpolation
// ============================================================================

float3 GetGeometricNormal(uint instanceID, uint primitiveID)
{
    GPUMaterial mat = g_materials[instanceID];
    uint base = mat.indexOffset + primitiveID * 3;
    uint i0 = g_indices.Load((base + 0) * 4);
    uint i1 = g_indices.Load((base + 1) * 4);
    uint i2 = g_indices.Load((base + 2) * 4);

    float3 p0 = LoadFloat3(g_vertices, mat.vertexOffset + i0);
    float3 p1 = LoadFloat3(g_vertices, mat.vertexOffset + i1);
    float3 p2 = LoadFloat3(g_vertices, mat.vertexOffset + i2);

    return normalize(cross(p1 - p0, p2 - p0));
}

float3 GetInterpolatedNormal(uint instanceID, uint primitiveID, float2 bary)
{
    GPUMaterial mat = g_materials[instanceID];
    uint base = mat.indexOffset + primitiveID * 3;
    uint i0 = g_indices.Load((base + 0) * 4);
    uint i1 = g_indices.Load((base + 1) * 4);
    uint i2 = g_indices.Load((base + 2) * 4);

    float3 n0 = LoadFloat3(g_normals, mat.vertexOffset + i0);
    float3 n1 = LoadFloat3(g_normals, mat.vertexOffset + i1);
    float3 n2 = LoadFloat3(g_normals, mat.vertexOffset + i2);

    float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    float3 n = n0 * w.x + n1 * w.y + n2 * w.z;

    float len2 = dot(n, n);
    if (len2 < 1e-8)
        return GetGeometricNormal(instanceID, primitiveID);
    return n * rsqrt(len2);
}

float2 GetInterpolatedUV(uint instanceID, uint primitiveID, float2 bary)
{
    GPUMaterial mat = g_materials[instanceID];
    uint base = mat.indexOffset + primitiveID * 3;
    uint i0 = g_indices.Load((base + 0) * 4);
    uint i1 = g_indices.Load((base + 1) * 4);
    uint i2 = g_indices.Load((base + 2) * 4);

    float2 uv0 = LoadFloat2(g_texcoords, mat.vertexOffset + i0);
    float2 uv1 = LoadFloat2(g_texcoords, mat.vertexOffset + i1);
    float2 uv2 = LoadFloat2(g_texcoords, mat.vertexOffset + i2);

    float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    return uv0 * w.x + uv1 * w.y + uv2 * w.z;
}

// Interpolate per-vertex hair tangent. Returns zero for non-hair meshes.
float3 GetInterpolatedTangent(uint instanceID, uint primitiveID, float2 bary)
{
    GPUMaterial mat = g_materials[instanceID];
    uint base = mat.indexOffset + primitiveID * 3;
    uint i0 = g_indices.Load((base + 0) * 4);
    uint i1 = g_indices.Load((base + 1) * 4);
    uint i2 = g_indices.Load((base + 2) * 4);

    float3 t0 = LoadFloat3(g_tangents, mat.vertexOffset + i0);
    float3 t1 = LoadFloat3(g_tangents, mat.vertexOffset + i1);
    float3 t2 = LoadFloat3(g_tangents, mat.vertexOffset + i2);

    float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    float3 t = t0 * w.x + t1 * w.y + t2 * w.z;
    float len2 = dot(t, t);
    return (len2 > 1e-8) ? t * rsqrt(len2) : float3(0, 0, 0);
}

// ============================================================================
// Texture LOD helpers
// ============================================================================

// approximate UV footprint of one screen pixel at the hit point.
float ComputeUVFootprint(uint materialID, uint primitiveID, float hitT, uint2 dims)
{
    GPUMaterial mat = g_materials[materialID];
    uint base = mat.indexOffset + primitiveID * 3;
    uint i0 = g_indices.Load((base + 0) * 4);
    uint i1 = g_indices.Load((base + 1) * 4);
    uint i2 = g_indices.Load((base + 2) * 4);

    float3 p0 = LoadFloat3(g_vertices, mat.vertexOffset + i0);
    float3 p1 = LoadFloat3(g_vertices, mat.vertexOffset + i1);
    float3 p2 = LoadFloat3(g_vertices, mat.vertexOffset + i2);

    float2 uv0 = LoadFloat2(g_texcoords, mat.vertexOffset + i0);
    float2 uv1 = LoadFloat2(g_texcoords, mat.vertexOffset + i1);
    float2 uv2 = LoadFloat2(g_texcoords, mat.vertexOffset + i2);

    float worldArea = 0.5 * length(cross(p1 - p0, p2 - p0));
    float uvArea = 0.5 * abs((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y));

    if (worldArea < 1e-12 || uvArea < 1e-12)
        return 0.0;

    float pixelAngle = 1.0 / float(max(dims.x, dims.y));
    float worldFootprint = hitT * pixelAngle;
    return worldFootprint * sqrt(uvArea / worldArea);
}

// Compute mip LOD for a specific texture given a UV footprint.
float ComputeTexLOD(Texture2D tex, float uvFootprint)
{
    uint texW, texH;
    tex.GetDimensions(texW, texH);
    float texelsPerPixel = uvFootprint * float(max(texW, texH));
    return max(log2(max(texelsPerPixel, 1.0)), 0.0);
}

// ============================================================================
// Coordinate frame helpers
// ============================================================================

void BuildONB(float3 N, out float3 T, out float3 B)
{
    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

float3 ToLocal(float3 v, float3 T, float3 B, float3 N)
{
    return float3(dot(v, T), dot(v, B), dot(v, N));
}

float3 ToWorld(float3 v, float3 T, float3 B, float3 N)
{
    return T * v.x + B * v.y + N * v.z;
}

// Offset a ray origin to avoid the shadow terminator problem
float3 OffsetRayOrigin(float3 hitPos, float3 Ng, float3 N)
{
    float3 p = hitPos + N * 0.002;
    p += Ng * max(0.0, -dot(p - hitPos, Ng));
    return p;
}

// ============================================================================
// Sampling primitives
// ============================================================================

float3 CosineSampleHemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * M_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

// ============================================================================
// Fresnel
// ============================================================================

float FresnelDielectric(float cosThetaI, float etaI, float etaT)
{
    if (etaI == etaT)
        return 0.0;
    if (cosThetaI < 0.0)
    {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = -cosThetaI;
    }
    float eta = etaI / etaT;
    float sinThetaTSq = eta * eta * (1.0 - cosThetaI * cosThetaI);
    if (sinThetaTSq > 1.0)
        return 1.0;
    float cosThetaT = sqrt(1.0 - sinThetaTSq);
    float Rs = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT) / (etaT * cosThetaI + etaI * cosThetaT);
    return 0.5 * (Rs * Rs + Rp * Rp);
}

// ============================================================================
// Color utilities
// ============================================================================

float Luminance(float3 c)
{
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

#endif // GEOMETRY_UTILS_HLSLI
