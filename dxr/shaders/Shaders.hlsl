// nori-dxr Shaders.hlsl

RaytracingAccelerationStructure g_scene : register(t0);
RWTexture2D<float4> g_output : register(u0);
RWTexture2D<float4> g_accum : register(u1);

cbuffer CameraParams : register(b0)
{
    float3 camPos;
    float pad0;
    float3 camLowerLeftCorner;
    float pad1;
    float3 camHorizontal;
    uint meshCount;
    float3 camVertical;
    uint frameCount;

    // Number of participating-medium instances; per-volume data lives in
    // g_volumes (StructuredBuffer<GPUVolume>) below. Zero means no volumes.
    uint volumeCount;
    float lensRadius; // 0 = pinhole camera
    float focalDistance;
    uint emitterCount; // number of emitter meshes; power CDF has (emitterCount+1) entries at g_emitterCdf offset 0
};

struct GPUMaterial
{
    uint type;
    float albedoR, albedoG, albedoB;
    float intIOR;
    float extIOR;
    float alpha;
    uint isEmitter;
    float radianceR, radianceG, radianceB;
    uint indexOffset;
    uint vertexOffset;
    uint indexCount;
    uint vertexCount;
    float surfaceArea;
    uint emitterCdfOffset;
    uint albedoTexIndex;
    uint normalTexIndex;
    uint roughnessTexIndex;
    uint metallicTexIndex;
    uint specularTexIndex;
    uint subsurfaceTexIndex;
    uint alphaTexIndex;

    float roughness;
    float metallic;
    float specular;
    float specularTint;
    float sheen;
    float sheenTint;
    float subsurface;
    float clearcoat;
    float clearcoatGloss;
    float anisotropic;
    float betaN;                // azimuthal roughness for hair
    float emitterSelectionProb; // power-weighted probability of selecting this emitter (0 for non-emitters)
};

float3 MatAlbedo(GPUMaterial m) { return float3(m.albedoR, m.albedoG, m.albedoB); }
float3 MatRadiance(GPUMaterial m) { return float3(m.radianceR, m.radianceG, m.radianceB); }

StructuredBuffer<GPUMaterial> g_materials : register(t1);
ByteAddressBuffer g_normals : register(t2);
ByteAddressBuffer g_indices : register(t3);
ByteAddressBuffer g_vertices : register(t4);
ByteAddressBuffer g_emitterCdf : register(t5);
ByteAddressBuffer g_texcoords : register(t6);

// Environment map
Texture2D<float4> g_envmap : register(t7);
ByteAddressBuffer g_envmapMarginalCdf : register(t8);
ByteAddressBuffer g_envmapConditionalCdf : register(t9);

// Fiber tangent buffer for hair
ByteAddressBuffer g_tangents : register(t10);

// Texture array and sampler for material textures
// Textures are bound starting at t11, indices stored in GPUMaterial
Texture2D g_textures[] : register(t11);
SamplerState g_sampler : register(s0);
SamplerState g_envmapSampler : register(s1);

// Participating-medium volumes. Multi-instance design: each entry in
// g_volumes carries its own AABB, scattering coefficients, phase param,
// and (for heterogeneous media) an index into g_volumeDensities[].
// See Common.hlsli for the canonical declaration.

#define VOLUME_FLAG_HETEROGENEOUS 0x1u
#define VOLUME_INVALID_TEX 0xFFFFFFFFu

struct GPUVolume
{
    float3 vMin;
    float pad0;
    float3 sigmaA;
    float pad1;
    float3 vMax;
    float pad2;
    float3 sigmaS;
    float phaseG;
    uint densityTexIndex;  // index into g_volumeDensities[], or VOLUME_INVALID_TEX
    uint flags;            // VOLUME_FLAG_*
    uint majorantTexIndex; // index of the brick-max-density coarse mip,
                           // or VOLUME_INVALID_TEX to fall back to global μ.
    uint pad3;
};

StructuredBuffer<GPUVolume> g_volumes : register(t0, space1);
Texture3D<float> g_volumeDensities[] : register(t1, space1);
SamplerState g_volumeSampler : register(s2);

static const float M_PI = 3.14159265358979323846;
static const float M_INV_PI = 0.31830988618379067154;
static const int MAX_BOUNCES = 32;

// TODO remove later:
//  When set to 1, the raygen shader replaces normal path tracing with a
//  Monte Carlo estimator of the total envmap integral: each sample computes
//  radiance / pdf from SampleEnvmap(). A correct sampler converges every
//  pixel to the same color (the sphere integral of envmap radiance), since
//  radiance / pdf is constant when pdf is proportional to radiance.
//  Set to 0 once verified.
#define ENVMAP_DEBUG_SAMPLER 0

struct HitPayload
{
    float hitT;
    float normalX, normalY, normalZ;
    float geoNormalX, geoNormalY, geoNormalZ;
    uint materialID;
    uint hit;
    float texU, texV;
    float envR, envG, envB;
    uint primitiveID;
    float tangentX, tangentY, tangentZ; // fiber tangent
    float hairH;                        // fiber offset h in [-1,1]
    uint rngState;                      // PCG state, propagated through any-hit
};
struct ShadowPayload
{
    uint shadowed;
    float3 transmission; // accumulated Fresnel transmission through glass
    uint rngState;       // PCG state, propagated through any-hit
};

uint PCGHash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct RNG
{
    uint state;
};

RNG InitRNG(uint2 pixel, uint2 dims, uint frame)
{
    RNG rng;
    rng.state = PCGHash(pixel.y * dims.x + pixel.x + frame * dims.x * dims.y);
    return rng;
}

float NextFloat(inout RNG rng)
{
    rng.state = PCGHash(rng.state);
    return float(rng.state) / 4294967295.0;
}

float BalanceHeuristic(float pdfA, float pdfB)
{
    float sum = pdfA + pdfB;
    if (sum < 1e-20)
        return 0.0;
    return pdfA / sum;
}

float3 LoadFloat3(ByteAddressBuffer buf, uint elementIndex)
{
    return asfloat(buf.Load3(elementIndex * 12));
}

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

float2 LoadFloat2(ByteAddressBuffer buf, uint elementIndex)
{
    return asfloat(buf.Load2(elementIndex * 8));
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

// Approximate UV footprint of one screen pixel at the hit point.
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

void BuildONB(float3 N, out float3 T, out float3 B)
{
    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

// offset a ray origin to avoid the shadow terminator problem!
float3 OffsetRayOrigin(float3 hitPos, float3 Ng, float3 N)
{
    float3 p = hitPos + N * 0.002;
    p += Ng * max(0.0, -dot(p - hitPos, Ng));
    return p;
}

// Evaluate equirectangular envmap for a world-space direction
float3 EvalEnvmap(float3 dir)
{
    float3 d = normalize(dir);
    float phi = atan2(d.z, d.x);
    if (phi < 0.0)
        phi += 2.0 * M_PI;
    float theta = acos(clamp(d.y, -1.0, 1.0));
    float2 uv = float2(phi / (2.0 * M_PI), theta / M_PI);
    return g_envmap.SampleLevel(g_envmapSampler, uv, 0).rgb;
}

// Binary search over a monotonic CDF in a ByteAddressBuffer.
uint EnvmapCdfSearch(ByteAddressBuffer buf, uint byteOffset, uint numEntries, float u)
{
    uint lo = 0;
    uint hi = numEntries - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        float val = asfloat(buf.Load(byteOffset + mid * 4));
        if (val <= u)
            lo = mid + 1;
        else
            hi = mid;
    }
    return max(0, int(lo) - 1);
}

// Sample a direction from the environment map proportional to radiance x sin(theta).
// u1, u2 are independent uniform samples in [0, 1).
// Outputs:
//   dir      - world-space unit direction toward the sampled pixel center
//   radiance - envmap radiance along dir
//   pdf      - probability density in solid-angle per steradian
// todo: the direction is built from the pixel cente; jittering should be added later
void SampleEnvmap(float u1, float u2, out float3 dir, out float3 radiance, out float pdf)
{
    uint W, H;
    g_envmap.GetDimensions(W, H);

    // 1. pick a row via the marginal CDF (H + 1 entries).
    uint y = EnvmapCdfSearch(g_envmapMarginalCdf, 0, H + 1, u1);

    // 2. pick a column within that row via the conditional CDF, where row y starts at float offset y * (W + 1), meaning byte offset y*(W+1)*4.
    uint rowBytes = (W + 1) * 4;
    uint x = EnvmapCdfSearch(g_envmapConditionalCdf, y * rowBytes, W + 1, u2);

    // 3. Compute the direction at the pixel center.
    float u = ((float)x + 0.5) / (float)W;
    float v = ((float)y + 0.5) / (float)H;
    float phi = 2.0 * M_PI * u;
    float theta = M_PI * v;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    dir = float3(sinTheta * cos(phi),
                 cosTheta,
                 sinTheta * sin(phi));

    // 4. compute radiance by reading the pixel directly via the sampler at the center UV.
    radiance = g_envmap.SampleLevel(g_envmapSampler, float2(u, v), 0).rgb;

    // 5. pixel-space pdf via CDF differencing, then convert to solid-angle pdf where p_pixel = p_marginal(y) * p_conditional(x | y)
    float pMar = asfloat(g_envmapMarginalCdf.Load((y + 1) * 4)) -
                 asfloat(g_envmapMarginalCdf.Load(y * 4));
    float pCon = asfloat(g_envmapConditionalCdf.Load(y * rowBytes + (x + 1) * 4)) -
                 asfloat(g_envmapConditionalCdf.Load(y * rowBytes + x * 4));
    float pPixel = pMar * pCon;

    // do the jacobian where : pixel-space -> direction-space, where p_omega = p_pixel * (W * H) / (2 * pi^2 * sin(theta)).
    float denom = 2.0 * M_PI * M_PI * max(sinTheta, 1e-8);
    pdf = pPixel * (float)(W * H) / denom;
}

// Compute the solid-angle pdf that SampleEnvmap would have produced for the given world-space direction, which is used for MIS weights when the BSDF/path
// sampler picks a direction that happens to hit the envmap.
float EnvmapPdfDirection(float3 dir)
{
    uint W, H;
    g_envmap.GetDimensions(W, H);

    float3 d = normalize(dir);
    float phi = atan2(d.z, d.x);
    if (phi < 0.0)
        phi += 2.0 * M_PI;
    float theta = acos(clamp(d.y, -1.0, 1.0));
    float sinTheta = sin(theta);

    float u = phi / (2.0 * M_PI);
    float v = theta / M_PI;
    uint x = min((uint)(u * (float)W), W - 1);
    uint y = min((uint)(v * (float)H), H - 1);

    uint rowBytes = (W + 1) * 4;
    float pMar = asfloat(g_envmapMarginalCdf.Load((y + 1) * 4)) -
                 asfloat(g_envmapMarginalCdf.Load(y * 4));
    float pCon = asfloat(g_envmapConditionalCdf.Load(y * rowBytes + (x + 1) * 4)) -
                 asfloat(g_envmapConditionalCdf.Load(y * rowBytes + x * 4));
    float pPixel = pMar * pCon;

    float denom = 2.0 * M_PI * M_PI * max(sinTheta, 1e-8);
    return pPixel * (float)(W * H) / denom;
}

float3 ToLocal(float3 v, float3 T, float3 B, float3 N)
{
    return float3(dot(v, T), dot(v, B), dot(v, N));
}

float3 ToWorld(float3 v, float3 T, float3 B, float3 N)
{
    return T * v.x + B * v.y + N * v.z;
}

#include "Volume.hlsl"

float3 CosineSampleHemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * M_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

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

// Microfacet functions
float BeckmannD(float3 wh, float alpha)
{
    if (wh.z <= 0.0)
        return 0.0;
    float cosTheta2 = wh.z * wh.z;
    float tanTheta2 = (1.0 - cosTheta2) / cosTheta2;
    float cosTheta3 = cosTheta2 * wh.z;
    float alpha2 = alpha * alpha;
    return exp(-tanTheta2 / alpha2) / (M_PI * alpha2 * cosTheta3);
}

float BeckmannDCosTheta(float3 wh, float alpha)
{
    if (wh.z <= 0.0)
        return 0.0;
    float cosTheta2 = wh.z * wh.z;
    float tanTheta2 = (1.0 - cosTheta2) / cosTheta2;
    float alpha2 = alpha * alpha;
    return exp(-tanTheta2 / alpha2) / (M_PI * alpha2 * cosTheta2);
}

float3 BeckmannSample(float2 u, float alpha)
{
    float phi = 2.0 * M_PI * u.y;
    float tanTheta2 = -alpha * alpha * log(max(1e-8, 1.0 - u.x));
    float cosTheta = 1.0 / sqrt(1.0 + tanTheta2);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float SmithG1(float3 v, float3 wh, float alpha)
{
    float dotVWh = dot(v, wh);
    float dotVN = v.z;
    if (dotVWh / dotVN <= 0.0)
        return 0.0;
    float cosTheta2 = v.z * v.z;
    if (cosTheta2 < 1e-10)
        return 0.0;
    float tanTheta = sqrt(max(0.0, 1.0 - cosTheta2)) / abs(v.z);
    if (tanTheta < 1e-10)
        return 1.0;
    float b = 1.0 / (alpha * tanTheta);
    if (b >= 1.6)
        return 1.0;
    float b2 = b * b;
    return (3.535 * b + 2.181 * b2) / (1.0 + 2.276 * b + 2.577 * b2);
}

float3 MicrofacetEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);
    float3 kd = MatAlbedo(mat);
    float ks = 1.0 - max(kd.x, max(kd.y, kd.z));
    float3 diffuse = kd * M_INV_PI;
    float3 wh = normalize(wi + wo);
    float D = BeckmannD(wh, mat.alpha);
    float F = FresnelDielectric(dot(wh, wi), mat.extIOR, mat.intIOR);
    float G = SmithG1(wi, wh, mat.alpha) * SmithG1(wo, wh, mat.alpha);
    float denom = 4.0 * wi.z * wo.z;
    return diffuse + float3(1, 1, 1) * (ks * D * F * G / max(denom, 1e-10));
}

float MicrofacetPdf(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return 0.0;
    float3 kd = MatAlbedo(mat);
    float ks = 1.0 - max(kd.x, max(kd.y, kd.z));
    float3 wh = normalize(wi + wo);
    float DwhCosTheta = BeckmannDCosTheta(wh, mat.alpha);
    float Jh = 1.0 / max(4.0 * dot(wh, wo), 1e-10);
    return ks * DwhCosTheta * Jh + (1.0 - ks) * wo.z * M_INV_PI;
}

float3 MicrofacetSample(float3 wi, inout RNG rng, GPUMaterial mat,
                        out float3 wo, out float bsdfPdf)
{
    if (wi.z <= 0.0)
    {
        wo = float3(0, 0, 1);
        bsdfPdf = 0.0;
        return float3(0, 0, 0);
    }
    float3 kd = MatAlbedo(mat);
    float ks = 1.0 - max(kd.x, max(kd.y, kd.z));
    float2 u = float2(NextFloat(rng), NextFloat(rng));
    if (u.x < ks)
    {
        u.x /= ks;
        float3 wh = BeckmannSample(u, mat.alpha);
        wo = 2.0 * dot(wh, wi) * wh - wi;
    }
    else
    {
        u.x = (u.x - ks) / (1.0 - ks);
        wo = CosineSampleHemisphere(u);
    }
    if (wo.z <= 0.0)
    {
        bsdfPdf = 0.0;
        return float3(0, 0, 0);
    }
    bsdfPdf = MicrofacetPdf(wi, wo, mat);
    if (bsdfPdf <= 0.0)
        return float3(0, 0, 0);
    return MicrofacetEval(wi, wo, mat) * wo.z / bsdfPdf;
}

// DISNEY BSDFS START HERE

float3 DisneyDiffuseEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);

    float3 wh = normalize(wi + wo);
    float cosThetaD = abs(dot(wo, wh));

    float FD90 = 0.5 + 2.0 * mat.roughness * cosThetaD * cosThetaD;

    // Schlick-style Fresnel in view and light directions
    float FV = 1.0 + (FD90 - 1.0) * pow(1.0 - wi.z, 5.0);
    float FL = 1.0 + (FD90 - 1.0) * pow(1.0 - wo.z, 5.0);

    return MatAlbedo(mat) * M_INV_PI * FV * FL;
}

// Hanrahan-Krueger subsurface-approximation diffuse term from Disney 2012
// When `subsurface > 0`, Disney blends this with the Burley lobe to fake
// SSS. (the todo here would be to change into real SSS if time provides)
// this also is the lobe a real BSSRDF would eventually replace
float3 DisneyHKSubsurfaceEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);

    float3 wh = normalize(wi + wo);
    float cosThetaD = abs(dot(wo, wh));

    float FSS90 = mat.roughness * cosThetaD * cosThetaD;
    float FSS_V = 1.0 + (FSS90 - 1.0) * pow(1.0 - wi.z, 5.0);
    float FSS_L = 1.0 + (FSS90 - 1.0) * pow(1.0 - wo.z, 5.0);

    // cosV + cosL both are > 0 here, but could be tiny at exterme angles,we'd get NaN fireflies if not careful!
    float cosSum = max(wi.z + wo.z, 1e-4);
    float ss = 1.25 * (FSS_V * FSS_L * (1.0 / cosSum - 0.5) + 0.5);

    return MatAlbedo(mat) * M_INV_PI * ss;
}

// Blend the two diffuse lobes by the `subsurface` parameter (0 = pure
// Burley, 1 = pure Hanrahan-Krueger)
float3 DisneyDiffuseLobe(float3 wi, float3 wo, GPUMaterial mat)
{
    float3 fBurley = DisneyDiffuseEval(wi, wo, mat);
    float3 fSS = DisneyHKSubsurfaceEval(wi, wo, mat);
    return lerp(fBurley, fSS, mat.subsurface);
}

// GGX microfacet which was taught in lecture but also that Disney requires
//
// The point of GGX is that it has longer tails that match the BRDFS of real materials
// better. The isotropic form is implemented here
//
// the convention is that, alpha = roughness squared so that `roughness` feels linear.
// For example, a roughness of 0.5 sits halfway between mirror and matte from a perceptual standpoiint

float Luminance(float3 c)
{
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

// GGX normal distribution, Returns D(h), which is the density of microfacet normals per projected unit area.
float DisneyGGX_D(float NdotH, float alpha)
{
    if (NdotH <= 0.0)
        return 0.0;
    float a2 = alpha * alpha;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(M_PI * denom * denom, 1e-10);
}

// Smith G1 masking for GGX
float DisneyGGX_G1(float NdotV, float alpha)
{
    if (NdotV <= 0.0)
        return 0.0;
    float a2 = alpha * alpha;
    float NdotV2 = NdotV * NdotV;
    float denom = NdotV + sqrt(a2 + (1.0 - a2) * NdotV2);
    return 2.0 * NdotV / max(denom, 1e-10);
}

// sample a half-vector wh from the GGX NDF in the local frame, using cosTheta^2 = (1-u) / (1 + (a^2-1)*u), phi = 2*pi*v.
// Returns wh with wh.z = cosTheta
float3 DisneyGGX_SampleWh(float2 u, float alpha)
{
    float a2 = alpha * alpha;
    float cosTheta2 = (1.0 - u.x) / max(1.0 + (a2 - 1.0) * u.x, 1e-10);
    float cosTheta = sqrt(saturate(cosTheta2));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta2));
    float phi = 2.0 * M_PI * u.y;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// this is the solid-angle pdf of `wo` under GGX half-vector sampling, derived from
//   p(wh) = D(wh) * wh.z        (half-vector pdf, since NDF is
//                                normalized by integral of D * cosTheta_h)
//   p(wo) = p(wh) / (4 * |wh . wo|)   Jacobian of wh -> wo transform
float DisneyGGX_PdfWo(float3 wi, float3 wo, float alpha)
{
    float3 wh = normalize(wi + wo);
    if (wh.z <= 0.0)
        return 0.0;
    float D = DisneyGGX_D(wh.z, alpha);
    return D * wh.z / max(4.0 * abs(dot(wh, wo)), 1e-10);
}

// Schlick Fresnel with a tinted F0, returns F0 + (1-F0)*(1-cosD)^5.
float3 FresnelSchlick(float3 F0, float cosThetaD)
{
    float t = pow(1.0 - cosThetaD, 5.0);
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * t;
}

// disney's tinted F0 for the specular lobe, where:
//   dielectric F0 = 0.08 * specular * lerp(white, Ctint, specularTint)
//   metal F0 = baseColor.
//   final = lerp(dielectric_F0, metal_F0, metallic)
float3 DisneySpecularF0(GPUMaterial mat)
{
    float3 base = MatAlbedo(mat);
    float lum = Luminance(base);
    float3 Ctint = (lum > 0.0) ? base / lum : float3(1.0, 1.0, 1.0);
    float3 Cspec0 = 0.08 * mat.specular *
                    lerp(float3(1.0, 1.0, 1.0), Ctint, mat.specularTint);
    return lerp(Cspec0, base, mat.metallic);
}

// Cook-Torrance specular BRDF: f = D * F * G / (4 * NdotV * NdotL).
float3 DisneySpecularEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);

    float3 wh = normalize(wi + wo);
    if (wh.z <= 0.0)
        return float3(0, 0, 0);

    float alpha = max(mat.roughness * mat.roughness, 1e-4);
    float NdotV = wi.z;
    float NdotL = wo.z;
    float VdotH = abs(dot(wi, wh));

    float D = DisneyGGX_D(wh.z, alpha);
    float3 F = FresnelSchlick(DisneySpecularF0(mat), VdotH);
    float G = DisneyGGX_G1(NdotV, alpha) * DisneyGGX_G1(NdotL, alpha);

    return D * F * G / max(4.0 * NdotV * NdotL, 1e-10);
}

float DisneySpecularPdf(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return 0.0;
    float alpha = max(mat.roughness * mat.roughness, 1e-4);
    return DisneyGGX_PdfWo(wi, wo, alpha);
}
// TODO: sheen? not sure if i need this
// Sheen is a grazing-angle brightening term for fibrous / cloth / fuzz-like
// surfaces. Not physically based. empirical lobe from Disney 2012 that
// fills in the bright-silhouette look specular + diffuse can't reproduce
// on its own. Goes to zero at facing angles (cosThetaD ~ 1), peaks at
// grazing (cosThetaD ~ 0).
//
// Sampling: no new code needed. Sheen is cosine-hemisphere sampled via
// the diffuse lobe branch. The Fresnel-like weighting naturally aligns
// with regions where cosine-hemisphere samples land, and because sheen
// is small in magnitude it's tolerable that no importance-sampling is
// done for it specifically.
float3 DisneySheenEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (mat.sheen <= 0.0 || wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);

    float3 wh = normalize(wi + wo);
    float cosThetaD = abs(dot(wo, wh));

    float3 base = MatAlbedo(mat);
    float lum = Luminance(base);
    float3 Ctint = (lum > 0.0) ? base / lum : float3(1.0, 1.0, 1.0);
    float3 Csheen = lerp(float3(1.0, 1.0, 1.0), Ctint, mat.sheenTint);

    float fresnel = pow(1.0 - cosThetaD, 5.0);
    return mat.sheen * Csheen * fresnel;
}

// CLEARCOAT
float DisneyGTR1_D(float NdotH, float alpha)
{
    if (NdotH <= 0.0)
        return 0.0;
    float a2 = alpha * alpha;
    if (a2 >= 0.9999)
        return M_INV_PI;
    float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
    return (a2 - 1.0) / (M_PI * log(a2) * t);
}

// Sample wh from GTR1.
// inverse CDF =    cos^2(th_h) = (1 - (a^2)^(1-u1)) / (1 - a^2)
// for alpha=1, it is a flat NDF and it falls back to uniform hemisphere.
float3 DisneyGTR1_SampleWh(float2 u, float alpha)
{
    float a2 = alpha * alpha;
    float cosTheta2;
    if (a2 >= 0.9999)
        cosTheta2 = 1.0 - u.x;
    else
        cosTheta2 = (1.0 - pow(a2, 1.0 - u.x)) / (1.0 - a2);
    float cosTheta = sqrt(saturate(cosTheta2));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta2));
    float phi = 2.0 * M_PI * u.y;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// outgoing-direction pdf under GTR1 half-vector sampling.
float DisneyGTR1_PdfWo(float3 wi, float3 wo, float alpha)
{
    float3 wh = normalize(wi + wo);
    if (wh.z <= 0.0)
        return 0.0;
    float D = DisneyGTR1_D(wh.z, alpha);
    return D * wh.z / max(4.0 * abs(dot(wh, wo)), 1e-10);
}

// remap gloss [0,1] to the alpha Disney uses where clearcoatGloss=1 is smoothest.
float DisneyClearcoatAlpha(GPUMaterial mat)
{
    return lerp(0.1, 0.001, mat.clearcoatGloss);
}

// Clearcoat BRDF which returns the scalar lobe value
float DisneyClearcoatEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (mat.clearcoat <= 0.0 || wi.z <= 0.0 || wo.z <= 0.0)
        return 0.0;

    float3 wh = normalize(wi + wo);
    if (wh.z <= 0.0)
        return 0.0;

    float alpha = DisneyClearcoatAlpha(mat);
    float D = DisneyGTR1_D(wh.z, alpha);

    // Schlick Fresnel with fixed F0 = 0.04 (dielectric, IOR ~1.5)
    float VdotH = abs(dot(wi, wh));
    float F = 0.04 + (1.0 - 0.04) * pow(1.0 - VdotH, 5.0);

    // Smith G with fixed alpha=0.25 (Disney's non-physical darkening).
    // Using the GGX G1 helper since it's just Smith masking. the alpha value is what Disney's paper specifies here.
    float Gv = DisneyGGX_G1(wi.z, 0.25);
    float Gl = DisneyGGX_G1(wo.z, 0.25);
    float G = Gv * Gl;

    return 0.25 * mat.clearcoat * D * F * G / max(4.0 * wi.z * wo.z, 1e-10);
}

float DisneyClearcoatPdf(float3 wi, float3 wo, GPUMaterial mat)
{
    if (mat.clearcoat <= 0.0 || wi.z <= 0.0 || wo.z <= 0.0)
        return 0.0;
    float alpha = DisneyClearcoatAlpha(mat);
    return DisneyGTR1_PdfWo(wi, wo, alpha);
}

// THREE LOBE SAMPLING PROBABILITIES FOR DISNEY
//
// Disney heuristic for picking which lobe to sample from:
//   w_diffuse   = 1 - metallic
//   w_specular  = 1
//   w_clearcoat = 0.25 * clearcoat
// And then they are normalized so they sum to 1
struct DisneyLobeProbs
{
    float pDiffuse;
    float pSpecular;
    float pClearcoat;
};

DisneyLobeProbs DisneyComputeLobeProbs(GPUMaterial mat)
{
    DisneyLobeProbs p;
    float wDiff = 1.0 - mat.metallic;
    float wSpec = 1.0;
    float wCC = 0.25 * mat.clearcoat;
    float total = wDiff + wSpec + wCC;

    float inv = (total > 0.0) ? 1.0 / total : 0.0;
    p.pDiffuse = wDiff * inv;
    p.pSpecular = wSpec * inv;
    p.pClearcoat = wCC * inv;
    return p;
}

// ============================================================================
// HAIR BCSDF (Chiang et al. 2016 / PBRT 4th ed. Section 9.9)
//
// References throughout:
//   [PBRT]   Section 9.9 "Scattering from Hair"
//   [Chiang] "A Practical and Controllable Hair and Fur Model"
//
// The hair local frame has: x = fiber tangent, y = bitangent, z = tube normal.
// Longitudinal angle: sinθ = ω.x     (PBRT Section 9.9.1)
// Azimuthal angle:    φ = atan2(ω.z, ω.y)
// pMax = 3: lobes R(p=0), TT(p=1), TRT(p=2), residual(p=3)
// ============================================================================

static const int HAIR_PMAX = 3;

// Modified Bessel function of the first kind, order 0.
// [PBRT] Section 9.9.3, used in the longitudinal scattering function Mp.
float HairI0(float x)
{
    float val = 0.0, x2i = 1.0, ifact = 1.0, i4 = 1.0;
    for (int i = 0; i < 10; i++)
    {
        if (i > 0)
        {
            x2i *= x * x;
            ifact *= float(i);
            i4 *= 4.0;
        }
        val += x2i / (i4 * ifact * ifact);
    }
    return val;
}

// log(I0(x))
float HairLogI0(float x)
{
    if (x > 12.0)
        return x + 0.5 * (-log(2.0 * M_PI) + log(1.0 / x) + 1.0 / (8.0 * x));
    return log(HairI0(x));
}

// Longitudinal scattering function Mp (from d'Eon et al. 2011).
// [PBRT] Eq. 9.49, [Chiang] Section 3.1
float HairMp(float cosTheta_i, float cosTheta_o, float sinTheta_i,
             float sinTheta_o, float v)
{
    float a = cosTheta_i * cosTheta_o / v;
    float b = sinTheta_i * sinTheta_o / v;
    float mp;
    if (v <= 0.1)
        mp = exp(HairLogI0(a) - b - 1.0 / v + 0.6931 + log(1.0 / (2.0 * v)));
    else
        mp = (exp(-b) * HairI0(a)) / (sinh(1.0 / v) * 2.0 * v);
    return mp;
}

// Logistic distribution and its trimmed/sampled variants.
// [Chiang] Section 3.3, Appendix A
float HairLogistic(float x, float s)
{
    float ex = exp(-abs(x) / s);
    return ex / (s * (1.0 + ex) * (1.0 + ex));
}

float HairLogisticCDF(float x, float s)
{
    return 1.0 / (1.0 + exp(-x / s));
}

float HairTrimmedLogistic(float x, float s, float a, float b)
{
    float denom = HairLogisticCDF(b, s) - HairLogisticCDF(a, s);
    return (denom > 1e-10) ? HairLogistic(x, s) / denom : 0.0;
}

float HairSampleTrimmedLogistic(float u, float s, float a, float b)
{
    float cdfA = HairLogisticCDF(a, s);
    float cdfB = HairLogisticCDF(b, s);
    float t = cdfA + u * (cdfB - cdfA);
    t = clamp(t, 1e-6, 1.0 - 1e-6);
    return -s * log(1.0 / t - 1.0);
}

// Specular exit azimuth Φ(p, γ_o, γ_t).
// [PBRT] Section 9.9.5, Figure 9.51
float HairPhi(int p, float gamma_o, float gamma_t)
{
    return 2.0 * float(p) * gamma_t - 2.0 * gamma_o + float(p) * M_PI;
}

// Azimuthal scattering function Np.
// [PBRT] Section 9.9.5, [Chiang] Eq. 4
float HairNp(float phi, int p, float s, float gamma_o, float gamma_t)
{
    float dphi = phi - HairPhi(p, gamma_o, gamma_t);
    // Remap to [-π, π]
    while (dphi > M_PI)
        dphi -= 2.0 * M_PI;
    while (dphi < -M_PI)
        dphi += 2.0 * M_PI;
    return HairTrimmedLogistic(dphi, s, -M_PI, M_PI);
}

// Convert hair color to absorption coefficient σ_a.
// [Chiang] Eq. 9:  least-squares fit relating multiple-scattering
// albedo C and azimuthal roughness β_N to single-fiber σ_a.
float3 HairSigmaAFromColor(float3 C, float betaN)
{
    float denom = 5.969 - 0.215 * betaN + 2.532 * betaN * betaN - 10.73 * betaN * betaN * betaN + 5.574 * betaN * betaN * betaN * betaN + 0.245 * betaN * betaN * betaN * betaN * betaN;
    float3 sigma_a;
    sigma_a.x = (C.x > 1e-4) ? pow(log(C.x) / denom, 2.0) : 100.0;
    sigma_a.y = (C.y > 1e-4) ? pow(log(C.y) / denom, 2.0) : 100.0;
    sigma_a.z = (C.z > 1e-4) ? pow(log(C.z) / denom, 2.0) : 100.0;
    return sigma_a;
}

// Precompute per-lobe roughness v[p] from β_M.
// [PBRT] "HairBxDF constructor implementation", [Chiang] Eq. 7
struct HairLobeParams
{
    float v[4];          // longitudinal roughness variance per lobe
    float s;             // azimuthal logistic scale
    float sin2kAlpha[3]; // cuticle tilt sin(2^k * α)
    float cos2kAlpha[3]; // cuticle tilt cos(2^k * α)
    float3 sigma_a;      // absorption coefficient
};

HairLobeParams HairComputeParams(GPUMaterial mat)
{
    HairLobeParams p;

    // Longitudinal roughness: Chiang Eq. 7
    float bm = mat.roughness;
    float v0 = 0.726 * bm + 0.812 * bm * bm + 3.7 * pow(bm, 20.0);
    v0 = v0 * v0;
    p.v[0] = v0;
    p.v[1] = 0.25 * v0; // TT lobe: sharper, refraction focuses
    p.v[2] = 4.0 * v0;  // TRT: broader, multiple reflections spread
    p.v[3] = p.v[2];

    // Azimuthal roughness from Chiang Eq. 8
    float bn = mat.betaN;
    float SqrtPiOver8 = 0.626657069;
    p.s = SqrtPiOver8 * (0.265 * bn + 1.194 * bn * bn + 5.372 * pow(bn, 22.0));

    // Cuticle scale tilt from PBRT Section 9.9.6
    float alphaRad = mat.alpha * M_PI / 180.0;
    p.sin2kAlpha[0] = sin(alphaRad);
    p.cos2kAlpha[0] = sqrt(max(0.0, 1.0 - p.sin2kAlpha[0] * p.sin2kAlpha[0]));
    for (int i = 1; i < 3; i++)
    {
        p.sin2kAlpha[i] = 2.0 * p.cos2kAlpha[i - 1] * p.sin2kAlpha[i - 1];
        p.cos2kAlpha[i] = p.cos2kAlpha[i - 1] * p.cos2kAlpha[i - 1] - p.sin2kAlpha[i - 1] * p.sin2kAlpha[i - 1];
    }

    // absorption from color
    p.sigma_a = HairSigmaAFromColor(MatAlbedo(mat), bn);

    return p;
}

// Compute attenuation Ap for all lobes
// [PBRT] Section 9.9.4, [Chiang] Eq. 6 for 4th lobe residual
void HairAp(float cosTheta_o, float sinTheta_o, float h, float eta,
            float3 sigma_a, out float3 ap[4])
{
    // Modified IOR for normal-plane refraction from [PBRT] Section 9.9.4
    float etap = sqrt(max(0.0, eta * eta - sinTheta_o * sinTheta_o)) / max(cosTheta_o, 1e-6);
    float sinGamma_t = clamp(h / etap, -1.0, 1.0);
    float cosGamma_t = sqrt(max(0.0, 1.0 - sinGamma_t * sinGamma_t));

    // Transmitted angle
    float sinTheta_t = sinTheta_o / eta;
    float cosTheta_t = sqrt(max(0.0, 1.0 - sinTheta_t * sinTheta_t));

    // Single-segment transmittance from Beer's law, PBRT Eq. 9.50
    float3 T = exp(-sigma_a * (2.0 * cosGamma_t / max(cosTheta_t, 1e-6)));

    // Fresnel at entry
    float cosGamma_o = sqrt(max(0.0, 1.0 - h * h));
    float f = FresnelDielectric(cosTheta_o * cosGamma_o, 1.0, eta);

    // Lobe attenuations: [PBRT] Section 9.9.4
    ap[0] = float3(f, f, f);           // R: surface reflection
    ap[1] = (1.0 - f) * (1.0 - f) * T; // TT: two transmissions
    ap[2] = ap[1] * T * f;             // TRT: +reflection +segment
    // Residual (Chiang Eq. 6) for geometric series for all p >= 3
    float3 tfProduct = T * f;
    float3 denom = max(float3(1, 1, 1) - tfProduct, float3(1e-10, 1e-10, 1e-10));
    ap[3] = ap[2] * f * T / denom;
}

// Apply cuticle scale tilt for lobe p.
// [PBRT] Section 9.9.6: p=0 rotates by -2α, p=1 by +α, p=2 by +4α
void HairScaleTilt(int p, float sinTheta_o, float cosTheta_o,
                   float sin2kAlpha[3], float cos2kAlpha[3],
                   out float sinThetap_o, out float cosThetap_o)
{
    if (p == 0)
    {
        sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
        cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
    }
    else if (p == 1)
    {
        sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
        cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
    }
    else if (p == 2)
    {
        sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
        cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
    }
    else
    {
        sinThetap_o = sinTheta_o;
        cosThetap_o = cosTheta_o;
    }
    cosThetap_o = abs(cosThetap_o);
}

// Full hair BCSDF evaluation. [PBRT] Section 9.9.6, Eq. 9.48
// wi, wo are in the hair local frame (tangent=x, normal=z).
// h is the fiber offset ∈ [-1,1] from the tessellator.
float3 HairBCSDF_Eval(float3 wi, float3 wo, GPUMaterial mat, float h)
{
    // No hemisphere check here! hair BCSDF is valid for all directions.
    // Light enters/exits from any angle around the fiber.

    // Hair coordinate angles: [PBRT] Section 9.9.1
    // Our wi = view direction = PBRT's "outgoing" (subscript _o)
    // Our wo = light direction = PBRT's "incident" (subscript _i)
    float sinTheta_o = wi.x;
    float cosTheta_o = sqrt(max(0.0, 1.0 - sinTheta_o * sinTheta_o));
    float phi_o = atan2(wi.z, wi.y);

    float sinTheta_i = wo.x;
    float cosTheta_i = sqrt(max(0.0, 1.0 - sinTheta_i * sinTheta_i));
    float phi_i = atan2(wo.z, wo.y);

    float phi = phi_i - phi_o;
    float gamma_o = asin(clamp(h, -1.0, 1.0));

    float eta = mat.intIOR;

    // Refracted azimuthal angle γ_t: [PBRT] Section 9.9.4
    float etap = sqrt(max(0.0, eta * eta - sinTheta_o * sinTheta_o)) / max(cosTheta_o, 1e-6);
    float gamma_t = asin(clamp(h / etap, -1.0, 1.0));

    // Precompute all lobe parameters
    HairLobeParams lp = HairComputeParams(mat);

    // Attenuation
    float3 ap[4];
    HairAp(cosTheta_o, sinTheta_o, h, eta, lp.sigma_a, ap);

    // Sum over lobes: [PBRT] "Evaluate hair BSDF"
    float3 fsum = float3(0, 0, 0);
    for (int p = 0; p < HAIR_PMAX; p++)
    {
        float sinThetap_o, cosThetap_o;
        HairScaleTilt(p, sinTheta_o, cosTheta_o,
                      lp.sin2kAlpha, lp.cos2kAlpha,
                      sinThetap_o, cosThetap_o);

        float mp = HairMp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, lp.v[p]);
        float np = HairNp(phi, p, lp.s, gamma_o, gamma_t);
        fsum += mp * ap[p] * np;
    }

    // Residual lobe: isotropic azimuthal (1/2π). [Chiang] Section 3.4
    fsum += HairMp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, lp.v[HAIR_PMAX]) * ap[HAIR_PMAX] / (2.0 * M_PI);

    // [PBRT] Eq. 9.48: divide by |cos θ_i| to cancel the cosθ factor
    // that the rendering equation/integrator multiplies by. In our convention,
    // wo = light direction (PBRT's incident), so wo.z = cos angle with tube normal.
    float absCosTheta = abs(wo.z);
    if (absCosTheta > 1e-6)
        fsum /= absCosTheta;

    return fsum;
}

// Hair PDF. [PBRT] Section 9.9.7
float HairBCSDF_Pdf(float3 wi, float3 wo, GPUMaterial mat, float h)
{
    // No hemisphere check — hair BCSDF works for all directions.

    // Our wi = view (PBRT _o), our wo = light (PBRT _i)
    float sinTheta_o = wi.x;
    float cosTheta_o = sqrt(max(0.0, 1.0 - sinTheta_o * sinTheta_o));
    float phi_o = atan2(wi.z, wi.y);

    float sinTheta_i = wo.x;
    float cosTheta_i = sqrt(max(0.0, 1.0 - sinTheta_i * sinTheta_i));
    float phi_i = atan2(wo.z, wo.y);

    float phi = phi_i - phi_o;
    float gamma_o = asin(clamp(h, -1.0, 1.0));
    float eta = mat.intIOR;
    float etap = sqrt(max(0.0, eta * eta - sinTheta_o * sinTheta_o)) / max(cosTheta_o, 1e-6);
    float gamma_t = asin(clamp(h / etap, -1.0, 1.0));

    HairLobeParams lp = HairComputeParams(mat);

    // Compute Ap luminance for discrete lobe probabilities
    float3 ap[4];
    HairAp(cosTheta_o, sinTheta_o, h, eta, lp.sigma_a, ap);

    float apLum[4];
    float sumLum = 0.0;
    for (int i = 0; i <= HAIR_PMAX; i++)
    {
        apLum[i] = Luminance(ap[i]);
        sumLum += apLum[i];
    }
    float invSum = (sumLum > 0.0) ? 1.0 / sumLum : 0.0;

    float pdf = 0.0;
    for (int p = 0; p < HAIR_PMAX; p++)
    {
        float sinThetap_o, cosThetap_o;
        HairScaleTilt(p, sinTheta_o, cosTheta_o,
                      lp.sin2kAlpha, lp.cos2kAlpha,
                      sinThetap_o, cosThetap_o);

        pdf += HairMp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, lp.v[p]) * apLum[p] * invSum * HairNp(phi, p, lp.s, gamma_o, gamma_t);
    }
    // Residual lobe for uniform azimuthal
    pdf += HairMp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, lp.v[HAIR_PMAX]) * apLum[HAIR_PMAX] * invSum * (1.0 / (2.0 * M_PI));

    return pdf;
}

// Hair BCSDF sampling. Returns f * cosθ / pdf (MC throughput weight).
// [PBRT] Section 9.9.7
float3 HairBCSDF_Sample(float3 wi, inout RNG rng, GPUMaterial mat, float h,
                        out float3 wo, out float pdf)
{
    float sinTheta_o = wi.x; // NOTE: wi is the viewing direction = "wo" in PBRT notation
    float cosTheta_o = sqrt(max(0.0, 1.0 - sinTheta_o * sinTheta_o));
    float phi_o = atan2(wi.z, wi.y);
    float gamma_o = asin(clamp(h, -1.0, 1.0));
    float eta = mat.intIOR;
    float etap = sqrt(max(0.0, eta * eta - sinTheta_o * sinTheta_o)) / max(cosTheta_o, 1e-6);
    float gamma_t = asin(clamp(h / etap, -1.0, 1.0));

    HairLobeParams lp = HairComputeParams(mat);

    // Attenuation for lobe selection
    float3 ap[4];
    HairAp(cosTheta_o, sinTheta_o, h, eta, lp.sigma_a, ap);

    // Discrete lobe probabilities from Ap luminance
    float apPDF[4];
    float sumLum = 0.0;
    for (int i = 0; i <= HAIR_PMAX; i++)
    {
        apPDF[i] = Luminance(ap[i]);
        sumLum += apPDF[i];
    }
    float invSum = (sumLum > 0.0) ? 1.0 / sumLum : 0.0;
    for (int j = 0; j <= HAIR_PMAX; j++)
        apPDF[j] *= invSum;

    // Choose lobe p: [PBRT] "Determine which term to sample"
    float u0 = NextFloat(rng);
    int p = 0;
    float cdf = apPDF[0];
    while (p < HAIR_PMAX && u0 > cdf)
    {
        p++;
        cdf += apPDF[p];
    }
    // Remap u0 for reuse
    float lobeStart = cdf - apPDF[p];
    u0 = clamp((u0 - lobeStart) / max(apPDF[p], 1e-10), 0.0, 1.0);

    // Apply cuticle tilt for the selected lobe
    float sinThetap_o, cosThetap_o;
    HairScaleTilt(p, sinTheta_o, cosTheta_o,
                  lp.sin2kAlpha, lp.cos2kAlpha,
                  sinThetap_o, cosThetap_o);

    // Sample Mp for θ_i: [PBRT] "Sample Mp to compute θ_i"
    float u1 = NextFloat(rng);
    float u2 = NextFloat(rng);
    float cosTheta = 1.0 + lp.v[p] * log(max(u1, 1e-5) + (1.0 - u1) * exp(-2.0 / lp.v[p]));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float cosPhi_sample = cos(2.0 * M_PI * u2);
    float sinTheta_i = -cosTheta * sinThetap_o + sinTheta * cosPhi_sample * cosThetap_o;
    float cosTheta_i = sqrt(max(0.0, 1.0 - sinTheta_i * sinTheta_i));

    // Sample Np for φ: [PBRT] "Sample Np to compute φ"
    float u3 = NextFloat(rng);
    float dphi;
    if (p < HAIR_PMAX)
        dphi = HairPhi(p, gamma_o, gamma_t) +
               HairSampleTrimmedLogistic(u3, lp.s, -M_PI, M_PI);
    else
        dphi = 2.0 * M_PI * u3;

    // Construct sampled direction: [PBRT] "Compute wi from sampled angles"
    float phi_i = phi_o + dphi;
    wo = float3(sinTheta_i,
                cosTheta_i * cos(phi_i),
                cosTheta_i * sin(phi_i));

    // No hemisphere check — hair scatters in all directions around fiber.

    // Compute combined PDF across all lobes
    pdf = HairBCSDF_Pdf(wi, wo, mat, h);
    if (pdf <= 0.0)
        return float3(0, 0, 0);

    // Evaluate full BCSDF at sampled direction
    float3 f = HairBCSDF_Eval(wi, wo, mat, h);

    // Return f * |cosθ| / pdf. Since HairBCSDF_Eval already divides by |cosθ|
    // (PBRT Eq. 9.48), multiplying by |cosθ| here cancels it, giving the
    // correct MC weight where the integrator's cosθ factor is handled.
    return f * abs(wo.z) / pdf;
}

// ============================================================================
// MATERIAL DISPATCH LAYER
// ============================================================================

bool MaterialIsDelta(GPUMaterial mat)
{
    return mat.type == 1 || mat.type == 2;
}

// Note: for hair (type 5), wi/wo must be in the hair frame (tangent=x, normal=z).
// The hair-specific 'h' parameter is passed via a global set before calling.
static float g_hairH = 0.0; // set by RayGen before dispatch calls for hair

float3 MaterialEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (mat.type == 0) // Diffuse
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return float3(0, 0, 0);
        return MatAlbedo(mat) * M_INV_PI;
    }
    else if (mat.type == 3) // Microfacet
    {
        return MicrofacetEval(wi, wo, mat);
    }
    else if (mat.type == 4) // Disney
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return float3(0, 0, 0);
        float3 fDiffuse = DisneyDiffuseLobe(wi, wo, mat);
        float3 fSpec = DisneySpecularEval(wi, wo, mat);
        float3 fSheen = DisneySheenEval(wi, wo, mat);
        float fCC = DisneyClearcoatEval(wi, wo, mat);
        return (1.0 - mat.metallic) * (fDiffuse + fSheen) + fSpec + fCC;
    }
    else if (mat.type == 5) // Hair
    {
        return HairBCSDF_Eval(wi, wo, mat, g_hairH);
    }
    return float3(0, 0, 0);
}

float MaterialPdf(float3 wi, float3 wo, GPUMaterial mat)
{
    if (mat.type == 0) // Diffuse
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return 0.0;
        return wo.z * M_INV_PI;
    }
    else if (mat.type == 3) // Microfacet
    {
        return MicrofacetPdf(wi, wo, mat);
    }
    else if (mat.type == 4) // Disney
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return 0.0;
        DisneyLobeProbs p = DisneyComputeLobeProbs(mat);
        float pdfDiff = wo.z * M_INV_PI;
        float pdfSpec = DisneySpecularPdf(wi, wo, mat);
        float pdfCC = DisneyClearcoatPdf(wi, wo, mat);
        return p.pDiffuse * pdfDiff + p.pSpecular * pdfSpec + p.pClearcoat * pdfCC;
    }
    else if (mat.type == 5) // Hair
    {
        return HairBCSDF_Pdf(wi, wo, mat, g_hairH);
    }
    return 0.0;
}

// Sample an outgoing direction. Returns f * cos(theta_o) / pdf (, which is the Monte Carlo throughput weight).
// Outputs wo in the local frame and the solid-angle pdf. Not intended for delta BSDFs
float3 MaterialSample(float3 wi, inout RNG rng, GPUMaterial mat,
                      out float3 wo, out float pdf)
{
    if (mat.type == 0) // Diffuse
    {
        if (wi.z <= 0.0)
        {
            wo = float3(0, 0, 1);
            pdf = 0.0;
            return float3(0, 0, 0);
        }
        float2 u = float2(NextFloat(rng), NextFloat(rng));
        wo = CosineSampleHemisphere(u);
        pdf = wo.z * M_INV_PI;
        return MatAlbedo(mat);
    }
    else if (mat.type == 3) // Microfacet
    {
        return MicrofacetSample(wi, rng, mat, wo, pdf);
    }
    else if (mat.type == 4) // Disney
    {
        if (wi.z <= 0.0)
        {
            wo = float3(0, 0, 1);
            pdf = 0.0;
            return float3(0, 0, 0);
        }

        DisneyLobeProbs p = DisneyComputeLobeProbs(mat);
        float u0 = NextFloat(rng);
        float2 u12 = float2(NextFloat(rng), NextFloat(rng));

        // pick a lobe which is one of diffuse, specular, or clearcoat.
        if (u0 < p.pDiffuse)
        {
            // Diffuse lobe covers Burley, HK subsurface, and sheen which are all cosine hemispehre sampled
            wo = CosineSampleHemisphere(u12);
        }
        else if (u0 < p.pDiffuse + p.pSpecular)
        {
            // specular GGX lobe
            float alpha = max(mat.roughness * mat.roughness, 1e-4);
            float3 wh = DisneyGGX_SampleWh(u12, alpha);
            wo = 2.0 * dot(wh, wi) * wh - wi;
        }
        else
        {
            // clearcoat (GTR1) lobe.
            float alpha = DisneyClearcoatAlpha(mat);
            float3 wh = DisneyGTR1_SampleWh(u12, alpha);
            wo = 2.0 * dot(wh, wi) * wh - wi;
        }

        if (wo.z <= 0.0)
        {
            pdf = 0.0;
            return float3(0, 0, 0);
        }

        // Combined pdf across all three sampling lobes
        float pdfDiff = wo.z * M_INV_PI;
        float pdfSpec = DisneySpecularPdf(wi, wo, mat);
        float pdfCC = DisneyClearcoatPdf(wi, wo, mat);
        pdf = p.pDiffuse * pdfDiff + p.pSpecular * pdfSpec + p.pClearcoat * pdfCC;
        if (pdf <= 0.0)
            return float3(0, 0, 0);

        // Full BSDF at the sampled direction.
        float3 fDiffuse = DisneyDiffuseLobe(wi, wo, mat);
        float3 fSpec = DisneySpecularEval(wi, wo, mat);
        float3 fSheen = DisneySheenEval(wi, wo, mat);
        float fCC = DisneyClearcoatEval(wi, wo, mat);
        float3 fTotal = (1.0 - mat.metallic) * (fDiffuse + fSheen) + fSpec + fCC;

        return fTotal * wo.z / pdf;
    }
    else if (mat.type == 5) // Hair (Chiang BCSDF)
    {
        return HairBCSDF_Sample(wi, rng, mat, g_hairH, wo, pdf);
    }
    wo = float3(0, 0, 1);
    pdf = 0.0;
    return float3(0, 0, 0);
}

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

    // Uniform emitter selection: P(emitter) = 1 / emitterCount.
    // The CPU still prepends a power CDF at the front of g_emitterCdf and
    // shifts each per-emitter triangle CDF past it, so eMat.emitterCdfOffset
    // remains valid; we just don't read the power CDF block here.
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
    // Match SampleEmitter: pdfArea = 1 / (surfaceArea * emitterCount).
    return (1.0 / (emitMat.surfaceArea * float(emitterCount))) * dist2 / cosL;
}

float3 MISDirectIllumination(float3 hitPos, float3 N, float3 Ng, float3 T, float3 B,
                             float3 wi_local, GPUMaterial mat, inout RNG rng)
{
    EmitterSample es = SampleEmitter(rng);
    if (!es.valid)
        return float3(0, 0, 0);
    float3 toLight = es.position - hitPos;
    float dist = length(toLight);
    float3 wi_world = toLight / dist;
    float cosTheta = dot(N, wi_world);
    float cosLight = dot(es.normal, -wi_world);
    if (cosTheta <= 0.0 || cosLight <= 0.0)
        return float3(0, 0, 0);

    float3 shadowOrigin = OffsetRayOrigin(hitPos, Ng, N);
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
    float3 f = MaterialEval(wi_local, wo_local, mat);
    float pdfBsdf = MaterialPdf(wi_local, wo_local, mat);

    float pdfEms = es.pdfArea * dist * dist / cosLight;
    float w = BalanceHeuristic(pdfEms, pdfBsdf);
    float3 volTr = MultiVolumeTransmittance(shadowOrigin, wi_world, dist, rng);
    return es.radiance * f * cosTheta / max(pdfEms, 1e-20) * w * volTr * shadow.transmission;
}

// Envmap next-event estimation, which samples one direction from the envmap's importance distribution, shoots a shadow ray evaluates the BSDF in that direction, and
// MIS-weights against the BSDF pdf. Called from raygen for diffuse and microfacet hits, in addition to MISDirectIllumination and the two contributions are summed.
float3 EnvmapDirectIllumination(float3 hitPos, float3 N, float3 Ng, float3 T, float3 B,
                                float3 wi_local, GPUMaterial mat, inout RNG rng)
{
    float u1 = NextFloat(rng);
    float u2 = NextFloat(rng);
    float3 wi_world, Lenv;
    float pdfEnv;
    SampleEnvmap(u1, u2, wi_world, Lenv, pdfEnv);
    if (pdfEnv <= 0.0)
        return float3(0, 0, 0);

    float cosTheta = dot(N, wi_world);
    if (cosTheta <= 0.0)
        return float3(0, 0, 0);

    float3 shadowOrigin = OffsetRayOrigin(hitPos, Ng, N);
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
    float3 f = MaterialEval(wi_local, wo_local, mat);
    float pdfBsdf = MaterialPdf(wi_local, wo_local, mat);

    float w = BalanceHeuristic(pdfEnv, pdfBsdf);
    float3 volTr = MultiVolumeTransmittance(shadowOrigin, wi_world, 1e20, rng);
    return Lenv * f * cosTheta / max(pdfEnv, 1e-20) * w * volTr * shadow.transmission;
}

// Volume NEE
//  at a medium scatter point, explicitly sample a light direction and evaluate the phase function.
// Unlike surface NEE, there is no cosine factor at the scatter point because volumes scatter isotropically w.r.t. geometry. the directional
// dependence is entirely in the phase function.
// MIS-weighted against the phase function sampling pdf.

float3 VolumeNEEAreaLight(float3 scatterPos, float3 wo, uint volumeIndex, inout RNG rng)
{
    // Early exit: if scatter point is deep inside the scattering volume in
    // every direction, transmittance to any light will be negligible. Skip
    // expensive shadow ray. Uses the volume the scatter happened in; ignores
    // other volumes the shadow ray may cross (cheap and conservative). For
    // RGB extinction we test against the dimmest channel — even one
    // surviving channel is reason enough to keep the shadow ray.
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

    // Emitter pdf in solid angle, phase function pdf = fp (already solid angle)
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
    // Early exit: envmap shadow rays must traverse the full remaining volume.
    // If the scatter point is deep inside *every* channel, transmittance
    // will be negligible.
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

// Ray Generation

[shader("raygeneration")] void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
    RNG rng = InitRNG(pixel, dims, frameCount);

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
        g_output[pixel] = float4(saturate(averaged), 1.0);
        return;
    }
#endif

    float2 jitter = float2(NextFloat(rng), NextFloat(rng));
    float2 uv = (float2(pixel) + jitter) / float2(dims);
    uv.y = 1.0 - uv.y;

    float3 dir = normalize(
        camLowerLeftCorner + uv.x * camHorizontal + uv.y * camVertical - camPos);

    RayDesc ray;
    ray.Origin = camPos;
    ray.Direction = dir;
    ray.TMin = 0.001;
    ray.TMax = 1e20;

    if (lensRadius > 0.0)
    {
        // Thin-lens depth of field — mirrors perspective.cpp sampleRay()
        // Reconstruct camera basis in world space from the image-plane vectors
        float3 camFwd = normalize(camLowerLeftCorner + 0.5 * camHorizontal + 0.5 * camVertical - camPos);
        float3 camRight = normalize(camHorizontal);
        float3 camUp = normalize(camVertical);

        // Focus point: walk along the pinhole ray until its projection onto
        // the optical axis equals focalDistance (equivalent to z = focalDistance
        // in camera space)
        float ft = focalDistance / dot(dir, camFwd);
        float3 focusPoint = camPos + dir * ft;

        // Sample a uniformly distributed point on the circular aperture disk
        // (squareToUniformDisk: r = sqrt(u1), theta = 2*pi*u2)
        float u1 = NextFloat(rng);
        float u2 = NextFloat(rng);
        float r = sqrt(u1) * lensRadius;
        float theta = 2.0 * M_PI * u2;
        float3 lensOffset = (r * cos(theta)) * camRight + (r * sin(theta)) * camUp;

        ray.Origin = camPos + lensOffset;
        ray.Direction = normalize(focusPoint - ray.Origin);
    }

    float3 Lo = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    float lastBsdfPdf = 0.0;
    float eta = 1.0;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        HitPayload payload;
        payload.hit = 0;
        payload.rngState = rng.state;
        TraceRay(g_scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
        rng.state = payload.rngState;

        // ── Volume free-flight check ──────────────────────────────────
        // [Marschner] §3, §5: before handling the surface hit, see whether
        // the ray scatters inside any participating medium first. The
        // multi-volume entry point walks all volumes the ray crosses and
        // returns the first scatter event (if any), along with the index
        // of the scattering volume so we can read its phase param.
        //
        // SampleVolumeFreeFlight uses null-collision tracking and threads a
        // per-channel weight back to us. We multiply throughput by it
        // unconditionally:
        //   - Pass-through: weight is the per-channel transmittance through
        //     every volume the ray crossed (dim-extinction channels survive
        //     better than the hero), so surfaces behind tinted media pick
        //     up the correct color.
        //   - Real scatter: weight already includes the σ_t_volume,c / μ
        //     factor at the scatter site; combined with the σ_s,c / σ_t,c
        //     albedo below, the scatter contribution becomes σ_s,c / μ
        //     per channel (matching the previous formulation).
        bool volumeScattered = false;
        if (volumeCount > 0)
        {
            float tSurface = payload.hit ? payload.hitT : 1e20;
            float tScatter;
            uint scatterVolIdx;
            float3 volWeight = float3(1, 1, 1);
            bool scattered = SampleVolumeFreeFlight(
                ray.Origin, ray.Direction, tSurface,
                rng, volWeight, tScatter, scatterVolIdx);
            throughput *= volWeight;

            if (scattered)
            {
                volumeScattered = true;
                GPUVolume scVol = g_volumes[scatterVolIdx];
                float3 scSigmaT = VolumeSigmaT(scVol);
                float scPhaseG = scVol.phaseG;
                float3 scatterPos = ray.Origin + tScatter * ray.Direction;

                // Per-channel single-scattering albedo σ_s,c / σ_t,c. The
                // null-collision sampling already contributed σ_t,c / μ, so
                // the combined per-channel weight at the scatter event
                // equals σ_s,c / μ.
                throughput *= scVol.sigmaS / max(scSigmaT, float3(1e-20, 1e-20, 1e-20));

                // [Marschner] §5 "Direct lighting for volumes": NEE
                Lo += throughput * VolumeNEEAreaLight(scatterPos,
                                                      ray.Direction, scatterVolIdx, rng);
                Lo += throughput * VolumeNEEEnvmap(scatterPos,
                                                   ray.Direction, scatterVolIdx, rng);

                // [Marschner] §5: importance-sample phase function
                // for the indirect bounce direction.
                float phasePdf;
                float3 newDir = SampleHG(ray.Direction, scPhaseG,
                                         rng, phasePdf);

                ray.Origin = scatterPos;
                ray.Direction = newDir;
                ray.TMin = 0.0;
                ray.TMax = 1e20;

                // Store phase pdf for envmap MIS on the next miss
                lastBsdfPdf = max(phasePdf, 1e-20);
            }
        }

        // Branch: volume scatter / miss / surface hit
        if (volumeScattered)
        {
            // Medium scatter — skip surface handling, proceed to
            // Russian roulette at the bottom of the loop.
        }
        else if (!payload.hit)
        {
            float3 env = float3(payload.envR, payload.envG, payload.envB);
            if (lastBsdfPdf == 0.0)
            {
                Lo += throughput * env;
            }
            else
            {
                float pdfEnv = EnvmapPdfDirection(ray.Direction);
                float w = BalanceHeuristic(lastBsdfPdf, pdfEnv);
                Lo += throughput * env * w;
            }
            break;
        }
        else
        {
            // surface hit

            float3 hitPos = ray.Origin + ray.Direction * payload.hitT;
            float3 N = normalize(float3(payload.normalX, payload.normalY, payload.normalZ));
            float3 Ng = normalize(float3(payload.geoNormalX, payload.geoNormalY, payload.geoNormalZ));
            GPUMaterial mat = g_materials[payload.materialID];
            float2 hitUV = float2(payload.texU, payload.texV);

            bool hitBackFace = (dot(Ng, ray.Direction) > 0.0);

            if (hitBackFace)
                Ng = -Ng;

            // BSDF returns zero, producing solid-black rectangles).
            if (dot(N, Ng) < 0.0)
                N = -N;

            bool hasAnyTex = (mat.albedoTexIndex != 0xFFFFFFFF) ||
                             (mat.normalTexIndex != 0xFFFFFFFF) ||
                             (mat.roughnessTexIndex != 0xFFFFFFFF) ||
                             (mat.specularTexIndex != 0xFFFFFFFF) ||
                             (mat.subsurfaceTexIndex != 0xFFFFFFFF);
            float uvFoot = 0.0;
            if (hasAnyTex)
                uvFoot = ComputeUVFootprint(payload.materialID, payload.primitiveID, payload.hitT, dims);

            // Texture sampling with LOD derived from actual texture resolution
            float3 texAlbedo = MatAlbedo(mat);
            if (mat.albedoTexIndex != 0xFFFFFFFF)
            {
                float lod = ComputeTexLOD(g_textures[mat.albedoTexIndex], uvFoot);
                texAlbedo = g_textures[mat.albedoTexIndex].SampleLevel(g_sampler, hitUV, lod).rgb;
            }

            // mormal map,  compute tangent frame from triangle UV derivatives
            if (mat.normalTexIndex != 0xFFFFFFFF)
            {
                uint base = mat.indexOffset + payload.primitiveID * 3;
                uint i0 = g_indices.Load((base + 0) * 4);
                uint i1 = g_indices.Load((base + 1) * 4);
                uint i2 = g_indices.Load((base + 2) * 4);

                float3 p0 = LoadFloat3(g_vertices, mat.vertexOffset + i0);
                float3 p1 = LoadFloat3(g_vertices, mat.vertexOffset + i1);
                float3 p2 = LoadFloat3(g_vertices, mat.vertexOffset + i2);

                float2 uv0 = LoadFloat2(g_texcoords, mat.vertexOffset + i0);
                float2 uv1 = LoadFloat2(g_texcoords, mat.vertexOffset + i1);
                float2 uv2 = LoadFloat2(g_texcoords, mat.vertexOffset + i2);

                float3 edge1 = p1 - p0;
                float3 edge2 = p2 - p0;
                float2 dUV1 = uv1 - uv0;
                float2 dUV2 = uv2 - uv0;

                float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;

                float3 T, B;
                if (abs(det) > 1e-8)
                {
                    float invDet = 1.0 / det;
                    T = normalize((dUV2.y * edge1 - dUV1.y * edge2) * invDet);
                    // orthogonalize T w.r.t. interpolated N, then derive B
                    T = normalize(T - N * dot(N, T));
                    B = cross(N, T);
                }
                else
                {
                    BuildONB(N, T, B);
                }

                float nLod = ComputeTexLOD(g_textures[mat.normalTexIndex], uvFoot);
                float3 tangentNormal = g_textures[mat.normalTexIndex].SampleLevel(g_sampler, hitUV, nLod).xyz;
                tangentNormal = tangentNormal * 2.0 - 1.0;
                N = normalize(T * tangentNormal.x + B * tangentNormal.y + N * tangentNormal.z);
            }

            mat.albedoR = texAlbedo.x;
            mat.albedoG = texAlbedo.y;
            mat.albedoB = texAlbedo.z;
            // For hair (type 5), mat.roughness = β_M and mat.alpha = cuticle tilt — leave them.
            // For everything else, only overwrite roughness/alpha if a roughness texture is actually bound.
            if (mat.type != 5 && mat.roughnessTexIndex != 0xFFFFFFFF)
            {
                float rLod = ComputeTexLOD(g_textures[mat.roughnessTexIndex], uvFoot);
                float texRough = g_textures[mat.roughnessTexIndex].SampleLevel(g_sampler, hitUV, rLod).r;
                mat.alpha = texRough;
                mat.roughness = texRough;
            }

            if (mat.specularTexIndex != 0xFFFFFFFF)
            {
                float sLod = ComputeTexLOD(g_textures[mat.specularTexIndex], uvFoot);
                mat.specular = g_textures[mat.specularTexIndex].SampleLevel(g_sampler, hitUV, sLod).r * 0.5;
            }
            if (mat.subsurfaceTexIndex != 0xFFFFFFFF)
            {
                float ssLod = ComputeTexLOD(g_textures[mat.subsurfaceTexIndex], uvFoot);
                mat.subsurface = g_textures[mat.subsurfaceTexIndex].SampleLevel(g_sampler, hitUV, ssLod).r;
            }

            if (mat.isEmitter)
            {
                if (lastBsdfPdf == 0.0)
                    Lo += throughput * MatRadiance(mat);
                else
                {

                    float pdfEms = EmitterPdfSolidAngle(mat, hitPos, ray.Origin, N);
                    Lo += throughput * MatRadiance(mat) * BalanceHeuristic(lastBsdfPdf, pdfEms);
                }
                break;
            }

            float3 T, B;
            if (mat.type == 5)
            {
                // Hair: build frame with fiber tangent as x-axis, tube surface normal as z.
                // Reference: [PBRT] Section 9.9.1 — sinθ = ω.x (tangent component)
                float3 hairTangent = float3(payload.tangentX, payload.tangentY, payload.tangentZ);
                bool hasStrandTangent = dot(hairTangent, hairTangent) > 1e-8;
                if (hasStrandTangent)
                {
                    // Strand geometry from tessellator — tangent buffer is valid.
                    // h was encoded by tessellator as UV.y = (h+1)/2.
                    hairTangent = normalize(hairTangent);
                    g_hairH = payload.hairH; // payload.hairH = texUV.y * 2 - 1
                }
                else
                {
                    // Hair card (OBJ mesh) — no tangent buffer.
                    // Derive fiber axis from UV: dP/dV runs along hair length (root→tip).
                    // h is the offset across the card width: UV.x remapped to [-1,1].
                    uint base2 = mat.indexOffset + payload.primitiveID * 3;
                    uint ci0 = g_indices.Load((base2 + 0) * 4);
                    uint ci1 = g_indices.Load((base2 + 1) * 4);
                    uint ci2 = g_indices.Load((base2 + 2) * 4);
                    float3 cp0 = LoadFloat3(g_vertices, mat.vertexOffset + ci0);
                    float3 cp1 = LoadFloat3(g_vertices, mat.vertexOffset + ci1);
                    float3 cp2 = LoadFloat3(g_vertices, mat.vertexOffset + ci2);
                    float2 cuv0 = LoadFloat2(g_texcoords, mat.vertexOffset + ci0);
                    float2 cuv1 = LoadFloat2(g_texcoords, mat.vertexOffset + ci1);
                    float2 cuv2 = LoadFloat2(g_texcoords, mat.vertexOffset + ci2);
                    float3 edge1c = cp1 - cp0, edge2c = cp2 - cp0;
                    float2 dUV1c = cuv1 - cuv0, dUV2c = cuv2 - cuv0;
                    float detc = dUV1c.x * dUV2c.y - dUV2c.x * dUV1c.y;
                    if (abs(detc) > 1e-8)
                    {
                        float invDetc = 1.0 / detc;
                        // dP/dV = (-dUV1c.x * edge1c + dUV2c.x * edge2c) * invDetc
                        hairTangent = normalize((-dUV1c.x * edge1c + dUV2c.x * edge2c) * invDetc);
                    }
                    else
                    {
                        BuildONB(N, hairTangent, B);
                    }
                    // h = fiber offset across card width (UV.x in [0,1] → [-1,1])
                    g_hairH = payload.texU * 2.0 - 1.0;
                }
                // Orthogonalize tangent against surface normal
                T = normalize(hairTangent - N * dot(hairTangent, N));
                B = cross(N, T);
            }
            else
            {
                BuildONB(N, T, B);
                g_hairH = 0.0;
            }

            // Shading-normal terminator handling: when the perturbed N points
            // away from the viewer but Ng faces it, the BSDF sees wi_local.z<=0
            // and returns black. Snapping all the way to Ng wipes the normal
            // map and inflates GGX specular at grazing angles. Bend N just
            // enough to bring the view direction onto the front side instead.
            {
                float3 V = -ray.Direction;
                float NdotV = dot(N, V);
                if (NdotV <= 0.0 && dot(V, Ng) > 0.0)
                {
                    const float kFrontEps = 1e-3;
                    N = normalize(N + V * (kFrontEps - NdotV));
                    BuildONB(N, T, B);
                }
            }

            float3 wi_local = ToLocal(-ray.Direction, T, B, N);

            // For Delta BSDFs, mirror and dielectric have their own branches. Next event estimation is
            // skipped for them because f*cos/pdf with a Dirac-delta is undefined at non-exact directions.

            // Mirror (delta)
            if (mat.type == 1)
            {
                float3 reflDir = reflect(ray.Direction, N);
                ray.Origin = hitPos + Ng * 0.001;
                ray.Direction = reflDir;
                ray.TMin = 0.0;
                ray.TMax = 1e20;
                lastBsdfPdf = 0.0;
            }
            // Dielectric (delta)
            else if (mat.type == 2)
            {
                float3 I = ray.Direction;

                float3 Nf = N;
                float etaI, etaT;
                if (!hitBackFace)
                {
                    etaI = mat.extIOR;
                    etaT = mat.intIOR;
                }
                else
                {
                    etaI = mat.intIOR;
                    etaT = mat.extIOR;
                }

                float cosThetaI = dot(-I, Nf);
                float Fr = FresnelDielectric(cosThetaI, etaI, etaT);

                float3 newDir;
                bool refracted = false;
                float rngVal = NextFloat(rng);
                if (rngVal < Fr)
                    newDir = reflect(I, Nf);
                else
                {
                    newDir = refract(I, Nf, etaI / etaT);
                    if (dot(newDir, newDir) < 0.001)
                        newDir = reflect(I, Nf);
                    else
                        refracted = true;
                }

                float3 offsetN = (dot(newDir, Ng) > 0.0) ? Ng : -Ng;
                ray.Origin = hitPos + offsetN * 0.001;
                ray.Direction = newDir;
                ray.TMin = 0.0;
                ray.TMax = 1e20;
                lastBsdfPdf = 0.0;

                if (refracted)
                    eta *= etaI / etaT;
            }
            else if (mat.type == 0 || mat.type == 3 || mat.type == 4 || mat.type == 5)
            {
                Lo += throughput * MISDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, rng);
                Lo += throughput * EnvmapDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, rng);

                float3 wo_local;
                float bsdfPdf;
                float3 weight = MaterialSample(wi_local, rng, mat, wo_local, bsdfPdf);
                if (bsdfPdf <= 0.0 || all(weight == 0.0))
                    break;

                ray.Origin = OffsetRayOrigin(hitPos, Ng, N);
                ray.Direction = ToWorld(wo_local, T, B, N);
                ray.TMin = 0.0;
                ray.TMax = 1e20;

                throughput *= weight;
                lastBsdfPdf = max(bsdfPdf, 1e-20);
            }
            else
            {
                break;
            }

        } // end surface hit

        if (bounce >= 3)
        {
            float q = min(max(throughput.x, max(throughput.y, throughput.z)) * eta * eta, 0.95);
            if (NextFloat(rng) >= q)
                break;
            throughput /= q;
        }
    }

    if (any(isnan(Lo)) || any(isinf(Lo)))
        Lo = float3(0, 0, 0);

    float4 prev = (frameCount == 0) ? float4(0, 0, 0, 0) : g_accum[pixel];
    float4 accum = prev + float4(Lo, 1.0);
    g_accum[pixel] = accum;

    float3 averaged = accum.xyz / accum.w;
    averaged = averaged / (1.0 + averaged);
    averaged = pow(averaged, 1.0 / 2.2);
    g_output[pixel] = float4(averaged, 1.0);
}

    [shader("closesthit")] void ClosestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.hit = 1;
    payload.hitT = RayTCurrent();
    payload.materialID = InstanceID();

    // interpolated shading normal
    float3 N = GetInterpolatedNormal(InstanceID(), PrimitiveIndex(), attr.barycentrics);
    payload.normalX = N.x;
    payload.normalY = N.y;
    payload.normalZ = N.z;

    // Geometric normal
    float3 Ng = GetGeometricNormal(InstanceID(), PrimitiveIndex());
    payload.geoNormalX = Ng.x;
    payload.geoNormalY = Ng.y;
    payload.geoNormalZ = Ng.z;

    // Texture coordinates
    float2 texUV = GetInterpolatedUV(InstanceID(), PrimitiveIndex(), attr.barycentrics);
    payload.texU = texUV.x;
    payload.texV = texUV.y;

    payload.primitiveID = PrimitiveIndex();

    // Hair fiber data: tangent direction + h parameter.
    // Reference: [PBRT] Section 9.9.1, [Chiang] Eq. 4
    // tangent comes from the per-vertex tangent buffer (zero for non-hair).
    // h is encoded in UV.y by the tessellator as (h+1)/2, so h = 2*v - 1.
    float3 tang = GetInterpolatedTangent(InstanceID(), PrimitiveIndex(), attr.barycentrics);
    payload.tangentX = tang.x;
    payload.tangentY = tang.y;
    payload.tangentZ = tang.z;
    payload.hairH = texUV.y * 2.0 - 1.0;
}

[shader("miss")] void Miss(inout HitPayload payload)
{
    payload.hit = 0;
    float3 env = EvalEnvmap(WorldRayDirection());
    payload.envR = env.x;
    payload.envG = env.y;
    payload.envB = env.z;
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
    float a = g_textures[mat.alphaTexIndex].SampleLevel(g_sampler, aUV, 0).r;

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
        float4 t = g_textures[mat.alphaTexIndex].SampleLevel(g_sampler, aUV, 0);
        float a = (t.a < 1.0) ? t.a : t.r;

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