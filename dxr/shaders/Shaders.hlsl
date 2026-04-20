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

// Texture array and sampler for material textures
// Textures are bound starting at t10, indices stored in GPUMaterial
Texture2D g_textures[] : register(t10);
SamplerState g_sampler : register(s0);
SamplerState g_envmapSampler : register(s1);

static const float M_PI = 3.14159265358979323846;
static const float M_INV_PI = 0.31830988618379067154;
static const int MAX_BOUNCES = 8;

// remove later:
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
};
struct ShadowPayload
{
    uint shadowed;
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

// compute the UV-space footprint of a single pixel at the hit point, and returns the approximate UV-space length that one screen pixel covers.
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

// evaluate the equirectangular environment map along a world-space direction, here y = up. phi in [0, 2*pi), theta in [0, pi], u = phi / (2*pi), v = theta / pi
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

// binary search over a CDF stored in a ByteAddressBuffer, starting at byte offset. CDF has a number of entries of monotonically increasing floats
// with the last equal to 1.0.
// returns the largest i s.t. cdf[i] <= u, which is the index of the cell containing u
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

// MATERIAL DISPATCH LAYER

// TODO:
//  The integrator (consisting of raygen, MISDirectIllumination, EnvmapDirectIllumination)
//  calls these instead of hardcoding per-type BSDF logic. Adding a new BSDF, such as adding Disney type 4 means adding a case here but callers are unchanged.
//
//  For example delta BSDFs like mirror or dielectric are handled by their own raygen branches
//
//  TODO: IF TIME PROVIDES, if I add BSSRDF, I'd have to wrap all of this in a higher-level MaterialDirectLighting(hit, scene, rng)
//  which dispatches to point-sampling for BSSRDF materials and to this direction-sampling path for BRDF
//  materials. Anyways the integrator stays the same either way.
//  ============================================================================

bool MaterialIsDelta(GPUMaterial mat)
{
    return mat.type == 1 || mat.type == 2;
}

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
    for (uint i = 0; i < meshCount; i++)
    {
        if (g_materials[i].isEmitter)
        {
            es.emitterID = i;
            es.valid = true;
            break;
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
    es.pdfArea = 1.0 / eMat.surfaceArea;
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
    return (1.0 / emitMat.surfaceArea) * dist2 / cosL;
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
    TraceRay(g_scene, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 0, 0, 1, shadowRay, shadow);
    if (shadow.shadowed)
        return float3(0, 0, 0);

    float3 wo_local = ToLocal(wi_world, T, B, N);
    float3 f = MaterialEval(wi_local, wo_local, mat);
    float pdfBsdf = MaterialPdf(wi_local, wo_local, mat);

    float pdfEms = es.pdfArea * dist * dist / cosLight;
    float w = BalanceHeuristic(pdfEms, pdfBsdf);
    return es.radiance * f * cosTheta / max(pdfEms, 1e-20) * w;
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
    TraceRay(g_scene,
             RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 0, 0, 1, shadowRay, shadow);
    if (shadow.shadowed)
        return float3(0, 0, 0);

    // BSDF evaluation at the envmap-sampled direction
    float3 wo_local = ToLocal(wi_world, T, B, N);
    float3 f = MaterialEval(wi_local, wo_local, mat);
    float pdfBsdf = MaterialPdf(wi_local, wo_local, mat);

    float w = BalanceHeuristic(pdfEnv, pdfBsdf);
    return Lenv * f * cosTheta / max(pdfEnv, 1e-20) * w;
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

    RayDesc ray;
    ray.Origin = camPos;
    ray.Direction = normalize(
        camLowerLeftCorner + uv.x * camHorizontal + uv.y * camVertical - camPos);
    ray.TMin = 0.001;
    ray.TMax = 1e20;

    float3 Lo = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    float lastBsdfPdf = 0.0;
    float eta = 1.0;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        HitPayload payload;
        payload.hit = 0;
        TraceRay(g_scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
        if (!payload.hit)
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

        float3 hitPos = ray.Origin + ray.Direction * payload.hitT;
        float3 N = normalize(float3(payload.normalX, payload.normalY, payload.normalZ));
        float3 Ng = normalize(float3(payload.geoNormalX, payload.geoNormalY, payload.geoNormalZ));
        GPUMaterial mat = g_materials[payload.materialID];
        float2 hitUV = float2(payload.texU, payload.texV);

        if (dot(Ng, ray.Direction) > 0.0)
            Ng = -Ng;

        bool hasAnyTex = (mat.albedoTexIndex != 0xFFFFFFFF) ||
                         (mat.normalTexIndex != 0xFFFFFFFF) ||
                         (mat.roughnessTexIndex != 0xFFFFFFFF);
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

        float texAlpha = mat.alpha;
        if (mat.roughnessTexIndex != 0xFFFFFFFF)
        {
            float rLod = ComputeTexLOD(g_textures[mat.roughnessTexIndex], uvFoot);
            texAlpha = g_textures[mat.roughnessTexIndex].SampleLevel(g_sampler, hitUV, rLod).r;
        }

        mat.albedoR = texAlbedo.x;
        mat.albedoG = texAlbedo.y;
        mat.albedoB = texAlbedo.z;
        mat.alpha = texAlpha;
        mat.roughness = texAlpha;

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
        BuildONB(N, T, B);
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
            float3 Nf;
            float etaI, etaT;
            if (dot(I, N) < 0.0)
            {
                Nf = N;
                etaI = mat.extIOR;
                etaT = mat.intIOR;
            }
            else
            {
                Nf = -N;
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
        else if (mat.type == 0 || mat.type == 3 || mat.type == 4)
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

    // interpolated shadign normal
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
}

[shader("miss")] void Miss(inout HitPayload payload)
{
    payload.hit = 0;
    float3 env = EvalEnvmap(WorldRayDirection());
    payload.envR = env.x;
    payload.envG = env.y;
    payload.envB = env.z;
}
    [shader("miss")] void ShadowMiss(inout ShadowPayload payload) { payload.shadowed = 0; }
