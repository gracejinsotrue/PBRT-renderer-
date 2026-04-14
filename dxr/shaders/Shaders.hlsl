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
};

float3 MatAlbedo(GPUMaterial m) { return float3(m.albedoR, m.albedoG, m.albedoB); }
float3 MatRadiance(GPUMaterial m) { return float3(m.radianceR, m.radianceG, m.radianceB); }

StructuredBuffer<GPUMaterial> g_materials : register(t1);
ByteAddressBuffer g_normals : register(t2);
ByteAddressBuffer g_indices : register(t3);
ByteAddressBuffer g_vertices : register(t4);
ByteAddressBuffer g_emitterCdf : register(t5);

static const float M_PI = 3.14159265358979323846;
static const float M_INV_PI = 0.31830988618379067154;
static const int MAX_BOUNCES = 8;

struct HitPayload
{
    float hitT;
    float normalX, normalY, normalZ;
    float geoNormalX, geoNormalY, geoNormalZ; // geometric normal
    uint materialID;
    uint hit;
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

float3 MISDirectIllumination(float3 hitPos, float3 N, float3 T, float3 B,
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

    RayDesc shadowRay;
    shadowRay.Origin = hitPos;
    shadowRay.Direction = wi_world;
    shadowRay.TMin = 0.001;
    shadowRay.TMax = dist - 0.001;
    ShadowPayload shadow;
    shadow.shadowed = 1;
    TraceRay(g_scene, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 0, 0, 1, shadowRay, shadow);
    if (shadow.shadowed)
        return float3(0, 0, 0);

    float3 wo_local = ToLocal(wi_world, T, B, N);
    float3 f;
    float pdfBsdf;
    if (mat.type == 0)
    {
        f = MatAlbedo(mat) * M_INV_PI;
        pdfBsdf = cosTheta * M_INV_PI;
    }
    else
    {
        f = MicrofacetEval(wi_local, wo_local, mat);
        pdfBsdf = MicrofacetPdf(wi_local, wo_local, mat);
    }

    float pdfEms = es.pdfArea * dist * dist / cosLight;
    float w = BalanceHeuristic(pdfEms, pdfBsdf);
    return es.radiance * f * cosTheta / max(pdfEms, 1e-20) * w;
}

// ============================================================
// Ray Generation
// ============================================================

[shader("raygeneration")] void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims = DispatchRaysDimensions().xy;
    RNG rng = InitRNG(pixel, dims, frameCount);

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

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        HitPayload payload;
        payload.hit = 0;
        TraceRay(g_scene, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
        if (!payload.hit)
            break;

        float3 hitPos = ray.Origin + ray.Direction * payload.hitT;
        float3 N = normalize(float3(payload.normalX, payload.normalY, payload.normalZ));
        float3 Ng = normalize(float3(payload.geoNormalX, payload.geoNormalY, payload.geoNormalZ));
        GPUMaterial mat = g_materials[payload.materialID];

        // Ensure geometric normal faces the incoming ray
        if (dot(Ng, ray.Direction) > 0.0)
            Ng = -Ng;

        // if (mat.type == 1)
        // { // mirror only
        //     Lo = N * 0.5 + 0.5;
        //     break;
        // }

        if (mat.isEmitter)
        {
            if (lastBsdfPdf == 0.0)
                Lo += throughput * MatRadiance(mat);
            else
            {
                // float pdfEms = EmitterPdfSolidAngle(mat, hitPos, ray.Origin, N);
                float pdfEms = EmitterPdfSolidAngle(mat, hitPos, ray.Origin, N);
                Lo += throughput * MatRadiance(mat) * BalanceHeuristic(lastBsdfPdf, pdfEms);
            }
            break;
        }

        float3 T, B;
        BuildONB(N, T, B);
        float3 wi_local = ToLocal(-ray.Direction, T, B, N);

        // --- Diffuse (type 0) ---
        if (mat.type == 0)
        {
            Lo += throughput * MISDirectIllumination(hitPos, N, T, B, wi_local, mat, rng);

            float2 u = float2(NextFloat(rng), NextFloat(rng));
            float3 wo_local = CosineSampleHemisphere(u);

            ray.Origin = hitPos;
            ray.Direction = ToWorld(wo_local, T, B, N);
            ray.TMin = 0.001;
            ray.TMax = 1e20;

            throughput *= MatAlbedo(mat);
            lastBsdfPdf = max(wo_local.z * M_INV_PI, 1e-20);
        }
        // --- Mirror (type 1) ---
        else if (mat.type == 1)
        {

            float3 reflDir = reflect(ray.Direction, N);
            ray.Origin = hitPos + Ng * 0.001;
            // ray.Origin = hitPos + N * 0.001;
            ray.Direction = reflDir;
            ray.TMin = 0.0;
            ray.TMax = 1e20;
            lastBsdfPdf = 0.0;
        }
        // --- Dielectric (type 2) ---
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
            if (NextFloat(rng) < Fr)
                newDir = reflect(I, Nf);
            else
            {
                newDir = refract(I, Nf, etaI / etaT);
                if (dot(newDir, newDir) < 0.001)
                    newDir = reflect(I, Nf);
            }

            float3 offsetN = (dot(newDir, Ng) > 0.0) ? Ng : -Ng;
            ray.Origin = hitPos + offsetN * 0.001;
            ray.Direction = newDir;
            ray.TMin = 0.0;
            ray.TMax = 1e20;
            lastBsdfPdf = 0.0;
        }
        // --- Microfacet (type 3) ---
        else if (mat.type == 3)
        {
            Lo += throughput * MISDirectIllumination(hitPos, N, T, B, wi_local, mat, rng);

            float3 wo_local;
            float bsdfPdf;
            float3 weight = MicrofacetSample(wi_local, rng, mat, wo_local, bsdfPdf);
            if (bsdfPdf <= 0.0 || all(weight == 0.0))
                break;

            ray.Origin = hitPos;
            ray.Direction = ToWorld(wo_local, T, B, N);
            ray.TMin = 0.001;
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
            float q = min(max(throughput.x, max(throughput.y, throughput.z)), 0.95);
            if (NextFloat(rng) >= q)
                break;
            throughput /= q;
        }
    }

    if (any(isnan(Lo)) || any(isinf(Lo)))
        Lo = float3(0, 0, 0);

    float4 prev = g_accum[pixel];
    float4 accum = prev + float4(Lo, 1.0);
    g_accum[pixel] = accum;

    float3 averaged = accum.xyz / accum.w;
    averaged = averaged / (1.0 + averaged);
    averaged = pow(averaged, 1.0 / 2.2);
    g_output[pixel] = float4(averaged, 1.0);
}

    // ============================================================
    // Closest Hit — passes both shading and geometric normals
    // ============================================================

    [shader("closesthit")] void ClosestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.hit = 1;
    payload.hitT = RayTCurrent();
    payload.materialID = InstanceID();

    // Shading normal (interpolated)
    float3 N = GetInterpolatedNormal(InstanceID(), PrimitiveIndex(), attr.barycentrics);
    payload.normalX = N.x;
    payload.normalY = N.y;
    payload.normalZ = N.z;

    // Geometric normal
    float3 Ng = GetGeometricNormal(InstanceID(), PrimitiveIndex());
    payload.geoNormalX = Ng.x;
    payload.geoNormalY = Ng.y;
    payload.geoNormalZ = Ng.z;
}

[shader("miss")] void Miss(inout HitPayload payload)
{ payload.hit = 0; }
    [shader("miss")] void ShadowMiss(inout ShadowPayload payload) { payload.shadowed = 0; }
