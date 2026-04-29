// Shaders.hlsl — DXR shader entry points (RayGen, ClosestHit, Miss).
// All BSDF, emitter, envmap, and volume logic lives in separate .hlsli
// files; this file is purely the integrator and hit/miss shaders.

#include "Common.hlsli"
#include "RNG.hlsli"
#include "GeometryUtils.hlsli"
#include "Envmap.hlsli"
#include "Volume.hlsl"
#include "Material.hlsli"
#include "Emitter.hlsli"

// When set to 1, RayGen replaces normal path tracing with a Monte Carlo
// estimator of the total envmap integral for sampler verification.
// Set to 0 once verified.
#define ENVMAP_DEBUG_SAMPLER 0

// ============================================================================
// Ray Generation — MIS path tracer
// ============================================================================

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

        // ── Volume free-flight check ──────────────────────────────────
        // [Marschner] §3, §5: before handling the surface hit, check
        // whether the ray scatters inside a participating medium first.
        bool volumeScattered = false;
        if (volumeEnabled)
        {
            float sigmaT = volumeSigmaA + volumeSigmaS;
            float3 vMin = float3(volumeMinX, volumeMinY, volumeMinZ);
            float3 vMax = float3(volumeMaxX, volumeMaxY, volumeMaxZ);
            float tNear, tFar;
            if (sigmaT > 1e-8 &&
                RayAABBIntersect(ray.Origin, ray.Direction, vMin, vMax, tNear, tFar))
            {
                tNear = max(tNear, 0.0);
                float tSurface = payload.hit ? payload.hitT : 1e20;
                tFar = min(tFar, tSurface);

                if (tFar > tNear)
                {
                    float tScatter;
                    bool scattered;

                    if (volumeHeterogeneous)
                    {
                        // [Marschner] §4.2: delta tracking
                        scattered = DeltaTracking(ray.Origin, ray.Direction,
                                                  tNear, tFar, sigmaT,
                                                  rng, tScatter);
                    }
                    else
                    {
                        // [Marschner] §3: homogeneous free-flight
                        float s = SampleFreeFlightHomogeneous(sigmaT, rng);
                        scattered = (s < (tFar - tNear));
                        tScatter = tNear + s;
                    }

                    if (scattered)
                    {
                        volumeScattered = true;
                        float3 scatterPos = ray.Origin + tScatter * ray.Direction;

                        // Throughput weight: σ_s / σ_t (single-scattering albedo)
                        throughput *= volumeSigmaS / sigmaT;

                        // [Marschner] §5 "Direct lighting for volumes": NEE
                        Lo += throughput * VolumeNEEAreaLight(scatterPos,
                                                              ray.Direction, volumePhaseG, rng);
                        Lo += throughput * VolumeNEEEnvmap(scatterPos,
                                                           ray.Direction, volumePhaseG, rng);

                        // [Marschner] §5: importance-sample phase function
                        float phasePdf;
                        float3 newDir = SampleHG(ray.Direction, volumePhaseG,
                                                 rng, phasePdf);

                        ray.Origin = scatterPos;
                        ray.Direction = newDir;
                        ray.TMin = 0.0;
                        ray.TMax = 1e20;

                        lastBsdfPdf = max(phasePdf, 1e-20);
                    }
                }
            }
        }

        // ── Branch: volume scatter / miss / surface hit ───────────────
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
            // ── Surface hit ───────────────────────────────────────────

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

            // Normal map: compute tangent frame from triangle UV derivatives
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
            // For hair (type 5), mat.roughness = β_M and mat.alpha = cuticle tilt.
            // These must NOT be overwritten by the texture roughness path.
            if (mat.type != 5)
            {
                mat.alpha = texAlpha;
                mat.roughness = texAlpha;
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

            // Build shading frame + hair fiber offset
            float3 T, B;
            float hairH = 0.0;
            if (mat.type == 5)
            {
                // Hair: build frame with fiber tangent as x-axis, tube surface normal as z.
                // Reference: [PBRT] Section 9.9.1 — sinθ = ω.x (tangent component)
                float3 hairTangent = normalize(float3(payload.tangentX, payload.tangentY, payload.tangentZ));
                // Orthogonalize tangent against surface normal
                T = normalize(hairTangent - N * dot(hairTangent, N));
                B = cross(N, T);
                hairH = payload.hairH;
            }
            else
            {
                BuildONB(N, T, B);
            }
            float3 wi_local = ToLocal(-ray.Direction, T, B, N);

            // ── Delta BSDFs (mirror, dielectric) ──────────────────────
            // NEE is skipped: f*cos/pdf with a Dirac-delta is undefined
            // at non-exact directions.

            if (mat.type == 1) // Mirror
            {
                float3 reflDir = reflect(ray.Direction, N);
                ray.Origin = hitPos + Ng * 0.001;
                ray.Direction = reflDir;
                ray.TMin = 0.0;
                ray.TMax = 1e20;
                lastBsdfPdf = 0.0;
            }
            else if (mat.type == 2) // Dielectric
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
                {
                    eta *= etaI / etaT;
                    // apply transmission tint via beer law approximation
                    throughput *= float3(mat.albedoR, mat.albedoG, mat.albedoB);
                }
            }
            // ── Non-delta BSDFs (diffuse, microfacet, Disney, hair) ───
            else if (mat.type == 0 || mat.type == 3 || mat.type == 4 || mat.type == 5)
            {
                Lo += throughput * MISDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, hairH, rng);
                Lo += throughput * EnvmapDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, hairH, rng);

                float3 wo_local;
                float bsdfPdf;
                float3 weight = MaterialSample(wi_local, rng, mat, hairH, wo_local, bsdfPdf);
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

        // Russian roulette
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

    // ============================================================================
    // Closest-hit shader
    // ============================================================================

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

// ============================================================================
// Miss shaders
// ============================================================================

[shader("miss")] void Miss(inout HitPayload payload)
{
    payload.hit = 0;
    float3 env = EvalEnvmap(WorldRayDirection());
    payload.envR = env.x;
    payload.envG = env.y;
    payload.envB = env.z;
}

    [shader("miss")] void ShadowMiss(inout ShadowPayload payload)
{
    payload.shadowed = 0;
}
