// nori-dxr Shaders.hlsl
//
// DXR entry-point translation unit. All shared declarations and the BSDF /
// emitter / volume math live in the modular .hlsli headers included below; this
// file is the only thing dxc compiles and contains just the shader entry points
// (RayGen, ClosestHit, Miss, PrimaryAnyHit, ShadowAnyHit, ShadowMiss).

#include "Common.hlsli"
#include "RNG.hlsli"
#include "GeometryUtils.hlsli"
#include "Microfacet.hlsli"
#include "Disney.hlsli"
#include "Hair.hlsli"
#include "Material.hlsli"
#include "Emitter.hlsli"
#include "Envmap.hlsli"
#include "Volume.hlsl"

// (debug) When 1, RayGen replaces path tracing with a Monte Carlo estimate
// of the envmap integral to validate the importance sampler. 0 = normal.
#define ENVMAP_DEBUG_SAMPLER 0

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

    // Denoiser feature buffers. Captured at the FIRST non-delta
    // interaction along the path, whre mirror/dielectric are skipped so the albedo /
    // normal describe the surface seen *through* the reflection/refraction. (This is juts because a
    // flat first-hit GBuffer on the specular spheres would make OIDN oversmooth
    // them)
    bool aovDone = false;
    float3 aovAlbedo = float3(0, 0, 0);
    float3 aovNormal = float3(0, 0, 0);

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
#if HAS_VOLUME
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
#endif // HAS_VOLUME

        // Branch: volume scatter / miss / surface hit
        if (volumeScattered)
        {
            // Medium scatter — skip surface handling, proceed to
            // Russian roulette at the bottom of the loop.
        }
        else if (!payload.hit)
        {
            float3 env = EvalEnvmap(ray.Direction);
            // Clamp to suppress fireflies from bright HDRI sun disks on high-variance paths.
            float envLum = dot(env, float3(0.2126, 0.7152, 0.0722));
            if (envLum > kFireflyClamp)
                env *= kFireflyClamp / envLum;
            if (!aovDone)
            {
                aovAlbedo = env;
                aovNormal = float3(0, 0, 0);
                aovDone = true;
            }
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
            float2 bary = float2(payload.baryX, payload.baryY);
            float3 N = normalize(GetInterpolatedNormal(payload.materialID, payload.primitiveID, bary));
            float3 Ng = normalize(GetGeometricNormal(payload.materialID, payload.primitiveID));
            GPUMaterial mat = g_materials[payload.materialID];
            float2 hitUV = GetInterpolatedUV(payload.materialID, payload.primitiveID, bary);

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

                if (!aovDone)
                {
                    aovAlbedo = float3(1, 1, 1);
                    aovNormal = N;
                    aovDone = true;
                }
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
            float h = 0.0; // hair fiber offset in [-1,1]
#if HAS_HAIR
            if (mat.type == 5)
            {
                // Hair: build frame with fiber tangent as x-axis, tube surface normal as z.
                // Reference: [PBRT] Section 9.9.1 — sinθ = ω.x (tangent component)
                float3 hairTangent = GetInterpolatedTangent(payload.materialID, payload.primitiveID, bary);
                bool hasStrandTangent = dot(hairTangent, hairTangent) > 1e-8;
                if (hasStrandTangent)
                {
                    // Strand geometry from tessellator — tangent buffer is valid.
                    // h was encoded by tessellator as UV.y = (h+1)/2.
                    hairTangent = normalize(hairTangent);
                    h = hitUV.y * 2.0 - 1.0; // matches former payload.hairH (texV*2-1)
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
                    h = hitUV.x * 2.0 - 1.0;
                }
                // Orthogonalize tangent against surface normal
                T = normalize(hairTangent - N * dot(hairTangent, N));
                B = cross(N, T);
            }
            else
            {
                BuildONB(N, T, B);
                h = 0.0;
            }
#else
            BuildONB(N, T, B);
#endif // HAS_HAIR

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
                if (!aovDone)
                {
                    aovAlbedo = texAlbedo;
                    aovNormal = N;
                    aovDone = true;
                }
                Lo += throughput * MISDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, h, rng);
                Lo += throughput * EnvmapDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, h, rng);

                float3 wo_local;
                float bsdfPdf;
                float3 weight = MaterialSample(wi_local, rng, mat, h, wo_local, bsdfPdf);
                if (bsdfPdf <= 0.0 || all(weight == 0.0))
                    break;

                float3 wo_world = ToWorld(wo_local, T, B, N);
                // For hair, TT/TRT lobes scatter *through* the fiber (dot(wo,Ng)<0).
                // Offsetting toward +Ng would put the origin on the wrong side and
                // cause immediate self-intersection. Always offset toward the outgoing side.
                float3 offsetNg = (mat.type == 5)
                                      ? (dot(wo_world, Ng) >= 0.0 ? Ng : -Ng)
                                      : Ng;
                ray.Origin = OffsetRayOrigin(hitPos, offsetNg, offsetNg);
                ray.Direction = wo_world;
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
    float4 prevA = (frameCount == 0) ? float4(0, 0, 0, 0) : g_albedo[pixel];
    g_albedo[pixel] = prevA + float4(aovAlbedo, 1.0);
    float4 prevN = (frameCount == 0) ? float4(0, 0, 0, 0) : g_normal[pixel];
    g_normal[pixel] = prevN + float4(aovNormal, 1.0);

    float3 averaged = accum.xyz / accum.w;
    averaged = averaged * pow(2.0, evCompensation); // exposure compensation in EV stops (0 = no change)

    // ACES filmic tonemapper (Hill 2016 approximation).
    // Preserves saturation and contrast in midtones better than Reinhard.
    // Input is assumed to be in scene-linear AP1-ish space.
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    averaged = saturate((averaged * (a * averaged + b)) / (averaged * (c * averaged + d) + e));

    averaged = pow(averaged, 1.0 / 2.2); // gamma 2.2
    g_output[pixel] = float4(averaged, 1.0);
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