// PathTrace.hlsli
//
// The path walker, shared by every driver that wants to trace one.
//
// Shaders.hlsl is compiled as a DXIL library because it carries the DXR entry
// points; a compute shader is a separate cs_6_5 compilation. So everything both
// need lives here: the path state, one bounce of a path, and the three pieces
// RayGen used to do inline -- open a pixel, build its primary ray, and write the
// result back.
//
// Include after Common / RNG / GeometryUtils / RayQueryTrace / the BSDF headers /
// Emitter / Envmap / Volume / Subsurface, all of which PathStep calls into.

#ifndef PATHTRACE_HLSLI
#define PATHTRACE_HLSLI

// Path state that has to survive from one bounce to the next. Everything else
// inside PathStep is scratch that dies with the iteration, which is the point:
// the whole path fits in registers and never spills to a global buffer the way a
// classical wavefront tracer's would.
struct PathState
{
    RayDesc ray;
    float3 throughput;
    float3 Lo;
    float3 aovAlbedo;
    float3 aovNormal;
    RNG rng;
    float lastBsdfPdf;
    float eta;
    int bounce;
    uint pathLen;
    bool aovDone;
    bool restirDone;
};

PathState InitPathState(RayDesc ray, RNG rng)
{
    PathState P;
    P.ray = ray;
    P.throughput = float3(1, 1, 1);
    P.Lo = float3(0, 0, 0);
    // Denoiser feature buffers. Captured at the FIRST non-delta interaction along
    // the path, where mirror/dielectric are skipped so the albedo / normal describe
    // the surface seen *through* the reflection/refraction.
    P.aovAlbedo = float3(0, 0, 0);
    P.aovNormal = float3(0, 0, 0);
    P.rng = rng;
    P.lastBsdfPdf = 0.0;
    P.eta = 1.0;
    P.bounce = 0;
    P.pathLen = 0;
    P.aovDone = false;
    // ReSTIR reuse happens once per path, at the first non-delta surface: it is a
    // screen-space technique and a pixel has only one primary shading point to
    // share with its neighbours. Deeper bounces fall back to plain RIS or NEE.
    P.restirDone = false;
    return P;
}

// One bounce of a path. Returns false when the path is finished, which is exactly
// what every `break` in the original bounce loop meant.
//
// This is the megakernel's loop body lifted verbatim. It is a function so that more
// than one driver can walk a path: RayGen calls it in a plain loop, one path per
// thread, and a persistent-thread driver can call it one bounce at a time across
// lanes holding paths at different depths -- which is what closes the path-length
// divergence measured by PROFILE_WAVES.
//
// The body is wrapped in a single-iteration loop so the original break statements
// compile unchanged; each is preceded by `alive = false`.
bool PathStep(inout PathState P, uint2 pixel, uint2 dims)
{
    // Unpack into the names the body already uses, so the body stays verbatim.
    RayDesc ray = P.ray;
    float3 throughput = P.throughput;
    float3 Lo = P.Lo;
    float3 aovAlbedo = P.aovAlbedo;
    float3 aovNormal = P.aovNormal;
    RNG rng = P.rng;
    float lastBsdfPdf = P.lastBsdfPdf;
    float eta = P.eta;
    int bounce = P.bounce;
    bool aovDone = P.aovDone;
    bool restirDone = P.restirDone;

    bool alive = true;
    [loop] for (uint _once = 0u; _once < 1u; _once++)
    {
        HitPayload payload;
        payload.hit = 0;
        payload.rngState = rng.state;
        TRACE_CLOSEST(g_scene, ray, payload);
        rng.state = payload.rngState;
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
                Lo += ClampContribution(throughput * VolumeNEEAreaLight(scatterPos,
                                                                        ray.Direction, scatterVolIdx, rng),
                                        bounce);
                Lo += ClampContribution(throughput * VolumeNEEEnvmap(scatterPos,
                                                                     ray.Direction, scatterVolIdx, rng),
                                        bounce);

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
            if (!aovDone)
            {
                aovAlbedo = env;
                aovNormal = float3(0, 0, 0);
                aovDone = true;
            }
            if (lastBsdfPdf == 0.0)
            {
                Lo += ClampContribution(throughput * env, bounce);
            }
            else
            {
                float pdfEnv = EnvmapPdfDirection(ray.Direction);
                float w = BalanceHeuristic(lastBsdfPdf, pdfEnv);
                Lo += ClampContribution(throughput * env * w, bounce);
            }
            { alive = false; break; }
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
                float lod = ComputeTexLOD(g_textures[NonUniformResourceIndex(mat.albedoTexIndex)], uvFoot);
                texAlbedo = g_textures[NonUniformResourceIndex(mat.albedoTexIndex)].SampleLevel(g_sampler, hitUV, lod).rgb;
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

                float nLod = ComputeTexLOD(g_textures[NonUniformResourceIndex(mat.normalTexIndex)], uvFoot);
                float3 tangentNormal = g_textures[NonUniformResourceIndex(mat.normalTexIndex)].SampleLevel(g_sampler, hitUV, nLod).xyz;
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
                float rLod = ComputeTexLOD(g_textures[NonUniformResourceIndex(mat.roughnessTexIndex)], uvFoot);
                float texRough = g_textures[NonUniformResourceIndex(mat.roughnessTexIndex)].SampleLevel(g_sampler, hitUV, rLod).r;
                mat.alpha = texRough;
                mat.roughness = texRough;
            }

            if (mat.specularTexIndex != 0xFFFFFFFF)
            {
                float sLod = ComputeTexLOD(g_textures[NonUniformResourceIndex(mat.specularTexIndex)], uvFoot);
                mat.specular = g_textures[NonUniformResourceIndex(mat.specularTexIndex)].SampleLevel(g_sampler, hitUV, sLod).r * 0.5;
            }
            if (mat.subsurfaceTexIndex != 0xFFFFFFFF)
            {
                float ssLod = ComputeTexLOD(g_textures[NonUniformResourceIndex(mat.subsurfaceTexIndex)], uvFoot);
                mat.subsurface = g_textures[NonUniformResourceIndex(mat.subsurfaceTexIndex)].SampleLevel(g_sampler, hitUV, ssLod).r;
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
                    Lo += ClampContribution(throughput * MatRadiance(mat), bounce);
                else
                {

                    float pdfEms = EmitterPdfSolidAngle(mat, hitPos, ray.Origin, N);
                    Lo += ClampContribution(
                        throughput * MatRadiance(mat) * BalanceHeuristic(lastBsdfPdf, pdfEms), bounce);
                }
                { alive = false; break; }
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
#ifndef USE_RIS
#define USE_RIS 0
#endif
#if USE_RIS
                // restirRadius > 0 turns on spatial reuse for the first
                // non-delta hit; everything after it, and every hit when reuse
                // is off, is plain RIS. Both paths write nothing the other
                // reads, so the toggle is a pure runtime switch.
                if (!restirDone && restirRadius > 0.0)
                {
                    restirDone = true;
                    Lo += ClampContribution(
                        throughput * ReSTIRDirectIllumination(pixel, dims, hitPos, N, Ng, T, B,
                                                              wi_local, mat, h, rng), bounce);
                }
                else
                {
                    Lo += ClampContribution(
                        throughput * RISDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, h, rng), bounce);
                }
#else
                Lo += ClampContribution(
                    throughput * MISDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, h, rng), bounce);
#endif
                Lo += ClampContribution(
                    throughput * EnvmapDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, h, rng), bounce);

                float3 wo_local;
                float bsdfPdf;
                float3 weight = MaterialSample(wi_local, rng, mat, h, wo_local, bsdfPdf);
                if (bsdfPdf <= 0.0 || all(weight == 0.0))
                    { alive = false; break; }

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
            // SSS
            else if (mat.type == 6)
            {
                // AOV: capture skin color/normal at the entry so OIDN has features.
                if (!aovDone)
                {
                    aovAlbedo = texAlbedo;
                    aovNormal = N;
                    aovDone = true;
                }
                float3 I = ray.Direction;

                mat.roughness *= lerp(1.0, 0.65, saturate(mat.specular * 2.0));
                float rough = mat.roughness;
                float aR = max(rough * rough, 1e-4);
                float3 wiL = ToLocal(-I, T, B, N); // .z = cos to N

                float3 wmL = (rough > 1e-3)
                                 ? BeckmannSample(float2(NextFloat(rng), NextFloat(rng)), aR)
                                 : float3(0, 0, 1);
                float3 wm = normalize(ToWorld(wmL, T, B, N));
                float cosThetaM = dot(-I, wm);
                if (cosThetaM <= 0.0) // back-facing microfacet at grazing, therefore use N
                {
                    wm = N;
                    wmL = float3(0, 0, 1);
                    cosThetaM = dot(-I, N);
                }
                float G1i = (rough > 1e-3) ? SmithG1(wiL, wmL, aR) : 1.0;
                float Fr = FresnelDielectric(cosThetaM, mat.extIOR, mat.intIOR);

                Lo += ClampContribution(
                    throughput * MISDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, h, rng), bounce);
                Lo += ClampContribution(
                    throughput * EnvmapDirectIllumination(hitPos, N, Ng, T, B, wi_local, mat, h, rng), bounce);

                if (NextFloat(rng) < Fr)
                {
                    float3 refl = reflect(I, wm);
                    float3 woL = ToLocal(refl, T, B, N);
                    if (woL.z <= 0.0)
                        { alive = false; break; } // reflected below the surface
                    float weight = (rough > 1e-3)
                                       ? G1i * SmithG1(woL, wmL, aR) * cosThetaM / max(wiL.z * wmL.z, 1e-6)
                                       : 1.0;
                    float pdfSpec = (rough > 1e-3)
                                        ? Fr * BeckmannDCosTheta(wmL, aR) / max(4.0 * cosThetaM, 1e-6)
                                        : 0.0;
                    ray.Origin = hitPos + Ng * 0.001;
                    ray.Direction = refl;
                    ray.TMin = 0.0;
                    ray.TMax = 1e20;
                    throughput *= weight;
                    lastBsdfPdf = pdfSpec;
                }
                else
                {
                    float3 refr = refract(I, wm, mat.extIOR / mat.intIOR);
                    if (dot(refr, refr) < 1e-8)
                    {
                        ray.Origin = hitPos + Ng * 0.001;
                        ray.Direction = reflect(I, wm);
                        ray.TMin = 0.0;
                        ray.TMax = 1e20;
                        lastBsdfPdf = 0.0;
                    }
                    else
                    {

                        float3 sssTint = float3(mat.sheen, mat.sheenTint, mat.clearcoat);
                        float3 aSharp = MatAlbedo(mat);
                        float3 aBlur = aSharp;
                        float3 sssDetail = float3(1, 1, 1);
                        if (mat.albedoTexIndex != 0xFFFFFFFF)
                        {
                            float blurLod = ComputeTexLOD(g_textures[NonUniformResourceIndex(mat.albedoTexIndex)], uvFoot) + 4.0;
                            aBlur = g_textures[NonUniformResourceIndex(mat.albedoTexIndex)].SampleLevel(g_sampler, hitUV, blurLod).rgb;
                            sssDetail = clamp(aSharp / max(aBlur, float3(1e-3, 1e-3, 1e-3)),
                                              float3(0.5, 0.5, 0.5), float3(2.0, 2.0, 2.0));
                        }

                        float sssSigmaT;
                        float3 sssAlpha;
                        SubsurfaceParams(aBlur * sssTint, mat.subsurface, sssSigmaT, sssAlpha);
                        float sssG = mat.anisotropic;

                        float3 exitPos, exitDir, exitN, tmul;
                        bool exited = SubsurfaceWalk(
                            hitPos - Ng * 0.001, normalize(refr), payload.materialID,
                            sssSigmaT, sssAlpha, sssG, mat.intIOR, mat.extIOR,
                            rng, exitPos, exitDir, exitN, tmul);

                        if (!exited)
                            { alive = false; break; }
                        throughput *= tmul * sssDetail;
                        ray.Origin = OffsetRayOrigin(exitPos, exitN, exitN);
                        ray.Direction = exitDir;
                        ray.TMin = 0.0;
                        ray.TMax = 1e20;
                        lastBsdfPdf = 0.0;
                    }
                }
            }
            else
            {
                { alive = false; break; }
            }

        } // end surface hit

        if (bounce >= 3)
        {
            float q = min(max(throughput.x, max(throughput.y, throughput.z)) * eta * eta, 0.95);
            if (NextFloat(rng) >= q)
                { alive = false; break; }
            throughput /= q;
        }
    }

    P.ray = ray;
    P.throughput = throughput;
    P.Lo = Lo;
    P.aovAlbedo = aovAlbedo;
    P.aovNormal = aovNormal;
    P.rng = rng;
    P.lastBsdfPdf = lastBsdfPdf;
    P.eta = eta;
    P.aovDone = aovDone;
    P.restirDone = restirDone;
    return alive;
}

// Camera ray for a pixel, jitter and thin-lens aperture included. Draws exactly
// the same RNG dimensions in the same order as the original inline code, which
// is what keeps a driver swap from changing the image.
RayDesc GeneratePrimaryRay(uint2 pixel, uint2 dims, inout RNG rng)
{
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
    return ray;
}

// Open a pixel: read its history, decide whether it still needs samples, seed the
// sampler from its own sample count, and stamp its reservoir slot. Returns false
// when the pixel is converged and should be skipped entirely.
bool BeginPixel(uint2 pixel, uint2 dims, out RNG rng,
                out float4 accumPrev, out float2 momentsPrev)
{
    // With adaptive sampling on, pixels stop at different frames, so accum.w is
    // no longer the same number as frameCount.
    accumPrev = (frameCount == 0) ? float4(0, 0, 0, 0) : g_accum[pixel];
    momentsPrev = (frameCount == 0) ? float2(0, 0) : g_moments[pixel];
    rng = (RNG)0;

    // Converged pixels cost nothing: no rays, no accumulator write, no AOV write.
    // Their sample count stays where it stopped and the resolve pass divides by
    // it, so a partially converged image is still a correct mean everywhere --
    // just computed from a different number of samples per pixel.
    if (PixelConverged(momentsPrev, accumPrev.w))
        return false;

    // The sampler is indexed by the pixel's own sample count, not by frameCount.
    // Owen-scrambled Sobol' is only stratified over a contiguous prefix of its
    // sequence, so a pixel that sat out some frames must still walk 0, 1, 2, ...
    // of its own rather than inherit the global frame number and sample a sparse
    // subset. With adaptive sampling off the two numbers are identical.
    rng = InitRNG(pixel, dims, (uint)accumPrev.w);

    // Stamp this pixel's reservoir slot invalid up front. A path that never
    // reaches a non-delta surface writes no reservoir, and without this its slot
    // would still hold the entry from two frames ago -- which after a camera move
    // describes a shading point that no longer exists.
    if (restirRadius > 0.0)
        g_reservoirs[ReservoirIndex(pixel, dims, frameCount & 1u)].valid = 0.0;
    return true;
}

// Retire a finished path into the accumulator and the denoiser AOVs.
void WritePathResult(uint2 pixel, PathState P, float4 accumPrev, float2 momentsPrev)
{
    float3 Lo = P.Lo;
    if (any(isnan(Lo)) || any(isinf(Lo)))
        Lo = float3(0, 0, 0);

#if PROFILE_WAVES
    // R = path length, G = wave utilisation, B = lanes that reached the end.
    // WaveActiveSum/Max count only lanes still active here, which is the right
    // denominator: a lane that returned early never entered the loop and should
    // not be charged for it.
    uint waveSum = WaveActiveSum(P.pathLen);
    uint waveMax = WaveActiveMax(P.pathLen);
    uint waveLanes = WaveActiveCountBits(true);
    float util = (waveMax > 0u)
                     ? float(waveSum) / (float(WaveGetLaneCount()) * float(waveMax))
                     : 0.0;
    g_accum[pixel] = accumPrev + float4(float(P.pathLen), util, float(waveLanes), 1.0);
#else
    g_accum[pixel] = accumPrev + float4(Lo, 1.0);
#endif
    float4 prevA = (frameCount == 0) ? float4(0, 0, 0, 0) : g_albedo[pixel];
    g_albedo[pixel] = prevA + float4(P.aovAlbedo, 1.0);
    float4 prevN = (frameCount == 0) ? float4(0, 0, 0, 0) : g_normal[pixel];
    g_normal[pixel] = prevN + float4(P.aovNormal, 1.0);

    // Moments track the clamped radiance that actually entered the accumulator,
    // so the variance estimate describes the image being formed rather than an
    // unclamped one nobody sees.
    float lum = dot(Lo, float3(0.2126, 0.7152, 0.0722));
    g_moments[pixel] = momentsPrev + float2(lum, lum * lum);
}

// Walk one path to completion. The megakernel driver: one thread, one path,
// every bounce back to back.
PathState TracePath(uint2 pixel, uint2 dims, RayDesc ray, RNG rng)
{
    PathState P = InitPathState(ray, rng);
    [loop] for (int b = 0; b < MAX_BOUNCES; b++)
    {
        P.bounce = b;
        P.pathLen++;
        if (!PathStep(P, pixel, dims))
            break;
    }
    return P;
}

#endif // PATHTRACE_HLSLI
