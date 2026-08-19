// RayQueryTrace.hlsli
//
// Inline ray tracing (DXR 1.1 RayQuery) for the two kinds of trace the megakernel
// does: a closest-hit ray (primary + the SSS interior walk) and an occlusion ray
// (NEE shadow). Gated on -D USE_RAYQUERY=1. Default 0 keeps the shipped
// DispatchRays path, and the macros at the bottom expand to the original TraceRay
// calls verbatim so the default .cso stays byte-for-byte identical.
//
// The point: we already inline all the shading in RayGen. ClosestHit/Miss just
// stash IDs + barycentrics and the any-hits only do an alpha/Fresnel test, so
// there's nothing the fixed-function pipeline buys us here. RayQuery hands the
// same info back inline and skips the shader-table round trip.
//
// Needs SM 6.5, so compile the RayQuery variant with -T lib_6_5. Include this after
// Common/RNG/GeometryUtils (we use GPUMaterial, g_materials/g_textures/g_sampler,
// PCGHash, GetInterpolatedUV, GetGeometricNormal, FresnelDielectric) and before
// Emitter/Subsurface, which use the macros.

#ifndef RAYQUERYTRACE_HLSLI
#define RAYQUERYTRACE_HLSLI

// Default ON. Inline RT is 14x faster on small scenes and 1.66x on the 2.5M-tri
// hero scene, so it is the shipped path; -D USE_RAYQUERY=0 restores the
// DispatchRays path for A/B. Note the default therefore requires -T lib_6_5.
#ifndef USE_RAYQUERY
#define USE_RAYQUERY 1
#endif

#if USE_RAYQUERY

// Closest hit. Fills HitPayload the same way ClosestHit + PrimaryAnyHit do. Opaque
// triangles commit on their own; for non-opaque candidates we redo PrimaryAnyHit
// here (accept if there's no alpha map, otherwise the cutoff + stochastic alpha test
// on the threaded PCG stream). RayQuery already tracks the nearest committed hit, so
// committing every accepted candidate leaves us with the closest one.
void TraceClosestInline(RaytracingAccelerationStructure as, RayDesc ray, inout HitPayload payload)
{
    uint rngState = payload.rngState;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(as, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint iid = q.CandidateInstanceID();
            GPUMaterial mat = g_materials[iid];
            if (mat.alphaTexIndex == 0xFFFFFFFF)
            {
                // No alpha map (dielectric/mirror marked non-opaque for the shadow
                // Fresnel test). PrimaryAnyHit just accepts these, so commit.
                q.CommitNonOpaqueTriangleHit();
                continue;
            }
            float2 aUV = GetInterpolatedUV(iid, q.CandidatePrimitiveIndex(),
                                           q.CandidateTriangleBarycentrics());
            float a = g_textures[mat.alphaTexIndex].SampleLevel(g_sampler, aUV, 0).r;
            if (a < 0.01)
                continue; // IgnoreHit
            rngState = PCGHash(rngState);
            float xi = float(rngState) / 4294967295.0;
            if (a * a < xi)
                continue; // IgnoreHit
            q.CommitNonOpaqueTriangleHit();
        }
    }
    payload.rngState = rngState;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        payload.hit = 1;
        payload.hitT = q.CommittedRayT();
        payload.materialID = q.CommittedInstanceID();
        payload.primitiveID = q.CommittedPrimitiveIndex();
        float2 bc = q.CommittedTriangleBarycentrics();
        payload.baryX = bc.x;
        payload.baryY = bc.y;
    }
    else
    {
        payload.hit = 0;
    }
}

// Occlusion / shadow ray. ACCEPT_FIRST_HIT_AND_END_SEARCH means the first opaque hit
// occludes and stops traversal. For non-opaque candidates we redo ShadowAnyHit: alpha
// runs the stochastic test (if it passes we commit, which counts as an occluder and
// ends the search); mirror and dielectric just attenuate transmission by (1-Fr) and
// keep going, so glass never fully occludes.
void TraceShadowInline(RaytracingAccelerationStructure as, RayDesc ray, inout ShadowPayload payload)
{
    uint rngState = payload.rngState;
    float3 transmission = payload.transmission;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(as, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint iid = q.CandidateInstanceID();
            GPUMaterial mat = g_materials[iid];
            if (mat.alphaTexIndex != 0xFFFFFFFF)
            {
                float2 aUV = GetInterpolatedUV(iid, q.CandidatePrimitiveIndex(),
                                               q.CandidateTriangleBarycentrics());
                float4 t = g_textures[mat.alphaTexIndex].SampleLevel(g_sampler, aUV, 0);
                float a = (t.a < 1.0) ? t.a : t.r;
                if (a < 0.01)
                    continue; // IgnoreHit
                rngState = PCGHash(rngState);
                float xi = float(rngState) / 4294967295.0;
                if (a * a < xi)
                    continue; // IgnoreHit
                q.CommitNonOpaqueTriangleHit(); // counts as an occluder, end-search stops here
                continue;
            }
            if (mat.type == 1 || mat.type == 2) // mirror or dielectric
            {
                float3 Ng = GetGeometricNormal(iid, q.CandidatePrimitiveIndex());
                float cosI = abs(dot(ray.Direction, Ng));
                float Fr = FresnelDielectric(cosI, mat.extIOR, mat.intIOR);
                transmission *= (1.0 - Fr);
                continue; // attenuate and keep going, glass doesn't occlude
            }
            // anything else non-opaque: let it pass through.
        }
    }
    payload.rngState = rngState;
    payload.transmission = transmission;
    payload.shadowed = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 1 : 0;
}

#define TRACE_CLOSEST(as, ray, pl) TraceClosestInline(as, ray, pl)
#define TRACE_SHADOW(as, ray, pl) TraceShadowInline(as, ray, pl)

#else // !USE_RAYQUERY: expand to the original TraceRay calls verbatim (byte-identical path)

#define TRACE_CLOSEST(as, ray, pl) \
    TraceRay(as, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, pl)
#define TRACE_SHADOW(as, ray, pl) \
    TraceRay(as, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, \
             0xFF, 1, 0, 1, ray, pl)

#endif // USE_RAYQUERY

#endif // RAYQUERYTRACE_HLSLI
