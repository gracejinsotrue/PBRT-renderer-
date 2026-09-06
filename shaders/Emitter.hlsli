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

    // Power-weighted emitter selection. Binary-search the CDF at the head of
    // g_emitterCdf, then jump straight to the mesh through the index table that
    // follows it (layout documented in DXRApp_Scene.cpp). This used to pick
    // uniformly and then linear-scan every mesh in the scene to find the k-th
    // emitter - O(meshCount) per light sample, and RIS draws several of those per
    // shading point. Now O(log emitterCount) with no scan, and bright lights get
    // sampled in proportion to their power.
    uint k = min(CdfSample(0, emitterCount + 1, NextFloat(rng)), emitterCount - 1);
    es.emitterID = g_emitterCdf.Load((emitterCount + 1 + k) * 4);
    es.valid = true;

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
    es.pdfArea = eMat.emitterSelectionProb / eMat.surfaceArea;
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

    return (emitMat.emitterSelectionProb / emitMat.surfaceArea) * dist2 / cosL;
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
    TRACE_SHADOW(g_scene, shadowRay, shadow);
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

// Fold the initial visibility test into the stored reservoir (see the long note
// at the kill site). 1 = the useful-but-biased variant, 0 = unbiased reuse of
// effective M only. Exists so the bias can be measured rather than argued about.
#ifndef RESTIR_VISIBILITY_REUSE
#define RESTIR_VISIBILITY_REUSE 1
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
    TRACE_SHADOW(g_scene, shadowRay, shadow);
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

// ---------------------------------------------------------------------------
// ReSTIR DI: spatial reuse on top of the RIS reservoir above.
//
// What this adds over RISDirectIllumination
// -----------------------------------------
// RIS alone draws RIS_M candidates per shading point and keeps one. Its quality
// is bounded by how large M can be made, and M is bought with candidate
// evaluations. Reuse buys effective M for free by borrowing the reservoirs
// neighbouring pixels already built -- and, more importantly, borrows samples
// that have already been *shadow tested*. That second part is the reason the
// plain-RIS measurement stalled: the target function p-hat is unshadowed, so RIS
// keeps picking bright occluded lights and the survivor carries a large W. No
// amount of extra M fixes a target function that is wrong; a visibility-tested
// sample from a neighbour does.
//
// Which variant this is
// ---------------------
// Neighbours are read from the *previous frame's* slice, so this is one dispatch
// with no barrier rather than the textbook build-then-reuse pass pair. That is a
// legitimate ReSTIR arrangement (a real-time implementation's spatial pass reads
// the previous spatiotemporal output), and here it is strictly the cheaper one:
// a same-frame split would have to re-trace the primary ray in the second pass,
// paying an extra primary hit per pixel per frame to buy nothing this does not
// already have -- the camera is static during accumulation, so a neighbour's
// frame-k-1 reservoir describes exactly the same surface point its frame-k one
// would.
//
// What is deliberately *not* done: the reservoir written out is the initial,
// pre-reuse one. Writing the combined reservoir back would compound reuse across
// frames and grow effective M without bound, but it also compounds the
// correlation between frames, and this renderer averages thousands of frames
// into one image. Bounded reuse, bounded correlation.
//
// Bias
// ----
// The combination uses the unbiased normalisation (Bitterli et al. 2020,
// "Spatiotemporal reservoir resampling", Algorithm 6): divide by Z, the number
// of candidates drawn from reservoirs whose domain actually contains the chosen
// sample, not by the raw total M. Dividing by M is the cheaper, biased variant
// and it darkens exactly where geometry disagrees, which is where the eye looks.
//
// The domain test below is geometric -- surface faces light, light faces surface
// -- rather than a full BSDF evaluation at the neighbour. For a diffuse surface
// those coincide exactly, since f > 0 wherever the cosines are positive, and the
// many-light test scene is diffuse by construction. For a glossy neighbour a
// direction with f == 0 but positive cosines would be counted, inflating Z and
// darkening slightly. Doing it exactly would mean storing each neighbour's full
// shading frame and material; the honest trade is recorded here rather than
// hidden.

// Rebuild the EmitterSample a stored reservoir represents, so the same
// EvalLightCandidate the initial pass uses can score it at a new shading point.
EmitterSample ReservoirSample(GPUReservoir r)
{
    EmitterSample es;
    es.position = r.lightPos;
    es.normal = r.lightNormal;
    es.radiance = r.radiance;
    es.pdfArea = r.pdfArea;
    es.emitterID = 0u;
    // W > 0 is what says the stored sample fields are meaningful: a reservoir
    // whose RIS loop found no surviving candidate still has valid == 1 (its M
    // draws count toward Z) but its sample fields are unset.
    es.valid = (r.valid > 0.0) && (r.M > 0.0) && (r.W > 0.0);
    return es;
}

// Could shading point (pos, N) have produced this light sample at all? Used for
// the Z normalisation, so a false positive darkens and a false negative
// brightens; see the bias note above.
//
// p-hat = lum(f * Le * cos), so it vanishes when any of three things does, and
// all three have to be tested or Z over-counts:
//   - the surface faces away from the light   (outgoing direction below N)
//   - the light faces away from the surface
//   - the BSDF is zero for that pair of directions. For a diffuse surface that
//     means the *view* direction is also below N, which is the condition a
//     purely light-facing test silently misses. The view direction is
//     reconstructed from camPos rather than stored, which is exact for a
//     reservoir built at a primary hit and approximate for one built after a
//     specular bounce.
bool SampleInDomain(float3 lightPos, float3 lightNormal, float3 pos, float3 N)
{
    float3 d = lightPos - pos;
    float dist2 = dot(d, d);
    if (dist2 < 1e-12)
        return false;
    float3 w = d * rsqrt(dist2);
    if (dot(N, w) <= 0.0 || dot(lightNormal, -w) <= 1e-8)
        return false;
    float3 v = camPos - pos;
    float v2 = dot(v, v);
    if (v2 < 1e-12)
        return false;
    return dot(N, v * rsqrt(v2)) > 0.0;
}

// Deterministic per-frame neighbour offset.
//
// It has to be deterministic because the Z pass walks the same neighbours a
// second time and must land on the same ones, and it has to be re-jittered every
// frame off frameCount because the camera is static: a fixed offset pattern
// would make every frame reuse the same neighbour, and the structured blotches
// that produces would survive into the accumulated average instead of averaging
// out.
int2 NeighbourOffset(uint2 pixel, uint i, float radius)
{
    uint seed = PCGHash(pixel.x * 73856093u ^ pixel.y * 19349663u ^
                        (frameCount * 83492791u + i * 2654435761u));
    float angle = float(seed & 0xFFFFu) * (1.0 / 65536.0) * 2.0 * M_PI;
    // sqrt() for a uniform density over the disk rather than a centre-heavy one
    float rad = radius * sqrt(float((seed >> 16) & 0xFFFFu) * (1.0 / 65536.0));
    return int2(int(round(cos(angle) * rad)), int(round(sin(angle) * rad)));
}

// A neighbour is only worth borrowing from if it is looking at roughly the same
// surface. Without this, reuse leaks light across silhouettes and corners.
bool NeighbourCompatible(GPUReservoir n, float3 hitPos, float3 N)
{
    if (n.valid <= 0.0)
        return false;
    if (dot(n.hitNormal, N) < 0.906) // ~25 degrees
        return false;
    // Depth similarity, measured along the view ray rather than as a raw
    // distance between the two points: neighbouring pixels on a plane seen at a
    // grazing angle are far apart in world space but are still the same surface.
    float dSelf = length(hitPos - camPos);
    float dOther = length(n.hitPos - camPos);
    return abs(dSelf - dOther) <= 0.1 * max(dSelf, 1e-4);
}

uint ReservoirIndex(uint2 pixel, uint2 dims, uint slice)
{
    return slice * (dims.x * dims.y) + pixel.y * dims.x + pixel.x;
}

float3 ReSTIRDirectIllumination(uint2 pixel, uint2 dims,
                                float3 hitPos, float3 N, float3 Ng, float3 T, float3 B,
                                float3 wi_local, GPUMaterial mat, float h, inout RNG rng)
{
    // ---- 1. initial candidates: ordinary streaming RIS, same as RIS_M above.
    Reservoir r;
    r.wSum = 0.0;
    r.pHat = 0.0;
    r.y.valid = false;

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
        float w = pHatC / pSAC;
        r.wSum += w;
        if (NextFloat(rng) < w / max(r.wSum, 1e-20))
        {
            r.y = c;
            r.pHat = pHatC;
        }
    }

    bool isHair = (mat.type == 5);

    // ---- 2. shadow-test our own survivor and fold the answer into its weight.
    // This is the step that makes the reservoir worth publishing: a neighbour
    // picking it up next frame inherits a sample already known to be visible.
    float Wself = 0.0;
    if (r.y.valid && r.pHat > 0.0)
    {
        float3 wiw;
        float dist, pHat, pSA;
        float3 integ;
        if (EvalLightCandidate(r.y, hitPos, N, wi_local, mat, h, T, B,
                               wiw, dist, integ, pHat, pSA))
        {
            float3 shadowNg = isHair ? (dot(wiw, Ng) >= 0.0 ? Ng : -Ng) : Ng;
            float3 shadowOrigin = OffsetRayOrigin(hitPos, shadowNg, shadowNg);
            RayDesc sr;
            sr.Origin = shadowOrigin;
            sr.Direction = wiw;
            sr.TMin = 0.0;
            sr.TMax = dist - 0.001;
            ShadowPayload sp;
            sp.shadowed = 1;
            sp.transmission = float3(1, 1, 1);
            sp.rngState = rng.state;
            TRACE_SHADOW(g_scene, sr, sp);
            rng.state = sp.rngState;
#if RESTIR_VISIBILITY_REUSE
            // Killing the survivor when it is occluded is what lets a neighbour
            // inherit a *visibility-tested* sample, which is the whole reason
            // reuse beats raw M. It is also a known source of bias: the stored
            // reservoir's W was derived for the unshadowed target, but the
            // sample it carries has been filtered by visibility at THIS pixel,
            // and a neighbour that borrows it re-applies visibility at its own
            // point without accounting for that filtering. Measured cost of
            // this line is in the results table.
            if (!sp.shadowed)
                Wself = (r.wSum / float(RIS_M)) / r.pHat;
#else
            // Unbiased variant: the reservoir carries the sample regardless of
            // visibility, and visibility is applied only where it belongs, in
            // the shading below. Reuse then buys effective M and nothing else.
            Wself = (r.wSum / float(RIS_M)) / r.pHat;
#endif
        }
    }

    // ---- 3. publish the initial reservoir for next frame's neighbours.
    GPUReservoir outRes;
    outRes.lightPos = r.y.position;
    outRes.pdfArea = r.y.pdfArea;
    outRes.lightNormal = r.y.normal;
    outRes.W = Wself;
    outRes.radiance = r.y.radiance;
    outRes.M = float(RIS_M);
    outRes.hitPos = hitPos;
    // valid means "this pixel drew RIS_M candidates and its shading point is the
    // one recorded below" -- NOT "the survivor was visible". Visibility lives in
    // W. Z counts candidate draws whose domain contains the chosen sample, so a
    // reservoir whose own survivor was occluded still has to be counted; folding
    // occlusion into valid would drop it from Z and brighten the result.
    outRes.valid = 1.0;
    outRes.hitNormal = N;
    outRes.pad0 = 0.0;
    g_reservoirs[ReservoirIndex(pixel, dims, frameCount & 1u)] = outRes;

    // ---- 4. combine our reservoir with the previous frame's neighbours.
    // Reservoir i contributes weight p-hat_here(y_i) * W_i * M_i, which is the
    // standard reservoir-combination weight: it re-scores the borrowed sample
    // against *this* pixel's integrand rather than trusting the neighbour's.
    EmitterSample chosen = r.y;
    float chosenPHat = 0.0;
    float wSum = 0.0;
    bool haveChosen = false;

    if (Wself > 0.0)
    {
        // p-hat of our own sample at our own point is r.pHat by construction.
        float w = r.pHat * Wself * float(RIS_M);
        if (w > 0.0)
        {
            wSum = w;
            chosen = r.y;
            chosenPHat = r.pHat;
            haveChosen = true;
        }
    }

    uint kN = (frameCount == 0u) ? 0u : min(restirNeighbours, 8u);
    uint prevSlice = (frameCount & 1u) ^ 1u;

    [loop] for (uint k = 0; k < kN; k++)
    {
        int2 off = NeighbourOffset(pixel, k, restirRadius);
        int2 np = int2(pixel) + off;
        if (np.x < 0 || np.y < 0 || np.x >= int(dims.x) || np.y >= int(dims.y))
            continue;
        GPUReservoir nb = g_reservoirs[ReservoirIndex(uint2(np), dims, prevSlice)];
        // Geometric rejection means the neighbour is not part of the combination
        // at all, so Z must not count it either -- the Z loop below repeats
        // exactly this test. A neighbour that IS combined but carries W == 0
        // contributes zero weight here yet still counts in Z.
        if (!NeighbourCompatible(nb, hitPos, N))
            continue;
        if (nb.W <= 0.0)
            continue;

        EmitterSample es = ReservoirSample(nb);
        if (!es.valid)
            continue;
        float3 wiwN;
        float distN, pHatN, pSAN;
        float3 integN;
        if (!EvalLightCandidate(es, hitPos, N, wi_local, mat, h, T, B,
                                wiwN, distN, integN, pHatN, pSAN))
            continue;

        float w = pHatN * nb.W * nb.M;
        if (w <= 0.0)
            continue;
        wSum += w;
        if (NextFloat(rng) < w / max(wSum, 1e-20))
        {
            chosen = es;
            chosenPHat = pHatN;
            haveChosen = true;
        }
    }

    if (!haveChosen || chosenPHat <= 0.0 || wSum <= 0.0)
        return float3(0, 0, 0);

    // ---- 5. unbiased normalisation: Z counts the candidates from reservoirs
    // that could actually have produced the chosen sample. The neighbour walk is
    // repeated rather than remembered because NeighbourOffset is deterministic,
    // and re-reading a handful of cached entries costs far less register
    // pressure than carrying nine shading points through the megakernel.
    float Z = 0.0;
    if (SampleInDomain(chosen.position, chosen.normal, hitPos, N))
        Z += float(RIS_M);

    [loop] for (uint k2 = 0; k2 < kN; k2++)
    {
        int2 off = NeighbourOffset(pixel, k2, restirRadius);
        int2 np = int2(pixel) + off;
        if (np.x < 0 || np.y < 0 || np.x >= int(dims.x) || np.y >= int(dims.y))
            continue;
        GPUReservoir nb = g_reservoirs[ReservoirIndex(uint2(np), dims, prevSlice)];
        if (!NeighbourCompatible(nb, hitPos, N))
            continue;
        if (SampleInDomain(chosen.position, chosen.normal, nb.hitPos, nb.hitNormal))
            Z += nb.M;
    }

    if (Z <= 0.0)
        return float3(0, 0, 0);

    float W = wSum / (Z * chosenPHat);

    // ---- 6. shade. The survivor may have come from a neighbour, where it was
    // visibility-tested against a different point, so it is shadow-tested again
    // here regardless of what its origin reservoir believed.
    float3 wiw;
    float dist, pHat, pSA;
    float3 integ;
    if (!EvalLightCandidate(chosen, hitPos, N, wi_local, mat, h, T, B,
                            wiw, dist, integ, pHat, pSA))
        return float3(0, 0, 0);

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
    TRACE_SHADOW(g_scene, shadowRay, shadow);
    rng.state = shadow.rngState;
    if (shadow.shadowed)
        return float3(0, 0, 0);

    // Same MIS convention as RISDirectIllumination: the naive solid-angle light
    // pdf on both sides. The weights only have to form a partition of unity, and
    // they do regardless of how the sample was actually chosen.
    float pdfBsdf = MaterialPdf(wi_local, ToLocal(wiw, T, B, N), mat, h);
    float wMis = BalanceHeuristic(pSA, pdfBsdf);
#if HAS_VOLUME
    float3 volTr = MultiVolumeTransmittance(shadowOrigin, wiw, dist, rng);
#else
    float3 volTr = float3(1, 1, 1);
#endif
    return integ * W * wMis * volTr * shadow.transmission;
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
    TRACE_SHADOW(g_scene, shadowRay, shadow);
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
    // Firefly control is not applied here. It lives at the call site in RayGen
    // (ClampContribution), where the contribution has been scaled by the path
    // throughput and the bounce index is known -- clamping the raw term here
    // would cap a bright light seen directly as hard as one seen through
    // sixteen bounces.
    return Lenv * f * absCosTheta / max(pdfEnv, 1e-20) * w * volTr * shadow.transmission;
}

// Volume NEE
//  at a medium scatter point, explicitly sample a light direction and evaluate the phase function.
// Unlike surface NEE, there is no cosine factor at the scatter point because volumes scatter isotropically w.r.t. geometry. the directional
// dependence is entirely in the phase function.
// MIS-weighted against the phase function sampling pdf.

float3 VolumeNEEAreaLight(float3 scatterPos, float3 wo, uint volumeIndex, inout RNG rng)
{

    GPUVolume vol = g_volumes[volumeIndex];
    float phaseG = vol.phaseG;

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
    TRACE_SHADOW(g_scene, shadowRay, shadow);
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
    // Envmap shadow rays traverse the full remaining volume; RatioTracking
    // gives an unbiased estimate of that transmittance and Russian-roulettes
    // itself once the estimate gets small, so there is deliberately no
    // geometric early-out here .
    GPUVolume vol = g_volumes[volumeIndex];
    float phaseG = vol.phaseG;

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
    TRACE_SHADOW(g_scene, shadowRay, shadow);
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
