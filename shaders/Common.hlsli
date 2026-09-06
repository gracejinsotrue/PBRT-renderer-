// Common.hlsli. Resource bindings, structs, and constants shared by all
// shader files.  Every .hlsli in this project should #include this first.

#ifndef COMMON_HLSLI
#define COMMON_HLSLI

// Feature toggles for per-scene specialization. Default would be a full kernel, everythign is on.
// A specialized build compiles a stripped .cso via dxc -D HAS_X=0 to drop unused material/volume code paths and shrink the kernel's register footprint.
#ifndef HAS_HAIR
#define HAS_HAIR 1
#endif
#ifndef HAS_DISNEY
#define HAS_DISNEY 1
#endif
#ifndef HAS_VOLUME
#define HAS_VOLUME 1
#endif

// Global resources

RaytracingAccelerationStructure g_scene : register(t0);
RWTexture2D<float4> g_output : register(u0);
RWTexture2D<float4> g_accum : register(u1);

// Denoiser feature buffers (AOVs)
RWTexture2D<float4> g_albedo : register(u2);
RWTexture2D<float4> g_normal : register(u3);

// Per-pixel luminance moments for adaptive sampling: x = sum of sample
// luminance, y = sum of squared sample luminance. The sample count is not
// duplicated here; it is g_accum[pixel].w. Only written when adaptive sampling
// is compiled in, but the resource is always bound so the root signature does
// not depend on a shader define.
RWTexture2D<float2> g_moments : register(u4);

// ReSTIR DI reservoir storage.
//
// One entry per pixel per parity slice: the buffer holds 2 * width * height
// entries and a frame writes slice (frameCount & 1) while reading slice
// (frameCount & 1) ^ 1. Ping-ponging inside one resource rather than swapping
// two descriptors keeps the descriptor table fixed, which matters because the
// heap layout is shared with the post-process passes.
//
// The G-buffer fields (hitPos, hitNormal) travel with the reservoir instead of
// living in their own texture. A neighbour's shading point is needed for two
// things -- the geometric similarity test and the unbiased Z normalisation --
// and both are needed exactly when the reservoir is read, so splitting them
// into a second resource would only double the number of loads.
struct GPUReservoir
{
    float3 lightPos;    // sampled point on the emitter
    float pdfArea;      // area-measure pdf it was drawn with
    float3 lightNormal; // emitter normal there
    float W;            // unbiased contribution weight; 0 = occluded or empty
    float3 radiance;    // emitted radiance at that point
    float M;            // number of candidates this reservoir represents
    float3 hitPos;      // shading point that produced it
    float valid;        // 1 = usable, 0 = miss / delta surface / converged pixel
    float3 hitNormal;   // shading normal there
    float pad0;
};

RWStructuredBuffer<GPUReservoir> g_reservoirs : register(u5);

// Constant buffer

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
    // g_volumes (StructuredBuffer<GPUVolume>) below, where zero means no volumes.
    uint volumeCount;
    float lensRadius;
    float focalDistance;
    uint emitterCount;
    float envmapScale;
    float evCompensation;
    float envmapRotation;  // yaw offset in radians applied to envmap phi lookup

    // Firefly clamp: maximum luminance of a single indirect contribution.
    // <= 0 disables it, which is the default and the validated offline path.
    float fireflyClamp;

    // Adaptive sampling: stop refining a pixel once the standard error of its
    // mean falls below adaptiveThreshold (relative). <= 0 disables it.
    float adaptiveThreshold;
    uint adaptiveMinSamples; // warm-up before the variance estimate is trusted

    // ReSTIR DI spatial reuse. Radius is in pixels; <= 0 disables reuse and
    // leaves RISDirectIllumination behaving exactly as it did.
    float restirRadius;
    uint restirNeighbours;
};

// Material structure

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

// Structured / byte-address buffers

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

// Participating-medium volumes. Multi-instance design where each entry in
// g_volumes carries its own AABB, scattering coefficients, phase param,
// and (for heterogeneous media) an index into g_volumeDensities[].

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

// Constants

static const float M_PI = 3.14159265358979323846;
static const float M_INV_PI = 0.31830988618379067154;
// Path length cap. Overridable via dxc -D MAX_BOUNCES=<n> so a real-time build
// can trade bounces for frame time without editing the source (32 = offline default).
#ifndef MAX_BOUNCES
#define MAX_BOUNCES 32
#endif
// Firefly control.
//
// A path tracer's variance is dominated by rare, enormous samples: a caustic
// path that finds a small bright light through a low-pdf chain, or (with RIS)
// a light picked by an unshadowed target that turns out to be barely visible,
// so the survivor carries a large unbiased weight W. One such sample can
// outweigh thousands of ordinary ones, and the mean it corrupts stays corrupted
// for the rest of the render.
//
// Clamping the luminance of each contribution caps that. It is *biased* -- it
// removes energy that genuinely belongs in the image -- so it is off by default
// and the offline reference path is unaffected.
//
// Two deliberate choices:
//  - it clamps each contribution as it is added, not the final per-sample sum,
//    so one runaway NEE term is capped without also crushing a pixel that is
//    legitimately bright because many ordinary terms added up.
//  - it never touches bounce 0. Directly visible emitters and the environment
//    seen down the primary ray are not fireflies, they are the picture; the
//    variance lives in what happens after the first scatter.
float3 ClampContribution(float3 c, int bounce)
{
    if (fireflyClamp <= 0.0 || bounce == 0)
        return c;
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    return (l > fireflyClamp) ? c * (fireflyClamp / l) : c;
}

// Adaptive sampling stop test.
//
// Given n samples of a pixel with running sums of luminance and squared
// luminance, estimate the standard error of the mean and compare it against a
// relative target. A pixel that passes stops being sampled for the rest of the
// render, so the cost per frame falls as the image converges and the remaining
// samples land where the noise actually is.
//
// The absolute floor matters: a near-black pixel has a tiny mean, so a purely
// relative test would chase precision there forever. kAdaptiveFloor is in
// scene-linear radiance, below which a pixel is considered good enough on
// absolute terms regardless of its relative error.
//
// Caveat, stated because it is easy to forget: deciding to stop using the same
// samples that form the estimate correlates the decision with the value, which
// biases the result slightly toward pixels that got lucky early. The warm-up
// (adaptiveMinSamples) is what keeps that negligible; do not set it low.
static const float kAdaptiveFloor = 1e-3;

bool PixelConverged(float2 moments, float n)
{
    if (adaptiveThreshold <= 0.0 || n < 2.0 || n < float(adaptiveMinSamples))
        return false;
    float mean = moments.x / n;
    // Sample variance, computed from the running sums. max() guards the
    // cancellation when every sample is identical and the two terms match.
    float var = max(0.0, (moments.y - moments.x * moments.x / n) / (n - 1.0));
    float stdErr = sqrt(var / n); // standard error of the mean
    return stdErr < max(adaptiveThreshold * mean, kAdaptiveFloor * adaptiveThreshold);
}

// Ray payloads

struct HitPayload
{
    float hitT;
    uint materialID;
    uint hit;
    uint primitiveID;
    float baryX, baryY; // hit barycentrics; N/UV/tangent/hairH recomputed in RayGen
    uint rngState;      // PCG state, propagated through any-hit
};

struct ShadowPayload
{
    uint shadowed;
    float3 transmission; // accumulated Fresnel transmission through glass
    uint rngState;       // PCG state, propagated through any-hit
};

#endif // COMMON_HLSLI
