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
    float2 _cbPad2;
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
static const int MAX_BOUNCES = 32;
static const float kFireflyClamp = 3.402823466e+38;

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
