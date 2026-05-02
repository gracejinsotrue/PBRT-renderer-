// Common.hlsli — Resource bindings, structs, and constants shared by all
// shader files.  Every .hlsli in this project should #include this first.

#ifndef COMMON_HLSLI
#define COMMON_HLSLI

// ============================================================================
// Global resources
// ============================================================================

RaytracingAccelerationStructure g_scene : register(t0);
RWTexture2D<float4> g_output : register(u0);
RWTexture2D<float4> g_accum : register(u1);

// ============================================================================
// Constant buffer
// ============================================================================

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

    // Volume parameters for homogeneous participating medium
    float volumeMinX, volumeMinY, volumeMinZ;
    float volumeSigmaA;
    float volumeMaxX, volumeMaxY, volumeMaxZ;
    float volumeSigmaS;
    float volumePhaseG;
    uint volumeEnabled;
    uint volumeHeterogeneous;
    uint volumeHasTexture;
};

// ============================================================================
// Material structure
// ============================================================================

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
    float betaN; // azimuthal roughness for hair
};

float3 MatAlbedo(GPUMaterial m) { return float3(m.albedoR, m.albedoG, m.albedoB); }
float3 MatRadiance(GPUMaterial m) { return float3(m.radianceR, m.radianceG, m.radianceB); }

// ============================================================================
// Structured / byte-address buffers
// ============================================================================

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

// Volume density for heterogeneous media
Texture3D<float> g_volumeDensity : register(t0, space1);
SamplerState g_volumeSampler : register(s2);

// ============================================================================
// Constants
// ============================================================================

static const float M_PI = 3.14159265358979323846;
static const float M_INV_PI = 0.31830988618379067154;
static const int MAX_BOUNCES = 32;

// ============================================================================
// Ray payloads
// ============================================================================

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
};

struct ShadowPayload
{
    uint shadowed;
};

#endif // COMMON_HLSLI