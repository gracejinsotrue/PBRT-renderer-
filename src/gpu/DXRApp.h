#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <memory>

namespace nori
{
    class Scene;
}

class Denoiser; // src/gpu/Denoiser.h OIDN wrapper

using Microsoft::WRL::ComPtr;

struct CameraConstants
{
    float camPos[3];
    float pad0;
    float camLowerLeftCorner[3];
    float pad1;
    float camHorizontal[3];
    uint32_t meshCount;
    float camVertical[3];
    uint32_t frameCount;

    // Number of participating-medium instances. Per-volume data is uploaded
    // separately as a StructuredBuffer<GPUVolume>; see m_volumeBuffer.
    uint32_t volumeCount;
    float lensRadius; // 0 = pinhole
    float focalDistance;
    uint32_t emitterCount; // number of emitter meshes
    float envmapScale;     // multiplied on every envmap sample (controls IBL brightness)
    float evCompensation;  // display EV stops: averaged *= pow(2, ev) before Reinhard
    float _cbPad2[2];      // pad to 16-byte boundary
};

// Mirror of the HLSL GPUVolume struct used for ray marching in the presence of participating media.
struct GPUVolume
{
    float vMin[3];             // 0..11
    float pad0;                // 12..15
    float sigmaA[3];           // 16..27
    float pad1;                // 28..31
    float vMax[3];             // 32..43
    float pad2;                // 44..47
    float sigmaS[3];           // 48..59
    float phaseG;              // 60..63
    uint32_t densityTexIndex;  // 64..67  0xFFFFFFFF if homogeneous / no texture
    uint32_t flags;            // 68..71  bit 0: heterogeneous
    uint32_t majorantTexIndex; // 72..75  index of the brick-max-density mip,
                               //         or 0xFFFFFFFF to fall back to global μ
    uint32_t pad3;             // 76..79
};

static_assert(sizeof(GPUVolume) == 80,
              "GPUVolume layout must match HLSL declaration");

static constexpr uint32_t VOLUME_FLAG_HETEROGENEOUS = 0x1u;
static constexpr uint32_t VOLUME_INVALID_TEX = 0xFFFFFFFFu;

struct GPUMaterial
{
    uint32_t type; // 0=diffuse, 1=mirror, 2=dielectric, 3=microfacet, 4=disney
    float albedo[3];
    float intIOR;
    float extIOR;
    float alpha;
    uint32_t isEmitter;
    float radiance[3];
    uint32_t indexOffset;        // first index,in elements, in global index buffer
    uint32_t vertexOffset;       // first vertex, in elements, in global normal/pos buffer
    uint32_t indexCount;         // number of indices for this mesh
    uint32_t vertexCount;        // number of vertices for this mesh
    float surfaceArea;           // total mesh surface area
    uint32_t emitterCdfOffset;   // first entry index in emitter CDF buffer
    uint32_t albedoTexIndex;     // texture index or 0xFFFFFFFF if none
    uint32_t normalTexIndex;     // texture index or 0xFFFFFFFF if none
    uint32_t roughnessTexIndex;  // texture index or 0xFFFFFFFF if none
    uint32_t metallicTexIndex;   // texture index or 0xFFFFFFFF if none
    uint32_t specularTexIndex;   // overrides scalar specular when valid
    uint32_t subsurfaceTexIndex; // overrides scalar subsurface when valid

    uint32_t alphaTexIndex; // texture index for alpha masking

    // Disney BRDF parameters from Burley 2012
    // baseColor is stored in the `albedo` field above to share texture
    // plumbing with other BSDFs.
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
    float betaN;                // azimuthal roughness β_N (hair only, Chiang Eq. 8)
    float emitterSelectionProb; // power-weighted probability of selecting this emitter (0 for non-emitters)
};

static_assert(sizeof(GPUMaterial) % 4 == 0,
              "GPUMaterial field sizes must be 4-byte aligned");

struct MeshGPUData
{
    ComPtr<ID3D12Resource> vertexBuffer;
    ComPtr<ID3D12Resource> indexBuffer;
    ComPtr<ID3D12Resource> blas;
    uint32_t indexCount;
};

class DXRApp
{
public:
    DXRApp(const std::string &scenePath, bool headless = false);
    ~DXRApp();

    void OnInit();
    void OnUpdate();
    void OnRender();
    void OnDestroy();

    // Enable GPU-timestamp profiling of the DispatchRays call (--profile).
    // Must be called before OnInit() so the query heap gets created.
    void SetProfiling(bool enable) { m_profile = enable; }

    void OnKeyDown(UINT8 key);
    void OnKeyUp(UINT8 key);
    void OnMouseDown(UINT button, int x, int y);
    void OnMouseUp(UINT button, int x, int y);
    void OnMouseMove(int x, int y);

    void SaveSnapshot();
    void SaveSnapshotEXR();
    void SaveAccumResourceEXR(ID3D12Resource *res, const char *filename);
    void DenoiseAndSaveEXR();
    void DenoiseToViewport();

    UINT GetWidth() const { return m_width; }
    UINT GetHeight() const { return m_height; }
    const wchar_t *GetTitle() const { return m_title.c_str(); }
    uint32_t GetFrameCount() const { return m_frameCount; }
    uint32_t GetTargetSamples() const { return m_targetSamples; }

private:
    static constexpr UINT FrameCount = 2;

    // Window
    UINT m_width;
    UINT m_height;
    std::wstring m_title;
    bool m_headless = false;

    // Nori scene
    std::string m_scenePath;
    nori::Scene *m_noriScene = nullptr;

    // Pipeline objects
    ComPtr<IDXGIFactory6> m_factory;
    ComPtr<ID3D12Device5> m_device;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<IDXGISwapChain3> m_swapChain;
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    UINT m_rtvDescriptorSize = 0;
    ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
    ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
    ComPtr<ID3D12GraphicsCommandList4> m_commandList;
    UINT m_frameIndex = 0;

    // Synchronization primtivies
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValues[FrameCount] = {};
    HANDLE m_fenceEvent = nullptr;

    // GPU-timestamp profiling. A 2-slot TIMESTAMP query heap
    // brackets the DispatchRays in PopulateCommandList; the ticks are resolved
    // to m_tsReadback and read back in OnRender to get pure kernel ms/frame.
    // Every frame is ~1 spp of identical work, the same as progressive accumulator, so
    // per-frame dispatch ms IS the metric. Authoritative measurement path is
    // --headless which is fully synchronous (or so I h ope); windowed sampling is
    // best-effort and may lag a frame. See PrintProfileSummary for stats.
    bool m_profile = false;
    ComPtr<ID3D12QueryHeap> m_tsQueryHeap;
    ComPtr<ID3D12Resource> m_tsReadback;
    UINT64 m_tsFrequency = 0;
    std::vector<double> m_frameMs;
    bool m_profileSummaryPrinted = false;

    // Acceleration structure
    std::vector<MeshGPUData> m_meshGPU;
    ComPtr<ID3D12Resource> m_tlas;

    // Scene data buffers for shader access
    ComPtr<ID3D12Resource> m_materialBuffer;       // StructuredBuffer<GPUMaterial>
    ComPtr<ID3D12Resource> m_globalNormalBuffer;   // ByteAddressBuffer — all normals
    ComPtr<ID3D12Resource> m_globalIndexBuffer;    // ByteAddressBuffer — all indices
    ComPtr<ID3D12Resource> m_globalVertexBuffer;   // ByteAddressBuffer — all positions
    ComPtr<ID3D12Resource> m_emitterCdfBuffer;     // ByteAddressBuffer — emitter triangle area CDFs
    ComPtr<ID3D12Resource> m_globalTexCoordBuffer; // ByteAddressBuffer — all UVs (float2 per vertex)
    ComPtr<ID3D12Resource> m_globalTangentBuffer;  // ByteAddressBuffer — all fiber tangents (float3 per vertex, hair only)
    uint32_t m_meshCount = 0;
    uint32_t m_emitterCount = 0;
    uint32_t m_frameCount = 0;
    uint32_t m_targetSamples = 0; // auto-save EXR and exit when reached (0 = disabled)

    // Texture resources
    std::vector<ComPtr<ID3D12Resource>> m_textures;
    std::vector<ComPtr<ID3D12Resource>> m_texUploads;

    std::vector<uint8_t> m_textureIsSRGB;
    uint32_t m_textureCount = 0;

    // Environment map (IBL)
    ComPtr<ID3D12Resource> m_envmap;
    ComPtr<ID3D12Resource> m_envmapUpload;
    std::vector<float> m_envmapPixels; // retained for CPU-side CDF construction
    uint32_t m_envmapWidth = 0;
    uint32_t m_envmapHeight = 0;
    bool m_envmapValid = false;

    // Envmap sampling CDFs (Distribution2D: marginal over rows,
    // conditional per row). Built on CPU from m_envmapPixels, uploaded as
    // ByteAddressBuffers.
    ComPtr<ID3D12Resource> m_envmapMarginalCdf;    // (H + 1) floats
    ComPtr<ID3D12Resource> m_envmapConditionalCdf; // H * (W + 1) floats

    // Participating-medium volumes
    // m_volumes is the CPU-side definition; m_volumeBuffer is the GPU-visible
    // StructuredBuffer<GPUVolume>. Heterogeneous volumes that load a density
    // file each push a Texture3D into m_volumeTextures and reference it by
    // index. m_volumeUploads holds upload heaps until the load command list
    // flushes.
    std::vector<GPUVolume> m_volumes;
    ComPtr<ID3D12Resource> m_volumeBuffer;
    std::vector<ComPtr<ID3D12Resource>> m_volumeTextures;
    std::vector<ComPtr<ID3D12Resource>> m_volumeUploads;

    // Ray tracing pipeline
    ComPtr<ID3D12StateObject> m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    ComPtr<ID3D12RootSignature> m_globalRootSig;

    // Output + descriptors
    ComPtr<ID3D12Resource> m_outputResource;
    ComPtr<ID3D12Resource> m_accumResource;
    ComPtr<ID3D12Resource> m_albedoResource;
    ComPtr<ID3D12Resource> m_normalResource;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;

    // In-app denoiser with OIDN, this is lazily initialized on first denoise request.
    // when the camera is stationary, the accumulator keeps running and we periodically denoise the current mean into m_denoisedResource
    // and display THAT, which is a clean preview that refines as spp grows.
    // if you move the camera, it resets it back to the live progressive view.
    std::unique_ptr<Denoiser> m_denoiser;
    ComPtr<ID3D12Resource> m_denoisedResource;
    bool m_showDenoised = false;
    bool m_autoDenoise = true;
    uint32_t m_nextDenoiseSpp = 16;
    static constexpr uint32_t kMaxDenoiseSpp = 2048;
    UINT m_srvUavDescriptorSize = 0;

    ComPtr<ID3D12Resource> m_shaderTable;

    CameraConstants m_camera = {};

    // Interactive camera state
    float m_camYaw = 0.0f;
    float m_camPitch = 0.0f;
    float m_camPos[3] = {};
    float m_camFovY = 0.0f; // vertical FOV in radians
    float m_camXFlip = 1.0f;
    float m_camSpeed = 1.0f;
    float m_mouseSensitivity = 0.003f;
    bool m_cameraDirty = true; // set when camera moved; resets accumulation

    // Input state
    bool m_keys[256] = {};
    bool m_mouseLeftDown = false;
    bool m_mouseRightDown = false;
    POINT m_lastMouse = {};

    std::chrono::high_resolution_clock::time_point m_lastFrameTime;

    void LoadNoriScene();
    void CreateDevice();
    void CreateCommandQueue();
    void CreateSwapChain();
    void CreateRTVHeap();
    void CreateCommandAllocatorsAndList();
    void CreateFence();
    void CreateProfiler();
    void PrintProfileSummary();
    void CreateAccelerationStructure();
    void SetupVolumes();
    void CreateSceneBuffers();
    void CreateTextures();
    void CreateRaytracingPipeline();
    void CreateOutputResource();
    void CreateShaderTable();
    void SetupCamera();

    // Texture helpers
    // uint32_t LoadTexture(const std::string &path = false);
    uint32_t LoadTexture(const std::string &path, bool isSRGB);
    void LoadEnvmap(const std::string &path);
    void BuildEnvmapDistribution();
    // Load a heterogeneous-volume density file (.vol) and append two
    // Texture3D's to m_volumeTextures: the dense density grid and a coarse
    // brick-max-density mip used for tracked-majorant volume tracking.
    // Returns the indices for GPUVolume::densityTexIndex and
    // GPUVolume::majorantTexIndex respectively, or VOLUME_INVALID_TEX in
    // both slots on failure.
    struct LoadedVolumeIndices
    {
        uint32_t densityIndex;
        uint32_t majorantIndex;
    };
    LoadedVolumeIndices LoadVolume(const std::string &path);

    void RecomputeCameraPlane();

    // Render helpers
    void PopulateCommandList();
    void WaitForGpu(UINT frameIndex);
    void FlushCommandQueue();

    // Denoiser helpers
    std::vector<float> ReadbackAccumResource(ID3D12Resource *res);
    bool RunDenoise(std::vector<float> &outRGB);
    void UploadRGBA8(ID3D12Resource *res, const std::vector<uint8_t> &rgba);

    // Resource helpers
    ComPtr<ID3D12Resource> CreateBuffer(
        UINT64 size, D3D12_RESOURCE_FLAGS flags,
        D3D12_RESOURCE_STATES initialState, D3D12_HEAP_TYPE heapType);

    static std::vector<uint8_t> ReadFileBytes(const std::wstring &path);
    static std::wstring GetExeDirectory();
    static void ThrowIfFailed(HRESULT hr, const char *msg);
};