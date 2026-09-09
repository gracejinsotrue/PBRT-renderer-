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
#include <cstring>
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
    float envmapScale;      // multiplied on every envmap sample (controls IBL brightness)
    float evCompensation;   // display EV stops: averaged *= pow(2, ev) before Reinhard
    float envmapRotation;   // yaw offset in radians applied to envmap phi lookup

    // Firefly clamp: max luminance of one indirect contribution (0 = off).
    float fireflyClamp;

    // Adaptive sampling: relative standard-error target (0 = off) and the
    // warm-up sample count before the per-pixel variance estimate is trusted.
    float adaptiveThreshold;
    uint32_t adaptiveMinSamples;

    // ReSTIR DI spatial reuse: neighbour search radius in pixels (0 = off) and
    // how many neighbours to combine.
    float restirRadius;
    uint32_t restirNeighbours;
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
    float translucency;         // thin-surface diffuse transmission, 0 = opaque reflector
};

static_assert(sizeof(GPUMaterial) % 4 == 0,
              "GPUMaterial field sizes must be 4-byte aligned");

struct MeshGPUData
{
    // The BLAS is the only per-mesh GPU resource. Its geometry is read
    // straight out of the global vertex/index buffers at these offsets,
    // rather than from a per-mesh copy: the global index buffer stores
    // mesh-local indices, so a vertex base plus an index base addresses
    // the mesh exactly.
    ComPtr<ID3D12Resource> blas;
    uint32_t vertexOffset = 0; // in vertices, into the global vertex buffer
    uint32_t indexOffset = 0;  // in indices, into the global index buffer
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
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

    // Present without waiting for vblank (--novsync). Must be called before
    // OnInit() so CreateSwapChain can request the tearing flag: SyncInterval 0
    // alone still gets composited at the refresh rate under DWM, so uncapping
    // for real needs ALLOW_TEARING on both the swap chain and the Present call.
    void SetVSync(bool enable) { m_vsync = enable; }

    // Firefly clamp (--clamp). Caps the luminance of any single contribution
    // added at bounce >= 1. This is a biased estimator, so it is off by
    // default and every existing reference render is unaffected.
    void SetFireflyClamp(float v) { m_fireflyClamp = v; }

    // Adaptive sampling (--adaptive). `threshold` is the relative standard
    // error at which a pixel stops being sampled; `minSamples` is the warm-up
    // before that test is allowed to fire.
    void SetAdaptive(float threshold, uint32_t minSamples)
    {
        m_adaptiveThreshold = threshold;
        m_adaptiveMinSamples = minSamples;
    }

    // Mean samples/pixel and converged fraction, read back from the
    // accumulator's .w channel. Only meaningful with adaptive sampling on.
    void ReportAdaptiveStats();
    // Where the scene's bytes physically live. Every buffer goes through
    // CreateBuffer, so tallying there catches all of them; the DXGI figures
    // additionally cover textures and the acceleration structures.
    void ReportMemory(const char *phase);

    // ReSTIR DI spatial reuse (--restir R[,K]). Only has an effect on a shader
    // built with -D USE_RIS=1; radius 0 leaves plain RIS untouched.
    void SetReSTIR(float radius, uint32_t neighbours)
    {
        m_restirRadius = radius;
        m_restirNeighbours = neighbours;
    }

    // Persistent-thread path tracer instead of DispatchRays (--wavefront [G]).
    // G is the number of 64-thread groups to keep resident; 0 leaves the
    // megakernel in charge.
    void SetWavefront(uint32_t groups) { m_wavefrontGroups = groups; }

    void OnKeyDown(UINT8 key);
    void OnKeyUp(UINT8 key);
    void OnMouseDown(UINT button, int x, int y);
    void OnMouseUp(UINT button, int x, int y);
    void OnMouseMove(int x, int y);

    void SaveSnapshot();
    // Display-referred PNG (exposure, bloom, ACES, gamma applied), unlike the
    // scene-linear EXR paths. `denoised` reads the denoised display texture.
    void SaveSnapshotPNG(bool denoised = false);
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
    ComPtr<IDXGIAdapter3> m_adapter; // retained for QueryVideoMemoryInfo
    struct HeapTally
    {
        UINT64 bytes[4] = {0, 0, 0, 0}; // indexed by D3D12_HEAP_TYPE (1..3)
        uint32_t count[4] = {0, 0, 0, 0};
    };
    HeapTally m_heapTally;
    // Upload buffers backing in-flight scene copies. Cleared once the
    // command list that consumes them has been flushed.
    std::vector<ComPtr<ID3D12Resource>> m_sceneUploadStaging;
    // Material records are finished in two phases: geometry and BSDF data in
    // CreateSceneBuffers, then texture indices once CreateTextures knows them.
    // The array stays on the host until then, so the GPU buffer is written once.
    std::vector<GPUMaterial> m_materialsCpu;
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

    // Whether the D3D12 debug layer is running (Debug builds only). Its messages
    // go to OutputDebugString, which a console run never sees, so DrainDebugMessages
    // pulls them explicitly — otherwise a clean-looking Debug run proves nothing.
    bool m_debugLayerActive = false;
    uint32_t DrainDebugMessages(const char *tag);

    // Presentation pacing. m_vsync is the user's request (--novsync clears it);
    // m_allowTearing records whether the adapter/OS actually supports tearing,
    // decided in CreateSwapChain. Both must hold to present uncapped.
    bool m_vsync = true;
    float m_fireflyClamp = 0.0f;      // 0 = disabled
    float m_adaptiveThreshold = 0.0f; // 0 = disabled
    uint32_t m_adaptiveMinSamples = 32;
    float m_restirRadius = 0.0f; // 0 = spatial reuse disabled
    uint32_t m_restirNeighbours = 4;
    uint32_t m_wavefrontGroups = 0; // 0 = use the DispatchRays megakernel
    bool m_allowTearing = false;

    // Wall-clock FPS for the windowed loop, printed once a second under
    // --profile. The GPU-timestamp path above brackets DispatchRays only, so it
    // is blind to presentation pacing — a vsync A/B moves this number and not
    // that one. Windowed only; headless already reports ms/spp at the end.
    std::chrono::high_resolution_clock::time_point m_fpsWindowStart;
    uint32_t m_fpsFrames = 0;
    bool m_fpsStarted = false;

    ComPtr<ID3D12QueryHeap> m_tsQueryHeap;
    ComPtr<ID3D12Resource> m_tsReadback;
    UINT64 m_tsFrequency = 0;
    std::vector<double> m_frameMs;
    std::vector<double> m_recordMs; // CPU record+submit per frame (--profile)
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
    // Per-pixel (sum of luminance, sum of luminance^2) for adaptive sampling.
    // Always created and bound; only written when adaptive sampling is on.
    ComPtr<ID3D12Resource> m_momentsResource;
    // ReSTIR reservoirs: 2 * width * height entries, one parity slice per frame.
    // Must match sizeof(GPUReservoir) in shaders/Common.hlsli.
    static constexpr UINT kReservoirStride = 80;
    ComPtr<ID3D12Resource> m_reservoirResource;
    UINT64 m_reservoirCount = 0;
    // Persistent-thread path tracer: a one-dword work queue plus the two compute
    // PSOs that reset and drain it.
    ComPtr<ID3D12Resource> m_pathQueueResource;
    ComPtr<ID3D12PipelineState> m_wavefrontPSO;
    ComPtr<ID3D12PipelineState> m_resetQueuePSO;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;

    // Display resolve pass (Resolve.hlsl / CSResolve). Turns an HDR
    // accumulation texture into the RGBA8 image that gets blitted to the
    // backbuffer: normalize by .w, exposure, ACES, gamma. RayGen used to do
    // this inline; it was split out so the curve lives in one place and so a
    // neighbourhood pass (bloom) has somewhere to sit between HDR and tonemap.
    //
    // The pass has its own root signature — the raytracing one is built around
    // unbounded texture arrays it has no use for — but it reads descriptors
    // from the tail of m_srvUavHeap so no heap swap is needed mid-frame.
    ComPtr<ID3D12RootSignature> m_postRootSig;
    ComPtr<ID3D12PipelineState> m_resolvePSO;
    ComPtr<ID3D12PipelineState> m_bloomPrefilterPSO;
    ComPtr<ID3D12PipelineState> m_bloomDownsamplePSO;
    ComPtr<ID3D12PipelineState> m_bloomUpsamplePSO;

    // Bloom mip chain: one RGBA16F texture, mip 0 at half the render
    // resolution, with a UAV per mip. RGBA16F rather than the more compact
    // R11G11B10 because the chain does typed UAV *loads*, which need
    // TypedUAVLoadAdditionalFormats — a cap this app already relies on, since
    // g_accum is R32G32B32A32_FLOAT and RayGen loads it.
    ComPtr<ID3D12Resource> m_bloomResource;
    UINT m_bloomMipCount = 0;

    // Scene-authored bloom controls (see Scene::getBloom*). Intensity is
    // runtime-adjustable with [ and ]; B toggles bloom off entirely.
    float m_bloomThreshold = 1.0f;
    float m_bloomKnee = 0.5f;
    float m_bloomIntensity = 0.0f;
    bool m_bloomEnabled = true;

    // Descriptor-heap index of the first post-process UAV. Every post pass binds
    // a 3-descriptor slot (u0 source, u1 dest, u2 aux) so they can share one
    // root signature; the bloom passes ignore u2. Slot layout, in units of 3:
    //   0                        resolve, live       (accum, output, bloom mip 0)
    //   1                        resolve, denoised   (denoisedHdr, denoised, bloom mip 0)
    //   2                        prefilter, live     (accum, bloom mip 0, -)
    //   3                        prefilter, denoised (denoisedHdr, bloom mip 0, -)
    //   4 .. 4+(M-2)             downsample i        (mip i, mip i+1, -)
    //   4+(M-1) .. 4+2(M-1)-1    upsample i          (mip i+1, mip i, -)
    UINT m_postDescriptorBase = 0;
    static constexpr UINT kPostSlotResolveLive = 0;
    static constexpr UINT kPostSlotResolveDenoised = 1;
    static constexpr UINT kPostSlotPrefilterLive = 2;
    static constexpr UINT kPostSlotPrefilterDenoised = 3;
    static constexpr UINT kPostSlotDownsampleFirst = 4;
    UINT PostSlotUpsampleFirst() const { return kPostSlotDownsampleFirst + (m_bloomMipCount - 1); }

    // Dimensions of bloom mip `level` (level 0 is half the render resolution).
    void BloomMipDims(UINT level, UINT &w, UINT &h) const
    {
        w = m_width >> (level + 1);
        h = m_height >> (level + 1);
        if (w == 0) w = 1;
        if (h == 0) h = 1;
    }

    // In-app denoiser with OIDN, this is lazily initialized on first denoise request.
    // when the camera is stationary, the accumulator keeps running and we periodically denoise the current mean into m_denoisedResource
    // and display THAT, which is a clean preview that refines as spp grows.
    // if you move the camera, it resets it back to the live progressive view.
    std::unique_ptr<Denoiser> m_denoiser;
    ComPtr<ID3D12Resource> m_denoisedResource;
    // OIDN's linear HDR output, uploaded with w = 1.0 so the resolve pass can
    // normalize it by .w exactly like the accumulation buffer and share one shader.
    ComPtr<ID3D12Resource> m_denoisedHdrResource;
    bool m_showDenoised = false;
    bool m_autoDenoise = true;
    uint32_t m_nextDenoiseSpp = 16;
    // Last denoise result, keyed on the sample count it was produced at, so
    // repeat callers at the same accumulation state don't re-run OIDN.
    std::vector<float> m_denoiseCache;
    uint32_t m_denoiseCacheFrame = 0xFFFFFFFFu;
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
    void CreatePostPipelines();
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

    // Record one post-process dispatch into m_commandList. `slot` is a
    // 3-descriptor slot index (see m_postDescriptorBase). Caller owns the
    // surrounding barriers.
    void RecordPostPass(ID3D12PipelineState *pso, UINT slot,
                        UINT srcW, UINT srcH, UINT dstW, UINT dstH,
                        float bloomIntensity);

    // Record the full bloom mip chain (prefilter, downsample, upsample) for the
    // given HDR source. No-op when bloom is disabled. Caller owns the barrier
    // that makes the HDR source visible.
    void RecordBloomChain(bool denoisedSource);

    // Record the display resolve. `denoisedSource` picks which HDR texture and
    // LDR destination to bind.
    void RecordResolve(bool denoisedSource);

    // Re-run bloom + resolve on the already-denoised HDR texture without
    // re-running OIDN, so bloom tweaks are visible while a preview is frozen.
    void RefreshDenoisedPreview();

    // Effective bloom intensity: zero when the user has toggled bloom off, which
    // makes the resolve shader skip the composite entirely.
    float EffectiveBloomIntensity() const { return m_bloomEnabled ? m_bloomIntensity : 0.0f; }

    // Denoiser helpers
    std::vector<float> ReadbackAccumRGBA(ID3D12Resource *res);
    std::vector<float> ReadbackAccumResource(ID3D12Resource *res);
    std::vector<uint8_t> ReadbackRGBA8(ID3D12Resource *res);
    bool RunDenoise(std::vector<float> &outRGB);
    // Upload interleaved float3 RGB into an RGBA32F texture, setting w = 1.0.
    void UploadHDR(ID3D12Resource *res, const std::vector<float> &rgb);

    // Resource helpers
    // Creates a device-local (DEFAULT heap) buffer and stages its contents
    // through a temporary UPLOAD buffer, recording the copy and the transition
    // on the currently open command list. `fill` is handed the mapped staging
    // pointer, so callers that build their data by concatenating per-mesh
    // arrays write straight into it - no intermediate host copy.
    //
    // The staging buffers are held in m_sceneUploadStaging and must outlive the
    // command list; ReleaseSceneUploadStaging() drops them after the flush.
    template <typename Fn>
    ComPtr<ID3D12Resource> CreateBufferFilled(UINT64 size, D3D12_RESOURCE_STATES finalState, Fn &&fill)
    {
        ComPtr<ID3D12Resource> dst = CreateBuffer(size, D3D12_RESOURCE_FLAG_NONE,
                                                  D3D12_RESOURCE_STATE_COPY_DEST,
                                                  D3D12_HEAP_TYPE_DEFAULT);
        ComPtr<ID3D12Resource> staging = CreateBuffer(size, D3D12_RESOURCE_FLAG_NONE,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ,
                                                      D3D12_HEAP_TYPE_UPLOAD);
        void *mapped = nullptr;
        ThrowIfFailed(staging->Map(0, nullptr, &mapped), "Map scene staging buffer");
        fill(static_cast<uint8_t *>(mapped));
        staging->Unmap(0, nullptr);

        m_commandList->CopyBufferRegion(dst.Get(), 0, staging.Get(), 0, size);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = dst.Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = finalState;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_commandList->ResourceBarrier(1, &barrier);

        m_sceneUploadStaging.push_back(staging);
        return dst;
    }

    // Convenience wrapper for data that is already contiguous on the host.
    ComPtr<ID3D12Resource> CreateBufferWithData(const void *data, UINT64 size,
                                                D3D12_RESOURCE_STATES finalState)
    {
        return CreateBufferFilled(size, finalState,
                                  [&](uint8_t *dst) { memcpy(dst, data, (size_t)size); });
    }

    void ReleaseSceneUploadStaging() { m_sceneUploadStaging.clear(); }

    ComPtr<ID3D12Resource> CreateBuffer(
        UINT64 size, D3D12_RESOURCE_FLAGS flags,
        D3D12_RESOURCE_STATES initialState, D3D12_HEAP_TYPE heapType);

    static std::vector<uint8_t> ReadFileBytes(const std::wstring &path);
    static std::wstring GetExeDirectory();
    static void ThrowIfFailed(HRESULT hr, const char *msg);
};