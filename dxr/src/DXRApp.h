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

namespace nori
{
    class Scene;
}

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

    // Volume (homogeneous participating medium)
    float volumeMin[3];
    float volumeSigmaA;
    float volumeMax[3];
    float volumeSigmaS;
    float volumePhaseG;
    uint32_t volumeEnabled;
    uint32_t volumeHeterogeneous;
    uint32_t volumeHasTexture;
};

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

    // Disney BRDF parameters (Burley 2012). Read only when type == 4.
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
    float betaN; // azimuthal roughness β_N (hair only, Chiang Eq. 8)
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
    DXRApp(const std::string &scenePath);
    ~DXRApp();

    void OnInit();
    void OnUpdate();
    void OnRender();
    void OnDestroy();

    void OnKeyDown(UINT8 key);
    void OnKeyUp(UINT8 key);
    void OnMouseDown(UINT button, int x, int y);
    void OnMouseUp(UINT button, int x, int y);
    void OnMouseMove(int x, int y);

    void SaveSnapshot();

    UINT GetWidth() const { return m_width; }
    UINT GetHeight() const { return m_height; }
    const wchar_t *GetTitle() const { return m_title.c_str(); }

private:
    static constexpr UINT FrameCount = 2;

    // Window
    UINT m_width;
    UINT m_height;
    std::wstring m_title;

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
    uint32_t m_frameCount = 0;

    // Texture resources
    std::vector<ComPtr<ID3D12Resource>> m_textures;
    std::vector<ComPtr<ID3D12Resource>> m_texUploads;

    std::vector<uint8_t> m_textureIsSRGB;
    uint32_t m_textureCount = 0;

    // Environment map (IBL)
    ComPtr<ID3D12Resource> m_envmap;
    ComPtr<ID3D12Resource> m_envmapUpload;
    std::vector<float> m_envmapPixels; // kept CPU-side for CDF build in Step 2
    uint32_t m_envmapWidth = 0;
    uint32_t m_envmapHeight = 0;
    bool m_envmapValid = false;

    // Envmap sampling CDFs (Distribution2D: marginal over rows,
    // conditional per row). Built on CPU from m_envmapPixels, uploaded as
    // ByteAddressBuffers.
    ComPtr<ID3D12Resource> m_envmapMarginalCdf;    // (H + 1) floats
    ComPtr<ID3D12Resource> m_envmapConditionalCdf; // H * (W + 1) floats

    // Volume density texture (heterogeneous media)
    ComPtr<ID3D12Resource> m_volumeTexture;
    ComPtr<ID3D12Resource> m_volumeUpload;
    bool m_hasVolumeTexture = false;

    // Ray tracing pipeline
    ComPtr<ID3D12StateObject> m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    ComPtr<ID3D12RootSignature> m_globalRootSig;

    // Output + descriptors
    ComPtr<ID3D12Resource> m_outputResource;
    ComPtr<ID3D12Resource> m_accumResource;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
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
    void CreateAccelerationStructure();
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
    void LoadVolume(const std::string &path);

    void RecomputeCameraPlane();

    // Render helpers
    void PopulateCommandList();
    void WaitForGpu(UINT frameIndex);
    void FlushCommandQueue();

    // Resource helpers
    ComPtr<ID3D12Resource> CreateBuffer(
        UINT64 size, D3D12_RESOURCE_FLAGS flags,
        D3D12_RESOURCE_STATES initialState, D3D12_HEAP_TYPE heapType);

    static std::vector<uint8_t> ReadFileBytes(const std::wstring &path);
    static std::wstring GetExeDirectory();
    static void ThrowIfFailed(HRESULT hr, const char *msg);
};