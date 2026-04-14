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
};

// Must match the HLSL GPUMaterial struct exactly, which is a scalar 4-byte packing
struct GPUMaterial
{
    uint32_t type; // 0=diffuse, 1=mirror, 2=dielectric, 3=microfacet
    float albedo[3];
    float intIOR;
    float extIOR;
    float alpha;
    uint32_t isEmitter;
    float radiance[3];
    uint32_t indexOffset;      // first index (in elements) in global index buffer
    uint32_t vertexOffset;     // first vertex (in elements) in global normal/pos buffer
    uint32_t indexCount;       // number of indices for this mesh
    uint32_t vertexCount;      // number of vertices for this mesh
    float surfaceArea;         // total mesh surface area (for emitter pdf)
    uint32_t emitterCdfOffset; // first entry index in emitter CDF buffer
};

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

    // Scene data buffers (for shader access)
    ComPtr<ID3D12Resource> m_materialBuffer;     // StructuredBuffer<GPUMaterial>
    ComPtr<ID3D12Resource> m_globalNormalBuffer; // ByteAddressBuffer — all normals
    ComPtr<ID3D12Resource> m_globalIndexBuffer;  // ByteAddressBuffer — all indices
    ComPtr<ID3D12Resource> m_globalVertexBuffer; // ByteAddressBuffer — all positions
    ComPtr<ID3D12Resource> m_emitterCdfBuffer;   // ByteAddressBuffer — emitter triangle area CDFs
    uint32_t m_meshCount = 0;
    uint32_t m_frameCount = 0;

    // Ray tracing pipeline
    ComPtr<ID3D12StateObject> m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    ComPtr<ID3D12RootSignature> m_globalRootSig;

    // Output + descriptors
    ComPtr<ID3D12Resource> m_outputResource;
    ComPtr<ID3D12Resource> m_accumResource; // R32G32B32A32_FLOAT accumulation buffer
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    UINT m_srvUavDescriptorSize = 0;

    ComPtr<ID3D12Resource> m_shaderTable;

    CameraConstants m_camera = {};

    void LoadNoriScene();
    void CreateDevice();
    void CreateCommandQueue();
    void CreateSwapChain();
    void CreateRTVHeap();
    void CreateCommandAllocatorsAndList();
    void CreateFence();
    void CreateAccelerationStructure();
    void CreateSceneBuffers();
    void CreateRaytracingPipeline();
    void CreateOutputResource();
    void CreateShaderTable();
    void SetupCamera();

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
