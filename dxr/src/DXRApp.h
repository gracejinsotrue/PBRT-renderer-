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
    float pad2;
    float camVertical[3];
    float pad3;
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

    // Nori scene (owned)
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

    // Ray tracing pipeline
    ComPtr<ID3D12StateObject> m_rtStateObject;
    ComPtr<ID3D12StateObjectProperties> m_rtStateObjectProps;
    ComPtr<ID3D12RootSignature> m_globalRootSig;

    // Output + descriptors
    ComPtr<ID3D12Resource> m_outputResource;
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
