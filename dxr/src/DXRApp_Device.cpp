#include "DXRApp.h"
#include "Win32Application.h"

// D3D12 device, command queue, swap chain, RTV heap, command allocators,
// fence, buffer creation, and GPU synchronization.

void DXRApp::CreateDevice()
{
    UINT flags = 0;
#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> dc;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dc))))
        {
            dc->EnableDebugLayer();
            flags |= DXGI_CREATE_FACTORY_DEBUG;
            printf("[init] D3D12 debug layer enabled\n");
        }
    }
#endif
    ThrowIfFailed(CreateDXGIFactory2(flags, IID_PPV_ARGS(&m_factory)), "Factory");
    ComPtr<IDXGIAdapter1> adapter;
    std::string dxrProbeLog;
    for (UINT i = 0; m_factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                           IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++i)
    {
        DXGI_ADAPTER_DESC1 d;
        adapter->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;

        ComPtr<ID3D12Device5> candidateDevice;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&candidateDevice))))
            continue;

        D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
        HRESULT featureHr = candidateDevice->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5));
        if (FAILED(featureHr) || options5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
        {
            char adapterName[128];
            std::snprintf(adapterName, sizeof(adapterName), "%ls", d.Description);
            dxrProbeLog += "  - ";
            dxrProbeLog += adapterName;
            dxrProbeLog += ": ";
            dxrProbeLog += FAILED(featureHr) ? "failed DXR feature query" : "no DXR support";
            dxrProbeLog += "\n";
            continue;
        }

        m_device = candidateDevice;
        printf("[init] Adapter: %ls\n", d.Description);
        printf("[init] DXR tier: %s\n", options5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1 ? "1.1" : "1.0");
        break;
    }
    if (!m_device)
    {
        if (!dxrProbeLog.empty())
            throw std::runtime_error("No DXR-capable D3D12 adapter found. Checked adapters:\n" + dxrProbeLog);
        throw std::runtime_error("No D3D12 device");
    }
}

void DXRApp::CreateCommandQueue()
{
    D3D12_COMMAND_QUEUE_DESC d{};
    d.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(m_device->CreateCommandQueue(&d, IID_PPV_ARGS(&m_commandQueue)), "CmdQueue");
}

void DXRApp::CreateSwapChain()
{
    DXGI_SWAP_CHAIN_DESC1 d{};
    d.BufferCount = FrameCount;
    d.Width = m_width;
    d.Height = m_height;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    d.SampleDesc = {1, 0};
    ComPtr<IDXGISwapChain1> sc1;
    ThrowIfFailed(m_factory->CreateSwapChainForHwnd(m_commandQueue.Get(),
                                                    Win32Application::GetHwnd(), &d, nullptr, nullptr, &sc1),
                  "SwapChain");
    m_factory->MakeWindowAssociation(Win32Application::GetHwnd(), DXGI_MWA_NO_ALT_ENTER);
    ThrowIfFailed(sc1.As(&m_swapChain), "SwapChain QI");
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
}

void DXRApp::CreateRTVHeap()
{
    D3D12_DESCRIPTOR_HEAP_DESC d{};
    d.NumDescriptors = FrameCount;
    d.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&d, IID_PPV_ARGS(&m_rtvHeap)), "RTV heap");
    m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    auto h = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < FrameCount; i++)
    {
        ThrowIfFailed(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_renderTargets[i])), "GetBuffer");
        m_device->CreateRenderTargetView(m_renderTargets[i].Get(), nullptr, h);
        h.ptr += m_rtvDescriptorSize;
    }
}

void DXRApp::CreateCommandAllocatorsAndList()
{
    for (UINT i = 0; i < FrameCount; i++)
        ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                       IID_PPV_ARGS(&m_commandAllocators[i])),
                      "CmdAlloc");
    ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              m_commandAllocators[0].Get(), nullptr, IID_PPV_ARGS(&m_commandList)),
                  "CmdList");
    m_commandList->Close();
}

void DXRApp::CreateFence()
{
    ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)), "Fence");
    m_fenceValues[m_frameIndex] = 1;
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent)
        throw std::runtime_error("CreateEvent");
}

ComPtr<ID3D12Resource> DXRApp::CreateBuffer(UINT64 size, D3D12_RESOURCE_FLAGS flags,
                                            D3D12_RESOURCE_STATES state, D3D12_HEAP_TYPE heap)
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> b;
    ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state,
                                                    nullptr, IID_PPV_ARGS(&b)),
                  "CreateBuffer");
    return b;
}

void DXRApp::FlushCommandQueue()
{
    ThrowIfFailed(m_commandList->Close(), "Close");
    ID3D12CommandList *cl[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(1, cl);
    WaitForGpu(m_frameIndex);
}

void DXRApp::WaitForGpu(UINT fi)
{
    const UINT64 v = m_fenceValues[fi];
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), v), "Signal");
    ThrowIfFailed(m_fence->SetEventOnCompletion(v, m_fenceEvent), "Fence");
    WaitForSingleObject(m_fenceEvent, INFINITE);
    m_fenceValues[fi]++;
}
