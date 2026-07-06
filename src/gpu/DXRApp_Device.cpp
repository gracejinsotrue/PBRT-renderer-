#include "DXRApp.h"
#include "Win32Application.h"
#include <algorithm>

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

void DXRApp::CreateProfiler()
{
    if (!m_profile)
        return;
    D3D12_QUERY_HEAP_DESC qd{};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = 2; // slot 0 = before DispatchRays, slot 1 = after
    ThrowIfFailed(m_device->CreateQueryHeap(&qd, IID_PPV_ARGS(&m_tsQueryHeap)), "QueryHeap");
    // Readback heap resources live permanently in COPY_DEST, which is what ResolveQueryData writes into.
    m_tsReadback = CreateBuffer(2 * sizeof(UINT64), D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);
    ThrowIfFailed(m_commandQueue->GetTimestampFrequency(&m_tsFrequency), "TSFreq");
    m_frameMs.reserve(4096);
    printf("[profile] GPU timestamp profiling ON (%.3f MHz timestamp clock)\n",
           (double)m_tsFrequency / 1e6);
}

void DXRApp::PrintProfileSummary()
{
    if (!m_profile || m_profileSummaryPrinted)
        return;
    m_profileSummaryPrinted = true;

    const size_t warmup = 8;
    if (m_frameMs.size() <= warmup)
    {
        printf("[profile] only %zu frames collected (<= %zu warmup) — no summary\n",
               m_frameMs.size(), warmup);
        return;
    }
    std::vector<double> s(m_frameMs.begin() + warmup, m_frameMs.end());
    std::sort(s.begin(), s.end());
    const size_t n = s.size();
    double sum = 0.0;
    for (double v : s)
        sum += v;
    const double mean = sum / (double)n;
    const double mn = s.front();
    const double median = s[n / 2];
    const double p95 = s[(size_t)(0.95 * (double)(n - 1))];
    printf("[profile] DispatchRays over %zu frames (dropped %zu warmup):\n", n, warmup);
    printf("[profile]   min %.4f  median %.4f  mean %.4f  p95 %.4f  (ms)\n",
           mn, median, mean, p95);
    // min is the least thermally-throttled sample
    printf("[profile]   min => %.1f dispatch/s\n", 1000.0 / mn);
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
