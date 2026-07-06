#include "DXRApp.h"
#include "Win32Application.h"
#include "Denoiser.h"

#include <scene.h>
#include <camera.h>
#include <filesystem/resolver.h>

#include <fstream>
#include <cstring>

using namespace nori;

void DXRApp::ThrowIfFailed(HRESULT hr, const char *msg)
{
    if (FAILED(hr))
    {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s (HRESULT 0x%08X)", msg, (unsigned)hr);
        throw std::runtime_error(buf);
    }
}

std::vector<uint8_t> DXRApp::ReadFileBytes(const std::wstring &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Failed to open shader file");
    auto size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> data((size_t)size);
    file.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

std::wstring DXRApp::GetExeDirectory()
{
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring dir(path);
    return dir.substr(0, dir.find_last_of(L"\\/") + 1);
}

DXRApp::DXRApp(const std::string &scenePath, bool headless)
    : m_scenePath(scenePath), m_title(L"nori-dxr"), m_headless(headless)
{
    LoadNoriScene();
    auto outputSize = m_noriScene->getCamera()->getOutputSize();
    m_width = (UINT)outputSize.x();
    m_height = (UINT)outputSize.y();
}

DXRApp::~DXRApp()
{
    delete m_noriScene;
}

void DXRApp::OnInit()
{
    CreateDevice();
    CreateCommandQueue();
    if (!m_headless)
    {
        CreateSwapChain();
        CreateRTVHeap();
    }
    CreateCommandAllocatorsAndList();
    CreateFence();
    CreateProfiler();
    CreateAccelerationStructure();
    SetupVolumes();
    CreateSceneBuffers();
    CreateTextures();
    CreateRaytracingPipeline();
    CreateOutputResource();
    CreateShaderTable();
    SetupCamera();
    printf("[init] DX12 + DXR initialization complete\n");
}

// OnUpdate, OnKeyDown/Up, OnMouseDown/Up/Move, SaveSnapshot, SaveSnapshotEXR
// are defined in DXRApp_Camera.cpp.

void DXRApp::OnRender()
{
    PopulateCommandList();
    ID3D12CommandList *cmdLists[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(1, cmdLists);

    if (!m_headless)
    {
        ThrowIfFailed(m_swapChain->Present(1, 0), "Present");
        const UINT64 cv = m_fenceValues[m_frameIndex];
        ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), cv), "Signal");
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
        if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
        {
            ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent), "Fence");
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }
        m_fenceValues[m_frameIndex] = cv + 1;
    }
    else
    {
        // Headless: just signal fence and wait
        const UINT64 cv = m_fenceValues[m_frameIndex];
        ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), cv), "Signal");
        if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
        {
            ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent), "Fence");
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }
        m_fenceValues[m_frameIndex] = cv + 1;
    }

    // Read back the DispatchRays timestamps. In headless the fence wait above is
    // fully synchronous so this frame's resolve has completed; in
    // windowed mode it may lag a frame, which is fine since work is identical.
    if (m_profile && m_tsReadback)
    {
        D3D12_RANGE rr{0, 2 * sizeof(UINT64)};
        void *p = nullptr;
        if (SUCCEEDED(m_tsReadback->Map(0, &rr, &p)) && p)
        {
            UINT64 t[2];
            memcpy(t, p, sizeof(t));
            D3D12_RANGE wr{0, 0};
            m_tsReadback->Unmap(0, &wr);
            if (t[1] > t[0] && m_tsFrequency)
                m_frameMs.push_back((double)(t[1] - t[0]) * 1000.0 / (double)m_tsFrequency);
        }
    }

    if (!m_headless && m_autoDenoise && m_nextDenoiseSpp <= kMaxDenoiseSpp &&
        m_frameCount >= m_nextDenoiseSpp)
    {
        DenoiseToViewport();
        m_nextDenoiseSpp *= 2;
    }

    // auto-save EXR and exit when the configured sample count is reached
    if (m_targetSamples > 0 && m_frameCount >= m_targetSamples)
    {
        SaveSnapshotEXR();
        if (!m_headless)
        {
            PostQuitMessage(0);
        }
    }
}

void DXRApp::OnDestroy()
{
    PrintProfileSummary();
    for (UINT i = 0; i < FrameCount; i++)
        WaitForGpu(i);
    if (m_fenceEvent)
        CloseHandle(m_fenceEvent);
}

// CreateDevice, CreateCommandQueue, CreateSwapChain, CreateRTVHeap,
// CreateCommandAllocatorsAndList, CreateFence, CreateBuffer, FlushCommandQueue,
// WaitForGpu are defined in DXRApp_Device.cpp.
// CreateAccelerationStructure, CreateSceneBuffers, SetupVolumes, LoadVolume
// are defined in DXRApp_Scene.cpp.
// LoadTexture, LoadEnvmap, BuildEnvmapDistribution, CreateTextures
// are defined in DXRApp_Textures.cpp.
// CreateRaytracingPipeline, CreateOutputResource, CreateShaderTable
// are defined in DXRApp_Pipeline.cpp.
// SetupCamera, RecomputeCameraPlane are defined in DXRApp_Camera.cpp.

void DXRApp::PopulateCommandList()
{
    auto *a = m_commandAllocators[m_frameIndex].Get();
    ThrowIfFailed(a->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(a, nullptr), "C");

    ID3D12DescriptorHeap *heaps[] = {m_srvUavHeap.Get()};
    m_commandList->SetDescriptorHeaps(1, heaps);
    m_commandList->SetComputeRootSignature(m_globalRootSig.Get());
    m_camera.frameCount = m_frameCount++;
    m_commandList->SetComputeRoot32BitConstants(0, sizeof(CameraConstants) / 4, &m_camera, 0);
    m_commandList->SetComputeRootDescriptorTable(1, m_srvUavHeap->GetGPUDescriptorHandleForHeapStart());

    D3D12_RESOURCE_BARRIER bb[2]{};
    bb[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bb[0].Transition.pResource = m_outputResource.Get();
    bb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    bb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bb[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(1, &bb[0]);

    // UAV barriers for the accumulation + AOV textures because we are doing a lot of cross-frame read-modify-write bullshit
    D3D12_RESOURCE_BARRIER uavBarriers[3]{};
    ID3D12Resource *uavRes[3] = {m_accumResource.Get(), m_albedoResource.Get(), m_normalResource.Get()};
    for (int i = 0; i < 3; i++)
    {
        uavBarriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavBarriers[i].UAV.pResource = uavRes[i];
    }
    m_commandList->ResourceBarrier(3, uavBarriers);

    m_commandList->SetPipelineState1(m_rtStateObject.Get());
    const UINT sa = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    auto ta = m_shaderTable->GetGPUVirtualAddress();
    D3D12_DISPATCH_RAYS_DESC dr{};
    dr.RayGenerationShaderRecord = {ta, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};

    dr.MissShaderTable = {ta + sa, sa * 2, sa};

    dr.HitGroupTable = {ta + sa * 3, sa * 2, sa}; // 2 hit groups (primary + shadow), stride = sa
    dr.Width = m_width;
    dr.Height = m_height;
    dr.Depth = 1;
    if (m_profile)
        m_commandList->EndQuery(m_tsQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    m_commandList->DispatchRays(&dr);
    if (m_profile)
        m_commandList->EndQuery(m_tsQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);

    bb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;

    if (!m_headless)
    {
        bb[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bb[1].Transition.pResource = m_renderTargets[m_frameIndex].Get();
        bb[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        bb[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        bb[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_commandList->ResourceBarrier(2, bb);

        // Present the denoised preview instead of the live noisy output when available. m_denoisedResource is kept in COPY_SOURCE.
        ID3D12Resource *presentSrc = (m_showDenoised && m_denoisedResource)
                                         ? m_denoisedResource.Get()
                                         : m_outputResource.Get();
        m_commandList->CopyResource(m_renderTargets[m_frameIndex].Get(), presentSrc);

        bb[0].Transition.pResource = m_renderTargets[m_frameIndex].Get();
        bb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        bb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        m_commandList->ResourceBarrier(1, &bb[0]);
    }
    else
    {
        m_commandList->ResourceBarrier(1, &bb[0]);
    }
    if (m_profile)
        m_commandList->ResolveQueryData(m_tsQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                        0, 2, m_tsReadback.Get(), 0);
    ThrowIfFailed(m_commandList->Close(), "Close");
}
