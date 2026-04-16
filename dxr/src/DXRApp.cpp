#include "DXRApp.h"
#include "Win32Application.h"

#include <nori/parser.h>
#include <nori/scene.h>
#include <nori/camera.h>
#include <nori/mesh.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/dpdf.h>
#include <filesystem/resolver.h>

#include <fstream>
#include <cmath>
#include <cstring>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
DXRApp::DXRApp(const std::string &scenePath)
    : m_scenePath(scenePath), m_title(L"nori-dxr")
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
void DXRApp::LoadNoriScene()
{
    filesystem::path xmlPath(m_scenePath);
    getFileResolver()->prepend(xmlPath.parent_path());

    NoriObject *root = loadFromXML(m_scenePath);
    if (root->getClassType() != NoriObject::EScene)
        throw std::runtime_error("Root object is not a Scene");

    m_noriScene = static_cast<Scene *>(root);
    printf("[nori] Scene loaded: %zu meshes\n", m_noriScene->getMeshes().size());
}

void DXRApp::OnInit()
{
    CreateDevice();
    CreateCommandQueue();
    CreateSwapChain();
    CreateRTVHeap();
    CreateCommandAllocatorsAndList();
    CreateFence();
    CreateAccelerationStructure();
    CreateSceneBuffers();
    CreateTextures();
    CreateRaytracingPipeline();
    CreateOutputResource();
    CreateShaderTable();
    SetupCamera();
    printf("[init] DX12 + DXR initialization complete\n");
}

void DXRApp::OnUpdate()
{
    // delta time
    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - m_lastFrameTime).count();
    m_lastFrameTime = now;
    dt = (dt > 0.1f) ? 0.1f : dt;

    // camera vectors
    float cy = cosf(m_camYaw), sy = sinf(m_camYaw);
    float cp = cosf(m_camPitch), sp = sinf(m_camPitch);
    float fwd[3] = {sy * cp, sp, cy * cp};
    float right[3] = {cy, 0.0f, -sy};
    float up[3] = {0.0f, 1.0f, 0.0f};

    float move = m_camSpeed * dt;
    bool moved = false;

    if (m_keys['W'])
    {
        for (int i = 0; i < 3; i++)
            m_camPos[i] += fwd[i] * move;
        moved = true;
    }
    if (m_keys['S'])
    {
        for (int i = 0; i < 3; i++)
            m_camPos[i] -= fwd[i] * move;
        moved = true;
    }
    if (m_keys['A'])
    {
        for (int i = 0; i < 3; i++)
            m_camPos[i] -= right[i] * move;
        moved = true;
    }
    if (m_keys['D'])
    {
        for (int i = 0; i < 3; i++)
            m_camPos[i] += right[i] * move;
        moved = true;
    }
    if (m_keys['Q'] || m_keys[VK_SPACE])
    {
        for (int i = 0; i < 3; i++)
            m_camPos[i] += up[i] * move;
        moved = true;
    }
    if (m_keys['E'] || m_keys[VK_SHIFT])
    {
        for (int i = 0; i < 3; i++)
            m_camPos[i] -= up[i] * move;
        moved = true;
    }

    if (moved || m_cameraDirty)
    {
        RecomputeCameraPlane();
        m_frameCount = 0;
        m_cameraDirty = false;
    }
}

// input handling

void DXRApp::OnKeyDown(UINT8 key)
{
    m_keys[key] = true;
    if (key == 'P')
        SaveSnapshot();
}

void DXRApp::OnKeyUp(UINT8 key)
{
    m_keys[key] = false;
}

void DXRApp::OnMouseDown(UINT button, int x, int y)
{
    if (button == 1) // right
    {
        m_mouseRightDown = true;
        m_lastMouse = {x, y};
    }
}

void DXRApp::OnMouseUp(UINT button, int x, int y)
{
    if (button == 1)
        m_mouseRightDown = false;
}

// for mouse movements for dxr camera
void DXRApp::OnMouseMove(int x, int y)
{
    if (m_mouseRightDown)
    {
        int dx = x - m_lastMouse.x;
        int dy = y - m_lastMouse.y;
        m_lastMouse = {x, y};

        m_camYaw += dx * m_mouseSensitivity;
        m_camPitch -= dy * m_mouseSensitivity;
        const float maxPitch = 1.5f;
        if (m_camPitch > maxPitch)
            m_camPitch = maxPitch;
        if (m_camPitch < -maxPitch)
            m_camPitch = -maxPitch;

        m_cameraDirty = true;
    }
}

void DXRApp::SaveSnapshot()
{
    WaitForGpu(m_frameIndex);

    D3D12_RESOURCE_DESC desc = m_outputResource->GetDesc();
    UINT64 rowPitch = ((desc.Width * 4 + 255) & ~255);
    UINT64 totalSize = rowPitch * desc.Height;

    auto readback = CreateBuffer(totalSize, D3D12_RESOURCE_FLAG_NONE,
                                 D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);

    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "C");

    // transition output to copy source
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = m_outputResource.Get();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dst.PlacedFootprint.Footprint.Width = (UINT)desc.Width;
    dst.PlacedFootprint.Footprint.Height = desc.Height;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;

    src.pResource = m_outputResource.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    FlushCommandQueue();

    // Map and write BMP
    uint8_t *data;
    readback->Map(0, nullptr, (void **)&data);

    char filename[256];
    snprintf(filename, sizeof(filename), "snapshot_%u.ppm", m_frameCount);

    FILE *f = fopen(filename, "wb");
    if (f)
    {
        fprintf(f, "P6\n%u %u\n255\n", (UINT)desc.Width, desc.Height);
        for (UINT y = 0; y < desc.Height; y++)
        {
            const uint8_t *row = data + y * rowPitch;
            for (UINT x = 0; x < (UINT)desc.Width; x++)
            {
                fwrite(row + x * 4, 1, 3, f); // RGB from RGBA
            }
        }
        fclose(f);
        printf("[snapshot] Saved %s (%u samples)\n", filename, m_frameCount);
    }

    readback->Unmap(0, nullptr);
}

void DXRApp::OnRender()
{
    PopulateCommandList();
    ID3D12CommandList *cmdLists[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(1, cmdLists);
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

void DXRApp::OnDestroy()
{
    for (UINT i = 0; i < FrameCount; i++)
        WaitForGpu(i);
    if (m_fenceEvent)
        CloseHandle(m_fenceEvent);
}

// Device, queue, swap chain, RTV, command list, fence
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

// Resource helpers
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

// build BLAS and TLAS
void DXRApp::CreateAccelerationStructure()
{
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "Alloc");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "CmdList");

    const auto &meshes = m_noriScene->getMeshes();
    m_meshGPU.resize(meshes.size());

    // scratch buffers must stay alive until FlushCommandQueue completes
    std::vector<ComPtr<ID3D12Resource>> tempBuffers;

    // builds one blas per mesh.
    for (size_t mi = 0; mi < meshes.size(); mi++)
    {
        const Mesh *mesh = meshes[mi];
        const auto &V = mesh->getVertexPositions();
        const auto &F = mesh->getIndices();

        UINT64 vbSize = V.size() * sizeof(float);
        UINT64 ibSize = F.size() * sizeof(uint32_t);

        // upload heaps are visible to both CPU and GPU and we need this type to transfer data
        m_meshGPU[mi].vertexBuffer = CreateBuffer(vbSize, D3D12_RESOURCE_FLAG_NONE,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
        void *mapped;
        m_meshGPU[mi].vertexBuffer->Map(0, nullptr, &mapped);
        memcpy(mapped, V.data(), vbSize);
        m_meshGPU[mi].vertexBuffer->Unmap(0, nullptr);

        m_meshGPU[mi].indexBuffer = CreateBuffer(ibSize, D3D12_RESOURCE_FLAG_NONE,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
        m_meshGPU[mi].indexBuffer->Map(0, nullptr, &mapped);
        memcpy(mapped, F.data(), ibSize);
        m_meshGPU[mi].indexBuffer->Unmap(0, nullptr);
        m_meshGPU[mi].indexCount = (uint32_t)F.size();

        printf("[accel] Mesh %zu: %u verts, %u tris\n",
               mi, (unsigned)V.cols(), (unsigned)F.cols());

        // NOW THAT THE RAW BYTES ARE IN VRAM,
        // define the triangle gometry and flag as opaque, which allows GPU to optimize its process by skipping alpha-testing for now

        D3D12_RAYTRACING_GEOMETRY_DESC geomDesc{};
        geomDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geomDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geomDesc.Triangles.VertexBuffer.StartAddress = m_meshGPU[mi].vertexBuffer->GetGPUVirtualAddress();
        geomDesc.Triangles.VertexBuffer.StrideInBytes = sizeof(float) * 3;
        geomDesc.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        geomDesc.Triangles.VertexCount = (UINT)V.cols();
        geomDesc.Triangles.IndexBuffer = m_meshGPU[mi].indexBuffer->GetGPUVirtualAddress();
        geomDesc.Triangles.IndexCount = (UINT)F.size();
        geomDesc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        inputs.NumDescs = 1;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.pGeometryDescs = &geomDesc;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO pre{};
        m_device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &pre);
        // use the size provided by the driver to allocate actual VRAM buffer
        // we put them in default heap so they reside in faster GPU memory
        // also push scratch buffer
        auto blasResult = CreateBuffer(pre.ResultDataMaxSizeInBytes,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_HEAP_TYPE_DEFAULT);
        auto blasScratch = CreateBuffer(pre.ScratchDataSizeInBytes,
                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                        D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
        tempBuffers.push_back(blasScratch);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd{};
        bd.Inputs = inputs;
        bd.DestAccelerationStructureData = blasResult->GetGPUVirtualAddress();
        bd.ScratchAccelerationStructureData = blasScratch->GetGPUVirtualAddress();
        m_commandList->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);

        // add an unordered access view barrier to have thhe GPU wait until all read/write operation in blasResult are fully complete b/f executing any subsequent commands that may try to access it

        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = blasResult.Get();
        m_commandList->ResourceBarrier(1, &uav);

        m_meshGPU[mi].blas = blasResult;
    }

    // tlas
    // create d3d12 raytracing structs on cpu. it looks like Nori doesn't use instancing (todo: fix later maybe but not a big deal).
    // so just create one instansce per mesh.
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances(meshes.size());
    for (size_t i = 0; i < meshes.size(); i++)
    {
        auto &inst = instances[i];
        memset(&inst, 0, sizeof(inst));
        inst.Transform[0][0] = 1;
        inst.Transform[1][1] = 1;
        inst.Transform[2][2] = 1;
        inst.InstanceID = (UINT)i;
        inst.InstanceMask = 0xFF;
        inst.AccelerationStructure = m_meshGPU[i].blas->GetGPUVirtualAddress();
    }

    // upload instance array to vram!
    // allocate upload buffer on GPU and use memcpy the array of instances to VRAM !

    UINT64 instSize = instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    auto instBuf = CreateBuffer(instSize, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
    tempBuffers.push_back(instBuf);
    void *im;
    instBuf->Map(0, nullptr, &im);
    memcpy(im, instances.data(), instSize);
    instBuf->Unmap(0, nullptr);

    // describe tlas to dxr

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS ti{};
    ti.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    ti.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    ti.NumDescs = (UINT)meshes.size();
    ti.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    ti.InstanceDescs = instBuf->GetGPUVirtualAddress();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tp{};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&ti, &tp);

    m_tlas = CreateBuffer(tp.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, D3D12_HEAP_TYPE_DEFAULT);
    auto tlasScratch = CreateBuffer(tp.ScratchDataSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_TYPE_DEFAULT);
    tempBuffers.push_back(tlasScratch);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tb{};
    tb.Inputs = ti;
    tb.DestAccelerationStructureData = m_tlas->GetGPUVirtualAddress();
    tb.ScratchAccelerationStructureData = tlasScratch->GetGPUVirtualAddress();
    m_commandList->BuildRaytracingAccelerationStructure(&tb, 0, nullptr);

    FlushCommandQueue();
    printf("[accel] %zu BLAS + TLAS built\n", meshes.size());
}

// Scene data buffers. material properties, normals, indices
void DXRApp::CreateSceneBuffers()
{
    const auto &meshes = m_noriScene->getMeshes();
    m_meshCount = (uint32_t)meshes.size();

    // Pass 1: compute total sizes and fill GPUMaterial array
    std::vector<GPUMaterial> materials(m_meshCount);
    uint32_t totalVertices = 0;
    uint32_t totalIndices = 0;

    for (uint32_t i = 0; i < m_meshCount; i++)
    {
        const Mesh *mesh = meshes[i];
        const BSDF *bsdf = mesh->getBSDF();
        BSDFGPUData gd = bsdf->getGPUData();

        GPUMaterial &mat = materials[i];
        memset(&mat, 0, sizeof(mat));
        mat.type = (uint32_t)gd.type;
        mat.albedo[0] = gd.albedo[0];
        mat.albedo[1] = gd.albedo[1];
        mat.albedo[2] = gd.albedo[2];
        mat.intIOR = gd.intIOR;
        mat.extIOR = gd.extIOR;
        mat.alpha = gd.alpha;

        if (mesh->isEmitter())
        {
            mat.isEmitter = 1;
            auto Le = mesh->getEmitter()->getRadiance();
            mat.radiance[0] = Le.r();
            mat.radiance[1] = Le.g();
            mat.radiance[2] = Le.b();
        }

        uint32_t vc = (uint32_t)mesh->getVertexCount();
        uint32_t ic = (uint32_t)mesh->getTriangleCount() * 3;
        mat.vertexOffset = totalVertices;
        mat.indexOffset = totalIndices;
        mat.vertexCount = vc;
        mat.indexCount = ic;
        mat.surfaceArea = mesh->getDiscretePDF().getSum();
        mat.emitterCdfOffset = 0;
        mat.albedoTexIndex = 0xFFFFFFFF;
        mat.normalTexIndex = 0xFFFFFFFF;
        mat.roughnessTexIndex = 0xFFFFFFFF;
        mat.metallicTexIndex = 0xFFFFFFFF;

        totalVertices += vc;
        totalIndices += ic;

        printf("[scene] Mesh %u: type=%u verts=%u tris=%u emitter=%u area=%.4f\n",
               i, mat.type, vc, ic / 3, mat.isEmitter, mat.surfaceArea);

        printf("[scene] Mesh %u: V.cols=%u N.cols=%u N.size=%zu V.size=%zu\n",
               i, (unsigned)meshes[i]->getVertexPositions().cols(),
               (unsigned)meshes[i]->getVertexNormals().cols(),
               meshes[i]->getVertexNormals().size(),
               meshes[i]->getVertexPositions().size());
    }

    printf("[scene] GPUMaterial size = %zu bytes\n", sizeof(GPUMaterial));

    // Build emitter CDF buffer from Nori's DiscretePDF
    // Concatenate normalized CDFs for all emitter meshes
    std::vector<float> allCdfData;
    for (uint32_t i = 0; i < m_meshCount; i++)
    {
        if (!materials[i].isEmitter)
            continue;
        const auto &cdf = meshes[i]->getDiscretePDF().getCDF();
        materials[i].emitterCdfOffset = (uint32_t)allCdfData.size();
        allCdfData.insert(allCdfData.end(), cdf.begin(), cdf.end());
        printf("[scene] Emitter %u: CDF has %zu entries, totalArea=%.4f\n",
               i, cdf.size(), materials[i].surfaceArea);
    }

    // Upload material structured buffer
    {
        UINT64 sz = m_meshCount * sizeof(GPUMaterial);
        m_materialBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                        D3D12_RESOURCE_STATE_GENERIC_READ,
                                        D3D12_HEAP_TYPE_UPLOAD);
        void *p;
        m_materialBuffer->Map(0, nullptr, &p);
        memcpy(p, materials.data(), sz);
        m_materialBuffer->Unmap(0, nullptr);
    }

    // concatenate vertex normals into one buffer
    {
        UINT64 sz = totalVertices * 3 * sizeof(float);
        m_globalNormalBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                            D3D12_RESOURCE_STATE_GENERIC_READ,
                                            D3D12_HEAP_TYPE_UPLOAD);
        uint8_t *dst;
        m_globalNormalBuffer->Map(0, nullptr, (void **)&dst);
        memset(dst, 0, sz);
        for (uint32_t i = 0; i < m_meshCount; i++)
        {
            const auto &N = meshes[i]->getVertexNormals();
            uint32_t vc = materials[i].vertexCount;
            if (N.size() > 0)
            {
                UINT64 bytes = N.size() * sizeof(float);
                memcpy(dst, N.data(), bytes);
            }
            dst += vc * 3 * sizeof(float); // always advance by vertex count
        }
        m_globalNormalBuffer->Unmap(0, nullptr);
    }

    // Concatenate triangle indices into one buffer. Indices are LOCAL per mesh!!, looked up with vertexOffset in the shader
    {
        UINT64 sz = totalIndices * sizeof(uint32_t);
        m_globalIndexBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                           D3D12_RESOURCE_STATE_GENERIC_READ,
                                           D3D12_HEAP_TYPE_UPLOAD);
        uint8_t *dst;
        m_globalIndexBuffer->Map(0, nullptr, (void **)&dst);
        for (uint32_t i = 0; i < m_meshCount; i++)
        {
            const auto &F = meshes[i]->getIndices();
            UINT64 bytes = F.size() * sizeof(uint32_t);
            memcpy(dst, F.data(), bytes);
            dst += bytes;
        }
        m_globalIndexBuffer->Unmap(0, nullptr);
    }

    // Concatenate vertex positions into one buffer, needed for emitter sampling
    {
        UINT64 sz = totalVertices * 3 * sizeof(float);
        m_globalVertexBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                            D3D12_RESOURCE_STATE_GENERIC_READ,
                                            D3D12_HEAP_TYPE_UPLOAD);
        uint8_t *dst;
        m_globalVertexBuffer->Map(0, nullptr, (void **)&dst);
        for (uint32_t i = 0; i < m_meshCount; i++)
        {
            const auto &V = meshes[i]->getVertexPositions();
            UINT64 bytes = V.size() * sizeof(float);
            memcpy(dst, V.data(), bytes);
            dst += bytes;
        }
        m_globalVertexBuffer->Unmap(0, nullptr);
    }

    // Upload emitter CDF buffer
    if (!allCdfData.empty())
    {
        UINT64 sz = allCdfData.size() * sizeof(float);
        m_emitterCdfBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ,
                                          D3D12_HEAP_TYPE_UPLOAD);
        void *p;
        m_emitterCdfBuffer->Map(0, nullptr, &p);
        memcpy(p, allCdfData.data(), sz);
        m_emitterCdfBuffer->Unmap(0, nullptr);
    }

    printf("[scene] Buffers uploaded: %u materials, %u verts, %u indices, %zu CDF entries\n",
           m_meshCount, totalVertices, totalIndices, allCdfData.size());

    // Concatenate vertex texture coordinates
    {
        UINT64 sz = totalVertices * 2 * sizeof(float);
        m_globalTexCoordBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                              D3D12_RESOURCE_STATE_GENERIC_READ,
                                              D3D12_HEAP_TYPE_UPLOAD);
        uint8_t *dst;
        m_globalTexCoordBuffer->Map(0, nullptr, (void **)&dst);
        memset(dst, 0, sz);
        for (uint32_t i = 0; i < m_meshCount; i++)
        {
            const auto &UV = meshes[i]->getVertexTexCoords();
            uint32_t vc = materials[i].vertexCount;
            if (UV.cols() > 0)
            {
                // UV is 2×N column-major (all U's then all V's).
                // Shader expects interleaved [u0,v0, u1,v1, ...].
                float *out = reinterpret_cast<float *>(dst);
                for (uint32_t v = 0; v < vc; v++)
                {
                    out[v * 2 + 0] = UV(0, v); // u
                    out[v * 2 + 1] = UV(1, v); // v
                }
            }
            dst += vc * 2 * sizeof(float);
        }
        m_globalTexCoordBuffer->Unmap(0, nullptr);
        printf("[scene] UV buffer uploaded: %u vertices\n", totalVertices);
    }
}

// Texture loading via stb_image lirbary
uint32_t DXRApp::LoadTexture(const std::string &path)
{
    int w, h, channels;
    unsigned char *pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!pixels)
    {
        printf("[texture] Failed to load: %s\n", path.c_str());
        return 0xFFFFFFFF;
    }

    uint32_t texIndex = (uint32_t)m_textures.size();
    UINT64 rowPitch = ((w * 4 + 255) & ~255); // D3D12 requires 256-byte row alignment
    UINT64 uploadSize = rowPitch * h;

    // Create the GPU texture resource on default heap vram
    D3D12_HEAP_PROPERTIES defaultHeap{};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w;
    td.Height = h;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags = D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> texture;
    ThrowIfFailed(m_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &td,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&texture)),
                  "Create texture");

    // Create upload buffer
    auto uploadBuf = CreateBuffer(uploadSize, D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);

    // Copy pixel data into upload buffer
    uint8_t *mapped;
    uploadBuf->Map(0, nullptr, (void **)&mapped);
    for (int y = 0; y < h; y++)
        memcpy(mapped + y * rowPitch, pixels + y * w * 4, w * 4);
    uploadBuf->Unmap(0, nullptr);
    stbi_image_free(pixels);

    // Copy from upload buffer to texture
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = texture.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;

    src.pResource = uploadBuf.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src.PlacedFootprint.Footprint.Width = w;
    src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;

    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    // Transition to shader resource
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(1, &barrier);

    m_textures.push_back(texture);
    m_texUploads.push_back(uploadBuf);

    printf("[texture] Loaded %s (%dx%d) as texture %u\n", path.c_str(), w, h, texIndex);
    return texIndex;
}

void DXRApp::CreateTextures()
{
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "Alloc");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "CmdList");

    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = 1;
        td.Height = 1;
        td.DepthOrArraySize = 1;
        td.MipLevels = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        ComPtr<ID3D12Resource> dummyTex;
        ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&dummyTex)),
                      "Dummy texture");

        auto uploadBuf = CreateBuffer(256, D3D12_RESOURCE_FLAG_NONE,
                                      D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
        uint8_t *mapped;
        uploadBuf->Map(0, nullptr, (void **)&mapped);
        uint8_t white[4] = {255, 255, 255, 255};
        memcpy(mapped, white, 4);
        uploadBuf->Unmap(0, nullptr);

        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = dummyTex.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = uploadBuf.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        src.PlacedFootprint.Footprint.Width = 1;
        src.PlacedFootprint.Footprint.Height = 1;
        src.PlacedFootprint.Footprint.Depth = 1;
        src.PlacedFootprint.Footprint.RowPitch = 256;
        m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = dummyTex.Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_commandList->ResourceBarrier(1, &barrier);

        m_textures.insert(m_textures.begin(), dummyTex); // slot 0 = dummy
        m_texUploads.push_back(uploadBuf);
    }

    // Load textures referenced by materials w/ deduplication
    const auto &meshes = m_noriScene->getMeshes();
    std::unordered_map<std::string, uint32_t> texCache;

    auto loadIfNotEmpty = [&](const std::string &relPath) -> uint32_t
    {
        if (relPath.empty())
            return 0xFFFFFFFF;
        filesystem::path resolved = getFileResolver()->resolve(relPath);
        std::string key = resolved.str();
        auto it = texCache.find(key);
        if (it != texCache.end())
            return it->second;
        uint32_t idx = LoadTexture(key);
        if (idx != 0xFFFFFFFF)
            texCache[key] = idx;
        return idx;
    };

    // per-mesh texture indices
    struct MeshTexIndices
    {
        uint32_t albedo, normal, roughness, metallic;
    };
    std::vector<MeshTexIndices> meshTexIndices(meshes.size());

    for (uint32_t i = 0; i < (uint32_t)meshes.size(); i++)
    {
        BSDFGPUData gd = meshes[i]->getBSDF()->getGPUData();
        meshTexIndices[i].albedo = loadIfNotEmpty(gd.albedoTexture);
        meshTexIndices[i].normal = loadIfNotEmpty(gd.normalTexture);
        meshTexIndices[i].roughness = loadIfNotEmpty(gd.roughnessTexture);
        meshTexIndices[i].metallic = loadIfNotEmpty(gd.metallicTexture);
    }

    m_textureCount = (uint32_t)m_textures.size();

    FlushCommandQueue();
    m_texUploads.clear();

    // patch texture indices into the material buffer
    {
        GPUMaterial *mats = nullptr;
        m_materialBuffer->Map(0, nullptr, (void **)&mats);
        for (uint32_t i = 0; i < (uint32_t)meshes.size(); i++)
        {
            mats[i].albedoTexIndex = meshTexIndices[i].albedo;
            mats[i].normalTexIndex = meshTexIndices[i].normal;
            mats[i].roughnessTexIndex = meshTexIndices[i].roughness;
            mats[i].metallicTexIndex = meshTexIndices[i].metallic;
        }
        m_materialBuffer->Unmap(0, nullptr);
    }

    printf("[texture] %u textures loaded (%u real + 1 dummy)\n",
           m_textureCount, m_textureCount - 1);
}

// camera
void DXRApp::SetupCamera()
{
    const Camera *cam = m_noriScene->getCamera();
    float w = (float)m_width, h = (float)m_height;

    // sample corner rays to extract position and orientation
    Ray3f ray_bl, ray_br, ray_tl, ray_ctr;
    Point2f ap(0, 0);
    cam->sampleRay(ray_bl, Point2f(0, h), ap);
    cam->sampleRay(ray_br, Point2f(w, h), ap);
    cam->sampleRay(ray_tl, Point2f(0, 0), ap);
    cam->sampleRay(ray_ctr, Point2f(w / 2, h / 2), ap);

    Point3f pos = ray_bl.o;
    Vector3f fwd = ray_ctr.d.normalized();

    // to extract yaw and pitch
    m_camPos[0] = pos.x();
    m_camPos[1] = pos.y();
    m_camPos[2] = pos.z();
    m_camYaw = atan2f(fwd.x(), fwd.z());
    m_camPitch = asinf(fwd.y());
    auto project = [&](const Ray3f &r) -> Point3f
    {
        float t = 1.0f / r.d.dot(fwd);
        return pos + r.d * t;
    };
    Point3f P_bl = project(ray_bl);
    Point3f P_br = project(ray_br);
    Point3f P_tl = project(ray_tl);
    float halfHeight = (P_tl - P_bl).norm() * 0.5f;
    m_camFovY = 2.0f * atanf(halfHeight);

    // Detect whether Nori's camera has a horizontal flip relative to the DXR yaw/pitch model.
    // DXR's "right" = (cos(yaw), 0, -sin(yaw)) = world_up x fwd = the camera-LEFT direction.
    // If Nori's horizontal axis (P_br - P_bl) is aligned with DXR right, the scene uses an
    // implicit x-flip (e.g. <scale value="-1,1,1"/>), so m_camXFlip = +1 (no change needed).
    // If anti-aligned (e.g. Ajax with no scale correction), m_camXFlip = -1 to negate the
    // horizontal axis and produce an image consistent with Nori's output.
    {
        float cy = cosf(m_camYaw), sy = sinf(m_camYaw);
        Vector3f dxrRight(cy, 0.0f, -sy);
        Vector3f noriHoriz = (P_br - P_bl).normalized();
        m_camXFlip = (noriHoriz.dot(dxrRight) >= 0.0f) ? 1.0f : -1.0f;
    }

    m_camera.meshCount = m_meshCount;
    m_camera.frameCount = 0;
    m_cameraDirty = true;

    m_lastFrameTime = std::chrono::high_resolution_clock::now();

    RecomputeCameraPlane();

    printf("[camera] pos=(%.3f,%.3f,%.3f) yaw=%.1f pitch=%.1f fov=%.1f\n",
           m_camPos[0], m_camPos[1], m_camPos[2],
           m_camYaw * 180.0f / 3.14159f, m_camPitch * 180.0f / 3.14159f,
           m_camFovY * 180.0f / 3.14159f);
    printf("[camera] Controls: WASD=move, QE=up/down, RightClick+drag=look, P=snapshot\n");
}

// recompute the camera image plane from yaw/pitch/position/fov
void DXRApp::RecomputeCameraPlane()
{
    float cy = cosf(m_camYaw), sy = sinf(m_camYaw);
    float cp = cosf(m_camPitch), sp = sinf(m_camPitch);

    // Forward = (sin(yaw)*cos(pitch), sin(pitch), cos(yaw)*cos(pitch))
    float fwd[3] = {sy * cp, sp, cy * cp};

    // World up = (0,1,0)
    // Right = normalize(fwd x up)
    float right[3] = {cy, 0.0f, -sy};

    // Camera up = fwd x right
    // NOT right x fwd, which would point down
    float up[3] = {
        fwd[1] * right[2] - fwd[2] * right[1],
        fwd[2] * right[0] - fwd[0] * right[2],
        fwd[0] * right[1] - fwd[1] * right[0]};

    float aspect = (float)m_width / (float)m_height;
    float halfH = tanf(m_camFovY * 0.5f);
    float halfW = halfH * aspect;

    float llc[3], horiz[3], vert[3];
    for (int i = 0; i < 3; i++)
    {
        llc[i] = m_camPos[i] + fwd[i] - m_camXFlip * halfW * right[i] - halfH * up[i];
        horiz[i] = m_camXFlip * 2.0f * halfW * right[i];
        vert[i] = 2.0f * halfH * up[i];
    }

    memcpy(m_camera.camPos, m_camPos, sizeof(float) * 3);
    memcpy(m_camera.camLowerLeftCorner, llc, sizeof(float) * 3);
    memcpy(m_camera.camHorizontal, horiz, sizeof(float) * 3);
    memcpy(m_camera.camVertical, vert, sizeof(float) * 3);
}

// Pipeline, output, shader table, render
void DXRApp::CreateRaytracingPipeline()
{
    // Compute total SRV count: 7 fixed (t0-t6) + N textures (t7+)
    UINT totalSRVs = 7 + m_textureCount;

    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[0].NumDescriptors = 2; // u0=output, u1=accumulation
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = totalSRVs; // t0-t6 fixed + t7..tN textures
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;

    D3D12_ROOT_PARAMETER rp[2]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[0].Constants.ShaderRegister = 0;
    rp[0].Constants.Num32BitValues = sizeof(CameraConstants) / 4;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[1].DescriptorTable.NumDescriptorRanges = 2;
    rp[1].DescriptorTable.pDescriptorRanges = ranges;
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Static sampler for texture sampling (bilinear, wrap)
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.ShaderRegister = 0; // s0
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 2;
    rsd.pParameters = rp;
    rsd.NumStaticSamplers = 1;
    rsd.pStaticSamplers = &sampler;
    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err), "RootSig serialize");
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                IID_PPV_ARGS(&m_globalRootSig)),
                  "RootSig");

    auto blob = ReadFileBytes(GetExeDirectory() + L"Shaders.cso");
    printf("[pipeline] Shader: %zu bytes\n", blob.size());

    D3D12_STATE_SUBOBJECT so[5]{};
    int idx = 0;
    D3D12_DXIL_LIBRARY_DESC ld{};
    ld.DXILLibrary = {blob.data(), blob.size()};
    so[idx].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    so[idx++].pDesc = &ld;
    D3D12_HIT_GROUP_DESC hg{};
    hg.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
    hg.ClosestHitShaderImport = L"ClosestHit";
    hg.HitGroupExport = L"HitGroup";
    so[idx].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    so[idx++].pDesc = &hg;
    D3D12_RAYTRACING_SHADER_CONFIG sc{};
    sc.MaxPayloadSizeInBytes = 48; // HitPayload: 11 fields x 4 bytes = 44, rounded up
    sc.MaxAttributeSizeInBytes = 8;
    so[idx].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    so[idx++].pDesc = &sc;
    D3D12_GLOBAL_ROOT_SIGNATURE grs{};
    grs.pGlobalRootSignature = m_globalRootSig.Get();
    so[idx].Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    so[idx++].pDesc = &grs;
    D3D12_RAYTRACING_PIPELINE_CONFIG pc{};
    pc.MaxTraceRecursionDepth = 1;
    so[idx].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    so[idx++].pDesc = &pc;

    D3D12_STATE_OBJECT_DESC sod{};
    sod.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    sod.NumSubobjects = idx;
    sod.pSubobjects = so;
    ThrowIfFailed(m_device->CreateStateObject(&sod, IID_PPV_ARGS(&m_rtStateObject)), "StateObject");
    ThrowIfFailed(m_rtStateObject->QueryInterface(IID_PPV_ARGS(&m_rtStateObjectProps)), "SOProps");
    printf("[pipeline] State object created\n");
}

void DXRApp::CreateOutputResource()
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    // output UAV texture R8G8B8A8_UNORM
    {
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = m_width;
        td.Height = m_height;
        td.DepthOrArraySize = 1;
        td.MipLevels = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&m_outputResource)),
                      "Output tex");
    }

    // Accumulation UAV texture
    {
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = m_width;
        td.Height = m_height;
        td.DepthOrArraySize = 1;
        td.MipLevels = 1;
        td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        td.SampleDesc.Count = 1;
        td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                        IID_PPV_ARGS(&m_accumResource)),
                      "Accum tex");
    }

    // Descriptor heap: 9 + N texture slots
    //  [0] u0  UAV output texture
    //  [1] u1  UAV accumulation texture
    //  [2] t0  SRV TLAS
    //  [3] t1  SRV material structured buffer
    //  [4] t2  SRV global vertex normals (raw)
    //  [5] t3  SRV global index buffer (raw)
    //  [6] t4  SRV global vertex positions (raw)
    //  [7] t5  SRV emitter CDF (raw)
    //  [8] t6  SRV global UV buffer (raw)
    //  [9+] t7+ SRV textures
    UINT totalDescriptors = 9 + m_textureCount;
    D3D12_DESCRIPTOR_HEAP_DESC dhd{};
    dhd.NumDescriptors = totalDescriptors;
    dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(&m_srvUavHeap)), "SRV/UAV heap");
    m_srvUavDescriptorSize = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto h = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();

    // [0] UAV — output texture
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        m_device->CreateUnorderedAccessView(m_outputResource.Get(), nullptr, &ud, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    // [1] UAV — accumulation texture
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        m_device->CreateUnorderedAccessView(m_accumResource.Get(), nullptr, &ud, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    // [2] SRV — TLAS
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.RaytracingAccelerationStructure.Location = m_tlas->GetGPUVirtualAddress();
        m_device->CreateShaderResourceView(nullptr, &sd, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    // [3] SRV — material structured buffer (t1)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Buffer.NumElements = m_meshCount;
        sd.Buffer.StructureByteStride = sizeof(GPUMaterial);
        m_device->CreateShaderResourceView(m_materialBuffer.Get(), &sd, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    // Helper for raw (ByteAddressBuffer) SRVs
    auto createRawSRV = [&](ID3D12Resource *buf)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = DXGI_FORMAT_R32_TYPELESS;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Buffer.NumElements = (UINT)(buf->GetDesc().Width / 4);
        sd.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        m_device->CreateShaderResourceView(buf, &sd, h);
        h.ptr += m_srvUavDescriptorSize;
    };

    // [4] SRV — global vertex normals (t2)
    createRawSRV(m_globalNormalBuffer.Get());

    // [5] SRV — global index buffer (t3)
    createRawSRV(m_globalIndexBuffer.Get());

    // [6] SRV — global vertex positions (t4)
    createRawSRV(m_globalVertexBuffer.Get());

    // [7] SRV — emitter CDF (t5)
    if (m_emitterCdfBuffer)
        createRawSRV(m_emitterCdfBuffer.Get());
    else
        h.ptr += m_srvUavDescriptorSize; // skip slot even if no CDF buffer

    // [8] SRV — global UV buffer (t6)
    createRawSRV(m_globalTexCoordBuffer.Get());

    // [9+] SRV — textures (t7+)
    for (uint32_t i = 0; i < m_textureCount; i++)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        m_device->CreateShaderResourceView(m_textures[i].Get(), &sd, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "C");
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = m_outputResource.Get();
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(1, &b);
    FlushCommandQueue();
    printf("[output] Output + accum textures + %u descriptors created\n", totalDescriptors);
}

void DXRApp::CreateShaderTable()
{
    const UINT id = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    const UINT al = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    // 4 entries: [RayGen] [Miss] [ShadowMiss] [HitGroup]
    m_shaderTable = CreateBuffer(al * 4, D3D12_RESOURCE_FLAG_NONE,
                                 D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
    void *rg = m_rtStateObjectProps->GetShaderIdentifier(L"RayGen");
    void *ms = m_rtStateObjectProps->GetShaderIdentifier(L"Miss");
    void *sm = m_rtStateObjectProps->GetShaderIdentifier(L"ShadowMiss");
    void *hg = m_rtStateObjectProps->GetShaderIdentifier(L"HitGroup");
    if (!rg || !ms || !sm || !hg)
        throw std::runtime_error("Shader ID not found");
    uint8_t *m;
    m_shaderTable->Map(0, nullptr, (void **)&m);
    memset(m, 0, al * 4);
    memcpy(m, rg, id);          // [0] RayGen
    memcpy(m + al, ms, id);     // [1] Miss
    memcpy(m + al * 2, sm, id); // [2] ShadowMiss
    memcpy(m + al * 3, hg, id); // [3] HitGroup
    m_shaderTable->Unmap(0, nullptr);
    printf("[shader table] Created (4 entries)\n");
}

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

    D3D12_RESOURCE_BARRIER uavBarrier{};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = m_accumResource.Get();
    m_commandList->ResourceBarrier(1, &uavBarrier);

    m_commandList->SetPipelineState1(m_rtStateObject.Get());
    const UINT sa = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    auto ta = m_shaderTable->GetGPUVirtualAddress();
    D3D12_DISPATCH_RAYS_DESC dr{};
    dr.RayGenerationShaderRecord = {ta, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};

    dr.MissShaderTable = {ta + sa, sa * 2, sa};

    dr.HitGroupTable = {ta + sa * 3, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES};
    dr.Width = m_width;
    dr.Height = m_height;
    dr.Depth = 1;
    m_commandList->DispatchRays(&dr);

    bb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    bb[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bb[1].Transition.pResource = m_renderTargets[m_frameIndex].Get();
    bb[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    bb[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    bb[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(2, bb);

    m_commandList->CopyResource(m_renderTargets[m_frameIndex].Get(), m_outputResource.Get());

    bb[0].Transition.pResource = m_renderTargets[m_frameIndex].Get();
    bb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    bb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    m_commandList->ResourceBarrier(1, &bb[0]);
    ThrowIfFailed(m_commandList->Close(), "Close");
}
