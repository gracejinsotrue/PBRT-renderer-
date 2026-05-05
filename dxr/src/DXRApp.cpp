#include "DXRApp.h"
#include "Win32Application.h"

#include <nori/parser.h>
#include <nori/scene.h>
#include <nori/camera.h>
#include <nori/mesh.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/medium.h>
#include <nori/dpdf.h>
#include <filesystem/resolver.h>

#include <algorithm>
#include <fstream>
#include <cmath>
#include <cstring>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// TODO: probably split some of this code into other files amazing
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
    SetupVolumes();
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
    if (key == 'L')
    {
        float cy = cosf(m_camYaw), sy = sinf(m_camYaw);
        float cp = cosf(m_camPitch), sp = sinf(m_camPitch);
        float fwd[3] = {sy * cp, sp, cy * cp};
        // target = origin + forward (1 unit ahead)
        float tx = m_camPos[0] + fwd[0];
        float ty = m_camPos[1] + fwd[1];
        float tz = m_camPos[2] + fwd[2];
        printf("[camera] pos=(%.4f, %.4f, %.4f)  yaw=%.2f deg  pitch=%.2f deg\n",
               m_camPos[0], m_camPos[1], m_camPos[2],
               m_camYaw * 180.0f / 3.14159f, m_camPitch * 180.0f / 3.14159f);
        printf("[camera] scene.xml lookat snippet:\n");
        printf("  <lookat origin=\"%.4f, %.4f, %.4f\"\n", m_camPos[0], m_camPos[1], m_camPos[2]);
        printf("          target=\"%.4f, %.4f, %.4f\"\n", tx, ty, tz);
        printf("          up=\"0, 1, 0\"/>\n");
    }
}

void DXRApp::OnKeyUp(UINT8 key)
{
    m_keys[key] = false;
}

void DXRApp::OnMouseDown(UINT button, int x, int y)
{
    if (button == 0) // left
    {
        m_mouseLeftDown = true;
        m_lastMouse = {x, y};
    }
    if (button == 1) // right
    {
        m_mouseRightDown = true;
        m_lastMouse = {x, y};
    }
}

void DXRApp::OnMouseUp(UINT button, int x, int y)
{
    if (button == 0)
        m_mouseLeftDown = false;
    if (button == 1)
        m_mouseRightDown = false;
}

// for mouse movements for dxr camera
void DXRApp::OnMouseMove(int x, int y)
{
    if (m_mouseLeftDown && m_mouseRightDown)
    {
        int dx = x - m_lastMouse.x;
        int dy = y - m_lastMouse.y;
        m_lastMouse = {x, y};

        float cy = cosf(m_camYaw), sy = sinf(m_camYaw);
        float cp = cosf(m_camPitch), sp = sinf(m_camPitch);
        float fwd[3] = {sy * cp, sp, cy * cp};
        float right[3] = {cy, 0.0f, -sy};
        float up[3] = {
            fwd[1] * right[2] - fwd[2] * right[1],
            fwd[2] * right[0] - fwd[0] * right[2],
            fwd[0] * right[1] - fwd[1] * right[0]};

        float panSpeed = 0.005f;
        for (int i = 0; i < 3; i++)
            m_camPos[i] -= right[i] * dx * panSpeed + up[i] * dy * panSpeed;

        m_cameraDirty = true;
    }
    else if (m_mouseRightDown)
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
                fwrite(row + x * 4, 1, 3, f);
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

        // Mark dielectric/mirror instances non-opaque so ShadowAnyHit runs for them
        BSDFGPUData gd = meshes[i]->getBSDF()->getGPUData();
        // if (gd.type == BSDFGPUData::MIRROR || gd.type == BSDFGPUData::DIELECTRIC)
        //     inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
        if (gd.type == BSDFGPUData::MIRROR || gd.type == BSDFGPUData::DIELECTRIC || !gd.alphaTexture.empty())
            inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
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
    printf("[scene] sizeof(GPUMaterial) on CPU = %zu bytes\n", sizeof(GPUMaterial));
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
        mat.specularTexIndex = 0xFFFFFFFF;
        mat.subsurfaceTexIndex = 0xFFFFFFFF;
        mat.alphaTexIndex = 0xFFFFFFFF;

        mat.roughness = gd.roughness;
        mat.metallic = gd.metallic;
        mat.specular = gd.specular;
        mat.specularTint = gd.specularTint;
        mat.sheen = gd.sheen;
        mat.sheenTint = gd.sheenTint;
        mat.subsurface = gd.subsurface;
        mat.clearcoat = gd.clearcoat;
        mat.clearcoatGloss = gd.clearcoatGloss;
        mat.anisotropic = gd.anisotropic;
        mat.betaN = gd.betaN;

        totalVertices += vc;
        totalIndices += ic;

        printf("[scene] Mesh %u: type=%u verts=%u tris=%u emitter=%u area=%.4f\n",
               i, mat.type, vc, ic / 3, mat.isEmitter, mat.surfaceArea);
        if (mat.type == 5)
            printf("[scene] Mesh %u (HAIR): albedo=(%.4f, %.4f, %.4f) betaN=%.4f roughness=%.4f\n",
                   i, mat.albedo[0], mat.albedo[1], mat.albedo[2], mat.betaN, mat.roughness);

        printf("[scene] Mesh %u: V.cols=%u N.cols=%u N.size=%zu V.size=%zu\n",
               i, (unsigned)meshes[i]->getVertexPositions().cols(),
               (unsigned)meshes[i]->getVertexNormals().cols(),
               meshes[i]->getVertexNormals().size(),
               meshes[i]->getVertexPositions().size());
    }

    printf("[scene] GPUMaterial size = %zu bytes\n", sizeof(GPUMaterial));

    // Build emitter CDF buffer from DiscretePDF and concatenate normalized CDFs for all emitter meshes
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

    // Build a power-weighted inter-emitter CDF and prepend it to allCdfData.
    // Power = luma(radiance) * surfaceArea. Sampling proportional to power
    // means bright/large lights are chosen more often, dramatically reducing
    // variance (graininess) when many lights are present instead of picking
    // uniformly at random (which would give each light probability 1/N).
    {
        std::vector<float> powers;
        std::vector<uint32_t> emitterIndices;
        for (uint32_t i = 0; i < m_meshCount; i++)
        {
            if (!materials[i].isEmitter)
                continue;
            float r = materials[i].radiance[0];
            float g = materials[i].radiance[1];
            float b = materials[i].radiance[2];
            float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            powers.push_back(luma * materials[i].surfaceArea);
            emitterIndices.push_back(i);
        }
        m_emitterCount = (uint32_t)powers.size();

        // Build normalized power CDF with (m_emitterCount + 1) entries.
        std::vector<float> powerCdf(m_emitterCount + 1);
        powerCdf[0] = 0.0f;
        for (uint32_t k = 0; k < m_emitterCount; k++)
            powerCdf[k + 1] = powerCdf[k] + powers[k];
        float totalPower = powerCdf[m_emitterCount];
        if (totalPower > 0.0f)
        {
            float inv = 1.0f / totalPower;
            for (uint32_t k = 1; k <= m_emitterCount; k++)
                powerCdf[k] *= inv;
        }
        else
        {
            // Fallback to uniform if all emitters report zero power.
            for (uint32_t k = 0; k <= m_emitterCount; k++)
                powerCdf[k] = float(k) / float(std::max(m_emitterCount, 1u));
        }
        powerCdf[m_emitterCount] = 1.0f; // clamp

        // Store per-emitter selection probability in GPUMaterial so the
        // shader can compute the correct pdf without re-traversing the CDF.
        for (uint32_t k = 0; k < m_emitterCount; k++)
            materials[emitterIndices[k]].emitterSelectionProb = powerCdf[k + 1] - powerCdf[k];

        // Shift all existing per-emitter triangle CDF offsets past the
        // power CDF block that we are about to prepend.
        uint32_t shift = m_emitterCount + 1;
        for (uint32_t i = 0; i < m_meshCount; i++)
            if (materials[i].isEmitter)
                materials[i].emitterCdfOffset += shift;

        // Prepend the power CDF so it lives at indices [0 .. m_emitterCount].
        allCdfData.insert(allCdfData.begin(), powerCdf.begin(), powerCdf.end());

        printf("[scene] Power CDF: %u emitters, totalPower=%.4f\n", m_emitterCount, totalPower);
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
                bool isHair = meshes[i]->hasTangents();
                for (uint32_t v = 0; v < vc; v++)
                {
                    out[v * 2 + 0] = UV(0, v); // u
                    // For OBJ meshes, flip V (OBJ V=0 bottom → DX V=0 top).
                    // For hair meshes, UV.y encodes the h parameter — do NOT flip.
                    out[v * 2 + 1] = isHair ? UV(1, v) : (1.0f - UV(1, v));
                }
            }
            dst += vc * 2 * sizeof(float);
        }
        m_globalTexCoordBuffer->Unmap(0, nullptr);
        printf("[scene] UV buffer uploaded: %u vertices\n", totalVertices);
    }

    // Concatenate per-vertex fiber tangents (float3 per vertex, zero for non-hair meshes)
    {
        UINT64 sz = totalVertices * 3 * sizeof(float);
        m_globalTangentBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_GENERIC_READ,
                                             D3D12_HEAP_TYPE_UPLOAD);
        uint8_t *dst;
        m_globalTangentBuffer->Map(0, nullptr, (void **)&dst);
        memset(dst, 0, sz); // zero for non-hair meshes
        for (uint32_t i = 0; i < m_meshCount; i++)
        {
            const auto &T = meshes[i]->getVertexTangents();
            uint32_t vc = materials[i].vertexCount;
            if (T.cols() > 0)
            {
                // T is 3×N column-major, same layout as normals
                UINT64 bytes = T.size() * sizeof(float);
                memcpy(dst, T.data(), bytes);
            }
            dst += vc * 3 * sizeof(float);
        }
        m_globalTangentBuffer->Unmap(0, nullptr);
        printf("[scene] Tangent buffer uploaded: %u vertices\n", totalVertices);
    }

    // Volume structured buffer. Always allocate at least one entry so the
    // descriptor table has a valid SRV target even when there are no
    // volumes. The texture index inside m_volumes[0] may still be patched
    // later by CreateTextures, so use an UPLOAD heap for persistent
    // mapping.
    {
        size_t numEntries = std::max<size_t>(1, m_volumes.size());
        UINT64 sz = numEntries * sizeof(GPUVolume);
        m_volumeBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                      D3D12_RESOURCE_STATE_GENERIC_READ,
                                      D3D12_HEAP_TYPE_UPLOAD);
        void *p = nullptr;
        m_volumeBuffer->Map(0, nullptr, &p);
        memset(p, 0, sz);
        if (!m_volumes.empty())
            memcpy(p, m_volumes.data(), m_volumes.size() * sizeof(GPUVolume));
        m_volumeBuffer->Unmap(0, nullptr);
        printf("[scene] Volume buffer uploaded: %zu entries\n", m_volumes.size());
    }
}

// Texture loading via stb_image lirbary
// Loads a 2D texture from disk and creates a GPU resource for it.
//
// isSRGB semantics:
//   - Pass `true` for color data (albedo / baseColor). The GPU will
//     linearize the 8-bit sRGB-encoded values on sample. This is the
//     correct behavior for color textures authored in standard sRGB,
//     which is every JPEG/PNG/TGA "color" texture from Poly Haven,
//     3D Scan Store, AmbientCG, etc.
//   - Pass `false` for non-color data (normal maps, roughness, metallic,
//     AO). Those textures carry scalar/vector data that's already in
//     linear space by convention. Applying sRGB linearization to them
//     would corrupt the values.
uint32_t DXRApp::LoadTexture(const std::string &path, bool isSRGB)
{
    int w, h, channels;
    unsigned char *pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!pixels)
    {
        printf("[texture] Failed to load: %s\n", path.c_str());
        return 0xFFFFFFFF;
    }

    const DXGI_FORMAT fmt = isSRGB ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                                   : DXGI_FORMAT_R8G8B8A8_UNORM;

    uint32_t texIndex = (uint32_t)m_textures.size();

    // Compute full mip chain count
    UINT mipCount = 1;
    {
        int dim = (std::max)(w, h);
        while (dim > 1)
        {
            dim >>= 1;
            mipCount++;
        }
    }

    // Generate mip chain on CPU using box filter
    std::vector<std::vector<uint8_t>> mipPixels(mipCount);
    std::vector<int> mipW(mipCount), mipH(mipCount);

    mipW[0] = w;
    mipH[0] = h;
    mipPixels[0].assign(pixels, pixels + w * h * 4);

    for (UINT m = 1; m < mipCount; m++)
    {
        int pw = mipW[m - 1], ph = mipH[m - 1];
        int mw = (std::max)(pw / 2, 1), mh = (std::max)(ph / 2, 1);
        mipW[m] = mw;
        mipH[m] = mh;
        mipPixels[m].resize(mw * mh * 4);

        const uint8_t *src = mipPixels[m - 1].data();
        uint8_t *dst = mipPixels[m].data();
        for (int y = 0; y < mh; y++)
        {
            for (int x = 0; x < mw; x++)
            {
                int sx = (std::min)(x * 2, pw - 1);
                int sy = (std::min)(y * 2, ph - 1);
                int sx1 = (std::min)(sx + 1, pw - 1);
                int sy1 = (std::min)(sy + 1, ph - 1);
                for (int c = 0; c < 4; c++)
                {
                    int v = (int)src[(sy * pw + sx) * 4 + c] + (int)src[(sy * pw + sx1) * 4 + c] + (int)src[(sy1 * pw + sx) * 4 + c] + (int)src[(sy1 * pw + sx1) * 4 + c];
                    dst[(y * mw + x) * 4 + c] = (uint8_t)(v / 4);
                }
            }
        }
    }
    stbi_image_free(pixels);

    // Compute upload buffer layout: each mip aligned to D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT
    std::vector<UINT64> mipOffsets(mipCount);
    std::vector<UINT64> mipRowPitches(mipCount);
    UINT64 totalUploadSize = 0;
    for (UINT m = 0; m < mipCount; m++)
    {
        UINT64 rowPitch = ((mipW[m] * 4 + 255) & ~255); // 256-byte row alignment
        mipRowPitches[m] = rowPitch;
        mipOffsets[m] = totalUploadSize;
        totalUploadSize += rowPitch * mipH[m];
        totalUploadSize = (totalUploadSize + 511) & ~511; // 512-byte placement alignment
    }

    // Create the GPU texture resource on default heap vram
    D3D12_HEAP_PROPERTIES defaultHeap{};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w;
    td.Height = h;
    td.DepthOrArraySize = 1;
    td.MipLevels = mipCount;
    td.Format = fmt;
    td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags = D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> texture;
    ThrowIfFailed(m_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &td,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&texture)),
                  "Create texture");

    auto uploadBuf = CreateBuffer(totalUploadSize, D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);

    uint8_t *mapped;
    uploadBuf->Map(0, nullptr, (void **)&mapped);
    for (UINT m = 0; m < mipCount; m++)
    {
        for (int y = 0; y < mipH[m]; y++)
            memcpy(mapped + mipOffsets[m] + y * mipRowPitches[m],
                   mipPixels[m].data() + y * mipW[m] * 4,
                   mipW[m] * 4);
    }
    uploadBuf->Unmap(0, nullptr);

    for (UINT m = 0; m < mipCount; m++)
    {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = texture.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = m;

        src.pResource = uploadBuf.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Offset = mipOffsets[m];
        src.PlacedFootprint.Footprint.Format = fmt;
        src.PlacedFootprint.Footprint.Width = mipW[m];
        src.PlacedFootprint.Footprint.Height = mipH[m];
        src.PlacedFootprint.Footprint.Depth = 1;
        src.PlacedFootprint.Footprint.RowPitch = (UINT)mipRowPitches[m];

        m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }

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
    m_textureIsSRGB.push_back(isSRGB ? 1 : 0);

    printf("[texture] Loaded %s (%dx%d, %u mips, %s) as texture %u\n",
           path.c_str(), w, h, mipCount, isSRGB ? "sRGB" : "linear", texIndex);
    return texIndex;
}

// HDR envmap loading. Stores RGBA32F on GPU and keeps CPU-side pixel data
// around for the CDF build in Step 2. Falls back to a flat gray 1x1 if the
// file is missing, so the renderer still works with no envmap present.
void DXRApp::LoadEnvmap(const std::string &path)
{
    int w = 0, h = 0, channels = 0;
    float *pixels = stbi_loadf(path.c_str(), &w, &h, &channels, 0);

    std::vector<float> rgba;
    if (!pixels)
    {
        printf("[envmap] No HDR at '%s' — falling back to flat gray.\n", path.c_str());
        w = 1;
        h = 1;
        rgba = {0.5f, 0.5f, 0.5f, 1.0f};
        m_envmapValid = false;
    }
    else
    {
        rgba.resize((size_t)w * h * 4);
        for (int i = 0; i < w * h; ++i)
        {
            float r = pixels[i * channels + 0];
            float g = (channels > 1) ? pixels[i * channels + 1] : r;
            float b = (channels > 2) ? pixels[i * channels + 2] : r;
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = 1.0f;
        }
        stbi_image_free(pixels);
        m_envmapValid = true;
    }

    m_envmapWidth = (uint32_t)w;
    m_envmapHeight = (uint32_t)h;
    m_envmapPixels = std::move(rgba);

    const UINT bytesPerPixel = 16; // RGBA32F
    UINT64 rowPitch = ((UINT64)w * bytesPerPixel + 255) & ~255ULL;
    UINT64 uploadSize = rowPitch * h;

    D3D12_HEAP_PROPERTIES defaultHeap{};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = w;
    td.Height = h;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    td.Flags = D3D12_RESOURCE_FLAG_NONE;

    ThrowIfFailed(m_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &td,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&m_envmap)),
                  "Create envmap texture");

    m_envmapUpload = CreateBuffer(uploadSize, D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);

    uint8_t *mapped = nullptr;
    m_envmapUpload->Map(0, nullptr, (void **)&mapped);
    for (int y = 0; y < h; ++y)
    {
        memcpy(mapped + y * rowPitch,
               m_envmapPixels.data() + (size_t)y * w * 4,
               (size_t)w * bytesPerPixel);
    }
    m_envmapUpload->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = m_envmap.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    src.pResource = m_envmapUpload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    src.PlacedFootprint.Footprint.Width = w;
    src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;
    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = m_envmap.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(1, &barrier);

    printf("[envmap] Loaded %s (%dx%d, %d channels -> RGBA32F)\n",
           path.c_str(), w, h, channels);
}

// Build the 2D piecewise-constant sampling distribution over the envmap.
// Factored as: marginal over rows, conditional per row over columns.
// Density is proportional to luminance(x, y) * sin(theta_y) so the
// solid-angle distortion of the equirectangular parameterization is cancelled.
//
// Outputs: two ByteAddressBuffers.
//   m_envmapMarginalCdf    : (H + 1) floats, CDF over row probabilities.
//   m_envmapConditionalCdf : H * (W + 1) floats, per-row CDF, row-major.
//   Both are normalized so their final entry equals 1.0.
void DXRApp::BuildEnvmapDistribution()
{
    const uint32_t W = m_envmapWidth;
    const uint32_t H = m_envmapHeight;
    if (W == 0 || H == 0 || m_envmapPixels.empty())
    {
        printf("[envmap-cdf] No envmap pixels — skipping.\n");
        return;
    }

    const float M_PI_F = 3.14159265358979323846f;

    // Rec. 709 luminance weights. Matches PBRT.
    auto luma = [](float r, float g, float b)
    { return 0.2126f * r + 0.7152f * g + 0.0722f * b; };

    std::vector<float> conditional((size_t)H * (W + 1));
    std::vector<float> rowSums(H);

    for (uint32_t y = 0; y < H; ++y)
    {
        // Sample latitude at the pixel center.
        float theta = M_PI_F * (y + 0.5f) / (float)H;
        float sinTheta = std::sin(theta);

        size_t rowBase = (size_t)y * (W + 1);
        conditional[rowBase + 0] = 0.0f;

        // Build the running sum across this row
        float rowSum = 0.0f;
        for (uint32_t x = 0; x < W; ++x)
        {
            size_t pixBase = ((size_t)y * W + x) * 4;
            float r = m_envmapPixels[pixBase + 0];
            float g = m_envmapPixels[pixBase + 1];
            float b = m_envmapPixels[pixBase + 2];
            float weight = luma(r, g, b) * sinTheta;

            if (weight < 0.0f)
                weight = 0.0f;
            rowSum += weight;
            conditional[rowBase + x + 1] = rowSum;
        }

        rowSums[y] = rowSum;

        if (rowSum > 0.0f)
        {
            float inv = 1.0f / rowSum;
            for (uint32_t x = 1; x <= W; ++x)
                conditional[rowBase + x] *= inv;
        }
        else
        {
            for (uint32_t x = 0; x <= W; ++x)
                conditional[rowBase + x] = (float)x / (float)W;
        }
    }

    // Build the marginal CDF from row sums.
    std::vector<float> marginal(H + 1);
    double total = 0.0;
    for (uint32_t y = 0; y < H; ++y)
        total += rowSums[y];

    marginal[0] = 0.0f;
    if (total > 0.0)
    {
        double accum = 0.0;
        double invTotal = 1.0 / total;
        for (uint32_t y = 0; y < H; ++y)
        {
            accum += rowSums[y];
            marginal[y + 1] = (float)(accum * invTotal);
        }

        marginal[H] = 1.0f;
    }
    else
    {
        for (uint32_t y = 0; y <= H; ++y)
            marginal[y] = (float)y / (float)H;
    }

    // upload both CDFs to GPU as ByteAddressBuffers on an UPLOAD heap.
    {
        UINT64 sz = marginal.size() * sizeof(float);
        m_envmapMarginalCdf = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                           D3D12_RESOURCE_STATE_GENERIC_READ,
                                           D3D12_HEAP_TYPE_UPLOAD);
        void *p;
        m_envmapMarginalCdf->Map(0, nullptr, &p);
        memcpy(p, marginal.data(), sz);
        m_envmapMarginalCdf->Unmap(0, nullptr);
    }
    {
        UINT64 sz = conditional.size() * sizeof(float);
        m_envmapConditionalCdf = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                              D3D12_RESOURCE_STATE_GENERIC_READ,
                                              D3D12_HEAP_TYPE_UPLOAD);
        void *p;
        m_envmapConditionalCdf->Map(0, nullptr, &p);
        memcpy(p, conditional.data(), sz);
        m_envmapConditionalCdf->Unmap(0, nullptr);
    }

    // TODO remove this debugging
    uint32_t nonZeroRows = 0;
    for (uint32_t y = 0; y < H; ++y)
        if (rowSums[y] > 0.0f)
            ++nonZeroRows;
    printf("[envmap-cdf] %ux%u, luminance integral = %.3f, "
           "marginal[H] = %.6f, conditional[0][W] = %.6f, "
           "%u/%u rows non-zero\n",
           W, H, (float)total,
           marginal[H], conditional[W], nonZeroRows, H);
}

// Build the m_volumes list from the scene's medium description. Runs
// before CreateSceneBuffers so the structured buffer can be sized correctly,
// and before CreateTextures so the medium's density texture (if any) can be
// patched in by index later.
void DXRApp::SetupVolumes()
{
    m_volumes.clear();
    const Medium *medium = m_noriScene->getMedium();
    if (medium)
    {
        BoundingBox3f vbox = medium->hasExplicitBounds()
                                 ? medium->getExplicitBounds()
                                 : m_noriScene->getBoundingBox();
        Color3f sigmaARGB = medium->getSigmaARGB();
        Color3f sigmaSRGB = medium->getSigmaSRGB();
        GPUVolume v{};
        v.vMin[0] = vbox.min.x();
        v.vMin[1] = vbox.min.y();
        v.vMin[2] = vbox.min.z();
        v.vMax[0] = vbox.max.x();
        v.vMax[1] = vbox.max.y();
        v.vMax[2] = vbox.max.z();
        v.sigmaA[0] = sigmaARGB.r();
        v.sigmaA[1] = sigmaARGB.g();
        v.sigmaA[2] = sigmaARGB.b();
        v.sigmaS[0] = sigmaSRGB.r();
        v.sigmaS[1] = sigmaSRGB.g();
        v.sigmaS[2] = sigmaSRGB.b();
        v.phaseG = medium->getPhaseG();
        v.flags = medium->isHeterogeneous() ? VOLUME_FLAG_HETEROGENEOUS : 0u;
        v.densityTexIndex = VOLUME_INVALID_TEX;  // patched in CreateTextures
        v.majorantTexIndex = VOLUME_INVALID_TEX; // patched in CreateTextures
        m_volumes.push_back(v);

        printf("[volume] 1 volume: sigmaA=(%.3f,%.3f,%.3f) sigmaS=(%.3f,%.3f,%.3f) g=%.2f bbox=(%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)%s%s\n",
               v.sigmaA[0], v.sigmaA[1], v.sigmaA[2],
               v.sigmaS[0], v.sigmaS[1], v.sigmaS[2], v.phaseG,
               vbox.min.x(), vbox.min.y(), vbox.min.z(),
               vbox.max.x(), vbox.max.y(), vbox.max.z(),
               medium->hasExplicitBounds() ? " [explicit bounds]" : " [scene bbox]",
               medium->isHeterogeneous() ? " [heterogeneous]" : " [homogeneous]");
    }
    else
    {
        printf("[volume] 0 volumes (no <medium> in scene XML)\n");
    }
}

// Helper: upload a dense W×H×D R32_FLOAT volume to a Texture3D and append
// it to m_volumeTextures. Returns the index in that array. Used for both
// the dense density grid and the coarse brick-max-density majorant mip.
static uint32_t UploadVolumeTexture3D(
    ID3D12Device5 *device,
    ID3D12GraphicsCommandList4 *cmdList,
    std::vector<ComPtr<ID3D12Resource>> &textures,
    std::vector<ComPtr<ID3D12Resource>> &uploads,
    const float *data, uint32_t W, uint32_t H, uint32_t D)
{
    ComPtr<ID3D12Resource> tex;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    td.Width = W;
    td.Height = H;
    td.DepthOrArraySize = (UINT16)D;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    td.SampleDesc.Count = 1;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    HRESULT hr = device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &td,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex));
    if (FAILED(hr))
        throw std::runtime_error("Volume Texture3D creation failed");

    UINT64 uploadSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT numRows = 0;
    UINT64 rowSize = 0;
    device->GetCopyableFootprints(&td, 0, 1, 0, &footprint, &numRows, &rowSize, &uploadSize);

    D3D12_HEAP_PROPERTIES uhp{};
    uhp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC ud{};
    ud.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    ud.Width = uploadSize;
    ud.Height = 1;
    ud.DepthOrArraySize = 1;
    ud.MipLevels = 1;
    ud.Format = DXGI_FORMAT_UNKNOWN;
    ud.SampleDesc.Count = 1;
    ud.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> upload;
    if (FAILED(device->CreateCommittedResource(
            &uhp, D3D12_HEAP_FLAG_NONE, &ud,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))))
        throw std::runtime_error("Volume upload buffer creation failed");

    uint8_t *mapped = nullptr;
    upload->Map(0, nullptr, (void **)&mapped);
    mapped += footprint.Offset;
    UINT srcRowPitch = W * sizeof(float);
    UINT dstRowPitch = footprint.Footprint.RowPitch;
    for (UINT z = 0; z < D; z++)
    {
        for (UINT y = 0; y < H; y++)
        {
            const float *srcRow = data + (size_t)W * (y + (size_t)H * z);
            uint8_t *dstRow = mapped + (size_t)dstRowPitch * y + (size_t)dstRowPitch * (size_t)numRows * z;
            memcpy(dstRow, srcRow, srcRowPitch);
        }
    }
    upload->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = tex.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = upload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = footprint;
    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = tex.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    uint32_t index = (uint32_t)textures.size();
    textures.push_back(tex);
    uploads.push_back(upload);
    return index;
}

DXRApp::LoadedVolumeIndices DXRApp::LoadVolume(const std::string &path)
{
    LoadedVolumeIndices result{VOLUME_INVALID_TEX, VOLUME_INVALID_TEX};

    // Read .vol file which consists of 40-byte header + dense float array
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        printf("[volume] WARNING: could not open '%s'\n", path.c_str());
        return result;
    }

    char magic[4];
    file.read(magic, 4);
    if (memcmp(magic, "VOL1", 4) != 0)
    {
        printf("[volume] WARNING: invalid magic in '%s'\n", path.c_str());
        return result;
    }

    uint32_t W, H, D;
    file.read((char *)&W, 4);
    file.read((char *)&H, 4);
    file.read((char *)&D, 4);

    float bboxMin[3], bboxMax[3];
    file.read((char *)bboxMin, 12);
    file.read((char *)bboxMax, 12);

    size_t numVoxels = (size_t)W * H * D;
    std::vector<float> data(numVoxels);
    file.read((char *)data.data(), numVoxels * sizeof(float));
    file.close();

    printf("[volume] Loaded '%s': %ux%ux%u (%.1f MB)\n",
           path.c_str(), W, H, D,
           (numVoxels * 4) / (1024.0f * 1024.0f));

    // Upload the dense density grid.
    result.densityIndex = UploadVolumeTexture3D(
        m_device.Get(), m_commandList.Get(),
        m_volumeTextures, m_volumeUploads,
        data.data(), W, H, D);

    // Build a coarse brick-max-density mip used as a tracked majorant for
    // null-collision tracking. Each majorant voxel holds the max density
    // of a BRICK_SIZE^3 block of dense voxels, so the local μ within a
    // brick becomes density_brick_max · σ_t,max — much smaller than the
    // global μ in low-density regions, which lets exponential free-flight
    // sampling take large steps through empty space.
    const uint32_t BRICK_SIZE = 4;
    const uint32_t mW = (W + BRICK_SIZE - 1) / BRICK_SIZE;
    const uint32_t mH = (H + BRICK_SIZE - 1) / BRICK_SIZE;
    const uint32_t mD = (D + BRICK_SIZE - 1) / BRICK_SIZE;
    std::vector<float> majorant((size_t)mW * mH * mD, 0.0f);
    for (uint32_t z = 0; z < D; z++)
    {
        for (uint32_t y = 0; y < H; y++)
        {
            for (uint32_t x = 0; x < W; x++)
            {
                float d = data[(size_t)x + (size_t)W * (y + (size_t)H * z)];
                size_t mIdx = (size_t)(x / BRICK_SIZE) +
                              (size_t)mW * ((y / BRICK_SIZE) + (size_t)mH * (z / BRICK_SIZE));
                majorant[mIdx] = std::max(majorant[mIdx], d);
            }
        }
    }
    result.majorantIndex = UploadVolumeTexture3D(
        m_device.Get(), m_commandList.Get(),
        m_volumeTextures, m_volumeUploads,
        majorant.data(), mW, mH, mD);

    printf("[volume] Density Texture3D (%ux%ux%u) uploaded as index %u\n",
           W, H, D, result.densityIndex);
    printf("[volume] Majorant Texture3D (%ux%ux%u, brick=%u) uploaded as index %u\n",
           mW, mH, mD, BRICK_SIZE, result.majorantIndex);
    return result;
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
        m_textureIsSRGB.insert(m_textureIsSRGB.begin(), 0);
        m_texUploads.push_back(uploadBuf);
    }

    // Load textures referenced by materials w/ deduplication
    const auto &meshes = m_noriScene->getMeshes();
    std::unordered_map<std::string, uint32_t> texCache;

    auto loadIfNotEmpty = [&](const std::string &relPath, bool isSRGB) -> uint32_t
    {
        if (relPath.empty())
            return 0xFFFFFFFF;
        filesystem::path resolved = getFileResolver()->resolve(relPath);
        std::string key = resolved.str();
        auto it = texCache.find(key);
        if (it != texCache.end())
            return it->second;
        uint32_t idx = LoadTexture(key, isSRGB);
        if (idx != 0xFFFFFFFF)
            texCache[key] = idx;
        return idx;
    };

    // per-mesh texture indices
    struct MeshTexIndices
    {
        uint32_t albedo, normal, roughness, metallic, specular, subsurface, alpha;
    };
    std::vector<MeshTexIndices> meshTexIndices(meshes.size());

    for (uint32_t i = 0; i < (uint32_t)meshes.size(); i++)
    {
        BSDFGPUData gd = meshes[i]->getBSDF()->getGPUData();
        // Albedo is COLOR data -> sRGB encoded on disk, must be linearized on sample.
        // Normal/roughness/metallic/specular/subsurface are DATA textures -> already linear, leave as-is.
        meshTexIndices[i].albedo = loadIfNotEmpty(gd.albedoTexture, true);
        meshTexIndices[i].normal = loadIfNotEmpty(gd.normalTexture, false);
        meshTexIndices[i].roughness = loadIfNotEmpty(gd.roughnessTexture, false);
        meshTexIndices[i].metallic = loadIfNotEmpty(gd.metallicTexture, false);
        meshTexIndices[i].specular = loadIfNotEmpty(gd.specularTexture, false);
        meshTexIndices[i].subsurface = loadIfNotEmpty(gd.subsurfaceTexture, false);
        meshTexIndices[i].alpha = loadIfNotEmpty(gd.alphaTexture, false);
    }

    m_textureCount = (uint32_t)m_textures.size();

    // Load the HDR environment map on the same command list so its upload and barrier transition flush together with the material textures.
    // TODO: this path is currently hardcoded, if I have time i need to wire through the scene XML so different scenes can specify their own HDR and parameters like intensity/rotation.
    {
        filesystem::path hdrPath =
            getFileResolver()->resolve("textures/sunset.hdr");
        // getFileResolver()->resolve("textures/white_furnace.hdr");
        LoadEnvmap(hdrPath.str());
    }

    BuildEnvmapDistribution();

    // Load any heterogeneous-volume density files referenced by the scene's
    // medium. LoadVolume uploads both the dense density and a coarse
    // brick-max-density majorant; both indices are patched into m_volumes
    // below so the shader can walk bricks during null-collision tracking.
    LoadedVolumeIndices mediumIdx{VOLUME_INVALID_TEX, VOLUME_INVALID_TEX};
    {
        const Medium *medium = m_noriScene->getMedium();
        if (medium && medium->isHeterogeneous() && !medium->getVolumePath().empty())
        {
            filesystem::path volPath = getFileResolver()->resolve(medium->getVolumePath());
            mediumIdx = LoadVolume(volPath.str());
        }
    }

    FlushCommandQueue();
    m_texUploads.clear();
    m_envmapUpload.Reset();
    m_volumeUploads.clear();

    // Patch the medium's volume entry (built in SetupVolumes with placeholder
    // texture indices) so its densityTexIndex / majorantTexIndex point at
    // the uploaded Texture3D's, then re-upload the volume buffer.
    if (!m_volumes.empty() && (m_volumes[0].flags & VOLUME_FLAG_HETEROGENEOUS))
    {
        m_volumes[0].densityTexIndex = mediumIdx.densityIndex;
        m_volumes[0].majorantTexIndex = mediumIdx.majorantIndex;
        void *p = nullptr;
        m_volumeBuffer->Map(0, nullptr, &p);
        memcpy(p, m_volumes.data(), m_volumes.size() * sizeof(GPUVolume));
        m_volumeBuffer->Unmap(0, nullptr);
    }

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
            mats[i].specularTexIndex = meshTexIndices[i].specular;
            mats[i].subsurfaceTexIndex = meshTexIndices[i].subsurface;
            mats[i].alphaTexIndex = meshTexIndices[i].alpha;
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
    {
        float cy = cosf(m_camYaw), sy = sinf(m_camYaw);
        Vector3f dxrRight(cy, 0.0f, -sy);
        Vector3f noriHoriz = (P_br - P_bl).normalized();
        m_camXFlip = (noriHoriz.dot(dxrRight) >= 0.0f) ? 1.0f : -1.0f;
    }

    m_camera.meshCount = m_meshCount;
    m_camera.emitterCount = m_emitterCount;
    m_camera.frameCount = 0;
    m_camera.lensRadius = cam->getLensRadius();
    m_camera.focalDistance = cam->getFocalDistance();
    m_cameraDirty = true;

    m_lastFrameTime = std::chrono::high_resolution_clock::now();

    RecomputeCameraPlane();

    printf("[camera] pos=(%.3f,%.3f,%.3f) yaw=%.1f pitch=%.1f fov=%.1f\n",
           m_camPos[0], m_camPos[1], m_camPos[2],
           m_camYaw * 180.0f / 3.14159f, m_camPitch * 180.0f / 3.14159f,
           m_camFovY * 180.0f / 3.14159f);
    printf("[camera] Controls: WASD=move, QE=up/down, RightClick+drag=look, Both+drag=pan, P=snapshot\n");

    m_camera.volumeCount = (uint32_t)m_volumes.size();
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

// pipeline, output, shader table, render
void DXRApp::CreateRaytracingPipeline()
{
    UINT totalSRVs = 7 + 1 + 2 + 1 + m_textureCount; // +1 for tangent buffer
    UINT volumeTexCount = (UINT)std::max<size_t>(1, m_volumeTextures.size());

    // ranges[2]: t0 space1 — StructuredBuffer<GPUVolume> (1 descriptor)
    // ranges[3]: t1 space1 — Texture3D<float>[] (volumeTexCount descriptors)
    D3D12_DESCRIPTOR_RANGE ranges[4]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[0].NumDescriptors = 2; // u0=output, u1=accumulation
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = totalSRVs;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;

    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 0;
    ranges[2].RegisterSpace = 1;
    ranges[2].OffsetInDescriptorsFromTableStart = 2 + totalSRVs;

    ranges[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[3].NumDescriptors = volumeTexCount;
    ranges[3].BaseShaderRegister = 1;
    ranges[3].RegisterSpace = 1;
    ranges[3].OffsetInDescriptorsFromTableStart = 2 + totalSRVs + 1;

    D3D12_ROOT_PARAMETER rp[2]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[0].Constants.ShaderRegister = 0;
    rp[0].Constants.Num32BitValues = sizeof(CameraConstants) / 4;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[1].DescriptorTable.NumDescriptorRanges = 4;
    rp[1].DescriptorTable.pDescriptorRanges = ranges;
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC samplers[3]{};
    samplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].ShaderRegister = 0; // s0
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    samplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[1].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].ShaderRegister = 1; // s1
    samplers[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    samplers[2].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[2].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[2].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[2].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[2].ShaderRegister = 2; // s2
    samplers[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 2;
    rsd.pParameters = rp;
    rsd.NumStaticSamplers = 3;
    rsd.pStaticSamplers = samplers;
    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err), "RootSig serialize");
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                IID_PPV_ARGS(&m_globalRootSig)),
                  "RootSig");

    auto blob = ReadFileBytes(GetExeDirectory() + L"Shaders.cso");
    printf("[pipeline] Shader: %zu bytes\n", blob.size());

    D3D12_STATE_SUBOBJECT so[6]{};
    int idx = 0;
    D3D12_DXIL_LIBRARY_DESC ld{};
    ld.DXILLibrary = {blob.data(), blob.size()};
    so[idx].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    so[idx++].pDesc = &ld;
    D3D12_HIT_GROUP_DESC hg{};
    hg.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
    hg.ClosestHitShaderImport = L"ClosestHit";
    hg.AnyHitShaderImport = L"PrimaryAnyHit";
    hg.HitGroupExport = L"HitGroup";
    so[idx].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    so[idx++].pDesc = &hg;
    // Shadow hit group: any-hit only (skips dielectrics with Fresnel attenuation)
    D3D12_HIT_GROUP_DESC shg{};
    shg.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
    shg.AnyHitShaderImport = L"ShadowAnyHit";
    shg.HitGroupExport = L"ShadowHitGroup";
    so[idx].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    so[idx++].pDesc = &shg;
    D3D12_RAYTRACING_SHADER_CONFIG sc{};
    sc.MaxPayloadSizeInBytes = 96;
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

    // Descriptor heap layout:
    //  [0]  u0  UAV output texture
    //  [1]  u1  UAV accumulation texture
    //  [2]  t0  SRV TLAS
    //  [3]  t1  SRV material structured buffer
    //  [4]  t2  SRV global vertex normals (raw)
    //  [5]  t3  SRV global index buffer (raw)
    //  [6]  t4  SRV global vertex positions (raw)
    //  [7]  t5  SRV emitter CDF (raw)
    //  [8]  t6  SRV global UV buffer (raw)
    //  [9]  t7  SRV environment map (RGBA32F)
    //  [10] t8  SRV envmap marginal CDF (raw)
    //  [11] t9  SRV envmap conditional CDF (raw)
    //  [12] t10 SRV global fiber tangent buffer (raw, hair only)
    //  [13..13+N) t11+ SRV material textures
    //  [13+N]     t0 (space1) SRV volume StructuredBuffer<GPUVolume>
    //  [14+N..)   t1+ (space1) SRV volume density Texture3D<float>[]
    UINT volumeTexCount = (UINT)std::max<size_t>(1, m_volumeTextures.size());
    UINT totalDescriptors = 14 + m_textureCount + volumeTexCount;
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

    // helper for raw ByteAddressBuffer SRVs
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
        h.ptr += m_srvUavDescriptorSize;

    // [8] SRV — global UV buffer (t6)
    createRawSRV(m_globalTexCoordBuffer.Get());

    // [9] SRV — environment map (t7)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        m_device->CreateShaderResourceView(m_envmap.Get(), &sd, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    // [10] SRV — envmap marginal CDF (t8, raw)
    if (m_envmapMarginalCdf)
        createRawSRV(m_envmapMarginalCdf.Get());
    else
        h.ptr += m_srvUavDescriptorSize;

    // [11] SRV — envmap conditional CDF (t9, raw)
    if (m_envmapConditionalCdf)
        createRawSRV(m_envmapConditionalCdf.Get());
    else
        h.ptr += m_srvUavDescriptorSize;

    // [12] SRV — global fiber tangent buffer (t10, raw)
    createRawSRV(m_globalTangentBuffer.Get());

    // [13+] SRV — material textures (t11+). The SRV's format must match
    // the resource's format (sRGB vs UNORM), tracked in m_textureIsSRGB.
    for (uint32_t i = 0; i < m_textureCount; i++)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = m_textureIsSRGB[i] ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                                       : DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture2D.MipLevels = 1;
        m_device->CreateShaderResourceView(m_textures[i].Get(), &sd, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    // [13+N] SRV — volume StructuredBuffer<GPUVolume> (t0, space1).
    // The descriptor table always points at a real buffer (CreateSceneBuffers
    // allocates at least one entry even when m_volumes is empty).
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Buffer.NumElements = (UINT)std::max<size_t>(1, m_volumes.size());
        sd.Buffer.StructureByteStride = sizeof(GPUVolume);
        m_device->CreateShaderResourceView(m_volumeBuffer.Get(), &sd, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    // [14+N..) SRV — volume density Texture3D[] (t1+, space1).
    // If the scene has no volume textures, fill the slot with one dummy
    // 1x1x1 R32_FLOAT resource so the descriptor table is valid.
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
        sd.Format = DXGI_FORMAT_R32_FLOAT;
        sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sd.Texture3D.MipLevels = 1;

        if (m_volumeTextures.empty())
        {
            D3D12_HEAP_PROPERTIES dhp{};
            dhp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC drd{};
            drd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            drd.Width = 1;
            drd.Height = 1;
            drd.DepthOrArraySize = 1;
            drd.MipLevels = 1;
            drd.Format = DXGI_FORMAT_R32_FLOAT;
            drd.SampleDesc.Count = 1;
            static ComPtr<ID3D12Resource> dummyVol;
            if (!dummyVol)
            {
                m_device->CreateCommittedResource(&dhp, D3D12_HEAP_FLAG_NONE, &drd,
                                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
                                                  IID_PPV_ARGS(&dummyVol));
            }
            m_device->CreateShaderResourceView(dummyVol.Get(), &sd, h);
            h.ptr += m_srvUavDescriptorSize;
        }
        else
        {
            for (size_t i = 0; i < m_volumeTextures.size(); i++)
            {
                m_device->CreateShaderResourceView(m_volumeTextures[i].Get(), &sd, h);
                h.ptr += m_srvUavDescriptorSize;
            }
        }
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
    // 5 entries: [RayGen] [Miss] [ShadowMiss] [HitGroup] [ShadowHitGroup]
    m_shaderTable = CreateBuffer(al * 5, D3D12_RESOURCE_FLAG_NONE,
                                 D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
    void *rg = m_rtStateObjectProps->GetShaderIdentifier(L"RayGen");
    void *ms = m_rtStateObjectProps->GetShaderIdentifier(L"Miss");
    void *sm = m_rtStateObjectProps->GetShaderIdentifier(L"ShadowMiss");
    void *hg = m_rtStateObjectProps->GetShaderIdentifier(L"HitGroup");
    void *shg = m_rtStateObjectProps->GetShaderIdentifier(L"ShadowHitGroup");
    if (!rg || !ms || !sm || !hg || !shg)
        throw std::runtime_error("Shader ID not found");
    uint8_t *m;
    m_shaderTable->Map(0, nullptr, (void **)&m);
    memset(m, 0, al * 5);
    memcpy(m, rg, id);           // [0] RayGen
    memcpy(m + al, ms, id);      // [1] Miss
    memcpy(m + al * 2, sm, id);  // [2] ShadowMiss
    memcpy(m + al * 3, hg, id);  // [3] HitGroup (primary rays, index 0)
    memcpy(m + al * 4, shg, id); // [4] ShadowHitGroup (shadow rays, index 1)
    m_shaderTable->Unmap(0, nullptr);
    printf("[shader table] Created (5 entries)\n");
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

    dr.HitGroupTable = {ta + sa * 3, sa * 2, sa}; // 2 hit groups (primary + shadow), stride = sa
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