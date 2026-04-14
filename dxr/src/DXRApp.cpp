#include "DXRApp.h"
#include "Win32Application.h"

// Nori headers
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
    CreateRaytracingPipeline();
    CreateOutputResource();
    CreateShaderTable();
    SetupCamera();
    printf("[init] DX12 + DXR initialization complete\n");
}

void DXRApp::OnUpdate() {}

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
    for (UINT i = 0; m_factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                           IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
         ++i)
    {
        DXGI_ADAPTER_DESC1 d;
        adapter->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&m_device))))
        {
            printf("[init] Adapter: %ls\n", d.Description);
            break;
        }
    }
    if (!m_device)
        throw std::runtime_error("No D3D12 device");
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
    ThrowIfFailed(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5)), "Features");
    if (o5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
        throw std::runtime_error("No DXR");
    printf("[init] DXR tier: %s\n", o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1 ? "1.1" : "1.0");
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

// ---------------------------------------------------------------------------
// Resource helpers
// ---------------------------------------------------------------------------
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

// Scene data buffers — material properties, normals, indices
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

    // Concatenate vertex normals into one buffer
    // Nori stores normals as 3×N column-major MatrixXf = interleaved [x0,y0,z0, x1,y1,z1,...]
    {
        UINT64 sz = totalVertices * 3 * sizeof(float);
        m_globalNormalBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                            D3D12_RESOURCE_STATE_GENERIC_READ,
                                            D3D12_HEAP_TYPE_UPLOAD);
        uint8_t *dst;
        m_globalNormalBuffer->Map(0, nullptr, (void **)&dst);
        for (uint32_t i = 0; i < m_meshCount; i++)
        {
            const auto &N = meshes[i]->getVertexNormals();
            UINT64 bytes = N.size() * sizeof(float);
            memcpy(dst, N.data(), bytes);
            dst += bytes;
        }
        m_globalNormalBuffer->Unmap(0, nullptr);
    }

    // Concatenate triangle indices into one buffer
    // Indices are LOCAL per mesh (0-based), looked up with vertexOffset in the shader
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

    // Concatenate vertex positions into one buffer
    // Needed for emitter sampling
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
}

// Camera: derive image plane from Nori's sampleRay at corners
void DXRApp::SetupCamera()
{
    const Camera *cam = m_noriScene->getCamera();
    float w = (float)m_width, h = (float)m_height;

    // Sample corner rays. Nori pixel coords: (0,0)=top-left, (w,h)=bottom-right
    // my shader: (u=0,v=0)=bottom-left, (u=1,v=1)=top-right
    // So: Nori(0, h) → this bottom-left,  Nori(w, h) → this bottom-right,
    //     Nori(0, 0) → this top-left
    Ray3f ray_bl, ray_br, ray_tl, ray_ctr;
    Point2f ap(0, 0);
    cam->sampleRay(ray_bl, Point2f(0, h), ap);
    cam->sampleRay(ray_br, Point2f(w, h), ap);
    cam->sampleRay(ray_tl, Point2f(0, 0), ap);
    cam->sampleRay(ray_ctr, Point2f(w / 2, h / 2), ap);

    // camera position is same for all rays
    Point3f pos = ray_bl.o;
    Vector3f fwd = ray_ctr.d.normalized();

    auto project = [&](const Ray3f &r) -> Point3f
    {
        float t = 1.0f / r.d.dot(fwd);
        return pos + r.d * t;
    };

    Point3f P_bl = project(ray_bl);
    Point3f P_br = project(ray_br);
    Point3f P_tl = project(ray_tl);

    Vector3f LLC = P_bl - Point3f(0, 0, 0);
    Vector3f H = P_br - P_bl;
    Vector3f V = P_tl - P_bl;

    m_camera.camPos[0] = pos.x();
    m_camera.camPos[1] = pos.y();
    m_camera.camPos[2] = pos.z();
    m_camera.camLowerLeftCorner[0] = P_bl.x();
    m_camera.camLowerLeftCorner[1] = P_bl.y();
    m_camera.camLowerLeftCorner[2] = P_bl.z();
    m_camera.camHorizontal[0] = H.x();
    m_camera.camHorizontal[1] = H.y();
    m_camera.camHorizontal[2] = H.z();
    m_camera.camVertical[0] = V.x();
    m_camera.camVertical[1] = V.y();
    m_camera.camVertical[2] = V.z();
    m_camera.meshCount = m_meshCount;
    m_camera.frameCount = 0;

    printf("[camera] pos=(%.3f,%.3f,%.3f) fwd=(%.3f,%.3f,%.3f)\n",
           pos.x(), pos.y(), pos.z(), fwd.x(), fwd.y(), fwd.z());
}

// Pipeline, output, shader table, render
void DXRApp::CreateRaytracingPipeline()
{
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[0].NumDescriptors = 2; // u0=output, u1=accumulation
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = 6; // t0=TLAS, t1=materials, t2=normals, t3=indices, t4=vertices, t5=emitterCdf
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

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 2;
    rsd.pParameters = rp;
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
    sc.MaxPayloadSizeInBytes = 48; // not sure if this payload size is correct
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

    // Accumulation UAV texture (R32G32B32A32_FLOAT for HDR accumulation)
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

    // Descriptor heap: 8 slots
    //  [0] u0  UAV output texture
    //  [1] u1  UAV accumulation texture
    //  [2] t0  SRV TLAS
    //  [3] t1  SRV material structured buffer
    //  [4] t2  SRV global vertex normals (raw)
    //  [5] t3  SRV global index buffer (raw)
    //  [6] t4  SRV global vertex positions (raw)
    //  [7] t5  SRV emitter CDF (raw)
    D3D12_DESCRIPTOR_HEAP_DESC dhd{};
    dhd.NumDescriptors = 8;
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
    printf("[output] Output + accum textures + 8 descriptors created\n");
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
    memcpy(m + al, ms, id);     // [1] Miss (primary, index 0)
    memcpy(m + al * 2, sm, id); // [2] ShadowMiss (index 1)
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
