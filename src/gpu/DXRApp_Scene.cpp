#include "DXRApp.h"

#include <parser.h>
#include <scene.h>
#include <mesh.h>
#include <bsdf.h>
#include <emitter.h>
#include <medium.h>
#include <dpdf.h>
#include <sampler.h>
#include <filesystem/resolver.h>

#include <fstream>
#include <cstring>

// Scene loading, BVH/acceleration structure build, GPU scene buffer upload,
// and participating-medium volume setup.

using namespace nori;

void DXRApp::LoadNoriScene()
{
    filesystem::path xmlPath(m_scenePath);
    getFileResolver()->prepend(xmlPath.parent_path());

    NoriObject *root = loadFromXML(m_scenePath);
    if (root->getClassType() != NoriObject::EScene)
        throw std::runtime_error("Root object is not a Scene");

    m_noriScene = static_cast<Scene *>(root);
    printf("[nori] Scene loaded: %zu meshes\n", m_noriScene->getMeshes().size());

    if (m_noriScene->getSampler())
        m_targetSamples = (uint32_t)m_noriScene->getSampler()->getSampleCount();
}

void DXRApp::CreateAccelerationStructure()
{
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "Alloc");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "CmdList");

    const auto &meshes = m_noriScene->getMeshes();
    m_meshGPU.resize(meshes.size());

    // Scratch buffers must stay alive until FlushCommandQueue completes.
    std::vector<ComPtr<ID3D12Resource>> tempBuffers;

    // Build one BLAS per mesh.
    for (size_t mi = 0; mi < meshes.size(); mi++)
    {
        const Mesh *mesh = meshes[mi];
        const auto &V = mesh->getVertexPositions();
        const auto &F = mesh->getIndices();

        UINT64 vbSize = V.size() * sizeof(float);
        UINT64 ibSize = F.size() * sizeof(uint32_t);

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

        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = blasResult.Get();
        m_commandList->ResourceBarrier(1, &uav);

        m_meshGPU[mi].blas = blasResult;
    }

    // Build TLAS — one instance per mesh (no instancing/transforms applied).
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

        // Non-opaque so ShadowAnyHit runs for dielectrics and alpha-masked meshes.
        BSDFGPUData gd = meshes[i]->getBSDF()->getGPUData();
        if (gd.type == BSDFGPUData::MIRROR || gd.type == BSDFGPUData::DIELECTRIC || !gd.alphaTexture.empty())
            inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
    }

    UINT64 instSize = instances.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    auto instBuf = CreateBuffer(instSize, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);
    tempBuffers.push_back(instBuf);
    void *im;
    instBuf->Map(0, nullptr, &im);
    memcpy(im, instances.data(), instSize);
    instBuf->Unmap(0, nullptr);

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

void DXRApp::CreateSceneBuffers()
{
    printf("[scene] sizeof(GPUMaterial) on CPU = %zu bytes\n", sizeof(GPUMaterial));
    const auto &meshes = m_noriScene->getMeshes();
    m_meshCount = (uint32_t)meshes.size();

    // Pass 1: fill per-mesh GPUMaterial and accumulate total vertex/index counts.
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
    }

    // Build inter-emitter power-weighted CDF, then per-emitter triangle area CDFs.
    std::vector<float> allCdfData;
    for (uint32_t i = 0; i < m_meshCount; i++)
    {
        if (!materials[i].isEmitter)
            continue;
        const auto &cdf = meshes[i]->getDiscretePDF().getCDF();
        materials[i].emitterCdfOffset = (uint32_t)allCdfData.size();
        allCdfData.insert(allCdfData.end(), cdf.begin(), cdf.end());
    }

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
            for (uint32_t k = 0; k <= m_emitterCount; k++)
                powerCdf[k] = float(k) / float(std::max(m_emitterCount, 1u));
        }
        powerCdf[m_emitterCount] = 1.0f;

        for (uint32_t k = 0; k < m_emitterCount; k++)
            materials[emitterIndices[k]].emitterSelectionProb = powerCdf[k + 1] - powerCdf[k];

        uint32_t shift = m_emitterCount + 1;
        for (uint32_t i = 0; i < m_meshCount; i++)
            if (materials[i].isEmitter)
                materials[i].emitterCdfOffset += shift;

        allCdfData.insert(allCdfData.begin(), powerCdf.begin(), powerCdf.end());

        printf("[scene] Power CDF: %u emitters, totalPower=%.4f\n", m_emitterCount, totalPower);
    }

    // Upload material structured buffer.
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

    // Concatenate per-mesh vertex normals.
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
                memcpy(dst, N.data(), N.size() * sizeof(float));
            dst += vc * 3 * sizeof(float);
        }
        m_globalNormalBuffer->Unmap(0, nullptr);
    }

    // Concatenate triangle indices (local per mesh; shader applies vertexOffset).
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

    // Concatenate vertex positions (needed for emitter sampling in shaders).
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

    // Emitter CDF buffer.
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

    // Concatenate UV coordinates (float2 per vertex, interleaved).
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
                float *out = reinterpret_cast<float *>(dst);
                bool isHair = meshes[i]->hasTangents();
                for (uint32_t v = 0; v < vc; v++)
                {
                    out[v * 2 + 0] = UV(0, v);
                    // Flip V for OBJ meshes (OBJ: V=0 at bottom; DX: V=0 at top).
                    // Do not flip for hair meshes where UV.y encodes the h parameter.
                    out[v * 2 + 1] = isHair ? UV(1, v) : (1.0f - UV(1, v));
                }
            }
            dst += vc * 2 * sizeof(float);
        }
        m_globalTexCoordBuffer->Unmap(0, nullptr);
    }

    // Concatenate per-vertex fiber tangents (float3 per vertex; zero for non-hair meshes).
    {
        UINT64 sz = totalVertices * 3 * sizeof(float);
        m_globalTangentBuffer = CreateBuffer(sz, D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_GENERIC_READ,
                                             D3D12_HEAP_TYPE_UPLOAD);
        uint8_t *dst;
        m_globalTangentBuffer->Map(0, nullptr, (void **)&dst);
        memset(dst, 0, sz);
        for (uint32_t i = 0; i < m_meshCount; i++)
        {
            const auto &T = meshes[i]->getVertexTangents();
            uint32_t vc = materials[i].vertexCount;
            if (T.cols() > 0)
                memcpy(dst, T.data(), T.size() * sizeof(float));
            dst += vc * 3 * sizeof(float);
        }
        m_globalTangentBuffer->Unmap(0, nullptr);
    }

    // Volume structured buffer. Always allocate at least one entry so the
    // descriptor table has a valid SRV target even when no volumes are present.
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

void DXRApp::SetupVolumes()
{
    m_volumes.clear();
    const Medium *medium = m_noriScene->getMedium();
    if (!medium)
    {
        printf("[volume] 0 volumes (no <medium> in scene XML)\n");
        return;
    }

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
    v.densityTexIndex = VOLUME_INVALID_TEX;
    v.majorantTexIndex = VOLUME_INVALID_TEX;
    m_volumes.push_back(v);

    printf("[volume] 1 volume: sigmaA=(%.3f,%.3f,%.3f) sigmaS=(%.3f,%.3f,%.3f) g=%.2f%s%s\n",
           v.sigmaA[0], v.sigmaA[1], v.sigmaA[2],
           v.sigmaS[0], v.sigmaS[1], v.sigmaS[2], v.phaseG,
           medium->hasExplicitBounds() ? " [explicit bounds]" : " [scene bbox]",
           medium->isHeterogeneous() ? " [heterogeneous]" : " [homogeneous]");
}

// Upload a dense W×H×D R32_FLOAT volume to a Texture3D and append it to
// the provided texture / upload-heap vectors. Returns the index in the array.
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

    // .vol format: 4-byte magic "VOL1" + dimensions (W, H, D) + bbox + dense float array.
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
           path.c_str(), W, H, D, (numVoxels * 4) / (1024.0f * 1024.0f));

    result.densityIndex = UploadVolumeTexture3D(
        m_device.Get(), m_commandList.Get(),
        m_volumeTextures, m_volumeUploads,
        data.data(), W, H, D);

    // Build a coarse brick-max-density mip for tracked-majorant null-collision
    // sampling. Each majorant voxel holds the max density within a BRICK_SIZE^3
    // block, giving tight local majorants that allow large free-flight steps.
    const uint32_t BRICK_SIZE = 4;
    const uint32_t mW = (W + BRICK_SIZE - 1) / BRICK_SIZE;
    const uint32_t mH = (H + BRICK_SIZE - 1) / BRICK_SIZE;
    const uint32_t mD = (D + BRICK_SIZE - 1) / BRICK_SIZE;
    std::vector<float> majorant((size_t)mW * mH * mD, 0.0f);
    for (uint32_t z = 0; z < D; z++)
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
            {
                float d = data[(size_t)x + (size_t)W * (y + (size_t)H * z)];
                size_t mIdx = (size_t)(x / BRICK_SIZE) +
                              (size_t)mW * ((y / BRICK_SIZE) + (size_t)mH * (z / BRICK_SIZE));
                majorant[mIdx] = std::max(majorant[mIdx], d);
            }

    result.majorantIndex = UploadVolumeTexture3D(
        m_device.Get(), m_commandList.Get(),
        m_volumeTextures, m_volumeUploads,
        majorant.data(), mW, mH, mD);

    printf("[volume] Density Texture3D (%ux%ux%u) index %u; majorant (%ux%ux%u, brick=%u) index %u\n",
           W, H, D, result.densityIndex, mW, mH, mD, BRICK_SIZE, result.majorantIndex);
    return result;
}
