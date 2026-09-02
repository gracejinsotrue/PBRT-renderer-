#include "DXRApp.h"

#include <algorithm>

// DXR pipeline state object, descriptor heap + output/accumulation resources,
// and shader table.  CreateRaytracingPipeline must be called before
// CreateOutputResource, which must be called before CreateShaderTable.

void DXRApp::CreateRaytracingPipeline()
{
    UINT totalSRVs = 7 + 1 + 2 + 1 + m_textureCount; // +1 for tangent buffer
    UINT volumeTexCount = (UINT)std::max<size_t>(1, m_volumeTextures.size());

    const UINT numUAV = 4; // u0=output, u1=accum, u2=albedo AOV, u3=normal AOV

    D3D12_DESCRIPTOR_RANGE ranges[4]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[0].NumDescriptors = numUAV;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = totalSRVs;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = numUAV;

    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 0;
    ranges[2].RegisterSpace = 1;
    ranges[2].OffsetInDescriptorsFromTableStart = numUAV + totalSRVs;

    ranges[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[3].NumDescriptors = volumeTexCount;
    ranges[3].BaseShaderRegister = 1;
    ranges[3].RegisterSpace = 1;
    ranges[3].OffsetInDescriptorsFromTableStart = numUAV + totalSRVs + 1;

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
    // HitPayload is 28 B (hitT, materialID, hit, primitiveID, baryX/Y, rngState);
    // ShadowPayload is 20 B. Must be >= the larger of the two.
    sc.MaxPayloadSizeInBytes = 28;
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

// Post-process pipelines: display resolve plus the three bloom chain passes.
//
// Deliberately does NOT reuse m_globalRootSig: that one is shaped around the
// raytracing pipeline's unbounded texture arrays and static samplers, none of
// which a post-process pass wants. It does read from m_srvUavHeap though
// (descriptors appended at the tail by CreateOutputResource), so no
// SetDescriptorHeaps call is needed between DispatchRays and Dispatch.
//
// All four passes share one root signature and one 3-wide binding table, which
// is why they can be dispatched back to back with nothing but a PSO swap.
void DXRApp::CreatePostPipelines()
{
    D3D12_DESCRIPTOR_RANGE range{};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    range.NumDescriptors = 3; // u0 = source, u1 = dest, u2 = aux (bloom mip 0)
    range.BaseShaderRegister = 0;
    range.OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER rp[2]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[0].Constants.ShaderRegister = 0;
    rp[0].Constants.Num32BitValues = 8; // see PostParams in PostCommon.hlsli
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[1].DescriptorTable.NumDescriptorRanges = 1;
    rp[1].DescriptorTable.pDescriptorRanges = &range;
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 2;
    rsd.pParameters = rp;
    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err),
                  "Post root sig serialize");
    ThrowIfFailed(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                IID_PPV_ARGS(&m_postRootSig)),
                  "Post root sig");

    struct PassDesc
    {
        const wchar_t *cso;
        ComPtr<ID3D12PipelineState> *pso;
    } passes[] = {
        {L"CSResolve.cso", &m_resolvePSO},
        {L"CSBloomPrefilter.cso", &m_bloomPrefilterPSO},
        {L"CSBloomDownsample.cso", &m_bloomDownsamplePSO},
        {L"CSBloomUpsample.cso", &m_bloomUpsamplePSO},
    };
    for (const auto &p : passes)
    {
        auto blob = ReadFileBytes(GetExeDirectory() + p.cso);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = m_postRootSig.Get();
        pd.CS = {blob.data(), blob.size()};
        ThrowIfFailed(m_device->CreateComputePipelineState(
                          &pd, IID_PPV_ARGS(p.pso->ReleaseAndGetAddressOf())),
                      "Post PSO");
    }
    printf("[pipeline] Post-process compute PSOs created (resolve + 3 bloom passes)\n");
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

    // Denoised-preview display texture (RGBA8), and a copy source for the
    // backbuffer blit when the denoise-while-still preview is active. Same
    // format as the output texture.
    //
    // It is now a UAV: the resolve pass writes it directly, so the denoised
    // preview goes through the exact same exposure/ACES/gamma code as the live
    // render instead of a duplicated CPU tonemap loop.
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
                                                        IID_PPV_ARGS(&m_denoisedResource)),
                      "Denoised display tex");
    }

    // OIDN's linear HDR output, staged on the GPU so the resolve pass can read
    // it. Uploaded with w = 1.0 (see UploadHDR) so the shader's divide-by-.w
    // normalization is a no-op for this path.
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
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        IID_PPV_ARGS(&m_denoisedHdrResource)),
                      "Denoised HDR tex");
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

    // Bloom mip chain. Mip 0 is half the render resolution; the level count is
    // derived rather than fixed so the blur covers a consistent fraction of the
    // frame instead of getting relatively tighter as the output grows.
    {
        UINT minDim = std::min(m_width, m_height);
        UINT levels = 1;
        while ((minDim >> (levels + 1)) >= 8 && levels < 6)
            levels++;
        m_bloomMipCount = levels;

        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = std::max(1u, m_width / 2);
        td.Height = std::max(1u, m_height / 2);
        td.DepthOrArraySize = 1;
        td.MipLevels = (UINT16)m_bloomMipCount;
        td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        td.SampleDesc.Count = 1;
        td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                        IID_PPV_ARGS(&m_bloomResource)),
                      "Bloom mip chain");
        printf("[bloom] %u mips, base %ux%u\n", m_bloomMipCount,
               (UINT)td.Width, (UINT)td.Height);
    }

    // Denoiser AOV textures albedo + shading normal accumulated like accum.

    // these get recreated with D3D12_HEAP_FLAG_SHARED + COMMON initial state so CUDA can import them;
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
                                                        IID_PPV_ARGS(&m_albedoResource)),
                      "Albedo AOV tex");
        ThrowIfFailed(m_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td,
                                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                        IID_PPV_ARGS(&m_normalResource)),
                      "Normal AOV tex");
    }

    // Descriptor heap layout (N = m_textureCount, V = volumeTexCount):
    //  [0]  u0  UAV output texture
    //  [1]  u1  UAV accumulation texture
    //  [2]  u2  UAV albedo AOV texture
    //  [3]  u3  UAV normal AOV texture
    //  [4]  t0  SRV TLAS
    //  [5]  t1  SRV material structured buffer
    //  [6]  t2  SRV global vertex normals (raw)
    //  [7]  t3  SRV global index buffer (raw)
    //  [8]  t4  SRV global vertex positions (raw)
    //  [9]  t5  SRV emitter CDF (raw)
    //  [10] t6  SRV global UV buffer (raw)
    //  [11] t7  SRV environment map (RGBA32F)
    //  [12] t8  SRV envmap marginal CDF (raw)
    //  [13] t9  SRV envmap conditional CDF (raw)
    //  [14] t10 SRV global fiber tangent buffer (raw, hair only)
    //  [15..15+N)   t11+ SRV material textures
    //  [15+N]       t0 (space1) SRV volume StructuredBuffer<GPUVolume>
    //  [16+N..16+N+V) t1+ (space1) SRV volume density Texture3D<float>[]
    //
    // Everything above is addressed by the raytracing root signature. The
    // entries below belong to the post-process passes, addressed by
    // m_postRootSig as 3-descriptor slots starting at m_postDescriptorBase (see
    // the slot map in DXRApp.h). They live at the tail so none of the offsets
    // above shift.
    UINT volumeTexCount = (UINT)std::max<size_t>(1, m_volumeTextures.size());
    const UINT rtDescriptors = 16 + m_textureCount + volumeTexCount;
    m_postDescriptorBase = rtDescriptors;
    const UINT postSlots = 4 + 2 * (m_bloomMipCount - 1);
    UINT totalDescriptors = rtDescriptors + postSlots * 3;
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

    // [2] UAV — albedo AOV, [3] UAV — normal AOV
    for (ID3D12Resource *aov : {m_albedoResource.Get(), m_normalResource.Get()})
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        m_device->CreateUnorderedAccessView(aov, nullptr, &ud, h);
        h.ptr += m_srvUavDescriptorSize;
    }

    // [4] SRV — TLAS
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

    // Post-process slots. Ordering within a slot is (source, dest, aux) to match
    // u0/u1/u2 in PostCommon.hlsli, which is the reverse of the raytracing
    // table's output-then-accum order at [0]/[1].
    {
        auto writeUav = [&](ID3D12Resource *res, DXGI_FORMAT fmt, UINT mip)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
            ud.Format = fmt;
            ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            ud.Texture2D.MipSlice = mip;
            m_device->CreateUnorderedAccessView(res, nullptr, &ud, h);
            h.ptr += m_srvUavDescriptorSize;
        };
        const DXGI_FORMAT kHdr = DXGI_FORMAT_R32G32B32A32_FLOAT;
        const DXGI_FORMAT kLdr = DXGI_FORMAT_R8G8B8A8_UNORM;
        const DXGI_FORMAT kBloom = DXGI_FORMAT_R16G16B16A16_FLOAT;
        auto bloomMip0 = [&]() { writeUav(m_bloomResource.Get(), kBloom, 0); };

        // slot 0: resolve, live
        writeUav(m_accumResource.Get(), kHdr, 0);
        writeUav(m_outputResource.Get(), kLdr, 0);
        bloomMip0();
        // slot 1: resolve, denoised
        writeUav(m_denoisedHdrResource.Get(), kHdr, 0);
        writeUav(m_denoisedResource.Get(), kLdr, 0);
        bloomMip0();
        // slot 2: prefilter, live  (u2 unused; bind mip 0 as filler)
        writeUav(m_accumResource.Get(), kHdr, 0);
        bloomMip0();
        bloomMip0();
        // slot 3: prefilter, denoised
        writeUav(m_denoisedHdrResource.Get(), kHdr, 0);
        bloomMip0();
        bloomMip0();
        // slots 4..: downsample i (mip i -> mip i+1)
        for (UINT i = 0; i + 1 < m_bloomMipCount; i++)
        {
            writeUav(m_bloomResource.Get(), kBloom, i);
            writeUav(m_bloomResource.Get(), kBloom, i + 1);
            bloomMip0();
        }
        // then upsample i (mip i+1 -> accumulated into mip i), walking back down
        for (UINT i = 0; i + 1 < m_bloomMipCount; i++)
        {
            writeUav(m_bloomResource.Get(), kBloom, i + 1);
            writeUav(m_bloomResource.Get(), kBloom, i);
            bloomMip0();
        }
    }

    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "C");
    D3D12_RESOURCE_BARRIER ib[3]{};
    ID3D12Resource *initToCopySrc[2] = {m_outputResource.Get(), m_denoisedResource.Get()};
    for (int i = 0; i < 2; i++)
    {
        ib[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        ib[i].Transition.pResource = initToCopySrc[i];
        ib[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        ib[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        ib[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }
    // The HDR staging texture rests in UNORDERED_ACCESS (the state the resolve
    // pass reads it in); UploadHDR transitions it to COPY_DEST and back.
    ib[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    ib[2].Transition.pResource = m_denoisedHdrResource.Get();
    ib[2].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    ib[2].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ib[2].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(3, ib);
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
