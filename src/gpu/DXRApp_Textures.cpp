#include "DXRApp.h"

#include <scene.h>
#include <mesh.h>
#include <bsdf.h>
#include <medium.h>
#include <filesystem/resolver.h>

#include <algorithm>
#include <unordered_map>
#include <cmath>

// Texture and environment map loading, mip generation, envmap CDF build,
// and final CreateTextures pass that wires everything into the heap.

// Single translation unit that compiles the stb/tinyexr implementations.
// Other files include these headers without the _IMPLEMENTATION defines.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

using namespace nori;

// isSRGB semantics:
//   true  — color data (albedo/baseColor): GPU linearizes 8-bit sRGB on sample.
//   false — non-color data (normals, roughness, metallic, etc.): already
//           in linear space, sRGB linearization would corrupt the values.
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
// for the CDF build. Falls back to a flat gray 1x1 if the file is missing.
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
// Factored as marginal over rows and conditional per row over columns.
// Density is proportional to luminance(x,y) * sin(theta_y), cancelling
// the solid-angle distortion of the equirectangular parameterization.
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

    auto luma = [](float r, float g, float b)
    { return 0.2126f * r + 0.7152f * g + 0.0722f * b; };

    std::vector<float> conditional((size_t)H * (W + 1));
    std::vector<float> rowSums(H);

    for (uint32_t y = 0; y < H; ++y)
    {
        float theta = M_PI_F * (y + 0.5f) / (float)H;
        float sinTheta = std::sin(theta);

        size_t rowBase = (size_t)y * (W + 1);
        conditional[rowBase + 0] = 0.0f;

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

void DXRApp::CreateTextures()
{
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "Alloc");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "CmdList");

    // Slot 0: white 1x1 dummy texture used for materials with no albedo texture.
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

    // Load textures referenced by materials with deduplication.
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

    struct MeshTexIndices
    {
        uint32_t albedo, normal, roughness, metallic, specular, subsurface, alpha;
    };
    std::vector<MeshTexIndices> meshTexIndices(meshes.size());

    for (uint32_t i = 0; i < (uint32_t)meshes.size(); i++)
    {
        BSDFGPUData gd = meshes[i]->getBSDF()->getGPUData();
        meshTexIndices[i].albedo = loadIfNotEmpty(gd.albedoTexture, true);
        meshTexIndices[i].normal = loadIfNotEmpty(gd.normalTexture, false);
        meshTexIndices[i].roughness = loadIfNotEmpty(gd.roughnessTexture, false);
        meshTexIndices[i].metallic = loadIfNotEmpty(gd.metallicTexture, false);
        meshTexIndices[i].specular = loadIfNotEmpty(gd.specularTexture, false);
        meshTexIndices[i].subsurface = loadIfNotEmpty(gd.subsurfaceTexture, false);
        meshTexIndices[i].alpha = loadIfNotEmpty(gd.alphaTexture, false);
    }

    m_textureCount = (uint32_t)m_textures.size();

    // Load the HDR environment map. Path comes from <string name="envmap" value="..."/>
    // in the scene XML; defaults to "textures/white_furnace.hdr" if not specified.
    {
        filesystem::path hdrPath = getFileResolver()->resolve(m_noriScene->getEnvmap());
        LoadEnvmap(hdrPath.str());
    }

    BuildEnvmapDistribution();

    // Load any heterogeneous-volume density files referenced by the scene's medium.
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

    // Patch the medium's volume entry with the uploaded Texture3D indices,
    // then re-upload the volume buffer so the shader sees the real indices.
    if (!m_volumes.empty() && (m_volumes[0].flags & VOLUME_FLAG_HETEROGENEOUS))
    {
        m_volumes[0].densityTexIndex = mediumIdx.densityIndex;
        m_volumes[0].majorantTexIndex = mediumIdx.majorantIndex;
        void *p = nullptr;
        m_volumeBuffer->Map(0, nullptr, &p);
        memcpy(p, m_volumes.data(), m_volumes.size() * sizeof(GPUVolume));
        m_volumeBuffer->Unmap(0, nullptr);
    }

    // Patch texture indices into the material buffer.
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
