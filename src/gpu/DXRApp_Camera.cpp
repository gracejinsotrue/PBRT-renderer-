#include "DXRApp.h"
#include "Denoiser.h"

#include <camera.h>
#include <scene.h>
#include <filesystem/resolver.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

// stb_image_write and tinyexr headers without _IMPLEMENTATION —
// the implementations are compiled once in DXRApp_Textures.cpp.
#include "stb_image_write.h"
#include "tinyexr.h"

// Camera initialization, image-plane recompute, interactive camera input,
// and snapshot saving (PPM via stb_image_write, EXR via tinyexr).

using namespace nori;

void DXRApp::SetupCamera()
{
    const Camera *cam = m_noriScene->getCamera();
    float w = (float)m_width, h = (float)m_height;

    Ray3f ray_bl, ray_br, ray_tl, ray_ctr;
    Point2f ap(0, 0);
    cam->sampleRay(ray_bl, Point2f(0, h), ap);
    cam->sampleRay(ray_br, Point2f(w, h), ap);
    cam->sampleRay(ray_tl, Point2f(0, 0), ap);
    cam->sampleRay(ray_ctr, Point2f(w / 2, h / 2), ap);

    Point3f pos = ray_bl.o;
    Vector3f fwd = ray_ctr.d.normalized();

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
    m_camera.envmapScale = m_noriScene->getEnvmapScale();
    m_camera.evCompensation = m_noriScene->getEvCompensation();
    m_camera.envmapRotation = m_noriScene->getEnvmapRotation();
    m_bloomThreshold = m_noriScene->getBloomThreshold();
    m_bloomKnee = m_noriScene->getBloomKnee();
    m_bloomIntensity = m_noriScene->getBloomIntensity();
    if (m_bloomIntensity > 0.0f)
        printf("[bloom] threshold %.2f, knee %.2f, intensity %.3f (B toggles, [ ] adjust)\n",
               m_bloomThreshold, m_bloomKnee, m_bloomIntensity);
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
    printf("[camera] Controls: WASD=move, QE=up/down, RightClick+drag=look, Both+drag=pan\n");
    printf("[camera]           P=snapshot, O=EXR, N=denoise now, M=toggle auto denoise-while-still (default ON)\n");

    m_camera.volumeCount = (uint32_t)m_volumes.size();
}

void DXRApp::RecomputeCameraPlane()
{
    float cy = cosf(m_camYaw), sy = sinf(m_camYaw);
    float cp = cosf(m_camPitch), sp = sinf(m_camPitch);

    float fwd[3] = {sy * cp, sp, cy * cp};
    float right[3] = {cy, 0.0f, -sy};
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

void DXRApp::OnUpdate()
{
    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - m_lastFrameTime).count();
    m_lastFrameTime = now;
    dt = (dt > 0.1f) ? 0.1f : dt;

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
        m_denoiseCacheFrame = 0xFFFFFFFFu; // accumulation restarted; cache is stale
        m_cameraDirty = false;
        m_showDenoised = false; // camera moved, therefore show the live (noisy) render again
        m_nextDenoiseSpp = 16;  // restart the denoise-while-still cadence
    }
}

void DXRApp::OnKeyDown(UINT8 key)
{
    m_keys[key] = true;
    if (key == 'P')
        SaveSnapshot();
    if (key == 'O')
        SaveSnapshotEXR();
    if (key == 'N')
        DenoiseToViewport(); // denoise the current accumulation now and show it
    if (key == 'M')
    {
        m_autoDenoise = !m_autoDenoise;
        if (!m_autoDenoise)
            m_showDenoised = false;
        printf("[denoise] auto denoise-while-still %s\n", m_autoDenoise ? "ON" : "OFF");
    }
    // Bloom tuning. A good intensity is entirely scene-dependent, and because
    // bloom is the last pass it re-evaluates on the next frame with no re-render
    // and no re-denoise, so these are meant to be dragged by eye.
    if (key == 'B')
    {
        m_bloomEnabled = !m_bloomEnabled;
        printf("[bloom] %s (intensity %.3f)\n", m_bloomEnabled ? "ON" : "OFF", m_bloomIntensity);
        if (m_showDenoised)
            RefreshDenoisedPreview();
    }
    if (key == VK_OEM_4 || key == VK_OEM_6) // '[' and ']'
    {
        m_bloomIntensity += (key == VK_OEM_6) ? 0.01f : -0.01f;
        if (m_bloomIntensity < 0.0f)
            m_bloomIntensity = 0.0f;
        printf("[bloom] intensity %.3f%s\n", m_bloomIntensity,
               m_bloomEnabled ? "" : " (bloom is OFF, press B)");
        if (m_showDenoised)
            RefreshDenoisedPreview();
    }
    if (key == 'L')
    {
        float cy = cosf(m_camYaw), sy = sinf(m_camYaw);
        float cp = cosf(m_camPitch), sp = sinf(m_camPitch);
        float fwd[3] = {sy * cp, sp, cy * cp};
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
    if (button == 0)
    {
        m_mouseLeftDown = true;
        m_lastMouse = {x, y};
    }
    if (button == 1)
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

// Read one of the RGBA8 display textures back into tightly packed RGBA bytes.
// Both live in COPY_SOURCE between frames, which is what CopyTextureRegion wants.
std::vector<uint8_t> DXRApp::ReadbackRGBA8(ID3D12Resource *res)
{
    WaitForGpu(m_frameIndex);

    D3D12_RESOURCE_DESC desc = res->GetDesc();
    UINT64 rowPitch = ((desc.Width * 4 + 255) & ~255ULL);
    UINT64 totalSize = rowPitch * desc.Height;

    auto readback = CreateBuffer(totalSize, D3D12_RESOURCE_FLAG_NONE,
                                 D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);

    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "C");

    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dst.PlacedFootprint.Footprint.Width = (UINT)desc.Width;
    dst.PlacedFootprint.Footprint.Height = desc.Height;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;

    src.pResource = res;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    FlushCommandQueue();

    std::vector<uint8_t> out((size_t)desc.Width * desc.Height * 4);
    uint8_t *data;
    readback->Map(0, nullptr, (void **)&data);
    for (UINT y = 0; y < desc.Height; y++)
        memcpy(out.data() + (size_t)y * desc.Width * 4,
               data + (size_t)y * rowPitch, (size_t)desc.Width * 4);
    readback->Unmap(0, nullptr);
    return out;
}

// Save the tonemapped display image as PNG. Unlike the EXR path this is the
// display-referred result: exposure, bloom, ACES and gamma all applied. It is
// the only way to get the post-processed image out of a headless run, and hence
// the only way to regression-test anything downstream of the accumulator.
void DXRApp::SaveSnapshotPNG(bool denoised)
{
    ID3D12Resource *srcRes = denoised ? m_denoisedResource.Get() : m_outputResource.Get();
    std::vector<uint8_t> rgba = ReadbackRGBA8(srcRes);

    std::string dir = filesystem::path(m_scenePath).parent_path().str();
    if (!dir.empty())
        dir += "/";
    char filename[512];
    snprintf(filename, sizeof(filename), "%ssnapshot_%u%s.png", dir.c_str(), m_frameCount,
             denoised ? "_denoised" : "");

    if (stbi_write_png(filename, (int)m_width, (int)m_height, 4, rgba.data(), (int)m_width * 4))
        printf("[snapshot] Saved %s (%u samples%s)\n", filename, m_frameCount,
               denoised ? ", denoised" : "");
    else
        printf("[snapshot] PNG save failed (%s)\n", filename);
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

    std::string snapshotDir = filesystem::path(m_scenePath).parent_path().str();
    if (!snapshotDir.empty())
        snapshotDir += "/";
    char filename[512];
    snprintf(filename, sizeof(filename), "%ssnapshot_%u.ppm", snapshotDir.c_str(), m_frameCount);

    FILE *f = fopen(filename, "wb");
    if (f)
    {
        fprintf(f, "P6\n%u %u\n255\n", (UINT)desc.Width, desc.Height);
        for (UINT y = 0; y < desc.Height; y++)
        {
            const uint8_t *row = data + y * rowPitch;
            for (UINT x = 0; x < (UINT)desc.Width; x++)
                fwrite(row + x * 4, 1, 3, f);
        }
        fclose(f);
        printf("[snapshot] Saved %s (%u samples)\n", filename, m_frameCount);
    }

    readback->Unmap(0, nullptr);
}

std::vector<float> DXRApp::ReadbackAccumResource(ID3D12Resource *res)
{
    WaitForGpu(m_frameIndex);

    D3D12_RESOURCE_DESC desc = res->GetDesc();
    UINT64 rowPitch = ((desc.Width * 16 + 255) & ~255ULL);
    UINT64 totalSize = rowPitch * desc.Height;

    auto readback = CreateBuffer(totalSize, D3D12_RESOURCE_FLAG_NONE,
                                 D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);

    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "C");

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(1, &b);

    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    dst.PlacedFootprint.Footprint.Width = (UINT)desc.Width;
    dst.PlacedFootprint.Footprint.Height = desc.Height;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;

    src.pResource = res;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    m_commandList->ResourceBarrier(1, &b);

    FlushCommandQueue();

    const float *data;
    readback->Map(0, nullptr, (void **)&data);

    const UINT W = (UINT)desc.Width;
    const UINT H = desc.Height;

    std::vector<float> rgb((size_t)W * H * 3);
    for (UINT y = 0; y < H; y++)
    {
        const float *row = data + y * (rowPitch / sizeof(float));
        for (UINT x = 0; x < W; x++)
        {
            float w = row[x * 4 + 3];
            float inv = (w > 0.0f) ? (1.0f / w) : 0.0f;
            size_t idx = (size_t)y * W + x;
            rgb[idx * 3 + 0] = row[x * 4 + 0] * inv;
            rgb[idx * 3 + 1] = row[x * 4 + 1] * inv;
            rgb[idx * 3 + 2] = row[x * 4 + 2] * inv;
        }
    }

    readback->Unmap(0, nullptr);
    return rgb;
}

void DXRApp::SaveAccumResourceEXR(ID3D12Resource *res, const char *filename)
{
    std::vector<float> rgb = ReadbackAccumResource(res);
    const char *err = nullptr;
    int ret = SaveEXR(rgb.data(), (int)m_width, (int)m_height, 3, 0, filename, &err);
    if (ret != TINYEXR_SUCCESS)
        printf("[snapshot] EXR save failed (%s): %s\n", filename, err ? err : "unknown error");
    else
        printf("[snapshot] Saved %s (%u samples, HDR linear)\n", filename, m_frameCount);
}

bool DXRApp::RunDenoise(std::vector<float> &outRGB)
{
    // Denoising the same accumulation twice always produces the same image, and
    // callers do stack up: `--denoise --png` wants one denoise for the EXR and
    // another to stage the HDR for the bloom pass. Cache on sample count so the
    // second caller is free. Invalidated in OnUpdate when the camera moves,
    // because that resets m_frameCount and a later frame could otherwise collide
    // with a stale entry at the same count.
    if (m_denoiseCacheFrame == m_frameCount && !m_denoiseCache.empty())
    {
        outRGB = m_denoiseCache;
        return true;
    }

    if (!m_denoiser)
    {
        m_denoiser = std::make_unique<Denoiser>();
        if (!m_denoiser->Init(m_width, m_height))
        {
            printf("[denoise] OIDN init failed — denoise unavailable\n");
            m_denoiser.reset();
            return false;
        }
    }

    std::vector<float> beauty = ReadbackAccumResource(m_accumResource.Get());
    std::vector<float> albedo = ReadbackAccumResource(m_albedoResource.Get());
    std::vector<float> normal = ReadbackAccumResource(m_normalResource.Get());

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = m_denoiser->Denoise(beauty.data(), albedo.data(), normal.data(), outRGB);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ok)
    {
        printf("[denoise] %u spp denoised in %.1f ms\n", m_frameCount,
               std::chrono::duration<float, std::milli>(t1 - t0).count());
        m_denoiseCache = outRGB;
        m_denoiseCacheFrame = m_frameCount;
    }
    return ok;
}

void DXRApp::DenoiseAndSaveEXR()
{
    std::vector<float> rgb;
    if (!RunDenoise(rgb))
        return;

    std::string dir = filesystem::path(m_scenePath).parent_path().str();
    if (!dir.empty())
        dir += "/";
    char filename[512];
    snprintf(filename, sizeof(filename), "%ssnapshot_%u_denoised.exr", dir.c_str(), m_frameCount);

    const char *err = nullptr;
    int ret = SaveEXR(rgb.data(), (int)m_width, (int)m_height, 3, 0, filename, &err);
    if (ret != TINYEXR_SUCCESS)
        printf("[denoise] EXR save failed (%s): %s\n", filename, err ? err : "unknown error");
    else
        printf("[denoise] Saved %s\n", filename);
}

void DXRApp::DenoiseToViewport()
{
    std::vector<float> rgb;
    if (!RunDenoise(rgb))
        return;

    // Stage OIDN's linear HDR output on the GPU, then run the *same* resolve
    // pass the live render uses. This used to be a hand-rolled copy of the ACES
    // curve on the CPU, which meant the tonemap existed in two places and could
    // drift; it also left nowhere for a bloom pass to apply to the denoised
    // image. Now both paths differ only in which texture pair they bind.
    UploadHDR(m_denoisedHdrResource.Get(), rgb);
    RefreshDenoisedPreview();

    m_showDenoised = true;
    printf("[denoise] showing denoised preview (move camera to resume live render)\n");
}

// Re-run bloom + resolve on the already-denoised HDR staging texture, without
// touching OIDN. Split out of DenoiseToViewport so that adjusting bloom while a
// denoised preview is frozen on screen updates it for the cost of a few
// dispatches instead of another ~260 ms denoise.
void DXRApp::RefreshDenoisedPreview()
{
    WaitForGpu(m_frameIndex);
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "C");

    ID3D12DescriptorHeap *heaps[] = {m_srvUavHeap.Get()};
    m_commandList->SetDescriptorHeaps(1, heaps);

    D3D12_RESOURCE_BARRIER toUav{};
    toUav.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toUav.Transition.pResource = m_denoisedResource.Get();
    toUav.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    toUav.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    toUav.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(1, &toUav);

    // Bloom runs on the *denoised* image, never before OIDN. Blurring first
    // would spread each firefly into a smooth low-frequency blob, which no
    // longer looks like Monte Carlo noise to the denoiser and so survives it.
    RecordBloomChain(/*denoisedSource=*/true);
    RecordResolve(/*denoisedSource=*/true);

    std::swap(toUav.Transition.StateBefore, toUav.Transition.StateAfter);
    m_commandList->ResourceBarrier(1, &toUav);

    FlushCommandQueue();
}

// Upload interleaved float3 RGB (OIDN's output layout) into an RGBA32F texture,
// expanding to float4 with w = 1.0. The alpha matters: the resolve pass divides
// by .w to turn the path tracer's running sum into a mean, so writing 1.0 here
// makes that normalization a no-op and lets both paths share one shader.
void DXRApp::UploadHDR(ID3D12Resource *res, const std::vector<float> &rgb)
{
    WaitForGpu(m_frameIndex);

    const UINT W = m_width, H = m_height;
    UINT64 rowPitch = ((UINT64)W * 16 + 255) & ~255ULL;
    auto upload = CreateBuffer(rowPitch * H, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD);

    uint8_t *p;
    upload->Map(0, nullptr, (void **)&p);
    for (UINT y = 0; y < H; y++)
    {
        float *row = reinterpret_cast<float *>(p + (size_t)y * rowPitch);
        const float *srcRow = rgb.data() + (size_t)y * W * 3;
        for (UINT x = 0; x < W; x++)
        {
            row[x * 4 + 0] = srcRow[x * 3 + 0];
            row[x * 4 + 1] = srcRow[x * 3 + 1];
            row[x * 4 + 2] = srcRow[x * 3 + 2];
            row[x * 4 + 3] = 1.0f;
        }
    }
    upload->Unmap(0, nullptr);

    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "C");

    // The staging texture rests in UNORDERED_ACCESS (the state the resolve pass
    // reads it in), so borrow it for the copy and hand it back.
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_commandList->ResourceBarrier(1, &b);

    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = res;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = upload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    src.PlacedFootprint.Footprint.Width = W;
    src.PlacedFootprint.Footprint.Height = H;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;
    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    m_commandList->ResourceBarrier(1, &b);

    FlushCommandQueue();
}

void DXRApp::SaveSnapshotEXR()
{
    std::string snapshotDir = filesystem::path(m_scenePath).parent_path().str();
    if (!snapshotDir.empty())
        snapshotDir += "/";
    char filename[512];
    snprintf(filename, sizeof(filename), "%ssnapshot_%u.exr", snapshotDir.c_str(), m_frameCount);
    SaveAccumResourceEXR(m_accumResource.Get(), filename);
    snprintf(filename, sizeof(filename), "%ssnapshot_%u_albedo.exr", snapshotDir.c_str(), m_frameCount);
    SaveAccumResourceEXR(m_albedoResource.Get(), filename);
    snprintf(filename, sizeof(filename), "%ssnapshot_%u_normal.exr", snapshotDir.c_str(), m_frameCount);
    SaveAccumResourceEXR(m_normalResource.Get(), filename);
}
