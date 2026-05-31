#include "DXRApp.h"

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
        m_cameraDirty = false;
    }
}

void DXRApp::OnKeyDown(UINT8 key)
{
    m_keys[key] = true;
    if (key == 'P')
        SaveSnapshot();
    if (key == 'O')
        SaveSnapshotEXR();
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

void DXRApp::SaveSnapshotEXR()
{
    WaitForGpu(m_frameIndex);

    D3D12_RESOURCE_DESC desc = m_accumResource->GetDesc();
    UINT64 rowPitch = ((desc.Width * 16 + 255) & ~255ULL);
    UINT64 totalSize = rowPitch * desc.Height;

    auto readback = CreateBuffer(totalSize, D3D12_RESOURCE_FLAG_NONE,
                                 D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK);

    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset(), "A");
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), nullptr), "C");

    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = m_accumResource.Get();
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

    src.pResource = m_accumResource.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

    m_commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    m_commandList->ResourceBarrier(1, &b);

    FlushCommandQueue();

    const float *data;
    readback->Map(0, nullptr, (void **)&data);

    const UINT W = (UINT)desc.Width;
    const UINT H = desc.Height;
    const UINT pixelCount = W * H;

    std::vector<float> rgb(pixelCount * 3);
    for (UINT y = 0; y < H; y++)
    {
        const float *row = data + y * (rowPitch / sizeof(float));
        for (UINT x = 0; x < W; x++)
        {
            float r = row[x * 4 + 0];
            float g = row[x * 4 + 1];
            float bv = row[x * 4 + 2];
            float w = row[x * 4 + 3];
            float inv = (w > 0.0f) ? (1.0f / w) : 0.0f;
            UINT idx = y * W + x;
            rgb[idx * 3 + 0] = r * inv;
            rgb[idx * 3 + 1] = g * inv;
            rgb[idx * 3 + 2] = bv * inv;
        }
    }

    readback->Unmap(0, nullptr);

    std::string snapshotDir = filesystem::path(m_scenePath).parent_path().str();
    if (!snapshotDir.empty())
        snapshotDir += "/";
    char filename[512];
    snprintf(filename, sizeof(filename), "%ssnapshot_%u.exr", snapshotDir.c_str(), m_frameCount);

    const char *err = nullptr;
    int ret = SaveEXR(rgb.data(), (int)W, (int)H, 3, 0, filename, &err);
    if (ret != TINYEXR_SUCCESS)
        printf("[snapshot] EXR save failed: %s\n", err ? err : "unknown error");
    else
        printf("[snapshot] Saved %s (%u samples, HDR linear)\n", filename, m_frameCount);
}
