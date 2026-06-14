#pragma once

// Thin wrapper around Intel Open Image Denoise (OIDN) for the in-app denoise
// path. Built interop-ready, the image data lives in OIDN device
// buffers, so B2 can later swap these for buffers imported from D3D12-shared
// resources (oidnNewBuffer(ptr,...) over a CUDA-mapped D3D12 heap) without
// touching the call sites.
//
// OIDN is included only in this header's .cpp; callers use the opaque interface.

#include <OpenImageDenoise/oidn.hpp>

#include <string>
#include <vector>

class Denoiser
{
public:
    // Create a CUDA device, allocate W*H Float3 buffers, and build + commit the RT filter. Returns false if no
    // device could be created. Commit loads the model weights (~seconds), so call this once and reuse.
    bool Init(unsigned width, unsigned height);
    bool Available() const { return (bool)m_device; }
    bool Denoise(const float *beauty, const float *albedo, const float *normal,
                 std::vector<float> &out);

private:
    oidn::DeviceRef m_device;
    oidn::FilterRef m_filter;
    oidn::BufferRef m_bColor, m_bAlbedo, m_bNormal, m_bOutput;
    unsigned m_w = 0, m_h = 0;
};
