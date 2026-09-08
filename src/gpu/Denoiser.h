#pragma once

// Thin wrapper around Intel Open Image Denoise (OIDN) for the in-app denoise
// path. Built interop-ready, the image data lives in OIDN device
// buffers, so B2 can later swap these for buffers imported from D3D12-shared
// resources (oidnNewBuffer(ptr,...) over a CUDA-mapped D3D12 heap) without
// touching the call sites.
//
// The OIDN SDK is optional (see NORI_ENABLE_OIDN in CMakeLists.txt). When it is
// absent NORI_HAS_OIDN is 0, the OIDN members and header drop out, and the two
// methods become stubs in the .cpp that report the denoiser unavailable -- the
// class and its signatures stay put so the call sites compile and link
// unchanged, and Init() failing is a path they already handle.
#ifndef NORI_HAS_OIDN
#define NORI_HAS_OIDN 0
#endif

#if NORI_HAS_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

#include <string>
#include <vector>

class Denoiser
{
public:
    // Create a CUDA device, allocate W*H Float3 buffers, and build + commit the RT filter. Returns false if no
    // device could be created (and always, when built without the SDK)
    bool Init(unsigned width, unsigned height);
    bool Denoise(const float *beauty, const float *albedo, const float *normal,
                 std::vector<float> &out);

#if NORI_HAS_OIDN
    bool Available() const { return (bool)m_device; }
#else
    bool Available() const { return false; }
#endif

private:
#if NORI_HAS_OIDN
    oidn::DeviceRef m_device;
    oidn::FilterRef m_filter;
    oidn::BufferRef m_bColor, m_bAlbedo, m_bNormal, m_bOutput;
#endif
    unsigned m_w = 0, m_h = 0;
};
