#include "Denoiser.h"

#include <cstdio>

bool Denoiser::Init(unsigned width, unsigned height)
{
    m_w = width;
    m_h = height;
    m_device = oidn::newDevice(oidn::DeviceType::CUDA);
    const char *err = nullptr;
    if (!m_device || m_device.getError(err) != oidn::Error::None)
        m_device = oidn::newDevice(oidn::DeviceType::Default);
    if (!m_device)
    {
        printf("[oidn] no device available\n");
        return false;
    }
    m_device.commit();
    if (m_device.getError(err) != oidn::Error::None)
    {
        printf("[oidn] device commit failed: %s\n", err ? err : "unknown");
        m_device = oidn::DeviceRef();
        return false;
    }

    printf("[oidn] device ready (%dx%d)\n", m_w, m_h);

    const size_t n = (size_t)m_w * m_h * 3 * sizeof(float);
    m_bColor = m_device.newBuffer(n);
    m_bAlbedo = m_device.newBuffer(n);
    m_bNormal = m_device.newBuffer(n);
    m_bOutput = m_device.newBuffer(n);

    m_filter = m_device.newFilter("RT");
    m_filter.setImage("color", m_bColor, oidn::Format::Float3, m_w, m_h);
    m_filter.setImage("albedo", m_bAlbedo, oidn::Format::Float3, m_w, m_h);
    m_filter.setImage("normal", m_bNormal, oidn::Format::Float3, m_w, m_h);
    m_filter.setImage("output", m_bOutput, oidn::Format::Float3, m_w, m_h);
    m_filter.set("hdr", true);
    m_filter.set("cleanAux", true);
    m_filter.set("quality", oidn::Quality::High);
    m_filter.commit();
    if (m_device.getError(err) != oidn::Error::None)
    {
        printf("[oidn] filter commit failed: %s\n", err ? err : "unknown");
        return false;
    }
    return true;
}

bool Denoiser::Denoise(const float *beauty, const float *albedo, const float *normal,
                       std::vector<float> &out)
{
    if (!m_device)
        return false;

    const size_t n = (size_t)m_w * m_h * 3 * sizeof(float);
    m_bColor.write(0, n, beauty);
    m_bAlbedo.write(0, n, albedo);
    m_bNormal.write(0, n, normal);

    m_filter.execute();

    const char *err = nullptr;
    if (m_device.getError(err) != oidn::Error::None)
    {
        printf("[oidn] execute failed: %s\n", err ? err : "unknown");
        return false;
    }

    out.resize((size_t)m_w * m_h * 3);
    m_bOutput.read(0, n, out.data());
    return true;
}
