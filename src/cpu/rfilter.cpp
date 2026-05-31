/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#include <nori/rfilter.h>
#include <tinyformat.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Gaussian image reconstruction filter
 *
 * This is a windowed Gaussian filter with configurable standard deviation
 * and radius.
 */
class GaussianFilter : public ReconstructionFilter
{
public:
    GaussianFilter(const PropertyList &propList)
    {
        m_radius = propList.getFloat("radius", 2.0f);
        m_stddev = propList.getFloat("stddev", 0.5f);
        m_alpha = std::exp(-m_stddev * m_stddev * m_radius * m_radius);
    }

    float eval(float x) const
    {
        return std::max(0.0f, std::exp(-m_stddev * m_stddev * x * x) - m_alpha);
    }

    std::string toString() const
    {
        return tfm::format("GaussianFilter[radius=%f, stddev=%f]", m_radius, m_stddev);
    }

private:
    float m_stddev, m_alpha;
};

NORI_REGISTER_CLASS(GaussianFilter, "gaussian");

NORI_NAMESPACE_END
