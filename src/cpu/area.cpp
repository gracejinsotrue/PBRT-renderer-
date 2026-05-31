#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN

class AreaLight : public Emitter
{
public:
    AreaLight(const PropertyList &props)
    {
        m_radiance = props.getColor("radiance");
    }

    Color3f getRadiance() const override
    {
        return m_radiance;
    }

    Color3f sample(const Point2f &sample, Point3f &p, Normal3f &n, float &pdf) const override
    {
        m_mesh->samplePosition(sample, p, n, pdf);
        return m_radiance;
    }

    float pdf() const override
    {
        return 1.0f / m_mesh->getDiscretePDF().getSum();
    }

    std::string toString() const override
    {
        return tfm::format(
            "AreaLight[\n"
            "  radiance = %s\n"
            "]",
            m_radiance.toString());
    }

private:
    Color3f m_radiance;
};

NORI_REGISTER_CLASS(AreaLight, "area");
NORI_NAMESPACE_END