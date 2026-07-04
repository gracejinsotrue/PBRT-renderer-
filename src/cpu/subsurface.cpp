/*
    for SSS.
    Scene usage:
      <bsdf type="subsurface">
        <color name="albedo" value="0.9 0.6 0.5"/>   <!-- scatter color A          -->
        <float name="radius" value="0.1"/>            <!-- mean free path d (world)  -->
        <float name="intIOR" value="1.33"/>           <!-- interior IOR (skin ~1.33) -->
        <float name="extIOR" value="1.000277"/>       <!-- exterior IOR (air)         -->
        <float name="g"      value="0.0"/>            <!-- HG phase anisotropy        -->
      </bsdf>

    Field reuse (keeps the GPUMaterial layout unchanged): albedo -> A,
    subsurface -> d, anisotropic -> g, intIOR/extIOR -> boundary.
*/

#include <bsdf.h>
#include <frame.h>
#include <warp.h>

NORI_NAMESPACE_BEGIN

class Subsurface : public BSDF
{
public:
    Subsurface(const PropertyList &propList)
    {
        m_albedo = propList.getColor("albedo", Color3f(0.8f, 0.5f, 0.4f));
        m_radius = propList.getFloat("radius", 0.1f);
        m_intIOR = propList.getFloat("intIOR", 1.33f);
        m_extIOR = propList.getFloat("extIOR", 1.000277f);
        m_g = propList.getFloat("g", 0.0f);
    }

    Color3f eval(const BSDFQueryRecord &bRec) const
    {
        if (bRec.measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0.f || Frame::cosTheta(bRec.wo) <= 0.f)
            return Color3f(0.f);
        return m_albedo * INV_PI;
    }

    float pdf(const BSDFQueryRecord &bRec) const
    {
        if (bRec.measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0.f || Frame::cosTheta(bRec.wo) <= 0.f)
            return 0.f;
        return INV_PI * Frame::cosTheta(bRec.wo);
    }

    Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const
    {
        if (Frame::cosTheta(bRec.wi) <= 0.f)
            return Color3f(0.f);
        bRec.measure = ESolidAngle;
        bRec.wo = Warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.f;
        return m_albedo;
    }

    bool isDiffuse() const { return true; }

    BSDFGPUData getGPUData() const
    {
        BSDFGPUData d;
        d.type = BSDFGPUData::SUBSURFACE;
        d.albedo[0] = m_albedo.r();
        d.albedo[1] = m_albedo.g();
        d.albedo[2] = m_albedo.b();
        d.intIOR = m_intIOR;
        d.extIOR = m_extIOR;
        d.subsurface = m_radius;
        d.anisotropic = m_g;
        return d;
    }

    std::string toString() const
    {
        return tfm::format(
            "Subsurface[\n"
            "  albedo = %s,\n"
            "  radius = %f,\n"
            "  intIOR = %f,\n"
            "  extIOR = %f,\n"
            "  g = %f\n"
            "]",
            m_albedo.toString(), m_radius, m_intIOR, m_extIOR, m_g);
    }

private:
    Color3f m_albedo;
    float m_radius;
    float m_intIOR;
    float m_extIOR;
    float m_g;
};

NORI_REGISTER_CLASS(Subsurface, "subsurface");
NORI_NAMESPACE_END
