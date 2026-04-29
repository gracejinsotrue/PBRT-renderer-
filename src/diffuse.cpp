/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>
#include <nori/texture.h>
#include <memory>

NORI_NAMESPACE_BEGIN

/**
 * \brief Diffuse / Lambertian BRDF model
 */
class Diffuse : public BSDF
{
public:
    Diffuse(const PropertyList &propList)
    {

        m_albedo = propList.getColor("albedo", Color3f(0.5f));
        m_albedoTexture = propList.getString("albedoTexture", "");
        m_normalTexture = propList.getString("normalTexture", "");
        m_roughnessTexture = propList.getString("roughnessTexture", "");
        m_metallicTexture = propList.getString("metallicTexture", "");
        // Load image if a path was given
        if (!m_albedoTexture.empty())
            m_texture = std::make_unique<Texture2D>(m_albedoTexture);
    }

    /// Evaluate the BRDF model
    Color3f eval(const BSDFQueryRecord &bRec) const
    {
        /* This is a smooth BRDF -- return zero if the measure
           is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0)
            return Color3f(0.0f);

        /* The BRDF is simply the albedo / pi */
        return getAlbedo(bRec.uv) * INV_PI;
    }

    /// Compute the density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord &bRec) const
    {
        /* This is a smooth BRDF -- return zero if the measure
           is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;

        /* Importance sampling density wrt. solid angles:
           cos(theta) / pi.

           Note that the directions in 'bRec' are in local coordinates,
           so Frame::cosTheta() actually just returns the 'z' component.
        */
        return INV_PI * Frame::cosTheta(bRec.wo);
    }

    /// Draw a a sample from the BRDF model
    Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const
    {
        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        /* Warp a uniformly distributed sample on [0,1]^2
           to a direction on a cosine-weighted hemisphere */
        bRec.wo = Warp::squareToCosineHemisphere(sample);

        /* Relative index of refraction: no change */
        bRec.eta = 1.0f;

        /* eval() / pdf() * cos(theta) = albedo. There
           is no need to call these functions. */
        return getAlbedo(bRec.uv);
    }

    bool isDiffuse() const
    {
        return true;
    }

    BSDFGPUData getGPUData() const
    {
        BSDFGPUData d;
        d.type = BSDFGPUData::DIFFUSE;
        d.albedo[0] = m_albedo.r();
        d.albedo[1] = m_albedo.g();
        d.albedo[2] = m_albedo.b();
        d.albedoTexture = m_albedoTexture;
        d.normalTexture = m_normalTexture;
        d.roughnessTexture = m_roughnessTexture;
        d.metallicTexture = m_metallicTexture;
        return d;
    }

    /// Return a human-readable summary
    std::string toString() const
    {
        return tfm::format(
            "Diffuse[\n"
            "  albedo = %s\n"
            "]",
            m_albedo.toString());
    }

    EClassType getClassType() const { return EBSDF; }

private:
    Color3f m_albedo;
    std::string m_albedoTexture;
    std::string m_normalTexture;
    std::string m_roughnessTexture;
    std::string m_metallicTexture;

    std::unique_ptr<Texture2D> m_texture;
    Color3f getAlbedo(const Point2f &uv) const
    {
        return m_texture ? m_texture->eval(uv) : m_albedo;
    }
};

NORI_REGISTER_CLASS(Diffuse, "diffuse");
NORI_NAMESPACE_END
