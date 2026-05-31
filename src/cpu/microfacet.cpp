/*
    This file is part of Nori, a simple educational ray tracer
    Copyright (c) 2015 by Wenzel Jakob
*/
#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class Microfacet : public BSDF
{
public:
    Microfacet(const PropertyList &propList)
    {
        /* RMS surface roughness */
        m_alpha = propList.getFloat("alpha", 0.1f);

        /* Interior IOR (default: BK7 borosilicate optical glass) */
        m_intIOR = propList.getFloat("intIOR", 1.5046f);

        /* Exterior IOR (default: air) */
        m_extIOR = propList.getFloat("extIOR", 1.000277f);

        /* Albedo of the diffuse base material (a.k.a "kd") */
        m_kd = propList.getColor("kd", Color3f(0.5f));

        m_ks = 1 - m_kd.maxCoeff();

        m_albedoTexture = propList.getString("albedoTexture", "");
        m_normalTexture = propList.getString("normalTexture", "");
        m_roughnessTexture = propList.getString("roughnessTexture", "");
        m_metallicTexture = propList.getString("metallicTexture", "");
    }

    /// Evaluate the BRDF for the given pair of directions
    Color3f eval(const BSDFQueryRecord &bRec) const
    {
        if (Frame::cosTheta(bRec.wi) <= 0.0f || Frame::cosTheta(bRec.wo) <= 0.0f)
            return Color3f(0.0f);

        // Diffuse term: kd / pi
        Color3f diffuse = m_kd * INV_PI;

        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        // D represents the Beckmann distribution. squareToBeckmannPdf returns D(m)*cos(theta_m),
        // so divide by cos(theta_h) to recover D alone.
        float D = Warp::squareToBeckmannPdf(wh, m_alpha) / Frame::cosTheta(wh);

        // fresnel reflectance at angle between wi and wh
        float F = fresnel(wh.dot(bRec.wi), m_extIOR, m_intIOR);

        // G(ωi,ωo,ωh)=G1(ωi,ωh) G1(ωo,ωh),
        float G = smithG1(bRec.wi, wh) * smithG1(bRec.wo, wh);

        // Specular term in the equation = ks * D * F * G / (4 * cos_i * cos_o)
        float cosI = Frame::cosTheta(bRec.wi);
        float cosO = Frame::cosTheta(bRec.wo);
        Color3f specular = Color3f(m_ks * D * F * G / (4.0f * cosI * cosO));

        return diffuse + specular;
    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord &bRec) const
    {
        if (Frame::cosTheta(bRec.wi) <= 0.0f || Frame::cosTheta(bRec.wo) <= 0.0f)
            return 0.0f;

        Vector3f wh = (bRec.wi + bRec.wo).normalized();
        float Dwh_cosTheta = Warp::squareToBeckmannPdf(wh, m_alpha);
        float Jh = 1.0f / (4.0f * wh.dot(bRec.wo));
        float specPdf = m_ks * Dwh_cosTheta * Jh;

        float diffPdf = (1.0f - m_ks) * Frame::cosTheta(bRec.wo) * INV_PI;

        return specPdf + diffPdf;
    }

    Color3f sample(BSDFQueryRecord &bRec, const Point2f &_sample) const
    {
        if (Frame::cosTheta(bRec.wi) <= 0.0f)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;
        bRec.eta = 1.0f;

        Point2f sample(_sample);
        if (sample.x() < m_ks)
        {
            sample.x() /= m_ks;
            Vector3f wh = Warp::squareToBeckmann(sample, m_alpha);
            bRec.wo = 2.0f * wh.dot(bRec.wi) * wh - bRec.wi;
        }
        else
        {
            sample.x() = (sample.x() - m_ks) / (1.0f - m_ks);
            bRec.wo = Warp::squareToCosineHemisphere(sample);
        }

        if (Frame::cosTheta(bRec.wo) <= 0.0f)
            return Color3f(0.0f);

        float pdfVal = pdf(bRec);
        if (pdfVal <= 0.0f)
            return Color3f(0.0f);

        return eval(bRec) * Frame::cosTheta(bRec.wo) / pdfVal;
    }

    bool isDiffuse() const
    {
        /* While microfacet BRDFs are not perfectly diffuse, they can be
           handled by sampling techniques for diffuse/non-specular materials,
           hence we return true here */
        return true;
    }

    BSDFGPUData getGPUData() const
    {
        BSDFGPUData d;
        d.type = BSDFGPUData::MICROFACET;
        d.albedo[0] = m_kd.r();
        d.albedo[1] = m_kd.g();
        d.albedo[2] = m_kd.b();
        d.intIOR = m_intIOR;
        d.extIOR = m_extIOR;
        d.alpha = m_alpha;
        d.albedoTexture = m_albedoTexture;
        d.normalTexture = m_normalTexture;
        d.roughnessTexture = m_roughnessTexture;
        d.metallicTexture = m_metallicTexture;
        return d;
    }

    std::string toString() const
    {
        return tfm::format(
            "Microfacet[\n"
            "  alpha = %f,\n"
            "  intIOR = %f,\n"
            "  extIOR = %f,\n"
            "  kd = %s,\n"
            "  ks = %f\n"
            "]",
            m_alpha,
            m_intIOR,
            m_extIOR,
            m_kd.toString(),
            m_ks);
    }

private:
    float m_alpha;
    float m_intIOR, m_extIOR;
    float m_ks;
    Color3f m_kd;
    std::string m_albedoTexture;
    std::string m_normalTexture;
    std::string m_roughnessTexture;
    std::string m_metallicTexture;

    float smithG1(const Vector3f &v, const Vector3f &wh) const
    {
        float dotVWh = v.dot(wh);
        float dotVN = Frame::cosTheta(v);

        // v must be on same side of macro and microsurface
        if (dotVWh / dotVN <= 0.0f)
            return 0.0f;

        float b = 1.0f / (m_alpha * Frame::tanTheta(v));

        if (b >= 1.6f)
            return 1.0f;

        float b2 = b * b;
        return (3.535f * b + 2.181f * b2) / (1.0f + 2.276f * b + 2.577f * b2);
    }
};

NORI_REGISTER_CLASS(Microfacet, "microfacet");
NORI_NAMESPACE_END