/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#include <nori/bsdf.h>
#include <nori/frame.h>

NORI_NAMESPACE_BEGIN

/// Ideal dielectric BSDF
class Dielectric : public BSDF
{
public:
    Dielectric(const PropertyList &propList)
    {
        /* Interior IOR (default: BK7 borosilicate optical glass) */
        m_intIOR = propList.getFloat("intIOR", 1.5046f);

        /* Exterior IOR (default: air) */
        m_extIOR = propList.getFloat("extIOR", 1.000277f);
    }

    Color3f eval(const BSDFQueryRecord &) const
    {
        /* Discrete BRDFs always evaluate to zero in Nori */
        return Color3f(0.0f);
    }

    float pdf(const BSDFQueryRecord &) const
    {
        /* Discrete BRDFs always evaluate to zero in Nori */
        return 0.0f;
    }
    // Given an incoming ray direction wi, sample() picks outgoing direction wo and stores it in the struct bRec (bRec saves all the in/out data for the BSDF),
    // and returns weight representing light throughput for this particular bounce.
    // for a diffuse BSDF, the distribution of outgoing directions is continous.
    // However, in a dielectric there are only two possible outgoing directions. 1 is reflection, the other is refraction.
    Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const
    {
        // take the cosine of the incoming ray with the surface normal.
        // In local coordinates, the normal is (0,0,1), so cosθ = wi.z()., and yes this can be negative if the ray hits from inside the object.
        float cosThetaI = Frame::cosTheta(bRec.wi);
        // The fresnel reflectance is dependent on the angle and two IORs.
        // The implementation of the Freshnel equations is done in common.h and common.cpp. Hooray!
        float Fr = fresnel(cosThetaI, m_extIOR, m_intIOR);

        // BSDF is nonzero in 2 directions, the direction is deterministically picked from a delta
        // distribution. Therefore there is no need to evlauate a continous PDF for it
        // tell the rest of Nori that eval() and pdf() will return 0
        bRec.measure = EDiscrete;

        if (sample.x() < Fr)
        {
            // Then this is a reflection case.
            // Reflection is easy, just flip x and y
            bRec.wo = Vector3f(-bRec.wi.x(), -bRec.wi.y(), bRec.wi.z());

            // and, the medium does not change so relative IOR = 1.
            bRec.eta = 1.0f;
        }
        else
        {
            // refraction is harder! first we need to decide which side of the surface we are on.
            // let the local normal point towards outside, think of it like the exterior
            float etaI = m_extIOR, etaT = m_intIOR;
            Normal3f n(0, 0, 1);

            // if the ray comes from inside of an object:
            if (cosThetaI < 0.0f)
            {
                std::swap(etaI, etaT);
                n = Normal3f(0, 0, -1);
                cosThetaI = -cosThetaI;
            }
            // continuing on, declare the necessary floats s.t. it can apply vector form of Snell's law
            // the ratio of IORs across the interface.
            float eta = etaI / etaT;
            // once again, Snell's law: sin²θt = η² · sin²θi = η² · (1 - cos²θi)
            float sinThetaTSqr = eta * eta * (1.0f - cosThetaI * cosThetaI);

            // If sin²θt > 1, there is total internal reflection and fresnel() returns 1.0 in that case. so, we shouldn't get
            // here, but there might have floating point edge cases:
            if (sinThetaTSqr > 1.0f)
            {
                bRec.wo = Vector3f(-bRec.wi.x(), -bRec.wi.y(), bRec.wi.z());
                bRec.eta = 1.0f;
                return Color3f(1.0f);
            }

            float cosThetaT = std::sqrt(1.0f - sinThetaTSqr);

            // vector form Snell's law
            //   ωt = η·(-ωi) + (η·cosθi - cosθt)·n
            bRec.wo = eta * (-bRec.wi) + (eta * cosThetaI - cosThetaT) * Vector3f(n);
            bRec.eta = eta;
        }
        return Color3f(1.0f);
    }

    std::string toString() const
    {
        return tfm::format(
            "Dielectric[\n"
            "  intIOR = %f,\n"
            "  extIOR = %f\n"
            "]",
            m_intIOR, m_extIOR);
    }

    BSDFGPUData getGPUData() const
    {
        BSDFGPUData d;
        d.type = BSDFGPUData::DIELECTRIC;
        d.albedo[0] = d.albedo[1] = d.albedo[2] = 1.0f;
        d.intIOR = m_intIOR;
        d.extIOR = m_extIOR;
        return d;
    }

private:
    float m_intIOR, m_extIOR;
};

NORI_REGISTER_CLASS(Dielectric, "dielectric");
NORI_NAMESPACE_END
