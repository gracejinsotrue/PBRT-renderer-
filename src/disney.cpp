/*
    This file is part of Nori, a simple educational ray tracer

    Disney "principled" BRDF (Burley 2012).

    STEP 2 STATUS: scaffolding only. The CPU-side eval/pdf/sample
    currently behave as Lambertian so the CPU Nori renderer still
    produces a reasonable fallback and so the plumbing can be
    validated before any Disney math is added. The GPU HLSL shader
    similarly routes type==DISNEY to the diffuse code path for now.

    Disney lobes will be filled in across Steps 3-7 on the GPU side:
      Step 3: Burley diffuse
      Step 4: Hanrahan-Krueger subsurface lobe
      Step 5: GGX specular (Schlick Fresnel, anisotropy-ready)
      Step 6: Sheen
      Step 7: Clearcoat
      Step 8: Lobe-sampling MIS and energy-conservation audit

    References:
      Burley, "Physically-Based Shading at Disney", SIGGRAPH 2012
      Burley, "Extending the Disney BRDF to a BSDF", SIGGRAPH 2015
      PBRT 3rd/4th ed., Chapter 11 (implementations of Disney BRDF)
*/

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>
#include <nori/texture.h>
#include <memory>

NORI_NAMESPACE_BEGIN

class Disney : public BSDF
{
public:
    Disney(const PropertyList &propList)
    {
        // Disney's "baseColor" is our unified albedo. Accept both
        // "baseColor" (Disney vocabulary) and "albedo" (our existing
        // vocabulary) so scene files can use either.
        if (propList.getColor("baseColor", Color3f(-1.f)).r() >= 0.f)
            m_baseColor = propList.getColor("baseColor");
        else
            m_baseColor = propList.getColor("albedo", Color3f(0.8f));

        m_roughness = propList.getFloat("roughness", 0.5f);
        m_metallic = propList.getFloat("metallic", 0.0f);
        m_specular = propList.getFloat("specular", 0.5f);
        m_specularTint = propList.getFloat("specularTint", 0.0f);
        m_sheen = propList.getFloat("sheen", 0.0f);
        m_sheenTint = propList.getFloat("sheenTint", 0.5f);
        m_subsurface = propList.getFloat("subsurface", 0.0f);
        m_clearcoat = propList.getFloat("clearcoat", 0.0f);
        m_clearcoatGloss = propList.getFloat("clearcoatGloss", 1.0f);
        m_anisotropic = propList.getFloat("anisotropic", 0.0f);

        // Shared texture plumbing with the other BSDFs
        m_albedoTexture = propList.getString("albedoTexture", "");
        m_normalTexture = propList.getString("normalTexture", "");
        m_roughnessTexture = propList.getString("roughnessTexture", "");
        m_metallicTexture = propList.getString("metallicTexture", "");

        // Alpha masking (e.g. eyelashes, hair cards, foliage)
        m_alphaTextureFile = propList.getString("alphaTexture", "");
        if (!m_alphaTextureFile.empty())
            m_alphaTex = std::make_unique<AlphaTexture>(m_alphaTextureFile);
    }

    // ---- CPU-side stubs (Step 2): behave as Lambertian ----
    // The CPU renderer isn't the target for Disney; it runs only as a
    // sanity fallback. Real Disney math lives in the HLSL shader.

    Color3f eval(const BSDFQueryRecord &bRec) const
    {
        if (bRec.measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0.f ||
            Frame::cosTheta(bRec.wo) <= 0.f)
            return Color3f(0.f);
        return m_baseColor * INV_PI;
    }

    float pdf(const BSDFQueryRecord &bRec) const
    {
        if (bRec.measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0.f ||
            Frame::cosTheta(bRec.wo) <= 0.f)
            return 0.f;
        return INV_PI * Frame::cosTheta(bRec.wo);
    }

    Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const
    {
        if (Frame::cosTheta(bRec.wi) <= 0.f)
            return Color3f(0.f);
        bRec.measure = ESolidAngle;
        bRec.wo = Warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.0f;
        return m_baseColor;
    }

    bool isDiffuse() const { return true; }

    BSDFGPUData getGPUData() const
    {
        BSDFGPUData d;
        d.type = BSDFGPUData::DISNEY;
        d.albedo[0] = m_baseColor.r();
        d.albedo[1] = m_baseColor.g();
        d.albedo[2] = m_baseColor.b();

        d.roughness = m_roughness;
        d.metallic = m_metallic;
        d.specular = m_specular;
        d.specularTint = m_specularTint;
        d.sheen = m_sheen;
        d.sheenTint = m_sheenTint;
        d.subsurface = m_subsurface;
        d.clearcoat = m_clearcoat;
        d.clearcoatGloss = m_clearcoatGloss;
        d.anisotropic = m_anisotropic;

        d.albedoTexture = m_albedoTexture;
        d.normalTexture = m_normalTexture;
        d.roughnessTexture = m_roughnessTexture;
        d.metallicTexture = m_metallicTexture;
        d.alphaTexture = m_alphaTextureFile;
        return d;
    }

    const AlphaTexture *getAlphaTexture() const override { return m_alphaTex.get(); }

    std::string toString() const
    {
        return tfm::format(
            "Disney[\n"
            "  baseColor = %s,\n"
            "  roughness = %f,\n"
            "  metallic = %f,\n"
            "  specular = %f,\n"
            "  specularTint = %f,\n"
            "  sheen = %f,\n"
            "  sheenTint = %f,\n"
            "  subsurface = %f,\n"
            "  clearcoat = %f,\n"
            "  clearcoatGloss = %f,\n"
            "  anisotropic = %f\n"
            "]",
            m_baseColor.toString(),
            m_roughness, m_metallic, m_specular, m_specularTint,
            m_sheen, m_sheenTint, m_subsurface,
            m_clearcoat, m_clearcoatGloss, m_anisotropic);
    }

private:
    Color3f m_baseColor;
    float m_roughness;
    float m_metallic;
    float m_specular;
    float m_specularTint;
    float m_sheen;
    float m_sheenTint;
    float m_subsurface;
    float m_clearcoat;
    float m_clearcoatGloss;
    float m_anisotropic;

    std::string m_albedoTexture;
    std::string m_normalTexture;
    std::string m_roughnessTexture;
    std::string m_metallicTexture;

    // Alpha masking
    std::string m_alphaTextureFile;
    std::unique_ptr<AlphaTexture> m_alphaTex;
};

NORI_REGISTER_CLASS(Disney, "disney");
NORI_NAMESPACE_END