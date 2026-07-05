// Material.hlsli
// Material dispatch layer: routes material type IDs to the correct BSDF.
//
// Requires: Common, RNG, GeometryUtils, Microfacet, Disney, Hair

#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

#include "Common.hlsli"
#include "RNG.hlsli"
#include "GeometryUtils.hlsli"
#include "Microfacet.hlsli"
#include "Disney.hlsli"
#include "Hair.hlsli"

// `h` is the hair fiber offset in [-1, 1]

float3 MaterialEval(float3 wi, float3 wo, GPUMaterial mat, float h)
{
    if (mat.type == 0) // Diffuse
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return float3(0, 0, 0);
        return MatAlbedo(mat) * M_INV_PI;
    }
    else if (mat.type == 3) // Microfacet
    {
        return MicrofacetEval(wi, wo, mat);
    }
#if HAS_DISNEY
    else if (mat.type == 4) // Disney
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return float3(0, 0, 0);
        float3 fDiffuse = DisneyDiffuseLobe(wi, wo, mat);
        float3 fSpec = DisneySpecularEval(wi, wo, mat);
        float3 fSheen = DisneySheenEval(wi, wo, mat);
        float fCC = DisneyClearcoatEval(wi, wo, mat);
        return (1.0 - mat.metallic) * (fDiffuse + fSheen) + fSpec + fCC;
    }
#endif
#if HAS_HAIR
    else if (mat.type == 5) // Hair
    {
        return HairBCSDF_Eval(wi, wo, mat, h);
    }
#endif
    else if (mat.type == 6)
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return float3(0, 0, 0);
        float a = max(mat.roughness * mat.roughness, 1e-4);
        float3 wh = normalize(wi + wo);
        float D = BeckmannD(wh, a);
        float Fr = FresnelDielectric(dot(wh, wi), mat.extIOR, mat.intIOR);
        float G = SmithG1(wi, wh, a) * SmithG1(wo, wh, a);
        return float3(1, 1, 1) * (D * Fr * G / max(4.0 * wi.z * wo.z, 1e-10));
    }
    return float3(0, 0, 0);
}

float MaterialPdf(float3 wi, float3 wo, GPUMaterial mat, float h)
{
    if (mat.type == 0) // Diffuse
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return 0.0;
        return wo.z * M_INV_PI;
    }
    else if (mat.type == 3) // Microfacet
    {
        return MicrofacetPdf(wi, wo, mat);
    }
#if HAS_DISNEY
    else if (mat.type == 4) // Disney
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return 0.0;
        DisneyLobeProbs p = DisneyComputeLobeProbs(mat);
        float pdfDiff = wo.z * M_INV_PI;
        float pdfSpec = DisneySpecularPdf(wi, wo, mat);
        float pdfCC = DisneyClearcoatPdf(wi, wo, mat);
        return p.pDiffuse * pdfDiff + p.pSpecular * pdfSpec + p.pClearcoat * pdfCC;
    }
#endif
#if HAS_HAIR
    else if (mat.type == 5) // Hair
    {
        return HairBCSDF_Pdf(wi, wo, mat, h);
    }
#endif
    else if (mat.type == 6)
    {
        if (wi.z <= 0.0 || wo.z <= 0.0)
            return 0.0;
        float a = max(mat.roughness * mat.roughness, 1e-4);
        float3 wh = normalize(wi + wo);
        float Fr = FresnelDielectric(dot(wh, wi), mat.extIOR, mat.intIOR);
        return Fr * BeckmannDCosTheta(wh, a) / max(4.0 * dot(wh, wo), 1e-10);
    }
    return 0.0;
}

// Sample an outgoing direction. Returns f * cos(theta_o) / pdf (, which is the Monte Carlo throughput weight).
// Outputs wo in the local frame and the solid-angle pdf. Not intended for delta BSDFs
float3 MaterialSample(float3 wi, inout RNG rng, GPUMaterial mat, float h,
                      out float3 wo, out float pdf)
{
    if (mat.type == 0) // Diffuse
    {
        if (wi.z <= 0.0)
        {
            wo = float3(0, 0, 1);
            pdf = 0.0;
            return float3(0, 0, 0);
        }
        float2 u = float2(NextFloat(rng), NextFloat(rng));
        wo = CosineSampleHemisphere(u);
        pdf = wo.z * M_INV_PI;
        return MatAlbedo(mat);
    }
    else if (mat.type == 3) // Microfacet
    {
        return MicrofacetSample(wi, rng, mat, wo, pdf);
    }
#if HAS_DISNEY
    else if (mat.type == 4) // Disney
    {
        if (wi.z <= 0.0)
        {
            wo = float3(0, 0, 1);
            pdf = 0.0;
            return float3(0, 0, 0);
        }

        DisneyLobeProbs p = DisneyComputeLobeProbs(mat);
        float u0 = NextFloat(rng);
        float2 u12 = float2(NextFloat(rng), NextFloat(rng));

        // pick a lobe which is one of diffuse, specular, or clearcoat.
        if (u0 < p.pDiffuse)
        {
            // Diffuse lobe covers Burley, HK subsurface, and sheen which are all cosine hemispehre sampled
            wo = CosineSampleHemisphere(u12);
        }
        else if (u0 < p.pDiffuse + p.pSpecular)
        {
            // specular GGX lobe
            float alpha = max(mat.roughness * mat.roughness, 1e-4);
            float3 wh = DisneyGGX_SampleWh(u12, alpha);
            wo = 2.0 * dot(wh, wi) * wh - wi;
        }
        else
        {
            // clearcoat (GTR1) lobe.
            float alpha = DisneyClearcoatAlpha(mat);
            float3 wh = DisneyGTR1_SampleWh(u12, alpha);
            wo = 2.0 * dot(wh, wi) * wh - wi;
        }

        if (wo.z <= 0.0)
        {
            pdf = 0.0;
            return float3(0, 0, 0);
        }

        // Combined pdf across all three sampling lobes
        float pdfDiff = wo.z * M_INV_PI;
        float pdfSpec = DisneySpecularPdf(wi, wo, mat);
        float pdfCC = DisneyClearcoatPdf(wi, wo, mat);
        pdf = p.pDiffuse * pdfDiff + p.pSpecular * pdfSpec + p.pClearcoat * pdfCC;
        if (pdf <= 0.0)
            return float3(0, 0, 0);

        // Full BSDF at the sampled direction.
        float3 fDiffuse = DisneyDiffuseLobe(wi, wo, mat);
        float3 fSpec = DisneySpecularEval(wi, wo, mat);
        float3 fSheen = DisneySheenEval(wi, wo, mat);
        float fCC = DisneyClearcoatEval(wi, wo, mat);
        float3 fTotal = (1.0 - mat.metallic) * (fDiffuse + fSheen) + fSpec + fCC;

        return fTotal * wo.z / pdf;
    }
#endif
#if HAS_HAIR
    else if (mat.type == 5) // hiar
    {
        return HairBCSDF_Sample(wi, rng, mat, h, wo, pdf);
    }
#endif
    wo = float3(0, 0, 1);
    pdf = 0.0;
    return float3(0, 0, 0);
}

#endif // MATERIAL_HLSLI
