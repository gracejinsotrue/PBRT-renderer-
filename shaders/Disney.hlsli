// Disney.hlsli
// Disney Principled BRDF (Burley 2012 / Disney 2015).
// Multi-lobe model, diffuse (Burley + HK subsurface), GGX specular, sheen, and GTR1 clearcoat.

#ifndef DISNEY_HLSLI
#define DISNEY_HLSLI

// Diffuse lobes

float3 DisneyDiffuseEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);

    float3 wh = normalize(wi + wo);
    float cosThetaD = abs(dot(wo, wh));

    float FD90 = 0.5 + 2.0 * mat.roughness * cosThetaD * cosThetaD;

    // Schlick-style Fresnel in view and light directions
    float FV = 1.0 + (FD90 - 1.0) * pow(1.0 - wi.z, 5.0);
    float FL = 1.0 + (FD90 - 1.0) * pow(1.0 - wo.z, 5.0);

    return MatAlbedo(mat) * M_INV_PI * FV * FL;
}

// Hanrahan-Krueger subsurface-approximation diffuse term from Disney 2012. When `subsurface > 0`, Disney blends this with the Burley lobe to fake
// SSS. the todo here would be to change into real SSS if time provides. this also is the lobe a real BSSRDF would eventually replace
float3 DisneyHKSubsurfaceEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);

    float3 wh = normalize(wi + wo);
    float cosThetaD = abs(dot(wo, wh));

    float FSS90 = mat.roughness * cosThetaD * cosThetaD;
    float FSS_V = 1.0 + (FSS90 - 1.0) * pow(1.0 - wi.z, 5.0);
    float FSS_L = 1.0 + (FSS90 - 1.0) * pow(1.0 - wo.z, 5.0);

    float cosSum = max(wi.z + wo.z, 1e-4);
    float ss = 1.25 * (FSS_V * FSS_L * (1.0 / cosSum - 0.5) + 0.5);

    return MatAlbedo(mat) * M_INV_PI * ss;
}

// blend the two diffuse lobes by the `subsurface` parameter
float3 DisneyDiffuseLobe(float3 wi, float3 wo, GPUMaterial mat)
{
    float3 fBurley = DisneyDiffuseEval(wi, wo, mat);
    float3 fSS = DisneyHKSubsurfaceEval(wi, wo, mat);
    return lerp(fBurley, fSS, mat.subsurface);
}

// GGX specular

// GGX microfacet which was taught in lecture but also that Disney requires
//
// The point of GGX is that it has longer tails that match the BRDFs of real materials
// better. The isotropic form is implemented here
//
// the convention is that, alpha = roughness squared so that `roughness` feels linear.
// For example, a roughness of 0.5 sits halfway between mirror and matte from a perceptual standpoint

float DisneyGGX_D(float NdotH, float alpha)
{
    if (NdotH <= 0.0)
        return 0.0;
    float a2 = alpha * alpha;
    float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
    return a2 / (M_PI * t * t);
}

float DisneyGGX_G1(float NdotV, float alpha)
{
    if (NdotV <= 0.0)
        return 0.0;
    float a2 = alpha * alpha;
    float denom = NdotV + sqrt(a2 + (1.0 - a2) * NdotV * NdotV);
    if (denom < 1e-12)
        return 0.0;
    return 2.0 * NdotV / denom;
}

float3 DisneyGGX_SampleWh(float2 u, float alpha)
{
    float a2 = alpha * alpha;
    float cosTheta2 = (1.0 - u.x) / (1.0 + (a2 - 1.0) * u.x);
    float cosTheta = sqrt(saturate(cosTheta2));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta2));
    float phi = 2.0 * M_PI * u.y;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float DisneyGGX_PdfWo(float3 wi, float3 wo, float alpha)
{
    float3 wh = normalize(wi + wo);
    if (wh.z <= 0.0)
        return 0.0;
    float D = DisneyGGX_D(wh.z, alpha);
    return D * wh.z / max(4.0 * abs(dot(wh, wo)), 1e-10);
}

float3 FresnelSchlick(float3 F0, float cosThetaD)
{
    float x = 1.0 - cosThetaD;
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return F0 + (1.0 - F0) * x5;
}

float3 DisneySpecularF0(GPUMaterial mat)
{
    float3 base = MatAlbedo(mat);
    float lum = Luminance(base);
    float3 Ctint = (lum > 0.0) ? base / lum : float3(1.0, 1.0, 1.0);
    float3 Ks = lerp(float3(1.0, 1.0, 1.0), Ctint, mat.specularTint);
    float3 F0_dielectric = 0.08 * mat.specular * Ks;
    return lerp(F0_dielectric, base, mat.metallic);
}

float3 DisneySpecularEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);

    float alpha = max(mat.roughness * mat.roughness, 1e-4);
    float3 wh = normalize(wi + wo);
    if (wh.z <= 0.0)
        return float3(0, 0, 0);

    float D = DisneyGGX_D(wh.z, alpha);
    float3 F = FresnelSchlick(DisneySpecularF0(mat), abs(dot(wi, wh)));
    float Gv = DisneyGGX_G1(wi.z, alpha);
    float Gl = DisneyGGX_G1(wo.z, alpha);
    float G = Gv * Gl;

    return D * F * G / max(4.0 * wi.z * wo.z, 1e-10);
}

float DisneySpecularPdf(float3 wi, float3 wo, GPUMaterial mat)
{
    if (wi.z <= 0.0 || wo.z <= 0.0)
        return 0.0;
    float alpha = max(mat.roughness * mat.roughness, 1e-4);
    return DisneyGGX_PdfWo(wi, wo, alpha);
}

// Sheen

float3 DisneySheenEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (mat.sheen <= 0.0 || wi.z <= 0.0 || wo.z <= 0.0)
        return float3(0, 0, 0);

    float3 wh = normalize(wi + wo);
    float cosThetaD = abs(dot(wo, wh));

    float3 base = MatAlbedo(mat);
    float lum = Luminance(base);
    float3 Ctint = (lum > 0.0) ? base / lum : float3(1.0, 1.0, 1.0);
    float3 Csheen = lerp(float3(1.0, 1.0, 1.0), Ctint, mat.sheenTint);

    float fresnel = pow(1.0 - cosThetaD, 5.0);
    return mat.sheen * Csheen * fresnel;
}

// Clearcoat (GTR1)

float DisneyGTR1_D(float NdotH, float alpha)
{
    if (NdotH <= 0.0)
        return 0.0;
    float a2 = alpha * alpha;
    if (a2 >= 0.9999)
        return M_INV_PI;
    float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
    return (a2 - 1.0) / (M_PI * log(a2) * t);
}

// Sample wh from GTR1.
// inverse CDF =    cos^2(th_h) = (1 - (a^2)^(1-u1)) / (1 - a^2)
// for alpha=1, it is a flat NDF and it falls back to uniform hemisphere.
float3 DisneyGTR1_SampleWh(float2 u, float alpha)
{
    float a2 = alpha * alpha;
    float cosTheta2;
    if (a2 >= 0.9999)
        cosTheta2 = 1.0 - u.x;
    else
        cosTheta2 = (1.0 - pow(a2, 1.0 - u.x)) / (1.0 - a2);
    float cosTheta = sqrt(saturate(cosTheta2));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta2));
    float phi = 2.0 * M_PI * u.y;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// outgoing-direction pdf under GTR1 half-vector sampling.
float DisneyGTR1_PdfWo(float3 wi, float3 wo, float alpha)
{
    float3 wh = normalize(wi + wo);
    if (wh.z <= 0.0)
        return 0.0;
    float D = DisneyGTR1_D(wh.z, alpha);
    return D * wh.z / max(4.0 * abs(dot(wh, wo)), 1e-10);
}

// remap gloss [0,1] to the alpha Disney uses where clearcoatGloss=1 is smoothest.
float DisneyClearcoatAlpha(GPUMaterial mat)
{
    return lerp(0.1, 0.001, mat.clearcoatGloss);
}

// Clearcoat BRDF which returns the scalar lobe value
float DisneyClearcoatEval(float3 wi, float3 wo, GPUMaterial mat)
{
    if (mat.clearcoat <= 0.0 || wi.z <= 0.0 || wo.z <= 0.0)
        return 0.0;

    float3 wh = normalize(wi + wo);
    if (wh.z <= 0.0)
        return 0.0;

    float alpha = DisneyClearcoatAlpha(mat);
    float D = DisneyGTR1_D(wh.z, alpha);

    // Schlick Fresnel with fixed F0 = 0.04
    float VdotH = abs(dot(wi, wh));
    float F = 0.04 + (1.0 - 0.04) * pow(1.0 - VdotH, 5.0);

    // Smith G with fixed alpha=0.25
    // Using the GGX G1 helper since it's just Smith masking. the alpha value is what Disney's paper specifies here.
    float Gv = DisneyGGX_G1(wi.z, 0.25);
    float Gl = DisneyGGX_G1(wo.z, 0.25);
    float G = Gv * Gl;

    return 0.25 * mat.clearcoat * D * F * G / max(4.0 * wi.z * wo.z, 1e-10);
}

float DisneyClearcoatPdf(float3 wi, float3 wo, GPUMaterial mat)
{
    if (mat.clearcoat <= 0.0 || wi.z <= 0.0 || wo.z <= 0.0)
        return 0.0;
    float alpha = DisneyClearcoatAlpha(mat);
    return DisneyGTR1_PdfWo(wi, wo, alpha);
}

// Lobe sampling probabilities

// Disney heuristic for picking which lobe to sample from:
//   w_diffuse   = 1 - metallic
//   w_specular  = 1
//   w_clearcoat = 0.25 * clearcoat
// And then they are normalized so they sum to 1

struct DisneyLobeProbs
{
    float pDiffuse;
    float pSpecular;
    float pClearcoat;
};

DisneyLobeProbs DisneyComputeLobeProbs(GPUMaterial mat)
{
    DisneyLobeProbs p;
    float wDiff = 1.0 - mat.metallic;
    float wSpec = 1.0;
    float wCC = 0.25 * mat.clearcoat;
    float total = wDiff + wSpec + wCC;

    float inv = (total > 0.0) ? 1.0 / total : 0.0;
    p.pDiffuse = wDiff * inv;
    p.pSpecular = wSpec * inv;
    p.pClearcoat = wCC * inv;
    return p;
}

#endif // DISNEY_HLSLI
