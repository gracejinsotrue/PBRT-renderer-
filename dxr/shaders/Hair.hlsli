// Hair.hlsli
// Implements Hair BCSDF (Chiang et al. 2016 / PBRT 4th ed. Section 9.9)
//
// References throughout:
//   [PBRT]   Section 9.9 "Scattering from Hair"
//   [Chiang] "A Practical and Controllable Hair and Fur Model"
//
// The hair local frame has: x = fiber tangent, y = bitangent, z = tube normal.
// Longitudinal angle: sinθ = ω.x     (PBRT Section 9.9.1)
// Azimuthal angle:    φ = atan2(ω.z, ω.y)
// pMax = 3: lobes R(p=0), TT(p=1), TRT(p=2), residual(p=3)
//
// Requires: Common.hlsli, GeometryUtils.hlsli, RNG.hlsli

#ifndef HAIR_HLSLI
#define HAIR_HLSLI

static const int HAIR_PMAX = 3;

// ============================================================================
// Special functions
// ============================================================================

// Modified Bessel function of the first kind, order 0.
// [PBRT] Section 9.9.3, used in the longitudinal scattering function Mp.
float HairI0(float x)
{
    float val = 0.0, x2i = 1.0, ifact = 1.0, i4 = 1.0;
    for (int i = 0; i < 10; i++)
    {
        if (i > 0)
        {
            x2i *= x * x;
            ifact *= float(i);
            i4 *= 4.0;
        }
        val += x2i / (i4 * ifact * ifact);
    }
    return val;
}

// log(I0(x))
float HairLogI0(float x)
{
    if (x > 12.0)
        return x + 0.5 * (-log(2.0 * M_PI) + log(1.0 / x) + 1.0 / (8.0 * x));
    return log(HairI0(x));
}

// ============================================================================
// Longitudinal scattering
// ============================================================================

// Longitudinal scattering function Mp
float HairMp(float cosTheta_i, float cosTheta_o, float sinTheta_i,
             float sinTheta_o, float v)
{
    float a = cosTheta_i * cosTheta_o / v;
    float b = sinTheta_i * sinTheta_o / v;
    float mp;
    if (v <= 0.1)
        mp = exp(HairLogI0(a) - b - 1.0 / v + 0.6931 + log(1.0 / (2.0 * v)));
    else
        mp = (exp(-b) * HairI0(a)) / (sinh(1.0 / v) * 2.0 * v);
    return mp;
}

// ============================================================================
// Azimuthal scattering
// ============================================================================

// Logistic distribution and its trimmed/sampled variants.
float HairLogistic(float x, float s)
{
    float ex = exp(-abs(x) / s);
    return ex / (s * (1.0 + ex) * (1.0 + ex));
}

float HairLogisticCDF(float x, float s)
{
    return 1.0 / (1.0 + exp(-x / s));
}

float HairTrimmedLogistic(float x, float s, float a, float b)
{
    float denom = HairLogisticCDF(b, s) - HairLogisticCDF(a, s);
    return (denom > 1e-10) ? HairLogistic(x, s) / denom : 0.0;
}

float HairSampleTrimmedLogistic(float u, float s, float a, float b)
{
    float cdfA = HairLogisticCDF(a, s);
    float cdfB = HairLogisticCDF(b, s);
    float t = cdfA + u * (cdfB - cdfA);
    t = clamp(t, 1e-6, 1.0 - 1e-6);
    return -s * log(1.0 / t - 1.0);
}

// Specular exit azimuth Φ(p, γ_o, γ_t).
// [PBRT] Section 9.9.5, Figure 9.51
float HairPhi(int p, float gamma_o, float gamma_t)
{
    return 2.0 * float(p) * gamma_t - 2.0 * gamma_o + float(p) * M_PI;
}

// Azimuthal scattering function Np.
// [PBRT] Section 9.9.5, [Chiang] Eq. 4
float HairNp(float phi, int p, float s, float gamma_o, float gamma_t)
{
    float dphi = phi - HairPhi(p, gamma_o, gamma_t);
    // Remap to [-π, π]
    while (dphi > M_PI)
        dphi -= 2.0 * M_PI;
    while (dphi < -M_PI)
        dphi += 2.0 * M_PI;
    return HairTrimmedLogistic(dphi, s, -M_PI, M_PI);
}

// ============================================================================
// Absorption & lobe parameters
// ============================================================================

// Convert hair color to absorption coefficient σ_a.
// [Chiang] Eq. 9:  least-squares fit relating multiple-scattering albedo C and azimuthal roughness β_N to single-fiber σ_a.
float3 HairSigmaAFromColor(float3 C, float betaN)
{
    float denom = 5.969 - 0.215 * betaN + 2.532 * betaN * betaN - 10.73 * betaN * betaN * betaN + 5.574 * betaN * betaN * betaN * betaN + 0.245 * betaN * betaN * betaN * betaN * betaN;
    float3 sigma_a;
    sigma_a.x = (C.x > 1e-4) ? pow(log(C.x) / denom, 2.0) : 100.0;
    sigma_a.y = (C.y > 1e-4) ? pow(log(C.y) / denom, 2.0) : 100.0;
    sigma_a.z = (C.z > 1e-4) ? pow(log(C.z) / denom, 2.0) : 100.0;
    return sigma_a;
}

// compute per-lobe roughness v[p] from β_M.
struct HairLobeParams
{
    float v[4];          // longitudinal roughness variance per lobe
    float s;             // azimuthal logistic scale
    float sin2kAlpha[3]; // cuticle tilt sin(2^k * α)
    float cos2kAlpha[3]; // cuticle tilt cos(2^k * α)
    float3 sigma_a;      // absorption coefficient
};

HairLobeParams HairComputeParams(GPUMaterial mat)
{
    HairLobeParams p;

    // Longitudinal roughness
    float bm = mat.roughness;
    float v0 = 0.726 * bm + 0.812 * bm * bm + 3.7 * pow(bm, 20.0);
    v0 = v0 * v0;
    p.v[0] = v0;
    p.v[1] = 0.25 * v0; // TT lobe
    p.v[2] = 4.0 * v0;  // TRT
    p.v[3] = p.v[2];

    // Azimuthal roughness from Chiang
    float bn = mat.betaN;
    float SqrtPiOver8 = 0.626657069;
    p.s = SqrtPiOver8 * (0.265 * bn + 1.194 * bn * bn + 5.372 * pow(bn, 22.0));

    // for cuticle scale tilt
    float alphaRad = mat.alpha * M_PI / 180.0;
    p.sin2kAlpha[0] = sin(alphaRad);
    p.cos2kAlpha[0] = sqrt(max(0.0, 1.0 - p.sin2kAlpha[0] * p.sin2kAlpha[0]));
    for (int i = 1; i < 3; i++)
    {
        p.sin2kAlpha[i] = 2.0 * p.cos2kAlpha[i - 1] * p.sin2kAlpha[i - 1];
        p.cos2kAlpha[i] = p.cos2kAlpha[i - 1] * p.cos2kAlpha[i - 1] - p.sin2kAlpha[i - 1] * p.sin2kAlpha[i - 1];
    }

    // absorption from color
    p.sigma_a = HairSigmaAFromColor(MatAlbedo(mat), bn);

    return p;
}

// Attenuation

// Compute attenuation Ap for all lobes
void HairAp(float cosTheta_o, float sinTheta_o, float h, float eta,
            float3 sigma_a, out float3 ap[4])
{
    // Modified IOR for normal-plane refraction
    float etap = sqrt(max(0.0, eta * eta - sinTheta_o * sinTheta_o)) / max(cosTheta_o, 1e-6);
    float sinGamma_t = clamp(h / etap, -1.0, 1.0);
    float cosGamma_t = sqrt(max(0.0, 1.0 - sinGamma_t * sinGamma_t));

    // Transmitted angle
    float sinTheta_t = sinTheta_o / eta;
    float cosTheta_t = sqrt(max(0.0, 1.0 - sinTheta_t * sinTheta_t));

    // Single-segment transmittance from Beer's law
    float3 T = exp(-sigma_a * (2.0 * cosGamma_t / max(cosTheta_t, 1e-6)));

    // Fresnel at entry
    float cosGamma_o = sqrt(max(0.0, 1.0 - h * h));
    float f = FresnelDielectric(cosTheta_o * cosGamma_o, 1.0, eta);

    // Lobe attenuations
    ap[0] = float3(f, f, f);           // R: surface reflection
    ap[1] = (1.0 - f) * (1.0 - f) * T; // TT: two transmissions
    ap[2] = ap[1] * T * f;             // TRT: +reflection +segment
    float3 tfProduct = T * f;
    float3 denom = max(float3(1, 1, 1) - tfProduct, float3(1e-10, 1e-10, 1e-10));
    ap[3] = ap[2] * f * T / denom;
}

// Cuticle scale tilt

// Apply cuticle scale tilt for lobe p.
// [PBRT] Section 9.9.6: p=0 rotates by -2α, p=1 by +α, p=2 by +4α
void HairScaleTilt(int p, float sinTheta_o, float cosTheta_o,
                   float sin2kAlpha[3], float cos2kAlpha[3],
                   out float sinThetap_o, out float cosThetap_o)
{
    if (p == 0)
    {
        sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
        cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
    }
    else if (p == 1)
    {
        sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
        cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
    }
    else if (p == 2)
    {
        sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
        cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
    }
    else
    {
        sinThetap_o = sinTheta_o;
        cosThetap_o = cosTheta_o;
    }
    cosThetap_o = abs(cosThetap_o);
}

// ============================================================================
// Eval / Pdf / Sample
// ============================================================================

// Full hair BCSDF evaluation.
// wi, wo are in the hair local frame (tangent=x, normal=z).
// h is the fiber offset ∈ [-1,1] from the tessellator.
float3 HairBCSDF_Eval(float3 wi, float3 wo, GPUMaterial mat, float h)
{
    // No hemisphere check here! hair BCSDF is valid for all directions.
    // Light enters/exits from any angle around the fiber.
    // Our wi = view direction = PBRT's "outgoing" (subscript _o)
    // Our wo = light direction = PBRT's "incident" (subscript _i)
    float sinTheta_o = wi.x;
    float cosTheta_o = sqrt(max(0.0, 1.0 - sinTheta_o * sinTheta_o));
    float phi_o = atan2(wi.z, wi.y);

    float sinTheta_i = wo.x;
    float cosTheta_i = sqrt(max(0.0, 1.0 - sinTheta_i * sinTheta_i));
    float phi_i = atan2(wo.z, wo.y);

    float phi = phi_i - phi_o;
    float gamma_o = asin(clamp(h, -1.0, 1.0));

    float eta = mat.intIOR;

    // Refracted azimuthal angle γ_t
    float etap = sqrt(max(0.0, eta * eta - sinTheta_o * sinTheta_o)) / max(cosTheta_o, 1e-6);
    float gamma_t = asin(clamp(h / etap, -1.0, 1.0));

    HairLobeParams lp = HairComputeParams(mat);

    // Attenuation
    float3 ap[4];
    HairAp(cosTheta_o, sinTheta_o, h, eta, lp.sigma_a, ap);

    // Sum over lobes
    float3 fsum = float3(0, 0, 0);
    for (int p = 0; p < HAIR_PMAX; p++)
    {
        float sinThetap_o, cosThetap_o;
        HairScaleTilt(p, sinTheta_o, cosTheta_o,
                      lp.sin2kAlpha, lp.cos2kAlpha,
                      sinThetap_o, cosThetap_o);

        float mp = HairMp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, lp.v[p]);
        float np = HairNp(phi, p, lp.s, gamma_o, gamma_t);
        fsum += mp * ap[p] * np;
    }

    // Residual lobe: isotropic azimuthal (1/2π)
    fsum += HairMp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, lp.v[HAIR_PMAX]) * ap[HAIR_PMAX] / (2.0 * M_PI);

    // divide by |cos θ_i| to cancel the cosθ factor that the rendering equation/integrator multiplies by.
    float absCosTheta = abs(wo.z);
    if (absCosTheta > 1e-6)
        fsum /= absCosTheta;

    return fsum;
}

// Hair PDF
float HairBCSDF_Pdf(float3 wi, float3 wo, GPUMaterial mat, float h)
{
    float sinTheta_o = wi.x;
    float cosTheta_o = sqrt(max(0.0, 1.0 - sinTheta_o * sinTheta_o));
    float phi_o = atan2(wi.z, wi.y);

    float sinTheta_i = wo.x;
    float cosTheta_i = sqrt(max(0.0, 1.0 - sinTheta_i * sinTheta_i));
    float phi_i = atan2(wo.z, wo.y);

    float phi = phi_i - phi_o;
    float gamma_o = asin(clamp(h, -1.0, 1.0));
    float eta = mat.intIOR;
    float etap = sqrt(max(0.0, eta * eta - sinTheta_o * sinTheta_o)) / max(cosTheta_o, 1e-6);
    float gamma_t = asin(clamp(h / etap, -1.0, 1.0));

    HairLobeParams lp = HairComputeParams(mat);

    // Compute Ap luminance for discrete lobe probabilities
    float3 ap[4];
    HairAp(cosTheta_o, sinTheta_o, h, eta, lp.sigma_a, ap);

    float apLum[4];
    float sumLum = 0.0;
    for (int i = 0; i <= HAIR_PMAX; i++)
    {
        apLum[i] = Luminance(ap[i]);
        sumLum += apLum[i];
    }
    float invSum = (sumLum > 0.0) ? 1.0 / sumLum : 0.0;

    float pdf = 0.0;
    for (int p = 0; p < HAIR_PMAX; p++)
    {
        float sinThetap_o, cosThetap_o;
        HairScaleTilt(p, sinTheta_o, cosTheta_o,
                      lp.sin2kAlpha, lp.cos2kAlpha,
                      sinThetap_o, cosThetap_o);

        pdf += HairMp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, lp.v[p]) * apLum[p] * invSum * HairNp(phi, p, lp.s, gamma_o, gamma_t);
    }
    // residual lobe for uniform azimuthal
    pdf += HairMp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, lp.v[HAIR_PMAX]) * apLum[HAIR_PMAX] * invSum * (1.0 / (2.0 * M_PI));

    return pdf;
}

// Hair BCSDF sampling, returns f * cosθ / pdf
float3 HairBCSDF_Sample(float3 wi, inout RNG rng, GPUMaterial mat, float h,
                        out float3 wo, out float pdf)
{
    float sinTheta_o = wi.x; // NOTE: wi is the viewing direction = "wo" in PBRT notation
    float cosTheta_o = sqrt(max(0.0, 1.0 - sinTheta_o * sinTheta_o));
    float phi_o = atan2(wi.z, wi.y);
    float gamma_o = asin(clamp(h, -1.0, 1.0));
    float eta = mat.intIOR;
    float etap = sqrt(max(0.0, eta * eta - sinTheta_o * sinTheta_o)) / max(cosTheta_o, 1e-6);
    float gamma_t = asin(clamp(h / etap, -1.0, 1.0));

    HairLobeParams lp = HairComputeParams(mat);

    // attenuation for lobe selection
    float3 ap[4];
    HairAp(cosTheta_o, sinTheta_o, h, eta, lp.sigma_a, ap);

    // discrete lobe probabilities from Ap luminance
    float apPDF[4];
    float sumLum = 0.0;
    for (int i = 0; i <= HAIR_PMAX; i++)
    {
        apPDF[i] = Luminance(ap[i]);
        sumLum += apPDF[i];
    }
    float invSum = (sumLum > 0.0) ? 1.0 / sumLum : 0.0;
    for (int j = 0; j <= HAIR_PMAX; j++)
        apPDF[j] *= invSum;

    float u0 = NextFloat(rng);
    int p = 0;
    float cdf = apPDF[0];
    while (p < HAIR_PMAX && u0 > cdf)
    {
        p++;
        cdf += apPDF[p];
    }
    // Remap u0 because we need to reuse it
    float lobeStart = cdf - apPDF[p];
    u0 = clamp((u0 - lobeStart) / max(apPDF[p], 1e-10), 0.0, 1.0);

    // Apply cuticle tilt for the selected lobe
    float sinThetap_o, cosThetap_o;
    HairScaleTilt(p, sinTheta_o, cosTheta_o,
                  lp.sin2kAlpha, lp.cos2kAlpha,
                  sinThetap_o, cosThetap_o);

    // Sample Mp for θ_i
    float u1 = NextFloat(rng);
    float u2 = NextFloat(rng);
    float cosTheta = 1.0 + lp.v[p] * log(max(u1, 1e-5) + (1.0 - u1) * exp(-2.0 / lp.v[p]));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float cosPhi_sample = cos(2.0 * M_PI * u2);
    float sinTheta_i = -cosTheta * sinThetap_o + sinTheta * cosPhi_sample * cosThetap_o;
    float cosTheta_i = sqrt(max(0.0, 1.0 - sinTheta_i * sinTheta_i));

    // Sample Np for φ
    float u3 = NextFloat(rng);
    float dphi;
    if (p < HAIR_PMAX)
        dphi = HairPhi(p, gamma_o, gamma_t) +
               HairSampleTrimmedLogistic(u3, lp.s, -M_PI, M_PI);
    else
        dphi = 2.0 * M_PI * u3;

    // Construct sampled direction
    float phi_i = phi_o + dphi;
    wo = float3(sinTheta_i,
                cosTheta_i * cos(phi_i),
                cosTheta_i * sin(phi_i));

    // hair scatters in all directions around fiber. Compute combined PDF across all lobes
    pdf = HairBCSDF_Pdf(wi, wo, mat, h);
    if (pdf <= 0.0)
        return float3(0, 0, 0);

    // Evaluate full BCSDF at sampled direction
    float3 f = HairBCSDF_Eval(wi, wo, mat, h);

    // Return f * |cosθ| / pdf. Since HairBCSDF_Eval already divides by |cosθ|, multiplying by |cosθ| here cancels it, giving the correct MC weight where the integrator's cosθ factor is handled
    return f * abs(wo.z) / pdf;
}

#endif // HAIR_HLSLI
