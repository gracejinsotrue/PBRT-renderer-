/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#pragma once

#include <object.h>
#include <string>

NORI_NAMESPACE_BEGIN

class AlphaTexture; // forward declare — full definition lives in texture.h

/**
 * \brief Convenience data structure used to pass multiple
 * parameters to the evaluation and sampling routines in \ref BSDF
 */
struct BSDFQueryRecord
{
    /// Incident direction (in the local frame)
    Vector3f wi;

    /// Outgoing direction (in the local frame)
    Vector3f wo;

    /// Relative refractive index in the sampled direction
    float eta;

    /// Measure associated with the sample
    EMeasure measure;

    Point2f uv;
    /// Create a new record for sampling the BSDF
    BSDFQueryRecord(const Vector3f &wi)
        : wi(wi), eta(1.f), measure(EUnknownMeasure) {}

    /// Create a new record for querying the BSDF
    BSDFQueryRecord(const Vector3f &wi,
                    const Vector3f &wo, EMeasure measure)
        : wi(wi), wo(wo), eta(1.f), measure(measure) {}
};

/// GPU-side material data extracted from a BSDF for the DXR port.
struct BSDFGPUData
{
    enum Type
    {
        DIFFUSE = 0,
        MIRROR = 1,
        DIELECTRIC = 2,
        MICROFACET = 3,
        DISNEY = 4,
        HAIR = 5,
        SUBSURFACE = 6
    };
    int type = DIFFUSE;
    float albedo[3] = {0.5f, 0.5f, 0.5f};
    float intIOR = 1.5046f;
    float extIOR = 1.000277f;
    float alpha = 0.1f;

    std::string albedoTexture;
    std::string normalTexture;
    std::string roughnessTexture;
    std::string metallicTexture;
    std::string specularTexture;   // overrides scalar `specular` if set (sampled from .r)
    std::string subsurfaceTexture; // overrides scalar `subsurface` if set (sampled from .r)
    std::string alphaTexture; // <-- NEW: path to RGBA image for alpha masking

    // Disney BRDF parameters (Burley 2012). All default to reasonable
    // neutral values so non-Disney materials ignore them safely.
    // Disney's baseColor is stored in `albedo` above to share texture
    // plumbing with other BSDFs.
    float roughness = 0.5f;
    float metallic = 0.0f;
    float specular = 0.5f;     // F0 multiplier for non-metals (0.5 => 4% F0)
    float specularTint = 0.0f; // tint specular lobe toward baseColor hue
    float sheen = 0.0f;
    float sheenTint = 0.5f;
    float subsurface = 0.0f; // blend toward Hanrahan-Krueger diffuse lobe
    float clearcoat = 0.0f;
    float clearcoatGloss = 1.0f; // 0=rough clearcoat, 1=glossy clearcoat
    float anisotropic = 0.0f;    // isotropic for now; implemented in later step

    // Hair BCSDF (Chiang 2016). Read only when type == HAIR (5).
    float betaN = 0.3f; // azimuthal roughness β_N
};

/**
 * \brief Superclass of all bidirectional scattering distribution functions
 */
class BSDF : public NoriObject
{
public:
    /**
     * \brief Sample the BSDF and return the importance weight (i.e. the
     * value of the BSDF * cos(theta_o) divided by the probability density
     * of the sample with respect to solid angles).
     *
     * \param bRec    A BSDF query record
     * \param sample  A uniformly distributed sample on \f$[0,1]^2\f$
     *
     * \return The BSDF value divided by the probability density of the sample
     *         sample. The returned value also includes the cosine
     *         foreshortening factor associated with the outgoing direction,
     *         when this is appropriate. A zero value means that sampling
     *         failed.
     */
    virtual Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const = 0;

    /**
     * \brief Evaluate the BSDF for a pair of directions and measure
     * specified in \code bRec
     *
     * \param bRec
     *     A record with detailed information on the BSDF query
     * \return
     *     The BSDF value, evaluated for each color channel
     */
    virtual Color3f eval(const BSDFQueryRecord &bRec) const = 0;

    /**
     * \brief Compute the probability of sampling \c bRec.wo
     * (conditioned on \c bRec.wi).
     *
     * This method provides access to the probability density that
     * is realized by the \ref sample() method.
     *
     * \param bRec
     *     A record with detailed information on the BSDF query
     *
     * \return
     *     A probability/density value expressed with respect
     *     to the specified measure
     */

    virtual float pdf(const BSDFQueryRecord &bRec) const = 0;

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.)
     * provided by this instance
     * */
    EClassType getClassType() const { return EBSDF; }

    /**
     * \brief Return whether or not this BRDF is diffuse. This
     * is primarily used by photon mapping to decide whether
     * or not to store photons on a surface
     */
    virtual bool isDiffuse() const { return false; }

    /**
     * \brief Return material parameters for GPU upload
     */
    virtual BSDFGPUData getGPUData() const { return BSDFGPUData{}; }

    /**
     * \brief Return the alpha mask texture, or nullptr if this
     * material has no alpha masking.  Used by the BVH traversal
     * to skip transparent hits.
     */
    virtual const AlphaTexture *getAlphaTexture() const { return nullptr; }
};

NORI_NAMESPACE_END