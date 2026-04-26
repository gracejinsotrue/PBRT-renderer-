/*
    Medium base class for participating media (fog, clouds, etc.)
*/

#pragma once

#include <nori/object.h>
#include <nori/bbox.h>

NORI_NAMESPACE_BEGIN

class Medium : public NoriObject
{
public:
    virtual ~Medium() {}

    /// Absorption coefficient
    virtual float getSigmaA() const = 0;

    /// Scattering coefficient
    virtual float getSigmaS() const = 0;

    /// Henyey-Greenstein asymmetry parameter
    virtual float getPhaseG() const = 0;

    /// Whether this medium has spatially varying density
    virtual bool isHeterogeneous() const { return false; }

    /// Path to volume density file (.vol), empty if none
    virtual std::string getVolumePath() const { return ""; }

    /// Whether explicit bounds were specified
    virtual bool hasExplicitBounds() const = 0;

    virtual BoundingBox3f getExplicitBounds() const = 0;

    EClassType getClassType() const { return EMedium; }
};

NORI_NAMESPACE_END