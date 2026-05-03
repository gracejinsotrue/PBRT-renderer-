/*
    Medium base class for participating media (fog, clouds, etc.)
*/

#pragma once

#include <nori/object.h>
#include <nori/color.h>
#include <nori/bbox.h>
#include <nori/proplist.h>
#include <nori/common.h>

NORI_NAMESPACE_BEGIN

/// Read a property that may be authored as either a scalar <float> or a
/// chromatic <color>. Color form takes precedence; scalar form broadcasts
/// to RGB; missing falls back to scalarDefault. Centralized here because
/// PropertyList::getColor / getFloat both throw on type mismatch, so
/// supporting both forms requires explicit catch handling.
inline Color3f readScalarOrColor(const PropertyList &propList,
                                 const std::string &name,
                                 float scalarDefault)
{
    try { return propList.getColor(name); }
    catch (const NoriException &) {}
    try { return Color3f(propList.getFloat(name)); }
    catch (const NoriException &) {}
    return Color3f(scalarDefault);
}

class Medium : public NoriObject
{
public:
    virtual ~Medium() {}

    /// Absorption coefficient (scalar — kept for backward compat).
    virtual float getSigmaA() const = 0;

    /// Scattering coefficient (scalar — kept for backward compat).
    virtual float getSigmaS() const = 0;

    /// Absorption coefficient as RGB. Default broadcasts the scalar so
    /// existing media stay valid; subclasses override to expose chromatic
    /// extinction (colored fog, tinted water, etc.).
    virtual Color3f getSigmaARGB() const { return Color3f(getSigmaA()); }

    /// Scattering coefficient as RGB. Same broadcast contract as
    /// getSigmaARGB.
    virtual Color3f getSigmaSRGB() const { return Color3f(getSigmaS()); }

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