/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#pragma once

#include <nori/object.h>
#include <nori/mesh.h>

NORI_NAMESPACE_BEGIN

class Emitter : public NoriObject
{
public:
    /// get radiance takes no direction because area lights emit uniformly
    virtual Color3f getRadiance() const = 0;

    /// return radiance and fills in position/normal/pdf
    virtual Color3f sample(const Point2f &sample, Point3f &p, Normal3f &n, float &pdf) const = 0;

    /// pdf is always 1/total area so we dont take args
    virtual float pdf() const = 0;

    /// set the mesh associated with this emitter
    void setMesh(Mesh *mesh) { m_mesh = mesh; }

    /// get the mesh associated with this emitter
    const Mesh *getMesh() const { return m_mesh; }

    EClassType getClassType() const { return EEmitter; }

protected:
    Mesh *m_mesh = nullptr;
};

NORI_NAMESPACE_END