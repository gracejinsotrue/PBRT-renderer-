/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#include <nori/warp.h>
#include <nori/vector.h>
#include <nori/frame.h>

NORI_NAMESPACE_BEGIN

Point2f Warp::squareToUniformSquare(const Point2f &sample)
{
    return sample;
}

float Warp::squareToUniformSquarePdf(const Point2f &sample)
{
    return ((sample.array() >= 0).all() && (sample.array() <= 1).all()) ? 1.0f : 0.0f;
}

static float tentInverseCDF(float xi)
{
    if (xi < 0.5f)
        return std::sqrt(2.0f * xi) - 1.0f;
    else
        return 1.0f - std::sqrt(2.0f * (1.0f - xi));
}
Point2f Warp::squareToTent(const Point2f &sample)
{
    return Point2f(tentInverseCDF(sample.x()), tentInverseCDF(sample.y()));
}
float Warp::squareToTentPdf(const Point2f &p)
{
    auto tent1D = [](float t) -> float
    {
        return (t >= -1.0f && t <= 1.0f) ? (1.0f - std::abs(t)) : 0.0f;
    };
    return tent1D(p.x()) * tent1D(p.y());
}
Point2f Warp::squareToUniformDisk(const Point2f &sample)
{
    float r = std::sqrt(sample.x());
    float theta = 2.0f * M_PI * sample.y();
    return Point2f(r * std::cos(theta), r * std::sin(theta));
}

float Warp::squareToUniformDiskPdf(const Point2f &p)
{
    return (p.x() * p.x() + p.y() * p.y() <= 1.0f) ? (1.0f / M_PI) : 0.0f;
}

Vector3f Warp::squareToUniformSphere(const Point2f &sample)
{
    float costheta = 1.0f - 2.0f * sample.x();
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - costheta * costheta));

    float phi = 2.0f * M_PI * sample.y();

    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), costheta);
}

float Warp::squareToUniformSpherePdf(const Vector3f &v)
{
    return INV_FOURPI;
}
Vector3f Warp::squareToUniformHemisphere(const Point2f &sample)
{
    float cosTheta = 1.0f - sample.x();
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * M_PI * sample.y();

    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}

float Warp::squareToUniformHemispherePdf(const Vector3f &v)
{
    return (v.z() >= 0.0f) ? INV_TWOPI : 0.0f;
}
Vector3f Warp::squareToCosineHemisphere(const Point2f &sample)
{
    Point2f d = squareToUniformDisk(sample);
    float z = std::sqrt(std::max(0.0f, 1.0f - d.x() * d.x() - d.y() * d.y()));
    return Vector3f(d.x(), d.y(), z);
}
float Warp::squareToCosineHemispherePdf(const Vector3f &v)
{
    return (v.z() > 0.0f) ? v.z() * INV_PI : 0.0f;
}

Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha)
{
    float phi = 2.0f * M_PI * sample.y();
    float tanTheta2 = -alpha * alpha * std::log(std::max(1e-8f, 1.0f - sample.x()));
    float cosTheta = 1.0f / std::sqrt(1.0f + tanTheta2);
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));

    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}
float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha)
{
    if (m.z() <= 0.0f)
        return 0.0f;

    float cosTheta = m.z();
    float cosTheta2 = cosTheta * cosTheta;
    float tanTheta2 = (1.0f - cosTheta2) / cosTheta2;
    float cosTheta3 = cosTheta2 * cosTheta;
    float alpha2 = alpha * alpha;

    return std::exp(-tanTheta2 / alpha2) / (M_PI * alpha2 * cosTheta3);
}

NORI_NAMESPACE_END
