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

// for tent t, we just use p(x,y) = p1(x)p1(y) because the axesa re distinct.
// basically when t is in the range of [01,0] we integrate p1(t) = 1+t - > p1(t) = (1+t)^2/2
// for t in the range of [0,1], integrate p1(t) = 1-t -> p1(t) = (1-t)^2/2 + 1/2
// invert by settingξ = P₁(t) and solve for t:

// ξ < 0.5 → ξ = (1+t)²/2 → t = √(2ξ) − 1
// ξ ≥ 0.5 → ξ = 1 − (1−t)²/2 → t = 1 − √(2(1−ξ))
//  this provides the helper:
static float tentInverseCDF(float xi)
{
    if (xi < 0.5f)
        return std::sqrt(2.0f * xi) - 1.0f;
    else
        return 1.0f - std::sqrt(2.0f * (1.0f - xi));
}
// the square to tent warp functions applies this inverse cdf to both axes
// reminder that both axes are independent, that is why we can transform each coordinate on its own
Point2f Warp::squareToTent(const Point2f &sample)
{
    return Point2f(tentInverseCDF(sample.x()), tentInverseCDF(sample.y()));
}
// pdf evaluates the tent on each coordiante ant multiplies
//  also note that outside of the range [-1,1] the pdf is 0 so real
float Warp::squareToTentPdf(const Point2f &p)
{
    auto tent1D = [](float t) -> float
    {
        return (t >= -1.0f && t <= 1.0f) ? (1.0f - std::abs(t)) : 0.0f;
    };
    return tent1D(p.x()) * tent1D(p.y());
}
// transform uniformly distributed 2d points on a unit square - > uniformly distributed points on unit SPHERE centered at origin
// the problem with a uniform disc is that if we naively do ts there will be bunching near the center.
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

// this is basically the way they taught us in lectur
Vector3f Warp::squareToUniformSphere(const Point2f &sample)
{
    float costheta = 1.0f - 2.0f * sample.x();                              // cos(theta) uniformly distributed in [-1, 1]
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - costheta * costheta)); // sin(theta) derived from cos(theta)

    float phi = 2.0f * M_PI * sample.y();

    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), costheta);
}

float Warp::squareToUniformSpherePdf(const Vector3f &v)
{
    return INV_FOURPI;
}
// literally the sphere thing but half
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
// transofrms 2d point to a point distributed on the unit hemisphere w/ cosine density function
// 1. this samples point unfiormly on unit disl
// 2. lift ts hemisphere by computing sqrt of 1- x^2 - y^2 which is just hemispehre equation
// cool
Vector3f Warp::squareToCosineHemisphere(const Point2f &sample)
{
    Point2f d = squareToUniformDisk(sample);
    float z = std::sqrt(std::max(0.0f, 1.0f - d.x() * d.x() - d.y() * d.y()));
    return Vector3f(d.x(), d.y(), z);
}
// therefore the pdf is jsut costheta/pi, where cos theta is the z component of the vector
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
// the distribution in the homework is D(θ,φ) = 1/(2π) · 2·e^(−tan²θ/α²) / (α²·cos³θ)
//  you can simplify it by canclleing out the twos. anwyas this is the code
//
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
