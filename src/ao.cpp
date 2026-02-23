#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/sampler.h>

NORI_NAMESPACE_BEGIN

// 1. sample a random direction on the hemisphere
// use square to cosine hemisphere for this
// 2. transform it from local coord to world coord
//  3. shoot a shadow rayt in that direction, check if it is blocked

class AOIntegrator : public Integrator
{
public:
    AOIntegrator(const PropertyList &props)
    {
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);

        // use the 2d sampler and get random point!
        Point2f sample = sampler->next2D();

        // wrap ts random point to a cosine-weighted direction on the hemisphere (in local coordinates)
        Vector3f localDir = Warp::squareToCosineHemisphere(sample);

        // transform from local (around z-axis) to world (around the surface normal)
        Vector3f worldDir = its.shFrame.toWorld(localDir);

        // thene just shoot a shadow ray in that direction
        Ray3f shadowRay(its.p, worldDir, Epsilon, std::numeric_limits<float>::infinity());
        if (scene->rayIntersect(shadowRay))
            return Color3f(0.0f);

        return Color3f(1.0f);
    }

    std::string toString() const
    {
        return "AOIntegrator[]";
    }
};

NORI_REGISTER_CLASS(AOIntegrator, "ao");
NORI_NAMESPACE_END