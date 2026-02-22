#include <nori/integrator.h>
#include <nori/warp.h>
#include <nori/scene.h>
NORI_NAMESPACE_BEGIN
class SimpleIntegrator : public Integrator
{
public:
    SimpleIntegrator(const PropertyList &props)
    {
        m_position = props.getPoint("position");
        m_power = props.getColor("power");
    }
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Intersection its; // find the intersection of the ray with the scene
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);              // ray miss so return black background
        Vector3f toLight = m_position - its.p; // compute the direction to the light source.

        float distSquared = toLight.squaredNorm();
        float dist = std::sqrt(distSquared);

        Vector3f toLightDir = toLight / dist; // normalize the direction to the light source
        float cosTheta = std::max(0.0f, its.shFrame.n.dot(toLightDir));
        // return Color3f(cosTheta);
        // // return Color3f(1.0f);
        if (cosTheta == 0.0f)
            return Color3f(0.0f); // light is parallel to the surface, so there is no contribution
        // for an actual shadow we need to check if anything blocks path from x to liigjt.
        Ray3f shadowRay(its.p, toLightDir, Epsilon, dist - Epsilon); // create a shadow ray from the intersection point to the light source
        if (scene->rayIntersect(shadowRay))                          // check if the shadow ray intersects anything
            return Color3f(0.0f);                                    // if it does, then the point shadow
        // otherwise we create light
        // L(x) = Phi / (4 * pi^2) * cos(theta) / ||x - p||^2
        Color3f radiance = m_power * (cosTheta / (4.0f * M_PI * M_PI * distSquared));
        return radiance;
    }
    std::string toString() const
    {
        return "six seven";
    }

private:
    Point3f m_position;
    Color3f m_power;
};
NORI_REGISTER_CLASS(SimpleIntegrator, "simple");
NORI_NAMESPACE_END