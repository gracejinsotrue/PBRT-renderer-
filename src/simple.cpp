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
        m_radius = props.getFloat("radius", 0.0f); // default value 0, if there is >0 then that is a spherical source
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

        // if radius is 0 therefore a point light, then do allat point light logic here
        if (m_radius == 0.0f)
        {

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
        else // this is for a spherical light source! raidus > 0 in ts case
        {
            // rthe half angle of a cone is: sin(θ_max) = radius / distance
            float sinThetaMax = m_radius / dist;
            sinThetaMax = std::min(sinThetaMax, 1.0f);
            float cosThetaMax = std::sqrt(1.0f - sinThetaMax * sinThetaMax);

            // uniformly sample a direction from within the cone
            Point2f sample = sampler->next2D();
            float cosAlpha = 1.0f - sample.x() * (1.0f - cosThetaMax);
            float sinAlpha = std::sqrt(std::max(0.0f, 1.0f - cosAlpha * cosAlpha));
            float phi = 2.0f * M_PI * sample.y();

            // build sampled direction around z-axis
            Vector3f localDir(sinAlpha * std::cos(phi), sinAlpha * std::sin(phi), cosAlpha);

            // then transform to world space
            Frame lightFrame(toLightDir);
            Vector3f worldDir = lightFrame.toWorld(localDir);

            // Reject samples that go below the surface
            if (its.shFrame.n.dot(worldDir) <= 0.0f)
                return Color3f(0.0f);
            Ray3f shadowRay(its.p, worldDir, Epsilon, std::numeric_limits<float>::infinity());
            if (scene->rayIntersect(shadowRay))
                return Color3f(0.0f);

            // the solid angle of a cone is 2π(1 - cos(θ_max))
            float solidAngle = 2.0f * M_PI * (1.0f - cosThetaMax);

            // radiance from a Lambertian sphere is total power Phi emitted over surface area 4πr² into hemisphere π
            // So surface radiance = Phi / (4πr² · π) = Phi / (4π²r²)
            Color3f Le = m_power / (4.0f * M_PI * M_PI * m_radius * m_radius);

            // the general monte carlo estimator is : Le * cos(θ_at_surface) * solidAngle / π
            // but we are uniformly sampling a cone! we can simplify this to: Le * solidAngle * cos(θ) / π  where θ is angle at shading point
            float cosAtSurface = its.shFrame.n.dot(worldDir);
            Color3f radiance = Le * solidAngle * cosAtSurface * INV_PI;

            return radiance;
        }
    }
    std::string toString() const
    {
        return "six seven";
    }

private:
    Point3f m_position;
    Color3f m_power;
    float m_radius;
};
NORI_REGISTER_CLASS(SimpleIntegrator, "simple");
NORI_NAMESPACE_END