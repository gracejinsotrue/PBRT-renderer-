#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/warp.h>
#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN
 
class LightTestIntegrator : public Integrator {
public:
    LightTestIntegrator(const PropertyList &props) {}
 
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {

        // This function should sample a point p on an emitter and return its coordinates as a color.
        Point3f p(0.5f, 0.5f, 0.5f);
        return Color3f(p.x(), p.y(), p.z());
    }
 
    /// Return a human-readable description for debugging purposes
    std::string toString() const {
        return tfm::format(
            "LightTestIntegrator[]"
        );
    }
};
 
NORI_REGISTER_CLASS(LightTestIntegrator, "lighttest");
NORI_NAMESPACE_END