#include <nori/integrator.h>
#include <nori/warp.h>
#include <nori/scene.h>
NORI_NAMESPACE_BEGIN
class SimpleIntegrator : public Integrator
{
public:
    SimpleIntegrator(const PropertyList &props) {}
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        return Color3f(6.7f);
    }
    std::string toString() const
    {
        return "six seven";
    }
};
NORI_REGISTER_CLASS(SimpleIntegrator, "simple");
NORI_NAMESPACE_END