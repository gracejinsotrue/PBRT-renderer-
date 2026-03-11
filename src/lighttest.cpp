#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/warp.h>
#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN

class LightTestIntegrator : public Integrator
{
public:
    LightTestIntegrator(const PropertyList &props) {}

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        const auto &meshes = scene->getMeshes();
        for (const auto *mesh : meshes)
        {
            if (mesh->isEmitter())
            {
                Point3f p;
                Normal3f n;
                float pdf;
                mesh->getEmitter()->sample(sampler->next2D(), p, n, pdf);
                return Color3f(p.x(), p.y(), p.z());
            }
        }
        return Color3f(0.0f);
    }

    /// Return a human-readable description for debugging purposes
    std::string toString() const
    {
        return tfm::format(
            "LightTestIntegrator[]");
    }
};

NORI_REGISTER_CLASS(LightTestIntegrator, "lighttest");
NORI_NAMESPACE_END