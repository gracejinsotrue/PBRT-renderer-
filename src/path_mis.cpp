#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

// combines both strategies from 1. path tracing with next event estimation, or
// 2. mats material sampling path tracer.
class PathMisIntegrator : public Integrator
{
public:
    PathMisIntegrator(const PropertyList &props) {}

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Color3f result(0.0f);
        Color3f throughput(1.0f);
        Ray3f currentRay = ray;
        float eta = 1.0f;
        int bounces = 0;
        bool lastBounceSpecular = true;

        while (true)
        {
            Intersection its;
            if (!scene->rayIntersect(currentRay, its))
                break;

            const BSDF *bsdf = its.mesh->getBSDF();
            if (its.mesh->isEmitter() && lastBounceSpecular)
                result += throughput * its.mesh->getEmitter()->getRadiance();

            if (bsdf->isDiffuse())
            {
                result += throughput * misDirectIllumination(scene, sampler, currentRay, its);
            }

            BSDFQueryRecord bRec(its.toLocal(-currentRay.d));
            bRec.uv = its.uv; // pass UV so textured BSDFs can look up the right color
            Color3f bsdfWeight = bsdf->sample(bRec, sampler->next2D());

            if (bsdfWeight.isZero())
                break;

            if (bsdf->isDiffuse())
            {
                Vector3f wo = its.toWorld(bRec.wo);
                Ray3f bsdfRay(its.p, wo, Epsilon, std::numeric_limits<float>::infinity());
                Intersection lightIts;
                if (scene->rayIntersect(bsdfRay, lightIts) && lightIts.mesh->isEmitter())
                {
                    Color3f Le = lightIts.mesh->getEmitter()->getRadiance();

                    float pb = bsdf->pdf(bRec);

                    float emitterSelectionPdf = 1.0f / emitterCount(scene);
                    float dist = (lightIts.p - its.p).norm();
                    float cosTheta_y = std::abs(lightIts.geoFrame.n.dot(-bsdfRay.d));
                    float pdfArea = lightIts.mesh->getEmitter()->pdf() * emitterSelectionPdf;
                    float pl = pdfArea * dist * dist / std::max(cosTheta_y, 1e-8f);

                    //  w_bsdf = p_bsdf / (p_bsdf + p_light)
                    float wb = pb / (pb + pl);

                    result += throughput * bsdfWeight * Le * wb;
                }
            }

            throughput *= bsdfWeight;
            eta *= bRec.eta;
            lastBounceSpecular = !bsdf->isDiffuse();
            bounces++;

            if (bounces >= 3)
            {
                float q = std::min(throughput.maxCoeff() * eta * eta, 0.99f);
                if (sampler->next1D() >= q)
                    break;
                throughput /= q;
            }

            currentRay = Ray3f(its.p, its.toWorld(bRec.wo), Epsilon,
                               std::numeric_limits<float>::infinity());
        }

        return result;
    }
    std::string toString() const
    {
        return "PathMisIntegrator[]";
    }

private:
    int emitterCount(const Scene *scene) const
    {
        int count = 0;
        for (const auto *mesh : scene->getMeshes())
            if (mesh->isEmitter())
                count++;
        return std::max(count, 1);
    }
    // this is just from the emitter block inside misSample() from direct.cpp.
    /// just from direct.cp.
    // 1. pick an emitter
    // 2. samsple a point
    // 3, test the shadow
    // 4 wl = pl / (pb + pl), return weighted contribution
    Color3f misDirectIllumination(const Scene *scene, Sampler *sampler,
                                  const Ray3f &ray, const Intersection &its) const
    {
        std::vector<const Mesh *> emitterMeshes;
        for (const auto *mesh : scene->getMeshes())
            if (mesh->isEmitter())
                emitterMeshes.push_back(mesh);
        if (emitterMeshes.empty())
            return Color3f(0.0f);

        float emitterSelectionPdf = 1.0f / emitterMeshes.size();
        const BSDF *bsdf = its.mesh->getBSDF();

        int emitterIdx = std::min(
            (int)(sampler->next1D() * emitterMeshes.size()),
            (int)emitterMeshes.size() - 1);
        const Mesh *emitterMesh = emitterMeshes[emitterIdx];

        Point3f y;
        Normal3f ny;
        float pdfArea;
        emitterMesh->getEmitter()->sample(sampler->next2D(), y, ny, pdfArea);
        pdfArea *= emitterSelectionPdf;

        Vector3f toLight = y - its.p;
        float distSquared = toLight.squaredNorm();
        float dist = std::sqrt(distSquared);
        Vector3f wi = toLight / dist;

        float cosTheta_x = its.shFrame.n.dot(wi);
        float cosTheta_y = std::abs(ny.dot(-wi));

        if (cosTheta_x <= 0.0f || cosTheta_y <= 0.0f)
            return Color3f(0.0f);

        Ray3f shadowRay(its.p, wi, Epsilon, dist - Epsilon);
        if (scene->rayIntersect(shadowRay))
            return Color3f(0.0f);
        BSDFQueryRecord bRec(its.toLocal(-ray.d), its.toLocal(wi), ESolidAngle);
        bRec.uv = its.uv;
        Color3f fr = bsdf->eval(bRec);

        float pl = pdfArea * distSquared / cosTheta_y;

        float pb = bsdf->pdf(bRec);

        float wl = pl / (pb + pl);

        Color3f Le = emitterMesh->getEmitter()->getRadiance();
        float G = cosTheta_x * cosTheta_y / distSquared;

        return fr * G * Le * wl / pdfArea;
    }
};

NORI_REGISTER_CLASS(PathMisIntegrator, "path_mis");
NORI_NAMESPACE_END