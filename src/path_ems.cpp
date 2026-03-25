#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN
//path tracer w/ next event 

class PathEmsIntegrator : public Integrator {
public:
    PathEmsIntegrator(const PropertyList &props) { }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
        Color3f result(0.0f);
        Color3f throughput(1.0f);
        Ray3f currentRay = ray;
        float eta = 1.0f;
        int bounces = 0;

        // on the very first ray from the camera, if we hit an emitter
        // we must count it. there's no previous next event that already
        // accounted for it
        bool lastBounceSpecular = true;

        while (true) {
            Intersection its;
            if (!scene->rayIntersect(currentRay, its))
                break;

            // add emitter contribution ONLY if:
            //   1. this is bounce 0, OR
            //   2. The previous bounce was specular, therefore we skipped NEE
            // Otherwise, Next at the previous bounce already counted
            // this light, so adding it again would double-count.
            if (its.mesh->isEmitter() && lastBounceSpecular)
                result += throughput * its.mesh->getEmitter()->getRadiance();

            const BSDF *bsdf = its.mesh->getBSDF();

            // applicable only for non-specular surfaces. for specular BSDFs,
            // eval() returns 0 for any specific direction, so light sampling would always contribute nothing.
            if (bsdf->isDiffuse()) {
                result += throughput * emitterSample(scene, sampler, currentRay, its);
            }
            // for the next bsdf bounce 
            BSDFQueryRecord bRec(its.toLocal(-currentRay.d));
            Color3f bsdfWeight = bsdf->sample(bRec, sampler->next2D());

            if (bsdfWeight.isZero())
                break;

            throughput *= bsdfWeight;
            eta *= bRec.eta;
            lastBounceSpecular = !bsdf->isDiffuse();

            bounces++;
            if (bounces >= 3) {
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
    // This line and below is just copied from directcpp emitter sampling. The logic is idenitical.


    Color3f emitterSample(const Scene *scene, Sampler *sampler,
                          const Ray3f &ray, const Intersection &its) const {
        std::vector<const Mesh *> emitterMeshes;
        for (const auto *mesh : scene->getMeshes()) {
            if (mesh->isEmitter())
                emitterMeshes.push_back(mesh);
        }
        if (emitterMeshes.empty())
            return Color3f(0.0f);

        int emitterIdx = std::min(
            (int)(sampler->next1D() * emitterMeshes.size()),
            (int)emitterMeshes.size() - 1);
        const Mesh *emitterMesh = emitterMeshes[emitterIdx];
        float emitterSelectionPdf = 1.0f / emitterMeshes.size();

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
        if (cosTheta_x <= 0.0f)
            return Color3f(0.0f);

        float cosTheta_y = std::abs(ny.dot(-wi));
        if (cosTheta_y <= 0.0f)
            return Color3f(0.0f);

        Ray3f shadowRay(its.p, wi, Epsilon, dist - Epsilon);
        if (scene->rayIntersect(shadowRay))
            return Color3f(0.0f);

        BSDFQueryRecord bRec(its.toLocal(-ray.d), its.toLocal(wi), ESolidAngle);
        Color3f fr = its.mesh->getBSDF()->eval(bRec);

        Color3f Le = emitterMesh->getEmitter()->getRadiance();
        float G = cosTheta_x * cosTheta_y / distSquared;

        return fr * G * Le / pdfArea;
    }

    std::string toString() const {
        return "PathEmsIntegrator[]";
    }
};

NORI_REGISTER_CLASS(PathEmsIntegrator, "path_ems");
NORI_NAMESPACE_END