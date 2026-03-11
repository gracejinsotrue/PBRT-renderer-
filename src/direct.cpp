#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class DirectIntegrator : public Integrator
{
public:
    DirectIntegrator(const PropertyList &props)
    {
        m_strategy = props.getString("strategy", "mis");
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);

        Color3f result(0.0f);
        if (its.mesh->isEmitter())
            result += its.mesh->getEmitter()->getRadiance();

        if (m_strategy == "bsdf")
        {
            result += bsdfSample(scene, sampler, ray, its);
        }
        else if (m_strategy == "emitter")
        {
            result += emitterSample(scene, sampler, ray, its);
        }
        else if (m_strategy == "mis")
        {
            result += misSample(scene, sampler, ray, its);
        }

        return result;
    }

    Color3f bsdfSample(const Scene *scene, Sampler *sampler, const Ray3f &ray, const Intersection &its) const
    {
        // 1. sample a direction from the BSDF
        const BSDF *bsdf = its.mesh->getBSDF();
        BSDFQueryRecord bRec(its.toLocal(-ray.d));
        Color3f bsdfWeight = bsdf->sample(bRec, sampler->next2D());

        if (bsdfWeight.isZero())
            return Color3f(0.0f);

        // 2. trace ray in sampled dir
        Vector3f wo = its.toWorld(bRec.wo);
        Ray3f shadowRay(its.p, wo, Epsilon, std::numeric_limits<float>::infinity());

        Intersection lightIts;
        if (!scene->rayIntersect(shadowRay, lightIts))
            return Color3f(0.0f);

        // do stuff if we hit an emitter
        if (!lightIts.mesh->isEmitter())
            return Color3f(0.0f);

        Color3f Le = lightIts.mesh->getEmitter()->getRadiance();
        return Le * bsdfWeight;
    }

    Color3f emitterSample(const Scene *scene, Sampler *sampler, const Ray3f &ray, const Intersection &its) const
    {
        // 1. collect all the emitter meshes
        std::vector<const Mesh *> emitterMeshes;
        for (const auto *mesh : scene->getMeshes())
        {
            if (mesh->isEmitter())
                emitterMeshes.push_back(mesh);
        }
        if (emitterMeshes.empty())
            return Color3f(0.0f);

        // 2. randomly pick emitter
        int emitterIdx = std::min((int)(sampler->next1D() * emitterMeshes.size()), (int)emitterMeshes.size() - 1);
        const Mesh *emitterMesh = emitterMeshes[emitterIdx];
        float emitterSelectionPdf = 1.0f / emitterMeshes.size();

        // sample point
        Point3f y;
        Normal3f ny;
        float pdfArea;
        emitterMesh->getEmitter()->sample(sampler->next2D(), y, ny, pdfArea);

        // what is the probabiltiy of choosing ts emitter
        pdfArea *= emitterSelectionPdf;

        Vector3f toLight = y - its.p;
        float distSquared = toLight.squaredNorm();
        float dist = std::sqrt(distSquared);
        Vector3f wi = toLight / dist;

        // cos at shade
        float cosTheta_x = its.shFrame.n.dot(wi);

        if (cosTheta_x <= 0.0f)
            return Color3f(0.0f);

        // cos at light
        float cosTheta_y = std::abs(ny.dot(-wi));
        if (cosTheta_y <= 0.0f)
            return Color3f(0.0f);
        // is our ray blocked in shadow
        Ray3f shadowRay(its.p, wi, Epsilon, dist - Epsilon);
        if (scene->rayIntersect(shadowRay))
            return Color3f(0.0f);

        //  evaluate the BSDF
        BSDFQueryRecord bRec(its.toLocal(-ray.d), its.toLocal(wi), ESolidAngle);
        Color3f fr = its.mesh->getBSDF()->eval(bRec);

        // fr * G * Le / pdf_area estimator
        Color3f Le = emitterMesh->getEmitter()->getRadiance();
        float G = cosTheta_x * cosTheta_y / distSquared;

        return fr * G * Le / pdfArea;
    }

    Color3f misSample(const Scene *scene, Sampler *sampler, const Ray3f &ray, const Intersection &its) const
    {
        Color3f result(0.0f);

        // 1. collect all the emitter meshes
        std::vector<const Mesh *> emitterMeshes;
        for (const auto *mesh : scene->getMeshes())
        {
            if (mesh->isEmitter())
                emitterMeshes.push_back(mesh);
        }
        if (emitterMeshes.empty())
            return Color3f(0.0f);

        float emitterSelectionPdf = 1.0f / emitterMeshes.size();
        const BSDF *bsdf = its.mesh->getBSDF();

        // BSDF
        {
            // sample a direction from the BSDF
            BSDFQueryRecord bRec(its.toLocal(-ray.d));
            Color3f bsdfWeight = bsdf->sample(bRec, sampler->next2D());

            if (!bsdfWeight.isZero())
            {
                // trace ray in sampled dir
                Ray3f bsdfRay(its.p, its.toWorld(bRec.wo));
                Intersection lightIts;
                if (scene->rayIntersect(bsdfRay, lightIts) && lightIts.mesh->isEmitter())
                {
                    Color3f Le = lightIts.mesh->getEmitter()->getRadiance();

                    // p_b BSDF pdf is already in solid angle
                    float pb = bsdf->pdf(bRec);

                    // p_l emitter pdf converted from area to solid angle
                    // pdf_solidAngle = pdf_area * dist^2 / |cos theta_y|
                    float dist = (lightIts.p - its.p).norm();
                    float cosTheta_y = std::abs(lightIts.geoFrame.n.dot(-bsdfRay.d));
                    float pdfArea = lightIts.mesh->getEmitter()->pdf() * emitterSelectionPdf;
                    float pl = pdfArea * dist * dist / std::max(cosTheta_y, 1e-8f);
                    float wb = pb / (pb + pl);
                    result += bsdfWeight * Le * wb;
                }
            }
        }

        // emitter
        {
            // randomly pick emitter
            int emitterIdx = std::min((int)(sampler->next1D() * emitterMeshes.size()), (int)emitterMeshes.size() - 1);
            const Mesh *emitterMesh = emitterMeshes[emitterIdx];

            // sample point on emitter
            Point3f y;
            Normal3f ny;
            float pdfArea;
            emitterMesh->getEmitter()->sample(sampler->next2D(), y, ny, pdfArea);
            pdfArea *= emitterSelectionPdf;

            Vector3f toLight = y - its.p;
            float distSquared = toLight.squaredNorm();
            float dist = std::sqrt(distSquared);
            Vector3f wi = toLight / dist;

            // cos at shade point and light point
            float cosTheta_x = its.shFrame.n.dot(wi);
            float cosTheta_y = std::abs(ny.dot(-wi));

            if (cosTheta_x > 0.0f && cosTheta_y > 0.0f)
            {
                Ray3f shadowRay(its.p, wi, Epsilon, dist - Epsilon);
                if (!scene->rayIntersect(shadowRay))
                {
                    // evaluate the BSDF
                    BSDFQueryRecord bRec(its.toLocal(-ray.d), its.toLocal(wi), ESolidAngle);
                    Color3f fr = bsdf->eval(bRec);

                    float pl = pdfArea * distSquared / cosTheta_y;
                    float pb = bsdf->pdf(bRec);
                    float wl = pl / (pb + pl);

                    Color3f Le = emitterMesh->getEmitter()->getRadiance();
                    float G = cosTheta_x * cosTheta_y / distSquared;

                    result += fr * G * Le * wl / pdfArea;
                }
            }
        }

        return result;
    }

    std::string toString() const
    {
        return tfm::format("DirectIntegrator[strategy=%s]", m_strategy);
    }

private:
    std::string m_strategy;
};

NORI_REGISTER_CLASS(DirectIntegrator, "direct");
NORI_NAMESPACE_END