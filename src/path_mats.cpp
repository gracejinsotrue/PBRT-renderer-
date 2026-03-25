#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>
// pa4 path_mats brute force ray yracer.
//strategy 1 is BSDF sampling, which shall lead to path trace that always extends its path by samplinga direction according to the BSDF and tracing a ray to find the next point
// for BSDF, we extend whitted style direct illumination to become a full path tracer that accounts for indirect illumination
// this is pretty similar to the direct integrator, but it includes reflected radiance, not only emitted radiance.


NORI_NAMESPACE_BEGIN

class PathMatsIntegrator : public Integrator {
public:
    PathMatsIntegrator(const PropertyList &props) { }
    
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
        Color3f result(0.0f);    
        Color3f throughput(1.0f); // this throughput represents the path throughput weight
        Ray3f currentRay = ray;
        float eta = 1.0f;        // accumulated IOR ratio product
        int bounces = 0;
        //it was recommended to do an iterative approach despite the recursive approach being closer to recursive estimator!
        while (true) {
            Intersection its;
            if (!scene->rayIntersect(currentRay, its))
                break;

            // if we hit an emitter, then just add its contribution weighted by the current path throughput. 
            if (its.mesh->isEmitter())
                result += throughput * its.mesh->getEmitter()->getRadiance();

            // Sample the BSDF to get the next direction
            const BSDF *bsdf = its.mesh->getBSDF();
            BSDFQueryRecord bRec(its.toLocal(-currentRay.d));
            Color3f bsdfWeight = bsdf->sample(bRec, sampler->next2D());
            if (bsdfWeight.isZero())
                break;
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            bounces++;

            //  "Additionally, we recommend to only start doing Russian Roulette after at least three bounces in order to avoid terminating very short paths which can lead to unnecessarily high variance. ""
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

    std::string toString() const {
        return "PathMatsIntegrator[]";
    }
};

NORI_REGISTER_CLASS(PathMatsIntegrator, "path_mats");
NORI_NAMESPACE_END