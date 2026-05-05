/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#include <nori/camera.h>
#include <nori/rfilter.h>
#include <nori/warp.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

/**
 * \brief Perspective camera with thin lens depth of field
 *
 * This class implements a perspective camera model with an optional thin lens
 * approximation for depth of field. When lensRadius > 0, rays are distributed
 * across a disk-shaped aperture and focused at a given focal distance.
 */
class PerspectiveCamera : public Camera
{
public:
    PerspectiveCamera(const PropertyList &propList)
    {
        /* Width and height in pixels. Default: 720p */
        m_outputSize.x() = propList.getInteger("width", 1280);
        m_outputSize.y() = propList.getInteger("height", 720);
        m_invOutputSize = m_outputSize.cast<float>().cwiseInverse();

        /* Specifies an optional camera-to-world transformation. Default: none */
        m_cameraToWorld = propList.getTransform("toWorld", Transform());

        /* Horizontal field of view in degrees */
        m_fov = propList.getFloat("fov", 30.0f);

        /* Near and far clipping planes in world-space units */
        m_nearClip = propList.getFloat("nearClip", 1e-4f);
        m_farClip = propList.getFloat("farClip", 1e4f);

        /* Thin lens parameters for depth of field.
           lensRadius = 0 reduces to a pinhole cameraa, which has infinite depth of field.
           focalDistance sets the distance to the plane that appears sharp. */
        m_lensRadius = propList.getFloat("lensRadius", 0.0f);
        m_focalDistance = propList.getFloat("focalDistance", 1.0f);

        m_rfilter = NULL;
    }

    void activate()
    {
        float aspect = m_outputSize.x() / (float)m_outputSize.y();

        /* Project vectors in camera space onto a plane at z=1:
         *
         *  xProj = cot * x / z
         *  yProj = cot * y / z
         *  zProj = (far * (z - near)) / (z * (far-near))
         *  The cotangent factor ensures that the field of view is
         *  mapped to the interval [-1, 1].
         */
        float recip = 1.0f / (m_farClip - m_nearClip),
              cot = 1.0f / std::tan(degToRad(m_fov / 2.0f));

        Eigen::Matrix4f perspective;
        perspective << cot, 0, 0, 0,
            0, cot, 0, 0,
            0, 0, m_farClip * recip, -m_nearClip * m_farClip * recip,
            0, 0, 1, 0;

        /**
         * Translation and scaling to shift the clip coordinates into the
         * range from zero to one. Also takes the aspect ratio into account.
         */
        m_sampleToCamera = Transform(
                               Eigen::DiagonalMatrix<float, 3>(Vector3f(-0.5f, -0.5f * aspect, 1.0f)) *
                               Eigen::Translation<float, 3>(-1.0f, -1.0f / aspect, 0.0f) * perspective)
                               .inverse();

        /* If no reconstruction filter was assigned, instantiate a Gaussian filter */
        if (!m_rfilter)
            m_rfilter = static_cast<ReconstructionFilter *>(
                NoriObjectFactory::createInstance("gaussian", PropertyList()));
    }

    Color3f sampleRay(Ray3f &ray,
                      const Point2f &samplePosition,
                      const Point2f &apertureSample) const
    {
        /* Compute the corresponding position on the
           near plane in local camera space */
        Point3f nearP = m_sampleToCamera * Point3f(
                                               samplePosition.x() * m_invOutputSize.x(),
                                               samplePosition.y() * m_invOutputSize.y(), 0.0f);

        /* Pinhole ray direction normalized, in camera space */
        Vector3f d = nearP.normalized();

        Point3f rayO(0.0f, 0.0f, 0.0f);

        if (m_lensRadius > 0.0f)
        {
            /* Thin lens model to sample a point on the circular aperture */
            Point2f pLens2 = m_lensRadius * Warp::squareToUniformDisk(apertureSample);
            Point3f pLens(pLens2.x(), pLens2.y(), 0.0f);

                float ft = m_focalDistance / d.z();
            Point3f pFocus(d.x() * ft, d.y() * ft, d.z() * ft);

            rayO = pLens;
            d = (pFocus - pLens).normalized();
        }

        float invZ = 1.0f / d.z();

        ray.o = m_cameraToWorld * rayO;
        ray.d = m_cameraToWorld * d;
        ray.mint = m_nearClip * invZ;
        ray.maxt = m_farClip * invZ;
        ray.update();

        return Color3f(1.0f);
    }

    float getLensRadius() const override { return m_lensRadius; }
    float getFocalDistance() const override { return m_focalDistance; }

    void addChild(NoriObject *obj)
    {
        switch (obj->getClassType())
        {
        case EReconstructionFilter:
            if (m_rfilter)
                throw NoriException("Camera: tried to register multiple reconstruction filters!");
            m_rfilter = static_cast<ReconstructionFilter *>(obj);
            break;

        default:
            throw NoriException("Camera::addChild(<%s>) is not supported!",
                                classTypeName(obj->getClassType()));
        }
    }

    /// Return a human-readable summary
    std::string toString() const
    {
        return tfm::format(
            "PerspectiveCamera[\n"
            "  cameraToWorld = %s,\n"
            "  outputSize = %s,\n"
            "  fov = %f,\n"
            "  clip = [%f, %f],\n"
            "  lensRadius = %f,\n"
            "  focalDistance = %f,\n"
            "  rfilter = %s\n"
            "]",
            indent(m_cameraToWorld.toString(), 18),
            m_outputSize.toString(),
            m_fov,
            m_nearClip,
            m_farClip,
            m_lensRadius,
            m_focalDistance,
            indent(m_rfilter->toString()));
    }

private:
    Vector2f m_invOutputSize;
    Transform m_sampleToCamera;
    Transform m_cameraToWorld;
    float m_fov;
    float m_nearClip;
    float m_farClip;
    float m_lensRadius;
    float m_focalDistance;
};

NORI_REGISTER_CLASS(PerspectiveCamera, "perspective");
NORI_NAMESPACE_END
