/*
    Homogeneous participating medium with constant extinction.

    how to use in XML:
      <medium type="homogeneous">
          <float name="sigmaA" value="0.02"/>
          <float name="sigmaS" value="0.3"/>
          <float name="g" value="0.0"/>
          <!-- optional explicit bounds; if omitted, uses scene bbox -->
          <point name="boundsMin" value="-1, 0, -1"/>
          <point name="boundsMax" value="1, 2, 1"/>
      </medium>
*/

#include <medium.h>

NORI_NAMESPACE_BEGIN

class HomogeneousMedium : public Medium
{
public:
    HomogeneousMedium(const PropertyList &propList)
    {
        m_sigmaARGB = readScalarOrColor(propList, "sigmaA", 0.0f);
        m_sigmaSRGB = readScalarOrColor(propList, "sigmaS", 0.3f);
        m_phaseG = propList.getFloat("g", 0.0f);

        Point3f defaultMin(1e18f, 1e18f, 1e18f);
        Point3f defaultMax(-1e18f, -1e18f, -1e18f);
        Point3f bmin = propList.getPoint("boundsMin", defaultMin);
        Point3f bmax = propList.getPoint("boundsMax", defaultMax);

        if (bmin.x() < 1e17f && bmax.x() > -1e17f)
        {
            m_hasExplicitBounds = true;
            m_bounds = BoundingBox3f(bmin, bmax);
        }
        else
        {
            m_hasExplicitBounds = false;
        }
    }

    // RGB is the source of truth; the scalar getters return the red
    // channel for backward compat (matches the original behavior for
    // grayscale scenes since Color3f(s) broadcasts).
    float getSigmaA() const override { return m_sigmaARGB.r(); }
    float getSigmaS() const override { return m_sigmaSRGB.r(); }
    Color3f getSigmaARGB() const override { return m_sigmaARGB; }
    Color3f getSigmaSRGB() const override { return m_sigmaSRGB; }
    float getPhaseG() const override { return m_phaseG; }
    bool hasExplicitBounds() const override { return m_hasExplicitBounds; }
    BoundingBox3f getExplicitBounds() const override { return m_bounds; }

    std::string toString() const override
    {
        return tfm::format(
            "HomogeneousMedium[\n"
            "  sigmaA = %s,\n"
            "  sigmaS = %s,\n"
            "  g = %f,\n"
            "  explicitBounds = %s\n"
            "]",
            m_sigmaARGB.toString(), m_sigmaSRGB.toString(), m_phaseG,
            m_hasExplicitBounds ? "yes" : "no (scene bbox)");
    }

private:
    Color3f m_sigmaARGB;
    Color3f m_sigmaSRGB;
    float m_phaseG;
    bool m_hasExplicitBounds = false;
    BoundingBox3f m_bounds;
};

NORI_REGISTER_CLASS(HomogeneousMedium, "homogeneous");
NORI_NAMESPACE_END