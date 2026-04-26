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

#include <nori/medium.h>

NORI_NAMESPACE_BEGIN

class HomogeneousMedium : public Medium
{
public:
    HomogeneousMedium(const PropertyList &propList)
    {
        m_sigmaA = propList.getFloat("sigmaA", 0.0f);
        m_sigmaS = propList.getFloat("sigmaS", 0.3f);
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

    float getSigmaA() const override { return m_sigmaA; }
    float getSigmaS() const override { return m_sigmaS; }
    float getPhaseG() const override { return m_phaseG; }
    bool hasExplicitBounds() const override { return m_hasExplicitBounds; }
    BoundingBox3f getExplicitBounds() const override { return m_bounds; }

    std::string toString() const override
    {
        return tfm::format(
            "HomogeneousMedium[\n"
            "  sigmaA = %f,\n"
            "  sigmaS = %f,\n"
            "  g = %f,\n"
            "  explicitBounds = %s\n"
            "]",
            m_sigmaA, m_sigmaS, m_phaseG,
            m_hasExplicitBounds ? "yes" : "no (scene bbox)");
    }

private:
    float m_sigmaA;
    float m_sigmaS;
    float m_phaseG;
    bool m_hasExplicitBounds = false;
    BoundingBox3f m_bounds;
};

NORI_REGISTER_CLASS(HomogeneousMedium, "homogeneous");
NORI_NAMESPACE_END