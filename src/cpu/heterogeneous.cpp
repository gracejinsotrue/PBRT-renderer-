/*
    heterogeneous.cpp: heterogeneous participating medium with spatially varying density
    XML usage:
      <medium type="heterogeneous">
          <float name="sigmaA" value="0.02"/>
          <float name="sigmaS" value="5.0"/>
          <float name="g" value="0.85"/>
          <point name="boundsMin" value="-0.5, 0.2, -0.5"/>
          <point name="boundsMax" value="0.5, 1.2, 0.5"/>
      </medium>
*/

#include <nori/medium.h>

NORI_NAMESPACE_BEGIN

class HeterogeneousMedium : public Medium
{
public:
    HeterogeneousMedium(const PropertyList &propList)
    {
        m_sigmaARGB = readScalarOrColor(propList, "sigmaA", 0.0f);
        m_sigmaSRGB = readScalarOrColor(propList, "sigmaS", 5.0f);
        m_phaseG = propList.getFloat("g", 0.85f);
        m_volumePath = propList.getString("volume", "");

        m_bounds = BoundingBox3f(
            propList.getPoint("boundsMin"),
            propList.getPoint("boundsMax"));
    }

    // RGB is the source of truth; scalar getters return the red channel
    // for backward compat with grayscale scenes (matches original
    // behavior since Color3f(s) broadcasts).
    float getSigmaA() const override { return m_sigmaARGB.r(); }
    float getSigmaS() const override { return m_sigmaSRGB.r(); }
    Color3f getSigmaARGB() const override { return m_sigmaARGB; }
    Color3f getSigmaSRGB() const override { return m_sigmaSRGB; }
    float getPhaseG() const override { return m_phaseG; }
    bool isHeterogeneous() const override { return true; }
    bool hasExplicitBounds() const override { return true; }
    BoundingBox3f getExplicitBounds() const override { return m_bounds; }
    std::string getVolumePath() const override { return m_volumePath; }

    std::string toString() const override
    {
        return tfm::format(
            "HeterogeneousMedium[\n"
            "  sigmaA = %s,\n"
            "  sigmaS = %s,\n"
            "  g = %f,\n"
            "  volume = \"%s\",\n"
            "  bounds = (%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)\n"
            "]",
            m_sigmaARGB.toString(), m_sigmaSRGB.toString(), m_phaseG,
            m_volumePath.empty() ? "(procedural)" : m_volumePath,
            m_bounds.min.x(), m_bounds.min.y(), m_bounds.min.z(),
            m_bounds.max.x(), m_bounds.max.y(), m_bounds.max.z());
    }

private:
    Color3f m_sigmaARGB;
    Color3f m_sigmaSRGB;
    float m_phaseG;
    std::string m_volumePath;
    BoundingBox3f m_bounds;
};

NORI_REGISTER_CLASS(HeterogeneousMedium, "heterogeneous");
NORI_NAMESPACE_END