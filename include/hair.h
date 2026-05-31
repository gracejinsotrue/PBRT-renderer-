/*
    Hair file loader and tessellated tube mesh generator.

    Loads Cem Yuksel's .hair binary format and tessellates each strand
    into a triangle tube mesh suitable for DXR raytracing (which has
    no native curve primitives).

    Reference:
      Cem Yuksel, "HAIR File Format Specification"
      cemyuksel.com/research/hairmodels/

    The tessellator stores per-vertex fiber tangent direction in m_T
    (a 3×N matrix parallel to m_V) and encodes the Chiang near-field
    offset h ∈ [-1,1] in m_UV.y (with m_UV.x = parametric position
    along the strand, 0 at root, 1 at tip).

    These are needed by the Chiang hair BCSDF:
      Chiang et al. 2016, "A Practical and Controllable Hair and Fur
      Model for Production Path Tracing"
      PBRT 4th ed., Section 9.9 "Scattering from Hair"
*/

#pragma once

#include <mesh.h>
#include <string>
#include <vector>

NORI_NAMESPACE_BEGIN

/// Raw strand data loaded from a .hair file.
struct HairFile
{
    /// Per-strand array of control points (root to tip).
    struct Strand
    {
        std::vector<Point3f> points;
        std::vector<float> thickness; // per-point; may be empty (use default)
    };

    std::vector<Strand> strands;
    uint32_t totalStrands = 0;
    uint32_t totalPoints = 0;
    float defaultThickness = 0.01f; // from header, in model units
    float defaultTransparency = 0.0f;
    Color3f defaultColor = Color3f(0.65f, 0.4f, 0.2f);

    /// Load from a .hair binary file. Throws NoriException on failure.
    static HairFile load(const std::string &path);
};

/**
 * \brief Triangle mesh generated from tessellating hair strand curves.
 *
 * Registered as "hair" so scene XML can reference:
 *   <mesh type="hair">
 *       <string name="filename" value="straight.hair"/>
 *       <integer name="sides" value="6"/>        <!-- tube sides, default 6 -->
 *       <float name="radiusScale" value="1.0"/>  <!-- multiply .hair thickness -->
 *       <integer name="maxStrands" value="5000"/> <!-- subsample for fast iteration, -1 = all -->
 *       <bsdf type="hair_bsdf"> ... </bsdf>
 *   </mesh>
 *
 * Fills m_V, m_N, m_F, m_UV from the base Mesh class, plus:
 *   m_T  — per-vertex fiber tangent direction (3×N, same layout as m_N)
 */
class HairMesh : public Mesh
{
public:
    HairMesh(const PropertyList &propList);

private:
    /// Number of sides for tube cross-section
    int m_sides = 6;

    /// Multiplier on .hair file thickness values
    float m_radiusScale = 1.0f;

    /// Tessellate loaded strands into m_V, m_N, m_F, m_UV, m_T
    void tessellate(const HairFile &hf, const Transform &trafo);
};

NORI_NAMESPACE_END