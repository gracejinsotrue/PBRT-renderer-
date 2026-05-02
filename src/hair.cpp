/*
    Hair file loader, tube tessellator, and hair BSDF class.

*/

#include <nori/hair.h>
#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>
#include <nori/timer.h>
#include <nori/texture.h>
#include <filesystem/resolver.h>
#include <fstream>
#include <cstring>
#include <cmath>
#include <memory>

NORI_NAMESPACE_BEGIN

// ============================================================
//  .hair binary file loader
//  Reference: [Yuksel] HAIR File Format Specification

// when I started writing for hair, I used yuksel's hair format specification which is why this code exists:
// ============================================================

/// .hair file header is 128 bytes, little-endian.
struct HairFileHeader
{
    char signature[4]; // "HAIR"
    uint32_t numStrands;
    uint32_t numPoints;
    uint32_t flags; // bit 0: segments, bit 1: points,
                    // bit 2: thickness, bit 3: transparency,
                    // bit 4: color
    uint32_t defaultSegments;
    float defaultThickness;
    float defaultTransparency;
    float defaultColor[3];
    char info[88];
};
static_assert(sizeof(HairFileHeader) == 128,
              "HairFileHeader must be exactly 128 bytes");

HairFile HairFile::load(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw NoriException("HairFile::load: cannot open \"%s\"", path.c_str());

    // read the header
    HairFileHeader hdr;
    f.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (!f || std::memcmp(hdr.signature, "HAIR", 4) != 0)
        throw NoriException("HairFile::load: \"%s\" has wrong signature", path.c_str());

    bool hasSegments = (hdr.flags & 1) != 0;
    bool hasPoints = (hdr.flags & 2) != 0;
    bool hasThickness = (hdr.flags & 4) != 0;
    bool hasTransparency = (hdr.flags & 8) != 0;
    bool hasColor = (hdr.flags & 16) != 0;

    if (!hasPoints)
        throw NoriException("HairFile::load: \"%s\" has no points array", path.c_str());

    HairFile hf;
    hf.totalStrands = hdr.numStrands;
    hf.totalPoints = hdr.numPoints;
    hf.defaultThickness = hdr.defaultThickness;
    hf.defaultTransparency = hdr.defaultTransparency;
    hf.defaultColor = Color3f(hdr.defaultColor[0],
                              hdr.defaultColor[1],
                              hdr.defaultColor[2]);

    // read the segments array, optional
    // every entry is uint16_t =  number of SEGMENTS for that strand, and the number of points per strand = segments + 1.
    std::vector<uint16_t> segments;
    if (hasSegments)
    {
        segments.resize(hdr.numStrands);
        f.read(reinterpret_cast<char *>(segments.data()),
               hdr.numStrands * sizeof(uint16_t));
        if (!f)
            throw NoriException("HairFile::load: failed reading segments");
    }

    // read points array
    std::vector<float> points(hdr.numPoints * 3);
    f.read(reinterpret_cast<char *>(points.data()),
           hdr.numPoints * 3 * sizeof(float));
    if (!f)
        throw NoriException("HairFile::load: failed reading points");

    // read thickness array
    std::vector<float> thickness;
    if (hasThickness)
    {
        thickness.resize(hdr.numPoints);
        f.read(reinterpret_cast<char *>(thickness.data()),
               hdr.numPoints * sizeof(float));
        if (!f)
            throw NoriException("HairFile::load: failed reading thickness");
    }

    // I don't use transparency or color
    if (hasTransparency)
        f.seekg(hdr.numPoints * sizeof(float), std::ios::cur);
    if (hasColor)
        f.seekg(hdr.numPoints * 3 * sizeof(float), std::ios::cur);

    // build the strand array
    hf.strands.resize(hdr.numStrands);
    uint32_t ptIdx = 0;
    for (uint32_t s = 0; s < hdr.numStrands; s++)
    {
        uint32_t nSegs = hasSegments ? (uint32_t)segments[s]
                                     : hdr.defaultSegments;
        uint32_t nPts = nSegs + 1;

        auto &strand = hf.strands[s];
        strand.points.resize(nPts);
        for (uint32_t p = 0; p < nPts; p++)
        {
            strand.points[p] = Point3f(points[(ptIdx + p) * 3 + 0],
                                       points[(ptIdx + p) * 3 + 1],
                                       points[(ptIdx + p) * 3 + 2]);
        }

        if (hasThickness)
        {
            strand.thickness.resize(nPts);
            for (uint32_t p = 0; p < nPts; p++)
                strand.thickness[p] = thickness[ptIdx + p];
        }

        ptIdx += nPts;
    }

    if (ptIdx != hdr.numPoints)
        throw NoriException("HairFile::load: point count mismatch "
                            "(%u read vs %u expected)",
                            ptIdx, hdr.numPoints);

    return hf;
}

// ============================================================
//  Tube tessellator
//  Reference: PBRT Section 9.9.1 — hair geometry, h parameter
// ============================================================

HairMesh::HairMesh(const PropertyList &propList)
{
    filesystem::path filename =
        getFileResolver()->resolve(propList.getString("filename"));

    m_sides = propList.getInteger("sides", 6);
    m_radiusScale = propList.getFloat("radiusScale", 1.0f);
    int maxStrands = propList.getInteger("maxStrands", -1);

    Transform trafo = propList.getTransform("toWorld", Transform());

    cout << "Loading hair \"" << filename << "\" .. ";
    cout.flush();
    Timer timer;

    HairFile hf = HairFile::load(filename.str());
    // TODO: when testing, I would subsample hair strands.
    if (maxStrands > 0 && (uint32_t)maxStrands < hf.totalStrands)
    {
        hf.strands.resize(maxStrands);
        hf.totalStrands = maxStrands;
        uint32_t pts = 0;
        for (auto &s : hf.strands)
            pts += (uint32_t)s.points.size();
        hf.totalPoints = pts;
    }

    cout << hf.totalStrands << " strands, "
         << hf.totalPoints << " points .. ";

    tessellate(hf, trafo);

    m_name = filename.str();
    cout << "done. (V=" << m_V.cols() << ", F=" << m_F.cols()
         << ", took " << timer.elapsedString() << ")" << endl;
}

void HairMesh::tessellate(const HairFile &hf, const Transform &trafo)
{
    const int S = m_sides; // sides per ring

    // pass 1: count total verts and tris
    // Each segment produces 2 rings of S verts = 2S verts, and 2S tris, and adjacent segments share a ring, so per strand with K segments:
    //   verts = (K + 1) * S
    //   tris  = K * 2 * S
    uint32_t totalVerts = 0;
    uint32_t totalTris = 0;
    for (auto &strand : hf.strands)
    {
        if (strand.points.size() < 2)
            continue;
        uint32_t nSegs = (uint32_t)(strand.points.size() - 1);
        totalVerts += (nSegs + 1) * S;
        totalTris += nSegs * 2 * S;
    }

    // allocate
    m_V.resize(3, totalVerts);  // positions
    m_N.resize(3, totalVerts);  // surface normals
    m_T.resize(3, totalVerts);  // fiber tangent directions
    m_UV.resize(2, totalVerts); // u = along strand [0,1], v = h mapped to [0,1]
    m_F.resize(3, totalTris);   // triangle indices

    uint32_t vi = 0; // vertex write cursor
    uint32_t fi = 0; // face write cursor

    for (auto &strand : hf.strands)
    {
        uint32_t nPts = (uint32_t)strand.points.size();
        if (nPts < 2)
            continue;
        uint32_t nSegs = nPts - 1;

        uint32_t ringStart = vi; // first vertex index of this strand

        for (uint32_t pi = 0; pi < nPts; pi++)
        {
            Point3f P = trafo * strand.points[pi];
            m_bbox.expandBy(P);

            // Fiber tangent at this control point
            // Use forward difference, backward at tip, averaged at interior. For consistency's sake, it is all computed from transformed points.
            Vector3f tangent;
            if (pi == 0)
                tangent = (trafo * strand.points[pi + 1]) -
                          (trafo * strand.points[pi]);
            else if (pi == nPts - 1)
                tangent = (trafo * strand.points[pi]) -
                          (trafo * strand.points[pi - 1]);
            else
                tangent = (trafo * strand.points[pi + 1]) -
                          (trafo * strand.points[pi - 1]);

            tangent.normalize();
            // if (tangent.squaredNorm() < 1e-8f)
            //     tangent = Vector3f(0, 1, 0); // fallback

            // build the  orthonormal frame for the cross-section ring.
            // We need two vectors (B, N_ring) perpendicular to tangent.
            // Use the Frame utility which builds an ONB from a single direction.
            // Frame expects the "z-axis" input, but we want tangent as x-axis
            // in the hair BCSDF coordinate system. For tessellation, we just
            // need any consistent pair perpendicular to tangent.
            Vector3f arbitrary = (std::abs(tangent.x()) < 0.9f)
                                     ? Vector3f(1, 0, 0)
                                     : Vector3f(0, 1, 0);
            Vector3f B = tangent.cross(arbitrary).normalized();
            Vector3f N_ring = tangent.cross(B).normalized();

            float radius;
            if (!strand.thickness.empty())
                radius = strand.thickness[pi] * 0.5f * m_radiusScale;
            else
                radius = hf.defaultThickness * 0.5f * m_radiusScale;

            // parametric position along strand where 0 at root, 1 at tip
            float u_along = (nPts > 1) ? (float)pi / (float)(nPts - 1) : 0.0f;

            // -generate a ring of 5 vertices
            for (int si = 0; si < S; si++)
            {
                // Angle around the tube cross-section.
                // Distribute S vertices evenly around [0, 2π).
                float angle = 2.0f * M_PI * (float)si / (float)S;

                // Radial direction in the normal plane
                Vector3f radial = std::cos(angle) * B + std::sin(angle) * N_ring;

                // Vertex position = center + radius * radial
                Point3f pos = P + radius * radial;

                // Surface normal = radial direction pointing outward
                Vector3f normal = radial.normalized();
                // TODO
                // h parameter: offset across fiber width, ∈ [-1, 1].
                //
                // Reference: [PBRT] Section 9.9.1, Figure 9.42
                //   h parameterizes the circle's diameter as seen from
                //   the incident ray direction. Since we don't know the
                //   ray direction at tessellation time, we store the
                //   angular position and let the shader compute h.
                //
                // We encode v_uv = (angle / 2π) so the shader can
                // reconstruct h = -cos(angle) = -cos(2π * v_uv).
                // Alternatively, for a simpler first pass, we just store
                // h directly: h = -cos(angle), mapped to [0,1] as
                //   v_uv = (h + 1) / 2.

                float h = -std::cos(angle);
                float v_uv = (h + 1.0f) * 0.5f;

                m_V.col(vi) = pos;
                m_N.col(vi) = normal;
                m_T.col(vi) = tangent;
                m_UV.col(vi) = Vector2f(u_along, v_uv);

                vi++;
            }

            // generate triangles connecting this ring to the previous
            if (pi > 0)
            {
                uint32_t prevRing = ringStart + (pi - 1) * S;
                uint32_t currRing = ringStart + pi * S;

                for (int si = 0; si < S; si++)
                {
                    int next = (si + 1) % S;

                    // two triangles per quad between adjacent rings
                    //   prevRing[si] — prevRing[next] — currRing[next]
                    //   prevRing[si] — currRing[next] — currRing[si]
                    m_F.col(fi) << (prevRing + si),
                        (prevRing + next),
                        (currRing + next);
                    fi++;

                    m_F.col(fi) << (prevRing + si),
                        (currRing + next),
                        (currRing + si);
                    fi++;
                }
            }
        }
    }

    cout << "tessellated (" << vi << " verts, " << fi << " tris) .. ";
}

NORI_REGISTER_CLASS(HairMesh, "hair");

//  Hair BSDF, CPU-side stub
//  Reference: [Chiang] Section 4 — six production parameters
//

class HairBSDF : public BSDF
{
public:
    HairBSDF(const PropertyList &propList)
    {
        // Hair color — will be converted to σ_a on the GPU via
        // Chiang Eq. 9 (albedo inversion).
        // deafult color here is medium brown but it is overwriten in the xml
        m_color = propList.getColor("color", Color3f(0.28f, 0.15f, 0.06f));

        // Longitudinal roughness β_M ∈ [0,1], mapped to variance v via Chiang Eq. 7 in the shader.
        // Default 0.25 gives a highlight similar to typical human hair, according to PBRT Figure 9.45(b).
        m_betaM = propList.getFloat("beta_m", 0.25f);

        // Azimuthal roughness β_N ∈ [0,1], mapped to logistic scale s via Chiang Eq. 8 in the respective shader.
        // Controls overall softness ([Chiang] Figure 12).
        m_betaN = propList.getFloat("beta_n", 0.3f);

        // cuticle scale tilt angle in degrees which shifts R highlight above specular, TT below.
        // Typically ~2° for human hair ([PBRT] Section 9.9.2).
        m_alpha = propList.getFloat("alpha", 2.0f);

        // Index of refraction of hair interior, typically 1.55
        m_eta = propList.getFloat("eta", 1.55f);

        // Alpha masking (e.g. hair cards)
        m_alphaTextureFile = propList.getString("alphaTexture", "");
        if (!m_alphaTextureFile.empty())
            m_alphaTex = std::make_unique<AlphaTexture>(m_alphaTextureFile);
    }

    Color3f eval(const BSDFQueryRecord &bRec) const override
    {
        if (bRec.measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0.f ||
            Frame::cosTheta(bRec.wo) <= 0.f)
            return Color3f(0.f);
        return m_color * INV_PI;
    }

    float pdf(const BSDFQueryRecord &bRec) const override
    {
        if (bRec.measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0.f ||
            Frame::cosTheta(bRec.wo) <= 0.f)
            return 0.f;
        return INV_PI * Frame::cosTheta(bRec.wo);
    }

    Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const override
    {
        if (Frame::cosTheta(bRec.wi) <= 0.f)
            return Color3f(0.f);
        bRec.measure = ESolidAngle;
        bRec.wo = Warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.0f;
        return m_color;
    }

    bool isDiffuse() const override { return false; }

    BSDFGPUData getGPUData() const override
    {
        BSDFGPUData d;
        d.type = 5;

        // Hair color stored in albedo[], where the shader will convert to σ_a via Chiang Eq. 9. refers to color → absorption coefficient
        d.albedo[0] = m_color.r();
        d.albedo[1] = m_color.g();
        d.albedo[2] = m_color.b();

        d.roughness = m_betaM; // longitudinal roughness β_M
        d.alpha = m_alpha;     // cuticle tilt angle in degrees
        d.intIOR = m_eta;      // hair IOR

        // Azimuthal roughness β_N (Chiang Eq. 8)
        d.betaN = m_betaN;
        d.alphaTexture = m_alphaTextureFile;

        return d;
    }

    const AlphaTexture *getAlphaTexture() const override { return m_alphaTex.get(); }

    std::string toString() const override
    {
        return tfm::format(
            "HairBSDF[\n"
            "  color = %s,\n"
            "  beta_m = %f,\n"
            "  beta_n = %f,\n"
            "  alpha = %f,\n"
            "  eta = %f\n"
            "]",
            m_color.toString(),
            m_betaM, m_betaN, m_alpha, m_eta);
    }

private:
    Color3f m_color;
    float m_betaM;
    float m_betaN;
    float m_alpha;
    float m_eta;

    // Alpha masking
    std::string m_alphaTextureFile;
    std::unique_ptr<AlphaTexture> m_alphaTex;
};

NORI_REGISTER_CLASS(HairBSDF, "hair_bsdf");

NORI_NAMESPACE_END