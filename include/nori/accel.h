/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#pragma once

#include <nori/mesh.h>
#include <vector>
#include <tbb/task_group.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Acceleration data structure for ray intersection queries
 *
 * The current implementation falls back to a brute force loop
 * through the geometry.
 */
class Accel
{
public:
    /**
     * \brief Register a triangle mesh for inclusion in the acceleration
     * data structure
     *
     * This function can only be used before \ref build() is called
     */
    void addMesh(Mesh *mesh);

    /// Build the acceleration data structure (currently a no-op)
    void build();

    /// Return an axis-aligned box that bounds the scene
    const BoundingBox3f &getBoundingBox() const { return m_bbox; }

    /**
     * \brief Intersect a ray against all triangles stored in the scene and
     * return detailed intersection information
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum extent
     *    information
     *
     * \param its
     *    A detailed intersection record, which will be filled by the
     *    intersection query
     *
     * \param shadowRay
     *    \c true if this is a shadow ray query, i.e. a query that only aims to
     *    find out whether the ray is blocked or not without returning detailed
     *    intersection information.
     *
     * \return \c true if an intersection was found
     */
    bool rayIntersect(const Ray3f &ray, Intersection &its, bool shadowRay) const;

private:
    // for tbb stuff
    // minimum number of triangles to bother parallelizing
    static constexpr uint32_t PARALLEL_THRESHOLD = 1024; // can tweak ts number
    // self contained result of building bvh tree

    /* === BVH constants === */

    static constexpr uint32_t N_BINS = 12;           ///< Number of SAH bins for split evaluation
    static constexpr uint32_t MAX_TRI_PER_LEAF = 10; ///< Max triangles before we stop splitting

    /**
     * \brief Compact BVH node (32 bytes)
     *
     * For leaf nodes: nTriangles > 0, start = first index into m_indices
     * For interior nodes: nTriangles == 0, start = index of right child in m_nodes
     *   (left child is always stored at current index + 1)
     */
    struct BVHNode
    {
        BoundingBox3f bbox;  ///< Axis-aligned bounding box (24 bytes)
        uint32_t start;      ///< Leaf: offset into m_indices. Interior: right child index.
        uint16_t nTriangles; ///< 0 = interior node, >0 = leaf node
        uint16_t axis;       ///< Split axis for interior nodes

        bool isLeaf() const { return nTriangles > 0; }
    };

    struct BVHBuildResult
    {
        std::vector<BVHNode> nodes;
        std::vector<uint32_t> indices;

        /**
         * \brief Build a BVH subtree in isolation (thread-safe, no shared state)
         * \param indices Triangle indices for this subtree (will be reordered)
         * \param bbox Bounding box enclosing all triangles
         * \return A self-contained subtree
         */
    };

    /**
     * \brief Recursively build the BVH
     * \param start Start index in m_indices (inclusive)
     * \param end End index in m_indices (exclusive)
     * \param bbox Bounding box enclosing all triangles in [start, end)
     * \return Index of the created node in m_nodes
     */
    uint32_t buildRecursive(uint32_t start, uint32_t end, const BoundingBox3f &bbox);
    BVHBuildResult buildParallel(std::vector<uint32_t> indices, const BoundingBox3f &bbox);
    Mesh *m_mesh = nullptr; ///< Mesh (only a single one for now)
    BoundingBox3f m_bbox;   ///< Bounding box of the entire scene

    /* === BVH storage === */
    std::vector<BVHNode> m_nodes;    ///< Flat array of BVH nodes
    std::vector<uint32_t> m_indices; ///< Triangle index array (permuted during build)
};

NORI_NAMESPACE_END
