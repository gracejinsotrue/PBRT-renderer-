/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#include <nori/accel.h>
#include <Eigen/Geometry>

#include <algorithm>
#include <numeric>

#include <chrono>
#include <iostream>
NORI_NAMESPACE_BEGIN

void Accel::addMesh(Mesh *mesh)
{
    if (m_mesh)
        throw NoriException("Accel: only a single mesh is supported!");
    m_mesh = mesh;
    m_bbox = m_mesh->getBoundingBox();
}

void Accel::build()
{
    if (!m_mesh)
        return;

    uint32_t nTris = m_mesh->getTriangleCount();
    m_indices.resize(nTris);
    for (uint32_t i = 0; i < nTris; ++i)
        m_indices[i] = i;
    // i add code for timing
    auto start = std::chrono::high_resolution_clock::now();

    buildRecursive(0, nTris, m_bbox);

    auto end = std::chrono::high_resolution_clock::now();
    double buildTime = std::chrono::duration<double, std::milli>(end - start).count();

    // just more code from timing section for report (asked chat to write this part only())
    uint32_t interiorCount = 0, leafCount = 0, totalTrisInLeaves = 0;
    for (const auto &node : m_nodes)
    {
        if (node.isLeaf())
        {
            leafCount++;
            totalTrisInLeaves += node.nTriangles;
        }
        else
        {
            interiorCount++;
        }
    }

    std::cout << "BVH Statistics:" << std::endl;
    std::cout << "  Build time:                " << buildTime << " ms" << std::endl;
    std::cout << "  Triangle count:            " << nTris << std::endl;
    std::cout << "  Interior nodes:            " << interiorCount << std::endl;
    std::cout << "  Leaf nodes:                " << leafCount << std::endl;
    std::cout << "  Avg triangles per leaf:    " << (float)totalTrisInLeaves / leafCount << std::endl;
    std::cout << "  Bytes per node:            " << sizeof(BVHNode) << std::endl;
    std::cout << "  Total nodes:               " << m_nodes.size() << std::endl;
}

// // median split
// uint32_t Accel::buildRecursive(uint32_t start, uint32_t end, const BoundingBox3f &bbox)
// {
//     uint32_t nodeIdx = m_nodes.size();
//     m_nodes.push_back(BVHNode());
//     BVHNode &node = m_nodes[nodeIdx];
//     node.bbox = bbox;

//     uint32_t count = end - start;

//     /* Base case: few enough triangles to make a leaf */
//     if (count <= MAX_TRI_PER_LEAF)
//     {
//         node.nTriangles = count;
//         node.start = start;
//         return nodeIdx;
//     }

//     /* Find the longest axis of the bounding box */
//     int axis = bbox.getLargestAxis();

//     /* Sort triangles by centroid along the longest axis */
//     std::sort(m_indices.begin() + start, m_indices.begin() + end,
//               [this, axis](uint32_t a, uint32_t b)
//               {
//                   return m_mesh->getCentroid(a)[axis] < m_mesh->getCentroid(b)[axis];
//               });

//     /* Split at the median */
//     uint32_t mid = start + count / 2;

//     /* Compute child bounding boxes */
//     BoundingBox3f leftBox, rightBox;
//     for (uint32_t i = start; i < mid; ++i)
//         leftBox.expandBy(m_mesh->getBoundingBox(m_indices[i]));
//     for (uint32_t i = mid; i < end; ++i)
//         rightBox.expandBy(m_mesh->getBoundingBox(m_indices[i]));

//     m_nodes[nodeIdx].axis = (uint16_t)axis;

//     // Build children (left child is at nodeIdx + 1)
//     buildRecursive(start, mid, leftBox);
//     // NOTE: m_nodes may have reallocated during recursion, so
//     // the 'node' reference is now dangling. Use nodeIdx instead.
//     uint32_t rightChild = buildRecursive(mid, end, rightBox);
//     m_nodes[nodeIdx].start = rightChild;
//     m_nodes[nodeIdx].nTriangles = 0; // Mark as interior node

//     return nodeIdx;
// }

// sah all 3 axes
// uint32_t Accel::buildRecursive(uint32_t start, uint32_t end, const BoundingBox3f &bbox)
// {
//     uint32_t nodeIdx = m_nodes.size();
//     m_nodes.push_back(BVHNode());
//     BVHNode &node = m_nodes[nodeIdx];
//     node.bbox = bbox;

//     uint32_t count = end - start;

//     /* Base case: few enough triangles to make a leaf */
//     if (count <= MAX_TRI_PER_LEAF)
//     {
//         node.nTriangles = count;
//         node.start = start;
//         return nodeIdx;
//     }

//     /* Compute the centroid bounds */
//     BoundingBox3f centroidBBox;
//     for (uint32_t i = start; i < end; ++i)
//         centroidBBox.expandBy(m_mesh->getCentroid(m_indices[i]));

//     /* === SAH binned split — try ALL 3 axes, pick the best === */

//     float globalBestCost = std::numeric_limits<float>::infinity();
//     uint32_t globalBestSplit = 0;
//     int globalBestAxis = -1;

//     for (int axis = 0; axis < 3; ++axis)
//     {
//         float cMin = centroidBBox.min[axis];
//         float cMax = centroidBBox.max[axis];

//         /* Skip this axis if all centroids are at the same position */
//         if (cMin == cMax)
//             continue;

//         /* Each bin tracks a bounding box and triangle count */
//         struct Bin
//         {
//             BoundingBox3f bbox;
//             uint32_t count = 0;
//         };
//         Bin bins[N_BINS];

//         /* Step 1: Assign each triangle to a bin based on its centroid */
//         float invBinWidth = N_BINS / (cMax - cMin);
//         for (uint32_t i = start; i < end; ++i)
//         {
//             float c = m_mesh->getCentroid(m_indices[i])[axis];
//             uint32_t bin = std::min((uint32_t)((c - cMin) * invBinWidth), N_BINS - 1);
//             bins[bin].count++;
//             bins[bin].bbox.expandBy(m_mesh->getBoundingBox(m_indices[i]));
//         }

//         /* Step 2: Evaluate SAH cost for each of the N_BINS-1 candidate splits */
//         for (uint32_t s = 1; s < N_BINS; ++s)
//         {
//             BoundingBox3f leftBox, rightBox;
//             uint32_t leftCount = 0, rightCount = 0;

//             for (uint32_t i = 0; i < s; ++i)
//             {
//                 leftBox.expandBy(bins[i].bbox);
//                 leftCount += bins[i].count;
//             }
//             for (uint32_t i = s; i < N_BINS; ++i)
//             {
//                 rightBox.expandBy(bins[i].bbox);
//                 rightCount += bins[i].count;
//             }

//             if (leftCount == 0 || rightCount == 0)
//                 continue;

//             /* SAH cost: C = S(A)*N_A + S(B)*N_B */
//             float cost = leftBox.getSurfaceArea() * leftCount + rightBox.getSurfaceArea() * rightCount;

//             if (cost < globalBestCost)
//             {
//                 globalBestCost = cost;
//                 globalBestSplit = s;
//                 globalBestAxis = axis;
//             }
//         }
//     }

//     /* If no valid split was found on any axis, fall back to a leaf */
//     if (globalBestAxis == -1)
//     {
//         m_nodes[nodeIdx].nTriangles = (uint16_t)count;
//         m_nodes[nodeIdx].start = start;
//         return nodeIdx;
//     }

//     /* Check if splitting is worthwhile compared to making a leaf */
//     float leafCost = bbox.getSurfaceArea() * count;
//     if (globalBestCost >= leafCost)
//     {
//         m_nodes[nodeIdx].nTriangles = (uint16_t)count;
//         m_nodes[nodeIdx].start = start;
//         return nodeIdx;
//     }

//     /* Partition the triangle indices around the best split */
//     int bestAxis = globalBestAxis;
//     float cMin = centroidBBox.min[bestAxis];
//     float cMax = centroidBBox.max[bestAxis];
//     float splitPos = cMin + globalBestSplit * (cMax - cMin) / N_BINS;
//     auto midIter = std::partition(
//         m_indices.begin() + start, m_indices.begin() + end,
//         [this, bestAxis, splitPos](uint32_t idx)
//         {
//             return m_mesh->getCentroid(idx)[bestAxis] < splitPos;
//         });
//     uint32_t mid = (uint32_t)(midIter - m_indices.begin());

//     /* Safety: if partition didn't actually split, force midpoint */
//     if (mid == start || mid == end)
//         mid = start + count / 2;

//     /* Compute child bounding boxes from the actual partitioned triangles */
//     BoundingBox3f leftBox, rightBox;
//     for (uint32_t i = start; i < mid; ++i)
//         leftBox.expandBy(m_mesh->getBoundingBox(m_indices[i]));
//     for (uint32_t i = mid; i < end; ++i)
//         rightBox.expandBy(m_mesh->getBoundingBox(m_indices[i]));

//     m_nodes[nodeIdx].axis = (uint16_t)bestAxis;

//     // Build children (left child is at nodeIdx + 1)
//     buildRecursive(start, mid, leftBox);
//     // NOTE: m_nodes may have reallocated during recursion, so
//     // the 'node' reference is now dangling. Use nodeIdx instead.
//     uint32_t rightChild = buildRecursive(mid, end, rightBox);
//     m_nodes[nodeIdx].start = rightChild;
//     m_nodes[nodeIdx].nTriangles = 0; // Mark as interior node

//     return nodeIdx;
// }

// sah longest axis only
uint32_t Accel::buildRecursive(uint32_t start, uint32_t end, const BoundingBox3f &bbox)
{
    uint32_t nodeIdx = m_nodes.size();
    m_nodes.push_back(BVHNode());
    BVHNode &node = m_nodes[nodeIdx];
    node.bbox = bbox;

    uint32_t count = end - start;

    if (count <= MAX_TRI_PER_LEAF)
    {
        node.nTriangles = count;
        node.start = start;
        return nodeIdx;
    }

    /* we find largest axis of bbox */
    int axis = bbox.getLargestAxis();

    /* centroid bound*/
    BoundingBox3f centroidBBox;
    for (uint32_t i = start; i < end; ++i)
        centroidBBox.expandBy(m_mesh->getCentroid(m_indices[i]));

    float cMin = centroidBBox.min[axis];
    float cMax = centroidBBox.max[axis];

    /* if all centroids are at the same position along this axis, we cannot split so this is just a leaf */
    if (cMin == cMax)
    {
        m_nodes[nodeIdx].nTriangles = (uint16_t)count;
        m_nodes[nodeIdx].start = start;
        return nodeIdx;
    }

    // this is where the SAH really starts

    // bin tracks triangle strucdture and bbox count
    struct Bin
    {
        BoundingBox3f bbox;
        uint32_t count = 0;
    };
    Bin bins[N_BINS];

    // 1) each triangle gets a bin based on its centroid
    float invBinWidth = N_BINS / (cMax - cMin);
    for (uint32_t i = start; i < end; ++i)
    {
        float c = m_mesh->getCentroid(m_indices[i])[axis];
        uint32_t bin = std::min((uint32_t)((c - cMin) * invBinWidth), N_BINS - 1);
        bins[bin].count++;
        bins[bin].bbox.expandBy(m_mesh->getBoundingBox(m_indices[i]));
    }

    // 2)  Evaluate SAH cost for each of the N_BINS-1 candidate splits
    float bestCost = std::numeric_limits<float>::infinity();
    uint32_t bestSplit = 0;

    for (uint32_t s = 1; s < N_BINS; ++s)
    {
        BoundingBox3f leftBox, rightBox;
        uint32_t leftCount = 0, rightCount = 0;

        for (uint32_t i = 0; i < s; ++i)
        {
            leftBox.expandBy(bins[i].bbox);
            leftCount += bins[i].count;
        }
        for (uint32_t i = s; i < N_BINS; ++i)
        {
            rightBox.expandBy(bins[i].bbox);
            rightCount += bins[i].count;
        }

        if (leftCount == 0 || rightCount == 0)
            continue;

        /* sah cost : C = S(A)*N_A + S(B)*N_B */
        float cost = leftBox.getSurfaceArea() * leftCount + rightBox.getSurfaceArea() * rightCount;

        if (cost < bestCost)
        {
            bestCost = cost;
            bestSplit = s;
        }
    }

    // now we check if splitting is worth compared to making leaf */
    float leafCost = bbox.getSurfaceArea() * count;
    if (bestCost >= leafCost)
    {
        m_nodes[nodeIdx].nTriangles = (uint16_t)count;
        m_nodes[nodeIdx].start = start;
        return nodeIdx;
    }

    /*4) now we partition the triangle indices around the best split */
    float splitPos = cMin + bestSplit * (cMax - cMin) / N_BINS;
    auto midIter = std::partition(
        m_indices.begin() + start, m_indices.begin() + end,
        [this, axis, splitPos](uint32_t idx)
        {
            return m_mesh->getCentroid(idx)[axis] < splitPos;
        });
    uint32_t mid = (uint32_t)(midIter - m_indices.begin());

    // if the partition didn't split, we just do midpoint
    if (mid == start || mid == end)
        mid = start + count / 2;

    // compute child bounding boxes from the actual partitioned triangles
    BoundingBox3f leftBox, rightBox;
    for (uint32_t i = start; i < mid; ++i)
        leftBox.expandBy(m_mesh->getBoundingBox(m_indices[i]));
    for (uint32_t i = mid; i < end; ++i)
        rightBox.expandBy(m_mesh->getBoundingBox(m_indices[i]));

    m_nodes[nodeIdx].axis = (uint16_t)axis;

    // build children (left child is at nodeIdx + 1)
    buildRecursive(start, mid, leftBox);

    uint32_t rightChild = buildRecursive(mid, end, rightBox);
    m_nodes[nodeIdx].start = rightChild;
    m_nodes[nodeIdx].nTriangles = 0; // Mark as interior node

    return nodeIdx;
}

bool Accel::rayIntersect(const Ray3f &ray_, Intersection &its, bool shadowRay) const
{
    bool foundIntersection = false;
    uint32_t f = (uint32_t)-1;

    Ray3f ray(ray_); /// Make a copy of the ray (we will need to update its '.maxt' value)

    // ray traversal with explict stack
    /* we avoid std::stack/std::vector here to prevent per-ray heap allocation */
    uint32_t stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        uint32_t nodeIdx = stack[--stackPtr];
        const BVHNode &node = m_nodes[nodeIdx];

        /* Skip this node if the ray doesn't hit its bounding box */
        if (!node.bbox.rayIntersect(ray))
            continue;

        if (node.isLeaf())
        {
            /* this is a leaf node so we just test all triangles */
            for (uint32_t i = node.start; i < node.start + node.nTriangles; ++i)
            {
                uint32_t triIdx = m_indices[i];
                float u, v, t;
                if (m_mesh->rayIntersect(triIdx, ray, u, v, t))
                {
                    if (shadowRay)
                        return true;
                    ray.maxt = its.t = t;
                    its.uv = Point2f(u, v);
                    its.mesh = m_mesh;
                    f = triIdx;
                    foundIntersection = true;
                }
            }
        }
        else
        {
            /* for interior node: push both children onto the stack */
            /* left child is always at nodeIdx + 1 */
            stack[stackPtr++] = nodeIdx + 1;
            stack[stackPtr++] = node.start; // right child
        }
    }

    if (foundIntersection)
    {
        /* At this point, we now know that there is an intersection,
           and we know the triangle index of the closest such intersection.

           The following computes a number of additional properties which
           characterize the intersection (normals, texture coordinates, etc..)
        */

        /* Find the barycentric coordinates */
        Vector3f bary;
        bary << 1 - its.uv.sum(), its.uv;

        /* References to all relevant mesh buffers */
        const Mesh *mesh = its.mesh;
        const MatrixXf &V = mesh->getVertexPositions();
        const MatrixXf &N = mesh->getVertexNormals();
        const MatrixXf &UV = mesh->getVertexTexCoords();
        const MatrixXu &F = mesh->getIndices();

        /* Vertex indices of the triangle */
        uint32_t idx0 = F(0, f), idx1 = F(1, f), idx2 = F(2, f);

        Point3f p0 = V.col(idx0), p1 = V.col(idx1), p2 = V.col(idx2);

        /* Compute the intersection positon accurately
           using barycentric coordinates */
        its.p = bary.x() * p0 + bary.y() * p1 + bary.z() * p2;

        /* Compute proper texture coordinates if provided by the mesh */
        if (UV.size() > 0)
            its.uv = bary.x() * UV.col(idx0) +
                     bary.y() * UV.col(idx1) +
                     bary.z() * UV.col(idx2);

        /* Compute the geometry frame */
        its.geoFrame = Frame((p1 - p0).cross(p2 - p0).normalized());

        if (N.size() > 0)
        {
            /* Compute the shading frame. Note that for simplicity,
               the current implementation doesn't attempt to provide
               tangents that are continuous across the surface. That
               means that this code will need to be modified to be able
               use anisotropic BRDFs, which need tangent continuity */

            its.shFrame = Frame(
                (bary.x() * N.col(idx0) +
                 bary.y() * N.col(idx1) +
                 bary.z() * N.col(idx2))
                    .normalized());
        }
        else
        {
            its.shFrame = its.geoFrame;
        }
    }

    return foundIntersection;
}

NORI_NAMESPACE_END
