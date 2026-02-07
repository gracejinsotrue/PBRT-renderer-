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

#include <tbb/tbb.h>
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
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, nTris),
                      [this](const tbb::blocked_range<uint32_t> &range)
                      {
                          for (uint32_t i = range.begin(); i < range.end(); ++i)
                              m_indices[i] = i;
                      });
    // m_indices.resize(nTris);
    // for (uint32_t i = 0; i < nTris; ++i)
    //     m_indices[i] = i;
    // i add code for timing
    auto start = std::chrono::high_resolution_clock::now();

    // buildRecursive(0, nTris, m_bbox); // this line here would call build recursive, i want to now call build():

    std::vector<uint32_t> allIndices(m_indices.begin(), m_indices.end());
    BVHBuildResult bvhResult = buildParallel(std::move(allIndices), m_bbox);
    m_nodes = std::move(bvhResult.nodes);
    m_indices = std::move(bvhResult.indices);

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

Accel::BVHBuildResult Accel::buildParallel(std::vector<uint32_t> indices, const BoundingBox3f &bbox)
{
    BVHBuildResult result;
    uint32_t count = indices.size();

    // create a root node
    BVHNode node;
    node.bbox = bbox;

    if (count <= MAX_TRI_PER_LEAF)
    {
        node.start = 0;
        node.nTriangles = (uint16_t)count;
        node.axis = 0;
        result.nodes.push_back(node);
        result.indices = std::move(indices);
        return result;
    }

    // compute the centroid bound
    int axis = bbox.getLargestAxis();
    BoundingBox3f centroidBBox;
    for (uint32_t idx : indices)
        centroidBBox.expandBy(m_mesh->getCentroid(idx));

    float cMin = centroidBBox.min[axis];
    float cMax = centroidBBox.max[axis];

    // all centroids coincide on this axis
    if (cMin == cMax)
    {
        node.start = 0;
        node.nTriangles = (uint16_t)count;
        node.axis = 0;
        result.nodes.push_back(node);
        result.indices = std::move(indices);
        return result;
    }

    // this is just SAH binned split, same as in accel_no_parallel.cpp (my original implementation)
    struct Bin
    {
        BoundingBox3f bbox;
        uint32_t count = 0;
    };
    Bin bins[N_BINS];

    float invBinWidth = N_BINS / (cMax - cMin);
    for (uint32_t idx : indices)
    {
        float c = m_mesh->getCentroid(idx)[axis];
        uint32_t bin = std::min((uint32_t)((c - cMin) * invBinWidth), N_BINS - 1);
        bins[bin].count++;
        bins[bin].bbox.expandBy(m_mesh->getBoundingBox(idx));
    }

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
        float cost = leftBox.getSurfaceArea() * leftCount + rightBox.getSurfaceArea() * rightCount;
        if (cost < bestCost)
        {
            bestCost = cost;
            bestSplit = s;
        }
    }

    float leafCost = bbox.getSurfaceArea() * count;
    if (bestCost >= leafCost) // is splitting worth? or do we just make leaf instead
    {
        node.start = 0;
        node.nTriangles = (uint16_t)count;
        node.axis = 0;
        result.nodes.push_back(node);
        result.indices = std::move(indices);
        return result;
    }

    // we partition
    float splitPos = cMin + bestSplit * (cMax - cMin) / N_BINS;
    auto midIter = std::partition(indices.begin(), indices.end(),
                                  [this, axis, splitPos](uint32_t idx)
                                  {
                                      return m_mesh->getCentroid(idx)[axis] < splitPos;
                                  });
    uint32_t mid = (uint32_t)(midIter - indices.begin());
    if (mid == 0 || mid == count)
        mid = count / 2;

    // split the index, compute child bboxes
    std::vector<uint32_t> leftIndices(indices.begin(), indices.begin() + mid);
    std::vector<uint32_t> rightIndices(indices.begin() + mid, indices.end());

    BoundingBox3f leftBox, rightBox;
    for (uint32_t idx : leftIndices)
        leftBox.expandBy(m_mesh->getBoundingBox(idx));
    for (uint32_t idx : rightIndices)
        rightBox.expandBy(m_mesh->getBoundingBox(idx));

    // if the subtrees are large enough we recurse in parallel
    BVHBuildResult leftResult, rightResult;

    if (count >= PARALLEL_THRESHOLD)
    {
        tbb::task_group tg;
        tg.run([&]()
               { leftResult = buildParallel(std::move(leftIndices), leftBox); });
        tg.run([&]()
               { rightResult = buildParallel(std::move(rightIndices), rightBox); });
        tg.wait();
    }
    else
    {
        leftResult = buildParallel(std::move(leftIndices), leftBox);
        rightResult = buildParallel(std::move(rightIndices), rightBox);
    }

    // now we merge, the layout being [thisNode, leftSubtree..., rightSubtree...]
    uint32_t leftSize = leftResult.nodes.size();
    uint32_t leftIdxSize = leftResult.indices.size();

    // interior node
    node.nTriangles = 0;
    node.axis = (uint16_t)axis;
    node.start = 1 + leftSize;
    result.nodes.reserve(1 + leftSize + rightResult.nodes.size());
    result.nodes.push_back(node);

    // append left subtree nodes, also adjust for index offset
    for (auto &n : leftResult.nodes)
    {
        if (n.isLeaf())
            n.start += 0; // left indices start at offset 0 in merged index array
        else
            n.start += 1; // offset by this node (parent)
        result.nodes.push_back(n);
    }

    // append right subtree nodes, also adjust for index offset
    for (auto &n : rightResult.nodes)
    {
        if (n.isLeaf())
            n.start += leftIdxSize; // right indices start after left indices
        else
            n.start += 1 + leftSize; // offset by this node + entire left subtree
        result.nodes.push_back(n);
    }

    // marge the index arrays [leftIndices, rightIndices]
    result.indices.reserve(leftIdxSize + rightResult.indices.size());
    result.indices = std::move(leftResult.indices);
    result.indices.insert(result.indices.end(),
                          rightResult.indices.begin(), rightResult.indices.end());

    return result;
}

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

    // 2)  here we evaluate SAH cost for each of the N_BINS-1 candidate splits
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
    m_nodes[nodeIdx].nTriangles = 0;

    return nodeIdx;
}

bool Accel::rayIntersect(const Ray3f &ray_, Intersection &its, bool shadowRay) const
{
    bool foundIntersection = false;
    uint32_t f = (uint32_t)-1;

    Ray3f ray(ray_);

    // ray traversal with explict stack
    /* we avoid std::stack/std::vector here to prevent per-ray heap allocation */
    uint32_t stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        uint32_t nodeIdx = stack[--stackPtr];
        const BVHNode &node = m_nodes[nodeIdx];

        // if the ray doesn't intersect the node's bounding box, skip it
        if (!node.bbox.rayIntersect(ray))
            continue;

        if (node.isLeaf())
        {
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
            // for interior node: push both children onto the stack
            // left child is always at nodeIdx + 1
            stack[stackPtr++] = nodeIdx + 1;
            stack[stackPtr++] = node.start;
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
