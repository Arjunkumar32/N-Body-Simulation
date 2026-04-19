/*
 * barnes_hut.cpp  +  octree.h (inline)
 *
 * Barnes-Hut hierarchical force approximation (O(N log N)).
 * theta = 0.5: if s/d < theta the cell is treated as a point mass.
 *
 * Structure:
 *  - OctNode: axis-aligned cube, stores mass & COM of contained bodies.
 *  - Octree : builds tree from bodies, then computes force on each body.
 */

#include "nbody.h"
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include <omp.h>

// ─── Octree Node ───────────────────────────────────────────────────────────
struct OctNode {
    Vec3   center;          // centre of cube
    double halfSize;        // half-length of cube edge

    double totalMass = 0.0;
    Vec3   com       = {};  // centre of mass

    // Children: 0=---,1=+--,2=-+-,3=++-,4=--+,5=+-+,6=-++,7=+++
    std::array<std::unique_ptr<OctNode>, 8> children;
    bool isLeaf   = true;
    int  bodyIdx  = -1;    // index into bodies array (-1 = empty)

    OctNode(Vec3 c, double hs) : center(c), halfSize(hs) {}

    // Determine which child octant a position belongs to
    int octantOf(const Vec3& p) const {
        return ((p.x >= center.x) ? 1 : 0)
             | ((p.y >= center.y) ? 2 : 0)
             | ((p.z >= center.z) ? 4 : 0);
    }

    Vec3 childCenter(int oct) const {
        double q = halfSize * 0.5;
        return {
            center.x + ((oct & 1) ? q : -q),
            center.y + ((oct & 2) ? q : -q),
            center.z + ((oct & 4) ? q : -q)
        };
    }
};

// ─── Insert body into tree ─────────────────────────────────────────────────
static void insertBody(OctNode* node, const std::vector<Body>& bodies, int idx) {
    if (node->isLeaf && node->bodyIdx == -1) {
        // Empty leaf — just store
        node->bodyIdx   = idx;
        node->totalMass = bodies[idx].mass;
        node->com       = bodies[idx].pos;
        return;
    }

    if (node->isLeaf) {
        // Occupied leaf — subdivide
        int existing = node->bodyIdx;
        node->bodyIdx = -1;
        node->isLeaf  = false;

        int oct = node->octantOf(bodies[existing].pos);
        if (!node->children[oct])
            node->children[oct] = std::make_unique<OctNode>(
                node->childCenter(oct), node->halfSize * 0.5);
        insertBody(node->children[oct].get(), bodies, existing);
    }

    // Internal node — update COM and descend
    double newMass = node->totalMass + bodies[idx].mass;
    node->com = (node->com * node->totalMass + bodies[idx].pos * bodies[idx].mass)
                / newMass;
    node->totalMass = newMass;

    int oct = node->octantOf(bodies[idx].pos);
    if (!node->children[oct])
        node->children[oct] = std::make_unique<OctNode>(
            node->childCenter(oct), node->halfSize * 0.5);
    insertBody(node->children[oct].get(), bodies, idx);
}

// ─── Compute force on body i from subtree ─────────────────────────────────
static Vec3 treeForce(const OctNode* node, const Body& bi) {
    if (!node || node->totalMass == 0.0) return {};
    if (node->isLeaf && node->bodyIdx == -1) return {};

    Vec3   r    = node->com - bi.pos;
    double dist = r.norm();
    if (dist < 1e-12) return {};              // same body

    // Barnes-Hut criterion: s/d < theta  → treat subtree as point mass
    double s = node->halfSize * 2.0;
    if (node->isLeaf || (s / dist) < THETA) {
        double r2   = dist * dist + SOFTENING * SOFTENING;
        double r3   = r2 * std::sqrt(r2);
        double fmag = G * bi.mass * node->totalMass / r3;
        return r * fmag;                       // force vector on bi
    }

    // Recurse into children
    Vec3 total = {};
    for (const auto& child : node->children)
        total += treeForce(child.get(), bi);
    return total;
}

// ─── Public API ────────────────────────────────────────────────────────────
void computeForcesBarnesHut(std::vector<Body>& bodies) {
    int n = static_cast<int>(bodies.size());

    // Find bounding box
    double maxCoord = 0.0;
    for (auto& b : bodies) {
        maxCoord = std::max(maxCoord,
            std::max(std::abs(b.pos.x), std::max(std::abs(b.pos.y), std::abs(b.pos.z))));
    }
    double halfSize = (maxCoord + 1.0) * 1.5;

    // Build tree on single thread (insertions aren't thread-safe)
    auto root = std::make_unique<OctNode>(Vec3{0,0,0}, halfSize);
    for (int i = 0; i < n; ++i)
        insertBody(root.get(), bodies, i);

    // Traverse tree in parallel — each body thread-safe (read-only tree)
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < n; ++i) {
        Vec3 f = treeForce(root.get(), bodies[i]);
        bodies[i].force = f;
        bodies[i].acc   = f / bodies[i].mass;
    }
}
