#include "src/datastructures.h"
#include <vector>
#include <queue>
#include <mutex>
#include <algorithm>
#include <limits>
#include <cmath>


struct Neighbor {
    double dist;
    const Point* pt;
};


struct NeighborCompare {
  bool operator()(Neighbor const &a, Neighbor const &b) {
    return a.dist < b.dist;
  }  
};


static double sqDist(const Point &a, const Point &b) {
    double d = 0;
    for (int i = 0; i < a.dimension; ++i) {
        double diff = a.coordinates[i] - b.coordinates[i];
        d += diff * diff;
    }
    return d;
}

class ConcurrentMaxHeap {
    std::priority_queue<Neighbor, std::vector<Neighbor>, NeighborCompare> heap;
    std::mutex mtx;
    size_t maxSize;

public:
    ConcurrentMaxHeap(size_t K) : maxSize(K) {}
    
    void pushCandidate(Neighbor cand) {
        std::lock_guard<std::mutex> lock(mtx);
        if (heap.size() < maxSize) {
            heap.push(cand);
        } else if (cand.dist < heap.top().dist) {
            heap.pop();
            heap.push(cand);
        }
    }

    double worstDist() {
        std::lock_guard<std::mutex> lock(mtx);
        if (heap.size() < maxSize) return std::numeric_limits<double>::infinity();
        return heap.top().dist;
    }

    std::vector<Neighbor> getAll() {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<Neighbor> out;
        out.reserve(heap.size());
        while (!heap.empty()) {
            out.push_back(heap.top());
            heap.pop();
        }
        return out;
    }
};


void searchKDTree(KDNode* node, const Dataset &ds, const Point &query, ConcurrentMaxHeap &globalHeap) {
    
    if (!node) return;

    // Visit this node
    const Point &p = ds.points[node->pointIndex];
    double d = sqDist(p, query);
    globalHeap.pushCandidate({d, &p});

    // Determine near/far child
    int axis = node->axis;
    double diff = query.coordinates[axis] - node->splitValue;
    KDNode *nearChild = (diff <= 0) ? node->left : node->right;
    KDNode *farChild  = (diff <= 0) ? node->right : node->left;

    // Recurse near side
    if (nearChild) searchKDTree(nearChild, ds, query, globalHeap);

    // Possibly recurse far side
    double worst = globalHeap.worstDist();
    if (diff*diff < worst && farChild) {
        searchKDTree(farChild, ds, query, globalHeap);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Parallel nearest-neighbor query using Parlay

// clusters: sequence of clusters
// query: the query point
// K: number of nearest neighbors to return
// P: number of top-priority trees to search
// std::vector<const Point*> parallelNearestNeighborQuery(
//     const parlay::sequence<Cluster> &clusters,
//     const Point &query,
//     size_t K,
//     size_t P)
// {
//     size_t C = clusters.size();

//     // 1) Compute (distance, index) for each cluster root
//     auto rootDists = parlay::tabulate(C, [&](size_t i) {
//       KDNode *root = clusters[i].points.kdtree;
//       if (!root) return std::make_pair(std::numeric_limits<double>::infinity(), i);
//       const Point &rp = clusters[i].points.points[root->pointIndex];
//       return std::make_pair(sqDist(rp, query), i);
//     });

//     // 2) Sort by distance
//     auto sorted = parlay::sort(rootDists, [&](auto &a, auto &b){
//       return a.first < b.first;
//     });

//     // 3) Clamp P to available clusters
//     size_t P2 = std::min(P, sorted.size());

//     // 4) Shared concurrent max-heap
//     ConcurrentMaxHeap globalHeap(K);

//     // 5) Parallel search top-P trees
//     parlay::parallel_for(0, P2, [&](size_t t) {
//       size_t idx = sorted[t].second;
//       const Dataset &ds = clusters[idx].points;
//       searchKDTree(ds.kdtree, ds, query, globalHeap);
//     });

//     // 6) Extract and sort neighbors by ascending distance
//     auto neighbors = globalHeap.getAll();
//     std::sort(neighbors.begin(), neighbors.end(),
//               [&](auto &a, auto &b){ return a.dist < b.dist; });

//     // 7) Collect pointers
//     std::vector<const Point*> result;
//     result.reserve(neighbors.size());
//     for (auto &n : neighbors) result.push_back(n.pt);
//     return result;
// }

// std::vector<const Point*> parallelNearestNeighborQuery(
//     const parlay::sequence<Cluster> &clusters,
//     const Point &query,
//     size_t K,
//     size_t P)
// {
//     size_t C = clusters.size();

//     // 1) Compute (distance, index) for each cluster root
//     auto rootDists = parlay::tabulate(C, [&](size_t i) {
//       KDNode *root = clusters[i].points.kdtree;
//       if (!root) return std::make_pair(std::numeric_limits<double>::infinity(), i);
//       const Point &rp = clusters[i].points.points[root->pointIndex];
//       return std::make_pair(sqDist(rp, query), i);
//     });

//     // 2) Sort by distance
//     auto sorted = parlay::sort(rootDists, [&](auto &a, auto &b){
//       return a.first < b.first;
//     });

//     // 3) Clamp P to available clusters
//     size_t P2 = std::min(P, sorted.size());

//     // 4) Shared concurrent max-heap
//     ConcurrentMaxHeap globalHeap(K);

//     // 5) Parallel search top-P trees
//     parlay::parallel_for(0, P2, [&](size_t t) {
//       size_t idx = sorted[t].second;
//       const Dataset &ds = clusters[idx].points;
//       searchKDTree(ds.kdtree, ds, query, globalHeap);
//     });

//     // 6) Extract and sort neighbors by ascending distance
//     auto neighbors = globalHeap.getAll();
//     std::sort(neighbors.begin(), neighbors.end(),
//               [&](auto &a, auto &b){ return a.dist < b.dist; });

//     // 7) Collect up to K distinct pointers
//     std::vector<const Point*> result;
//     result.reserve(std::min(neighbors.size(), K));
//     for (auto &n : neighbors) {
//       if (std::find(result.begin(), result.end(), n.pt) == result.end()) {
//         if (n.dist != std::numeric_limits<double>::infinity()) {
//             result.push_back(n.pt);
//         }
//         if (result.size() == K) break;
//       }
//     }
//     return result;
// }

std::vector<const Point*> parallelNearestNeighborQuery(
    const parlay::sequence<Cluster> &clusters,
    const Point &query,
    size_t K,
    size_t P)
{
    size_t C = clusters.size();

    // 1) Build (distance, index) only for clusters with a tree
    auto rootDistsAll = parlay::tabulate(C, [&](size_t i) {
      KDNode *root = clusters[i].points.kdtree;
      if (!root) return std::make_pair(std::numeric_limits<double>::infinity(), i);
      const Point &rp = clusters[i].points.points[root->pointIndex];
      return std::make_pair(sqDist(rp, query), i);
    });

    // 2) Remove the infinite‐distance entries (no tree)
    auto rootDists = parlay::remove_if(rootDistsAll, [&](auto &pr) {
      return pr.first == std::numeric_limits<double>::infinity();
    });

    // 3) Sort by distance
    auto sorted = parlay::sort(rootDists, [&](auto &a, auto &b){
      return a.first < b.first;
    });

    // 4) Clamp to at most P real clusters
    size_t P2 = std::min(P, sorted.size());

    // 5) Shared concurrent max‐heap
    ConcurrentMaxHeap globalHeap(K);

    // 6) Parallel search top‐P2 real trees
    parlay::parallel_for(0, P2, [&](size_t t) {
      size_t idx = sorted[t].second;
      const Dataset &ds = clusters[idx].points;
      searchKDTree(ds.kdtree, ds, query, globalHeap);
    });

    // 7) Extract, sort and collect up to K distinct points
    auto neighbors = globalHeap.getAll();
    std::sort(neighbors.begin(), neighbors.end(),
              [&](auto &a, auto &b){ return a.dist < b.dist; });

    std::vector<const Point*> result;
    result.reserve(std::min(neighbors.size(), K));
    for (auto &n : neighbors) {
      if (std::find(result.begin(), result.end(), n.pt) == result.end()) {
        result.push_back(n.pt);
        if (result.size() == K) break;
      }
    }
    return result;
}