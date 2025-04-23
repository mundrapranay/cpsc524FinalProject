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
    std::unordered_set<const Point*> seen;
    std::mutex mtx;
    size_t maxSize;

public:
    ConcurrentMaxHeap(size_t K) 
      : maxSize(K) {}

    void pushCandidate(Neighbor cand) {
        std::lock_guard<std::mutex> lock(mtx);
        // 1) skip if already seen
        if (seen.count(cand.pt)) return;

        if (heap.size() < maxSize) {
            heap.push(cand);
            seen.insert(cand.pt);
        } else if (cand.dist < heap.top().dist) {
            // evict the worst
            const Point* worstPt = heap.top().pt;
            heap.pop();
            seen.erase(worstPt);
            // insert the new candidate
            heap.push(cand);
            seen.insert(cand.pt);
        }
    }

    double worstDist() {
        std::lock_guard<std::mutex> lock(mtx);
        return (heap.size() < maxSize)
               ? std::numeric_limits<double>::infinity()
               : heap.top().dist;
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

    size_t heapSize() {
      std::lock_guard<std::mutex> lock(mtx);
      return heap.size();
    }
};


void searchKDTree(KDNode* node, const Dataset &ds, const Point &query, ConcurrentMaxHeap &globalHeap, size_t k) {
    
    // if (!node) return;

    // // Visit this node
    // const Point &p = ds.points[node->pointIndex];
    // double d = sqDist(p, query);
    // globalHeap.pushCandidate({d, &p});

    // // Determine near/far child
    // int axis = node->axis;
    // double diff = query.coordinates[axis] - node->splitValue;
    // KDNode *nearChild = (diff <= 0) ? node->left : node->right;
    // KDNode *farChild  = (diff <= 0) ? node->right : node->left;

    // // Recurse near side
    // if (nearChild) searchKDTree(nearChild, ds, query, globalHeap);

    // // Possibly recurse far side
    // double worst = globalHeap.worstDist();
    // if (diff*diff < worst && farChild) {
    //     searchKDTree(farChild, ds, query, globalHeap);
    // }

    if (!node) return;
    const Point &p = ds.points[node->pointIndex];
    double d = sqDist(p, query);
    globalHeap.pushCandidate({d, &p});

    int axis = node->axis;
    double diff = query.coordinates[axis] - node->splitValue;
    KDNode *nearChild = (diff <= 0) ? node->left : node->right;
    KDNode *farChild  = (diff <= 0) ? node->right : node->left;


    parlay::par_do(
      [&]{
        searchKDTree(nearChild, ds, query, globalHeap, k);
      },
      [&]{
        bool go_far;
        {
          go_far = (globalHeap.heapSize() < k || (diff * diff < globalHeap.worstDist()));
        }
        if (go_far) searchKDTree(farChild, ds, query, globalHeap, k);
      }
    );

}

////////////////////////////////////////////////////////////////////////////////
// Parallel nearest-neighbor query using Parlay

// clusters: sequence of clusters
// query: the query point
// K: number of nearest neighbors to return
// P: number of top-priority trees to search
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

    // 2) Filter out empty clusters
    auto rootDists = parlay::remove_if(rootDistsAll, [&](auto &pr) {
      return pr.first == std::numeric_limits<double>::infinity();
    });

    // 3) Sort by distance
    auto sorted = parlay::sort(rootDists, [&](auto &a, auto &b){
      return a.first < b.first;
    });

    // 4) Clamp to at most P clusters
    size_t P2 = std::min(P, sorted.size());

    // 5) Shared concurrent max-heap for the top-K candidates
    ConcurrentMaxHeap globalHeap(K);

    // 6) Parallel search the top-P2 trees
    parlay::parallel_for(0, P2, [&](size_t t) {
      size_t idx = sorted[t].second;
      const Dataset &ds = clusters[idx].points;
      searchKDTree(ds.kdtree, ds, query, globalHeap, K);
    });

    // 7) Extract all candidates, sort them ascending
    auto neighbors = globalHeap.getAll();
    neighbors.erase(
      std::remove_if(neighbors.begin(), neighbors.end(),
        [](const Neighbor &n){
          return !std::isfinite(n.dist);
        }),
      neighbors.end()
    );
    std::sort(neighbors.begin(), neighbors.end(),
              [&](auto &a, auto &b){ return a.dist < b.dist; });

    // 8) Simply take the first K (no dedupe)
    std::vector<const Point*> result;
    result.reserve(std::min(neighbors.size(), K));
    for (size_t i = 0; i < neighbors.size() && result.size() < K; ++i) {
      // if (neighbors[i].dist != std::numeric_limits<double>::infinity()) {
      //   result.push_back(neighbors[i].pt);
      // }
      result.push_back(neighbors[i].pt);
    }
    return result;
}
