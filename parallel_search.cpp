#include "src/datastructures.h"
#include "src/distance_metrics.h"

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

// NOTE: THIS IS FOR LOCAL / CLUSTER kD TREE SEARCHES

void searchKDTree(
    KDNode* node, 
    const Dataset &ds, 
    const Point &query, 
    ConcurrentMaxHeap &globalHeap, 
    size_t k) 
{
    if (!node) return;
    
    // Process current node
    const Point &p = ds.points[node->pointIndex];
    double d = cosineDistance(p, query);
    globalHeap.pushCandidate({d, &p});
    
    // If leaf node, return early
    if (!node->left && !node->right) {
        return;
    }
    
    // Determine which child to visit first
    int axis = node->axis;
    double diff = query.coordinates[axis] - node->splitValue;
    KDNode *nearChild = (diff <= 0) ? node->left : node->right;
    KDNode *farChild  = (diff <= 0) ? node->right : node->left;
    
    // Parallelize at all levels
    parlay::par_do(
        [&]() {
            if (nearChild) {
                searchKDTree(nearChild, ds, query, globalHeap, k);
            }
        },
        [&]() {
            // Check if we might find better points in the far subtree
            bool go_far = (globalHeap.heapSize() < k || (diff * diff < globalHeap.worstDist()));
            
            if (go_far && farChild) {
                searchKDTree(farChild, ds, query, globalHeap, k);
            }
        }
    );
}

// void searchKDTreeLimitDepth(
//     KDNode* node, 
//     const Dataset &ds, 
//     const Point &query, 
//     ConcurrentMaxHeap &globalHeap, 
//     size_t k,
//     int depth = 0) 
// {
//     if (!node) return;
    
//     // Process current node
//     const Point &p = ds.points[node->pointIndex];
//     double d = cosineDistance(p, query);
//     globalHeap.pushCandidate({d, &p});
    
//     // If leaf node, return early
//     if (!node->left && !node->right) {
//         return;
//     }
    
//     // Determine which child to visit first
//     int axis = node->axis;
//     double diff = query.coordinates[axis] - node->splitValue;
//     KDNode *nearChild = (diff <= 0) ? node->left : node->right;
//     KDNode *farChild  = (diff <= 0) ? node->right : node->left;
    
//     // Only parallelize up to depth 3
//     if (depth < 3) {
//         // Parallelize at shallow depths
//         parlay::par_do(
//             [&]() {
//                 if (nearChild) {
//                     searchKDTree(nearChild, ds, query, globalHeap, k, depth + 1);
//                 }
//             },
//             [&]() {
//                 // Check if we might find better points in the far subtree
//                 bool go_far = (globalHeap.heapSize() < k || (diff * diff < globalHeap.worstDist()));
                
//                 if (go_far && farChild) {
//                     searchKDTree(farChild, ds, query, globalHeap, k, depth + 1);
//                 }
//             }
//         );
//     } else {
//         // Sequential search for deeper levels
//         if (nearChild) {
//             searchKDTree(nearChild, ds, query, globalHeap, k, depth + 1);
//         }
        
//         // More aggressive pruning at deeper levels
//         bool go_far = (globalHeap.heapSize() < k || (diff * diff < globalHeap.worstDist()));
        
//         if (go_far && farChild) {
//             searchKDTree(farChild, ds, query, globalHeap, k, depth + 1);
//         }
//     }
// }

// void searchKDTreeSequential(
//     KDNode* node, 
//     const Dataset &ds, 
//     const Point &query, 
//     ConcurrentMaxHeap &globalHeap, 
//     size_t k) 
// {
//     if (!node) return;
    
//     // Process current node
//     const Point &p = ds.points[node->pointIndex];
//     double d = cosineDistance(p, query);
//     globalHeap.pushCandidate({d, &p});
    
//     // If leaf node, return early
//     if (!node->left && !node->right) {
//         return;
//     }
    
//     // Determine which child to visit first
//     int axis = node->axis;
//     double diff = query.coordinates[axis] - node->splitValue;
//     KDNode *nearChild = (diff <= 0) ? node->left : node->right;
//     KDNode *farChild  = (diff <= 0) ? node->right : node->left;
    
//     // Always search near child first (sequentially)
//     if (nearChild) {
//         searchKDTreeSequential(nearChild, ds, query, globalHeap, k);
//     }
    
//     // Only search far subtree if potentially useful
//     bool go_far = (globalHeap.heapSize() < k || (diff * diff < globalHeap.worstDist()));
    
//     if (go_far && farChild) {
//         searchKDTreeSequential(farChild, ds, query, globalHeap, k);
//     }
// }

////////////////////////////////////////////////////////////////////////////////
// Parallel nearest-neighbor query using Parlay
// NOTE: THIS IS FOR LOCAL CLUSTER SEARCHES (main.cpp)

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
      return std::make_pair(cosineDistance(rp, query), i);
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
    // ConcurrentMaxHeap globalHeap(K);

    parlay::sequence<std::vector<Neighbor>> allLocal
    = parlay::tabulate(P2, [&](size_t t) {
        size_t idx = sorted[t].second;
        const Dataset &ds = clusters[idx].points;
        // each thread gets its own heap
        ConcurrentMaxHeap localHeap(K);

        searchKDTree(ds.kdtree, ds, query, localHeap, K);

        // searchKDTreeSequential(ds.kdtree, ds, query, localHeap, K);

        // extract and return its sorted candidates
        auto v = localHeap.getAll();
        std::sort(v.begin(), v.end(),
                  [](auto &a, auto &b){ return a.dist < b.dist; });
        return v;
    });

    // 7) Flatten all per‐cluster lists
    auto flat = parlay::flatten(allLocal);

    // 8) pick the global top-K from 'flat' (no locks needed)
    if (flat.size() > K) {
      std::nth_element(flat.begin(), flat.begin()+K, flat.end(),
                      [](auto &a, auto &b){ return a.dist < b.dist; });
      flat.resize(K);
    }
    std::sort(flat.begin(), flat.end(),
              [](auto &a, auto &b){ return a.dist < b.dist; });

    // 9) collect pointers
    std::vector<const Point*> result;
    result.reserve(flat.size());
    for (auto &n : flat) result.push_back(n.pt);
    return result;
  }

// Fully parallel search for global kD-tree using concurrent max heap
void searchGlobalTreeParallelWithHeap(
    KDNode* node, 
    const Dataset &ds, 
    const Point &query, 
    ConcurrentMaxHeap &globalHeap, 
    int k,
    int depth = 0) 
{
    if (!node) return;
    
    // Process current node
    const Point &p = ds.points[node->pointIndex];
    double d = cosineDistance(p, query);
    globalHeap.pushCandidate({d, &p});
    
    // If leaf node, return early
    if (!node->left && !node->right) {
        return;
    }
    
    // Determine which child to visit first
    int axis = node->axis;
    double queryVal = query.coordinates[axis];
    double splitVal = node->splitValue;
    
    KDNode *nearChild = (queryVal <= splitVal) ? node->left : node->right;
    KDNode *farChild = (queryVal <= splitVal) ? node->right : node->left;
    
    // Only parallelize up to depth 3
    if (depth < 3) {
        // Parallelize at shallow depths
        parlay::par_do(
            [&]() {
                if (nearChild) {
                    searchGlobalTreeParallelWithHeap(nearChild, ds, query, globalHeap, k, depth + 1);
                }
            },
            [&]() {
                // Only search far subtree if potentially useful
                double axisDist = std::abs(queryVal - splitVal);
                double axisDist2 = axisDist * axisDist;
                
                // Check if we might find better points in the far subtree
                bool go_far = (globalHeap.heapSize() < static_cast<size_t>(k) || axisDist2 < globalHeap.worstDist());
                
                if (go_far && farChild) {
                    searchGlobalTreeParallelWithHeap(farChild, ds, query, globalHeap, k, depth + 1);
                }
            }
        );
    } else {
        // Sequential search for deeper levels
        if (nearChild) {
            searchGlobalTreeParallelWithHeap(nearChild, ds, query, globalHeap, k, depth + 1);
        }
        
        // Only search far subtree if potentially useful
        double axisDist = std::abs(queryVal - splitVal);
        double axisDist2 = axisDist * axisDist;
        
        // More aggressive pruning at deeper levels
        bool go_far = (globalHeap.heapSize() < static_cast<size_t>(k) || axisDist2 < globalHeap.worstDist());
        
        if (go_far && farChild) {
            searchGlobalTreeParallelWithHeap(farChild, ds, query, globalHeap, k, depth + 1);
        }
    }
}


// void searchGlobalTreeSequential(
//     KDNode* node, 
//     const Dataset &ds, 
//     const Point &query, 
//     ConcurrentMaxHeap &globalHeap, 
//     int k) 
// {
//     if (!node) return;
    
//     // Process current node
//     const Point &p = ds.points[node->pointIndex];
//     double d = cosineDistance(p, query);
//     globalHeap.pushCandidate({d, &p});
    
//     // If leaf node, return early
//     if (!node->left && !node->right) {
//         return;
//     }
    
//     // Determine which child to visit first
//     int axis = node->axis;
//     double queryVal = query.coordinates[axis];
//     double splitVal = node->splitValue;
    
//     KDNode *nearChild = (queryVal <= splitVal) ? node->left : node->right;
//     KDNode *farChild = (queryVal <= splitVal) ? node->right : node->left;
    
//     // Always search near child first (sequentially)
//     if (nearChild) {
//         searchGlobalTreeSequential(nearChild, ds, query, globalHeap, k);
//     }
    
//     // Only search far subtree if potentially useful
//     double axisDist = std::abs(queryVal - splitVal);
//     double axisDist2 = axisDist * axisDist;
    
//     // Check if we might find better points in the far subtree
//     bool go_far = (globalHeap.heapSize() < static_cast<size_t>(k) || axisDist2 < globalHeap.worstDist());
    
//     if (go_far && farChild) {
//         searchGlobalTreeSequential(farChild, ds, query, globalHeap, k);
//     }
// }


// Public interface for fully parallel global kD-tree search
std::vector<const Point*> kNearestNeighborsParallelGlobal(
    KDNode* root, 
    const Dataset& data, 
    const Point& query, 
    int k) 
{
    // Use a ConcurrentMaxHeap for the global search
    ConcurrentMaxHeap globalHeap(k);
    
    // Search tree directly with the heap
    searchGlobalTreeParallelWithHeap(root, data, query, globalHeap, k);
    // searchGlobalTreeSequential(root, data, query, globalHeap, k);
    
    // Get all candidates from the heap (already sorted)
    auto candidates = globalHeap.getAll();
    
    // Extract the points
    std::vector<const Point*> result;
    result.reserve(candidates.size());
    
    for (const auto& candidate : candidates) {
        result.push_back(candidate.pt);
    }
    
    return result;
  }
