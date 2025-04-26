#include "src/datastructures.h"
#include <algorithm>
#include <memory>
#include <atomic>


int findSplitDimension(const parlay::sequence<Point>& points) {
    /*
    At each level, finds dimension which reduces maximum variance
    */
    int dim = points[0].dimension;

    // calculate means for each dimension
    std::vector<double> means(dim, 0.0);
    for (const auto& point : points) {
        for (int d = 0; d < dim; ++d) {
            means[d] += point.coordinates[d];
        }
    }

    for (int d = 0; d < dim; ++d) {
        means[d] /= points.size();
    }

    // Recall: Var(X_d) = \frac{1}{n-1} \sum^{n} (X_{i, d} - \mu_{d})^2
    std::vector<double> variances(dim, 0.0);
    for (const auto& point : points) {
        for (int d = 0; d < dim; ++d) {
            double diff = point.coordinates[d] - means[d];
            variances[d] += diff * diff;
        }
    }

    // find maximum varying dimension
    int maxVarDim = 0;
    double maxVar = variances[0];
    for (int d = 1; d < dim; ++d) {
        if (variances[d] > maxVar) {
            maxVar = variances[d];
            maxVarDim = d;
        } 
    }
    return maxVarDim;
}

// split along median of given dim
// double findSplitValue(const parlay::sequence<Point>& points, int splitDim) {
//     auto values = parlay::tabulate(points.size(), [&](size_t i) {
//         return points[i].coordinates[splitDim]
//     });

//     size_t medianIdx = values.size() / 2;
//     // partial sorting
//     parlay::nth_element(values, medianIdx);
//     return values[medianIdx];
// }

// find split along median of given dim and return point
std::pair<double, int> findSplitValueAndIndex(const parlay::sequence<Point>& indexedPoints, const parlay::sequence<int>& indices, int splitDim) {
    // (value, originalIdx)
    auto valuePairs = parlay::tabulate(indices.size(), [&](size_t i) {
        return std::make_pair(indexedPoints[i].coordinates[splitDim], indices[i]);
    });

    std::vector<std::pair<double, int>> vec_valuePairs(valuePairs.begin(), valuePairs.end());

    size_t medianIdx = vec_valuePairs.size() / 2;

    // partial sort, only find medianIdx rank statistic
    std::nth_element(vec_valuePairs.begin(),
                     vec_valuePairs.begin() + medianIdx,
                     vec_valuePairs.end(),
                     [](auto& a, auto& b) { return a.first < b.first; });

    // parlay::nth_element(valuePairs, medianIdx, [](auto& a, auto& b) {return a.first < b.first; });
    return {vec_valuePairs[medianIdx].first, vec_valuePairs[medianIdx].second};
}


KDNode* buildKDTreeRecursive(const parlay::sequence<Point>& points, const parlay::sequence<int>& indices,
                             int depth, int maxDepth) {
    if (indices.size() == 0) {
        return nullptr;
    }

    if (indices.size() == 1 || depth >= maxDepth) {
        KDNode* node = new KDNode();
        node->axis = -1; // leaf node
        node->pointIndex = indices[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }

    // obtain points
    parlay::sequence<Point> indexedPoints = parlay::tabulate(indices.size(), [&](size_t i) {
        return points[indices[i]];
    });

    int splitDim = findSplitDimension(indexedPoints);

    // double splitVal = findSplitValue(indexedPoints, splitDim);
    auto [splitVal, medianPointIdx] = findSplitValueAndIndex(indexedPoints, indices, splitDim);

    // partition via splitVal
    auto leftIndices = parlay::filter(indices, [&](int idx) {
        return points[idx].coordinates[splitDim] <= splitVal;
    });

    auto rightIndices = parlay::filter(indices, [&](int idx) {
        return points[idx].coordinates[splitDim] > splitVal;
    });

    KDNode* node = new KDNode();
    node->axis = splitDim;
    node->splitValue = splitVal;
    node->pointIndex = medianPointIdx;

    // parallelize recursive subtree construction (up to a certain depth which we can prob tune)
    if (depth < 3) {
        
        KDNode* leftNode = nullptr;
        KDNode* rightNode = nullptr;

        parlay::par_do(
            [&]() { leftNode = buildKDTreeRecursive(points, leftIndices, depth + 1, maxDepth); },
            [&]() { rightNode = buildKDTreeRecursive(points, rightIndices, depth + 1, maxDepth); },
            false // non-conservative mode
        );
        node->left = leftNode;
        node->right = rightNode;
    } else {
        // Sequential for deeper levels
        node->left = buildKDTreeRecursive(points, leftIndices, depth + 1, maxDepth);
        node->right = buildKDTreeRecursive(points, rightIndices, depth + 1, maxDepth);
    }
    return node;
}

// Main function to build each kD tree
KDNode* buildKDTreeParallel(const Dataset& dataset) {
    int maxDepth = 2 * (int)std::log2(dataset.n) + 1;

    auto indices = parlay::tabulate(dataset.n, [](size_t i) {
        return (int)i;
    });

    return buildKDTreeRecursive(dataset.points, indices, 0, maxDepth);
}

void deleteKDTree(KDNode* node) {
    if (node) {
        deleteKDTree(node->left);
        deleteKDTree(node->right);
        delete node;
    }
}

// parallelize tree construction across clusters
void buildClusterKDTrees(std::vector<Cluster>& clusters) {
    parlay::parallel_for(0, clusters.size(), [&](size_t i) {
        // delete any existing trees
        if (clusters[i].points.kdtree) {
            deleteKDTree(clusters[i].points.kdtree);
        }

        // create new one
        clusters[i].points.kdtree = buildKDTreeParallel(clusters[i].points);
    });
}
