#ifndef PARALLEL_KDTREE_H
#define PARALLEL_KDTREE_H

#include "datastructures.h"
#include <vector>

// Find dimension with maximum variance for splitting
int findSplitDimension(const parlay::sequence<Point>& points);

// Find the median value and corresponding point index for splitting
std::pair<double, int> findSplitValueAndIndex(
    const parlay::sequence<Point>& indexedPoints, 
    const parlay::sequence<int>& indices, 
    int splitDim);

// Recursively build kD-tree with parallelism at upper levels
KDNode* buildKDTreeRecursive(
    const parlay::sequence<Point>& points, 
    const parlay::sequence<int>& indices,
    int depth, 
    int maxDepth);

// Build kD-tree for a dataset in parallel
KDNode* buildKDTreeParallel(const Dataset& dataset);

// Clean up kD-tree memory
void deleteKDTree(KDNode* node);

// Build kD-trees for all clusters in parallel
void buildClusterKDTrees(std::vector<Cluster>& clusters);

#endif // PARALLEL_KDTREE_H