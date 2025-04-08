#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <include/parlay/primitives.h>
#include <include/parlay/parallel.h>
#include <include/parlay/sequence.h>
#include <include/parlay/sort.h>




struct Point {
    int dimension;
    std::vector<double> coordinates;
};

struct KDNode {
    int axis;
    double splitValue;
    int pointIndex;
    KDNode* left;
    KDNode* right;
}

struct Dataset {
    int n;
    parlay::sequence<Point> points;
    Point centroid;
    KDNode* kdtree;
}

