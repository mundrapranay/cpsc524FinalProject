#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <parlay/primitives.h>
#include <parlay/parallel.h>
#include <parlay/sequence.h>
// #include <parlay/sort.h>

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
};

struct Dataset {
    int n;
    parlay::sequence<Point> points;
    Point centroid;
    KDNode* kdtree;
};

struct Cluster {
    Dataset points;
    Point centroid;
};

#endif // DATASTRUCTURES_H