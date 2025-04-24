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

struct QueryStats {
    double totalTime;
    double minTime;
    double maxTime;
    double avgTime;
    std::vector<double> queryTimes;
    std::vector<double> nearestDistances;
    
    void computeStats() {
        if (queryTimes.empty()) return;
        
        minTime = *std::min_element(queryTimes.begin(), queryTimes.end());
        maxTime = *std::max_element(queryTimes.begin(), queryTimes.end());
        totalTime = std::accumulate(queryTimes.begin(), queryTimes.end(), 0.0);
        avgTime = totalTime / queryTimes.size();
    }
};

#endif // DATASTRUCTURES_H