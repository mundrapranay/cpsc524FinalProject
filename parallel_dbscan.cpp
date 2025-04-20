#include "src/datastructures.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <queue>
#include <atomic>

// --- Function to read example points from CSV ---
Dataset readCSV(const std::string& filename) {
    Dataset dataset;
    std::ifstream file(filename);
    std::string line;
    parlay::sequence<Point> points;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> coords;

        while (std::getline(ss, cell, ',')) {
            coords.push_back(std::stod(cell));
        }

        Point p;
        p.dimension = coords.size();
        p.coordinates = std::move(coords);
        points.push_back(p);
    }

    dataset.n = points.size();
    dataset.points = std::move(points);
    dataset.kdtree = nullptr;

    return dataset;
}

double distance(const Point& a, const Point& b) {
    double sum = 0;
    for (int i = 0; i < a.dimension; ++i) {
        double diff = a.coordinates[i] - b.coordinates[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Find neighbors within eps radius
parlay::sequence<int> regionQuery(const Dataset& data, int index, double eps) {
    const Point& point = data.points[index];
    return parlay::filter(
        parlay::iota<int>(data.n),
        [&](int j) {
            return distance(point, data.points[j]) <= eps;
        }
    );
}

// Parallel DBSCAN implementation
std::pair<std::vector<Cluster>, parlay::sequence<int>> parallelDBSCAN(
    const Dataset& data, 
    double eps, 
    int minPts
) {
    int n = data.n;
    parlay::sequence<int> labels(n, -1); // -1 = unvisited, -2 = noise
    std::atomic<int> clusterId{0};
    std::mutex label_mutex;

    auto expandCluster = [&](int pointIdx, int currentClusterId) {
        std::queue<int> q;
        q.push(pointIdx);
        labels[pointIdx] = currentClusterId;

        while (!q.empty()) {
            int p = q.front();
            q.pop();

            auto neighbors = regionQuery(data, p, eps);

            if (neighbors.size() >= static_cast<size_t>(minPts)) {
                for (int nIdx : neighbors) {
                    // int expected = -1;
                    if (labels[nIdx] == -1) {
                        labels[nIdx] = currentClusterId;
                        q.push(nIdx);
                    }
                }
            }
        }
    };

    parlay::parallel_for(0, n, [&](int i) {
        if (labels[i] != -1) return;

        auto neighbors = regionQuery(data, i, eps);

        if (neighbors.size() < static_cast<size_t>(minPts)) {
            labels[i] = -2; // noise
        } else {
            int id = clusterId.fetch_add(1);
            expandCluster(i, id);
        }
    });

    // Group points by cluster
    int totalClusters = clusterId.load();
    std::vector<Cluster> clusters(totalClusters);

    for (int i = 0; i < n; ++i) {
        int label = labels[i];
        if (label >= 0) {
            clusters[label].points.points.push_back(data.points[i]);
        }
    }

    // Set metadata
    for (Cluster& c : clusters) {
        c.points.n = c.points.points.size();
        c.points.kdtree = nullptr;
        // Optionally compute centroid here
    }

    return {clusters, labels};
}

void writeClusteredPointsToCSV(const std::string& filename,
                                const Dataset& data,
                                const parlay::sequence<int>& labels) {
    std::ofstream file(filename);
    file << "x,y,cluster\n";

    for (int i = 0; i < data.n; ++i) {
        const auto& point = data.points[i];
        if (point.dimension < 2) continue; // skip if not 2D
        file << point.coordinates[0] << "," << point.coordinates[1] << "," << labels[i] << "\n";
    }

    file.close();
}

// --- Main ---
// int main() {
//     std::string filename = "/Users/syedrizvi/Desktop/Life/Yale/Courses/04_Spring_2025/CPSC424/Final_Project/cpsc524FinalProject/example_data/sample2d.csv";

//     Dataset data = readCSV(filename);
//     std::cout << "Loaded " << data.n << " points.\n";

//     double eps = 0.2;  // 0.2
//     int minNeighbors = 10;
//     std::cout << "Testing parallel DBSCAN with eps = " << eps << " and minNeighbors = " << minNeighbors << ".\n";
//     auto [clusters, labels] = parallelDBSCAN(data, eps, minNeighbors);
//     std::cout << "Clustered into " << clusters.size() << " clusters.\n";

//     for (int i = 0; i < clusters.size(); ++i) {
//         std::cout << "Cluster " << i << ": " << clusters[i].points.n << " points\n";
//     }

//     // Write clustered points to CSV
//     writeClusteredPointsToCSV("example_data/clustered_points.csv", data, labels);

//     return 0;
// }
