#include "src/datastructures.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <random>

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


// Helper: Euclidean distance
double distance(const Point& a, const Point& b) {
    double sum = 0;
    for (int i = 0; i < a.dimension; ++i) {
        double diff = a.coordinates[i] - b.coordinates[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<Cluster> naiveClusterData(const Dataset& data, int k) {
    std::vector<Cluster> clusters(k);
    std::mt19937 gen(42); // fixed seed
    std::uniform_int_distribution<> dis(0, data.points.size() - 1);

    // Randomly assign centroids
    for (int i = 0; i < k; ++i) {
        clusters[i].centroid = data.points[dis(gen)];
    }

    // Initialize sequences to store points per cluster
    std::vector<parlay::sequence<Point>> cluster_points(k);

    // Assign each point to the nearest centroid
    for (const auto& point : data.points) {
        double bestDist = std::numeric_limits<double>::max();
        int bestCluster = 0;

        for (int i = 0; i < k; ++i) {
            double d = distance(point, clusters[i].centroid);
            if (d < bestDist) {
                bestDist = d;
                bestCluster = i;
            }
        }

        cluster_points[bestCluster].push_back(point);
    }

    // Set the points dataset for each cluster
    for (int i = 0; i < k; ++i) {
        Cluster& c = clusters[i];
        c.points.n = cluster_points[i].size();
        c.points.points = std::move(cluster_points[i]);
        c.points.kdtree = nullptr;
        // Optionally recompute centroid here
    }

    return clusters;
}

// --- Main ---
int main() {
    std::string filename = "/Users/syedrizvi/Desktop/Life/Yale/Courses/04_Spring_2025/CPSC424/Final_Project/cpsc524FinalProject/example_data/sample2d.csv";
    int k = 5;

    Dataset data = readCSV(filename);
    std::cout << "Loaded " << data.n << " points.\n";

    std::cout << "Testing naive clustering algorithm.\n";
    auto clusters = naiveClusterData(data, k);
    std::cout << "Clustered into " << clusters.size() << " clusters.\n";

    for (int i = 0; i < clusters.size(); ++i) {
        std::cout << "Cluster " << i << ": " << clusters[i].points.n << " points\n";
    }

    return 0;
}
