#include "src/datastructures.h"
#include "src/parallel_kdtree.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

// Function declarations from parallel_dbscan.cpp
Dataset readCSV(const std::string& filename);

std::pair<std::vector<Cluster>, parlay::sequence<int>> parallelDBSCAN(
    const Dataset& data, double eps, int minPts);

void writeClusteredPointsToCSV(const std::string& filename,
                              const Dataset& data,
                              const parlay::sequence<int>& labels);

// Function declarations from parallel_search.cpp
std::vector<const Point*> parallelNearestNeighborQuery(
    const parlay::sequence<Cluster> &clusters,
    const Point &query,
    size_t K,
    size_t P);

// Function to compute centroid for a dataset
Point computeCentroid(const Dataset& dataset) {
    Point centroid;
    if (dataset.n == 0) return centroid;
    
    centroid.dimension = dataset.points[0].dimension;
    centroid.coordinates.resize(centroid.dimension, 0.0);
    
    for (const auto& point : dataset.points) {
        for (int d = 0; d < centroid.dimension; ++d) {
            centroid.coordinates[d] += point.coordinates[d];
        }
    }
    
    for (int d = 0; d < centroid.dimension; ++d) {
        centroid.coordinates[d] /= dataset.n;
    }
    
    return centroid;
}

// Function to update cluster centroids
void updateClusterCentroids(std::vector<Cluster>& clusters) {
    parlay::parallel_for(0, clusters.size(), [&](size_t i) {
        clusters[i].centroid = computeCentroid(clusters[i].points);
    });
}

int main(int argc, char** argv) {
    // Default parameters
    std::string inputFilename = "example_data/rnaseq_sample50d_1000_cells.csv";
    std::string outputFilename = "example_data/rnaseq_sample50d_1000_cells_clustered.csv";
    double eps = 0.2;
    int minNeighbors = 10;
    bool buildTrees = true; // Whether to build kD-trees
    int k = 5;
    int p = 2;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) inputFilename = argv[++i];
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) outputFilename = argv[++i];
        } else if (arg == "--eps" || arg == "-e") {
            if (i + 1 < argc) eps = std::stod(argv[++i]);
        } else if (arg == "--min-pts" || arg == "-m") {
            if (i + 1 < argc) minNeighbors = std::stoi(argv[++i]);
        } else if (arg == "--k-neighbors" || arg == "-k") {
            if (i + 1 < argc) k = std::stoi(argv[++i]);
        } else if (arg == "--clusters-search" || arg == "-p") {
            if (i + 1 < argc) p = std::stoi(argv[++i]);
        } else if (arg == "--no-trees") {
            buildTrees = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                    << "Options:\n"
                    << "  -i, --input FILE          Input CSV file (default: " << inputFilename << ")\n"
                    << "  -o, --output FILE         Output CSV file (default: " << outputFilename << ")\n"
                    << "  -e, --eps VALUE           DBSCAN epsilon value (default: " << eps << ")\n"
                    << "  -m, --min-pts VALUE       DBSCAN minimum points (default: " << minNeighbors << ")\n"
                    << "  -k, --k-neighbors VALUE   Number of nearest neighbors to find (default: " << k << ")\n"
                    << "  -p, --clusters-search NUM Number of closest clusters to search (default: " << p << ")\n"
                    << "  --no-trees                Skip building kD-trees\n"
                    << "  --no-query                Skip nearest neighbor query\n"
                    << "  -h, --help                Show this help message\n";
            return 0;
        }
    }
    
    // Load dataset
    auto start = std::chrono::high_resolution_clock::now();
    Dataset data = readCSV(inputFilename);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Loaded " << data.n << " points in " << duration << " ms.\n";
    
    // Compute global centroid
    data.centroid = computeCentroid(data);
    
    // Build global kD-tree
    if (buildTrees) {
        start = std::chrono::high_resolution_clock::now();
        data.kdtree = buildKDTreeParallel(data);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Built global kD-tree in " << duration << " ms.\n";
    }
    
    // Run DBSCAN clustering
    std::cout << "Running parallel DBSCAN with eps = " << eps
              << " and minNeighbors = " << minNeighbors << ".\n";
    
    start = std::chrono::high_resolution_clock::now();
    auto [clusters, labels] = parallelDBSCAN(data, eps, minNeighbors);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Clustered into " << clusters.size() << " clusters in " << duration << " ms.\n";
    
    // Print cluster statistics
    for (size_t i = 0; i < clusters.size(); ++i) {
        std::cout << "Cluster " << i << ": " << clusters[i].points.n << " points\n";
    }
    
    // Update cluster centroids
    updateClusterCentroids(clusters);
    
    // Build kD-trees for each cluster
    if (buildTrees) {
        start = std::chrono::high_resolution_clock::now();
        buildClusterKDTrees(clusters);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Built cluster kD-trees in " << duration << " ms.\n";
    }
    
    // Write clustered points to CSV
    writeClusteredPointsToCSV(outputFilename, data, labels);
    std::cout << "Wrote clustered points to " << outputFilename << ".\n";
    
    // Example of nearest-neighbor query
    if (buildTrees && !clusters.empty() && data.n > 0) {
        // Use a random point for the query
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.n - 1);
        int randomIdx = dis(gen);
        
        const Point& queryPoint = data.points[randomIdx];

        std::cout << "Query point coordinates: ";
        for (int d = 0; d < queryPoint.dimension; ++d) {
            std::cout << queryPoint.coordinates[d] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\nPerforming " << k << "-nearest neighbor query using " 
                << p << " closest clusters...\n";
        
        start = std::chrono::high_resolution_clock::now();
        auto nearestPoints = parallelNearestNeighborQuery(
            parlay::sequence<Cluster>(clusters.begin(), clusters.end()),
            queryPoint, k, p);
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout << "Found " << nearestPoints.size() << " nearest neighbors in " 
                  << duration << " µs.\n";
        
        // Print distances to nearest neighbors
        std::cout << "Distances to nearest neighbors:\n";
        for (size_t i = 0; i < nearestPoints.size(); ++i) {
            double dist = 0.0;
            for (int d = 0; d < queryPoint.dimension; ++d) {
                double diff = queryPoint.coordinates[d] - nearestPoints[i]->coordinates[d];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            std::cout << "  Neighbor " << i + 1 << ": distance=" << dist << ", coords=(";
            for (int d = 0; d < nearestPoints[i]->dimension; ++d) {
                std::cout << nearestPoints[i]->coordinates[d];
                if (d < nearestPoints[i]->dimension - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
        }
    }
    
    // Clean up kD-tree memory
    if (buildTrees) {
        deleteKDTree(data.kdtree);
        for (auto& cluster : clusters) {
            deleteKDTree(cluster.points.kdtree);
        }
    }
    
    return 0;
}