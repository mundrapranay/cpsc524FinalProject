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
#include <queue>

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

// Compute squared Euclidean distance
double squaredDistance(const Point& a, const Point& b) {
    double dist = 0.0;
    for (int i = 0; i < a.dimension; ++i) {
        double diff = a.coordinates[i] - b.coordinates[i];
        dist += diff * diff;
    }
    return dist;
}

// Recursive kNN helper
void kNearestNeighborsRecursive(KDNode* node, const Dataset& data,
                                const Point& query, int k,
                                std::priority_queue<std::pair<double, const Point*>>& maxHeap) {
    if (!node) return;

    const Point& currentPoint = data.points[node->pointIndex];
    double dist = squaredDistance(query, currentPoint);

    if (maxHeap.size() < static_cast<size_t>(k)) {
        maxHeap.emplace(dist, &currentPoint);
    } else if (dist < maxHeap.top().first) {
        maxHeap.pop();
        maxHeap.emplace(dist, &currentPoint);
    }

    if (node->axis == -1) return;

    double queryVal = query.coordinates[node->axis];
    double splitVal = node->splitValue;

    KDNode* first = (queryVal <= splitVal) ? node->left : node->right;
    KDNode* second = (queryVal <= splitVal) ? node->right : node->left;

    // Recurse into the closer subtree first
    kNearestNeighborsRecursive(first, data, query, k, maxHeap);

    // Recurse into the other subtree if needed
    double axisDist = (queryVal - splitVal) * (queryVal - splitVal);
    if (maxHeap.size() < static_cast<size_t>(k) || axisDist < maxHeap.top().first) {
        kNearestNeighborsRecursive(second, data, query, k, maxHeap);
    }
}

// Public interface
std::vector<const Point*> kNearestNeighbors(KDNode* root, const Dataset& data, const Point& query, int k) {
    std::priority_queue<std::pair<double, const Point*>> maxHeap;
    kNearestNeighborsRecursive(root, data, query, k, maxHeap);

    std::vector<const Point*> result;
    while (!maxHeap.empty()) {
        result.push_back(maxHeap.top().second);
        maxHeap.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

// Generate same query points to use for benchmarking across methods
std::vector<int> generateQueryIndices(const Dataset& data, int Q) {
    std::vector<int> queryIndices;
    if (data.n == 0 || Q <= 0) return queryIndices;
    
    queryIndices.reserve(Q);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.n - 1);
    
    for (int q = 0; q < Q; ++q) {
        queryIndices.push_back(dis(gen));
    }
    
    return queryIndices;
}

// Perform single query here using one (i.e. global) kD tree 
std::pair<double, std::vector<const Point*>> performSingleGlobalQuery(
    const Dataset& data, const Point& queryPoint, int k) {
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<const Point*> nearest = kNearestNeighbors(data.kdtree, data, queryPoint, k);
    auto end = std::chrono::high_resolution_clock::now();
    
    double queryTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return {queryTime, nearest};
}

// Perform single query using cluster kD trees here
std::pair<double, std::vector<const Point*>> performSingleClusterQuery(
    const parlay::sequence<Cluster>& parlayData, const Point& queryPoint, int k, int p) {
    
    auto start = std::chrono::high_resolution_clock::now();
    auto nearest = parallelNearestNeighborQuery(parlayData, queryPoint, k, p);
    auto end = std::chrono::high_resolution_clock::now();
    
    double queryTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return {queryTime, nearest};
}

// Perform Q queries using global kD tree with predefined query points
QueryStats performGlobalQueries(const Dataset& data, int k, const std::vector<int>& queryIndices) {
    QueryStats stats;
    if (!data.kdtree || data.n == 0 || queryIndices.empty()) return stats;
    
    int Q = queryIndices.size();
    stats.queryTimes.reserve(Q);
    stats.nearestDistances.reserve(Q);

    // Parallel search 
    struct QueryResult {
        double queryTime;
        double nearestDistance;
        size_t numNeighbors;
    };

    auto results = parlay::tabulate(Q, [&](size_t q) {
        const Point& queryPoint = data.points[queryIndices[q]];
        auto [queryTime, nearest] = performSingleGlobalQuery(data, queryPoint, k);
        
        double nearestDist = std::numeric_limits<double>::infinity();
        if (!nearest.empty()) {
            nearestDist = squaredDistance(queryPoint, *nearest[0]);
        }
        
        return QueryResult{queryTime, nearestDist, nearest.size()};
    });

    for (size_t q= 0; q < results.size(); ++q) {
        stats.queryTimes.push_back(results[q].queryTime);
        stats.nearestDistances.push_back(results[q].nearestDistance);
        
        // some logging for early queries
        if (q < 3) {
            std::cout << "Query " << q + 1 << " (Global): Found " << results[q].numNeighbors 
                      << " neighbors in " << results[q].queryTime << " µs\n";
            std::cout << "  Nearest neighbor distance: " << results[q].nearestDistance << std::endl;
        }
    }

    stats.computeStats();
    return stats;
}

// Perform Q queries using clustered kD trees with predefined query points
QueryStats performClusterQueries(const std::vector<Cluster>& clusters, const Dataset& data, 
                               int k, int p, const std::vector<int>& queryIndices) {
    QueryStats stats;
    if (clusters.empty() || data.n == 0 || queryIndices.empty()) return stats;
    
    int Q = queryIndices.size();
    stats.queryTimes.reserve(Q);
    stats.nearestDistances.reserve(Q);
    
    parlay::sequence<Cluster> parlayData(clusters.begin(), clusters.end());
    
    struct QueryResult {
        double queryTime;
        double nearestDistance;
        size_t numNeighbors;
    };

    auto results = parlay::tabulate(Q, [&](size_t q) {
        const Point& queryPoint = data.points[queryIndices[q]];
        auto [queryTime, nearest] = performSingleClusterQuery(parlayData, queryPoint, k, p);
        
        double nearestDist = std::numeric_limits<double>::infinity();
        if (!nearest.empty()) {
            nearestDist = squaredDistance(queryPoint, *nearest[0]);
        }
        
        return QueryResult{queryTime, nearestDist, nearest.size()};
    });

    for (size_t q = 0; q < results.size(); ++q) {
        stats.queryTimes.push_back(results[q].queryTime);
        stats.nearestDistances.push_back(results[q].nearestDistance);
        
        // Optionally log first few queries
        if (q < 3) {
            std::cout << "Query " << q + 1 << " (Cluster): Found " << results[q].numNeighbors 
                      << " neighbors in " << results[q].queryTime << " µs\n";
            std::cout << "  Nearest neighbor distance: " << results[q].nearestDistance << std::endl;
        }
    }
    
    stats.computeStats();
    return stats;
}

// Print out compiled stats
void printQueryStats(const QueryStats& stats, const std::string& method) {
    std::cout << "Summary for " << method << " (" << stats.queryTimes.size() << " queries):\n";
    std::cout << "  Total time: " << stats.totalTime << " µs\n";
    std::cout << "  Min query time: " << stats.minTime << " µs\n";
    std::cout << "  Max query time: " << stats.maxTime << " µs\n";
    std::cout << "  Avg query time: " << stats.avgTime << " µs\n";
}

int main(int argc, char** argv) {
    // Default parameters
    std::string inputFilename;  // = "example_data/rnaseq_sample50d_1000_cells.csv";
    std::string outputFilename;  // = "example_data/rnaseq_sample50d_1000_cells_clustered.csv";
    double eps = 0.2; // default
    int minNeighbors = 10; // minimum number of points to be considered a core point in DBSCAN
    bool buildTrees = true; // Whether to build kD-trees
    int k = 5; // Number of global neighbors to find / query point
    int p = 2; // Number of cluster trees to consider (via distance to root)
    int Q = 1; // Number of queries to perform

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
        } else if (arg == "--queries" || arg == "-q") {
            if (i + 1 < argc) Q = std::stoi(argv[++i]);
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
                    << "  -q, --queries NUM         Number of queries to perform (default: " << Q << ")\n"
                    << "  --no-trees                Skip building kD-trees\n"
                    << "  -h, --help                Show this help message\n";
            return 0;
        }
    }

    if (inputFilename.empty()) {
        std::cerr << "Error: Input file is required\n";
        std::cerr << "Use --help for usage information\n";
        return 1;
    }

    if (outputFilename.empty()) {
        // Create default output filename based on input
        size_t lastDot = inputFilename.find_last_of(".");
        size_t lastSlash = inputFilename.find_last_of("/\\");
        
        // Convert numeric values to strings
        std::ostringstream epsStr, minPtsStr;
        epsStr << eps;
        minPtsStr << minNeighbors;
        
        if (lastDot != std::string::npos && (lastSlash == std::string::npos || lastDot > lastSlash)) {
            outputFilename = inputFilename.substr(0, lastDot) + "_eps_" + epsStr.str() + 
                            "_minpts_" + minPtsStr.str() + "_clustered" + inputFilename.substr(lastDot);
        } else {
            outputFilename = inputFilename + "_clustered.csv";
        }
    }
    
    // Load dataset
    auto start = std::chrono::high_resolution_clock::now();
    Dataset data = readCSV(inputFilename);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Loaded " << data.n << " points in " << duration << " ms.\n" << std::endl;
    
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



    // CHECK BELOW -- INTEGRATED QUERYING TOGETHER
    // if (buildTrees && data.n > 0) {
    //     // Use a random point for the query
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_int_distribution<> dis(0, data.n - 1);
    //     int randomIdx = dis(gen);
        
    //     const Point& queryPoint = data.points[randomIdx];

    //     // std::cout << "\nQuery point coordinates (global kD-tree): ";
    //     // for (int d = 0; d < queryPoint.dimension; ++d) {
    //     //     std::cout << queryPoint.coordinates[d] << " ";
    //     // }
    //     // std::cout << std::endl;

    //     // Run nearest neighbor query on global tree
    //     std::cout << "\nPerforming " << k << "-nearest neighbor query using global kD-tree...\n";

    //     start = std::chrono::high_resolution_clock::now();
    //     std::vector<const Point*> globalNearest = kNearestNeighbors(data.kdtree, data, queryPoint, k);

    //     end = std::chrono::high_resolution_clock::now();
        
    //     duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //     std::cout << "Global KD-Tree query found " << globalNearest.size() 
    //             << " nearest neighbors in " << duration << " µs.\n";

    //     std::cout << "Distances to global nearest neighbors:\n";
    //     for (size_t i = 0; i < globalNearest.size(); ++i) {
    //         double dist = 0.0;
    //         for (int d = 0; d < queryPoint.dimension; ++d) {
    //             double diff = queryPoint.coordinates[d] - globalNearest[i]->coordinates[d];
    //             dist += diff * diff;
    //         }
    //         dist = std::sqrt(dist);
    //         std::cout << "  Neighbor " << i + 1 << ": distance=" << dist << std::endl;  // ", coords=(";
    //         // for (int d = 0; d < globalNearest[i]->dimension; ++d) {
    //         //     std::cout << globalNearest[i]->coordinates[d];
    //         //     if (d < globalNearest[i]->dimension - 1) std::cout << ", ";
    //         // }
    //         // std::cout << ")\n";
    //     }
    // }



    // Our Algorithm Phase 1: Run DBSCAN clustering
    std::cout << "--------------------------------------" << std::endl;
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

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Testing Cluster KD Tree Algorithm" << std::endl;

    // Update cluster centroids
    updateClusterCentroids(clusters);
    


    // Our Algorithm Phase 2: Build kD-trees for each cluster
    if (buildTrees) {
        start = std::chrono::high_resolution_clock::now();
        buildClusterKDTrees(clusters);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Built cluster kD-trees in " << duration << " ms.\n";
    }
    
    // Write clustered points to CSV
    // outputFilename --> include both eps and min_pts
    writeClusteredPointsToCSV(outputFilename, data, labels);
    std::cout << "Wrote clustered points to " << outputFilename << ".\n";


    
    // Our Algorithm Phase 3: Example of nearest-neighbor query
    // std::cout << "buildTrees:" << buildTrees << std::endl;
    // std::cout << "clusters.empty():" << clusters.empty() << std::endl;
    // std::cout << "data.n:" << data.n << std::endl;
    if (buildTrees && !clusters.empty() && data.n > 0) {
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Performing " << Q << " random queries with k=" << k << " nearest neighbors" << std::endl;
        // Use a random point for the query
        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::uniform_int_distribution<> dis(0, data.n - 1);
        // int randomIdx = dis(gen);
        
        // const Point& queryPoint = data.points[randomIdx];

        std::vector<int> queryIndices = generateQueryIndices(data, Q);
    
        // std::cout << "Query point coordinates: ";
        // for (int d = 0; d < queryPoint.dimension; ++d) {
        //     std::cout << queryPoint.coordinates[d] << " ";
        // }
        // std::cout << std::endl;

        // First perform queries using global kD tree
        std::cout << "\nRunning global kD-tree queries...\n";
        QueryStats globalStats = performGlobalQueries(data, k, queryIndices);
        printQueryStats(globalStats, "Global kD-tree");
        
        // Then perform queries using clustered kD-trees
        std::cout << "\nRunning clustered kD-tree queries with p=" << p << " clusters...\n";
        QueryStats clusterStats = performClusterQueries(clusters, data, k, p, queryIndices);
        printQueryStats(clusterStats, "Cluster kD-trees");

        // start = std::chrono::high_resolution_clock::now();
        // auto nearestPoints = parallelNearestNeighborQuery(
        //     parlay::sequence<Cluster>(clusters.begin(), clusters.end()),
        //     queryPoint, k, p);
        // end = std::chrono::high_resolution_clock::now();
        
        // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // std::cout << "Found " << nearestPoints.size() << " nearest neighbors in " 
        //           << duration << " µs.\n";
        
        // // Print distances to nearest neighbors
        // std::cout << "Distances to nearest neighbors:\n";
        // for (size_t i = 0; i < nearestPoints.size(); ++i) {
        //     double dist = 0.0;
        //     for (int d = 0; d < queryPoint.dimension; ++d) {
        //         double diff = queryPoint.coordinates[d] - nearestPoints[i]->coordinates[d];
        //         dist += diff * diff;
        //     }
        //     dist = std::sqrt(dist);
        //     std::cout << "  Neighbor " << i + 1 << ": distance=" << dist << std::endl;  // ", coords=(";
            // for (int d = 0; d < nearestPoints[i]->dimension; ++d) {
            //     std::cout << nearestPoints[i]->coordinates[d];
            //     if (d < nearestPoints[i]->dimension - 1) std::cout << ", ";
            // }
            // std::cout << ")" << std::endl;
        // }

        // Compare performance
        if (!globalStats.queryTimes.empty() && !clusterStats.queryTimes.empty()) {
            double speedup = globalStats.avgTime / clusterStats.avgTime;
            std::cout << "\nPerformance comparison:" << std::endl;
            std::cout << "  Speedup (cluster/global): " << speedup << "x" << std::endl;
            
            // Calculate result quality metrics
            double avgDistanceDiff = 0.0;
            int matchCount = 0;
            for (size_t i = 0; i < std::min(globalStats.nearestDistances.size(), clusterStats.nearestDistances.size()); ++i) {
                if (std::isfinite(globalStats.nearestDistances[i]) && std::isfinite(clusterStats.nearestDistances[i])) {
                    avgDistanceDiff += std::abs(globalStats.nearestDistances[i] - clusterStats.nearestDistances[i]);
                    
                    // Consider results matching if within 0.1% relative difference
                    double relDiff = std::abs(globalStats.nearestDistances[i] - clusterStats.nearestDistances[i]) / 
                                    std::max(globalStats.nearestDistances[i], 1e-10);
                    if (relDiff < 0.001) matchCount++;
                }
            }
            
            if (matchCount > 0) {
                avgDistanceDiff /= matchCount;
                std::cout << "  Quality metrics:" << std::endl;
                std::cout << "    Exact matches: " << matchCount << " / " 
                          << std::min(globalStats.nearestDistances.size(), clusterStats.nearestDistances.size())
                          << " (" << (100.0 * matchCount / std::min(globalStats.nearestDistances.size(), clusterStats.nearestDistances.size())) 
                          << "%)" << std::endl;
                std::cout << "    Average distance difference: " << avgDistanceDiff << std::endl;
            }
            
            // Write performance metrics to a file
            std::string resultsDir = "./results";
            std::string baseFilename;
            size_t lastSlash = inputFilename.find_last_of("/\\");
            if (lastSlash != std::string::npos) {
                baseFilename = inputFilename.substr(lastSlash + 1);
            } else {
                baseFilename = inputFilename;
            }
            
            // Create metrics filename
            std::string metricsFilename = resultsDir + "/query_performance_" + baseFilename;
            std::ofstream metricsFile(metricsFilename);
            
            metricsFile << "Query,GlobalTime,ClusterTime,GlobalNearestDist,ClusterNearestDist\n";
            size_t numQueries = std::min(globalStats.queryTimes.size(), clusterStats.queryTimes.size());
            
            for (size_t i = 0; i < numQueries; ++i) {
                metricsFile << i + 1 << "," 
                           << globalStats.queryTimes[i] << "," 
                           << clusterStats.queryTimes[i] << ","
                           << globalStats.nearestDistances[i] << ","
                           << clusterStats.nearestDistances[i] << "\n";
            }
            
            metricsFile.close();
            std::cout << "Wrote performance metrics to " << metricsFilename << std::endl;
        }
    }
    
    // Clean up kD-tree memory
    if (buildTrees) {
        deleteKDTree(data.kdtree);
        for (auto& cluster : clusters) {
            deleteKDTree(cluster.points.kdtree);
        }
    }
    std::cout << "Script finished." << std::endl<< "------------------" << std::endl;
    
    return 0;
}