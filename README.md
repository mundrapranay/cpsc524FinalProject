# Project Plan: Parallel Clustering and Nearest Neighbor Search Using KD-Trees/Ball Trees

## Overview

This project aims to develop a parallel system for clustering and performing efficient nearest neighbor (NN) searches on high-dimensional data using KD-Trees or Ball Trees. The system will be fully implemented in C++ and designed for shared-memory parallelism.

Google Doc: https://docs.google.com/document/d/1OMdeZtnvKImjsmKK4mV8en7d_ntYKo8lF7ty2kFxyRQ/edit?pli=1&tab=t.xgjg7v2dpjin
---

## Objectives

1. Implement a clustering algorithm that divides a high-dimensional dataset into `k` clusters.
2. Build a KD-tree (or Ball tree) for each cluster in parallel.
3. Use a **parallel nearest neighbor search** strategy across these trees with a **shared concurrent max-heap** to manage candidates.
4. Optimize the system using **priority-based tree selection**, memory locality improvements, and smart task scheduling.

---

## System Architecture

### Input

- A dataset `D` consisting of `N` high-dimensional points.
- An integer `k` specifying the number of clusters.
- A query point `q`.

### Output

- `k` clusters (partitions of the dataset).
- One KD/Ball tree per cluster.
- A set of nearest neighbors for a given query.

---

## Data Types and Structures

### Basic Type Definitions (in C++)

```cpp
using Point = std::vector<double>;
using Dataset = std::vector<Point>;
```

### Cluster

```cpp
struct Cluster {
    Dataset points;
    Point centroid; // Optional, useful for optimization
};
```

### KD-Tree Node

```cpp
struct KDNode {
    Point point;
    int splitDim;
    double splitValue;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;
};
```

### KD-Tree

```cpp
class KDTree {
public:
    std::unique_ptr<KDNode> root;
    void build(const Dataset& points);
    std::vector<Point> query(const Point& target, int k) const;
};
```

### Concurrent Max Heap

```cpp
template<typename T, typename Compare = std::less<T>>
class ConcurrentPriorityQueue {
private:
    std::priority_queue<T, std::vector<T>, Compare> heap;
    mutable std::mutex mtx;
public:
    void push(const T& value);
    bool tryPop(T& value);
    bool empty() const;
};
```

---

## Module Details

### 1. Clustering Module

- **Goal:** Partition the dataset into `k` clusters.
- **Algorithm:** Use k-means or any suitable clustering algorithm.
- **Function Signature:**
  ```cpp
  std::vector<Cluster> clusterData(const Dataset& data, int k);
  ```

---

### 2. Tree Construction Module

- **Goal:** Build a KD-tree (or Ball tree) for each cluster.
- **Parallelism:** One tree per cluster can be built in parallel.
- **Function Signature:**
  ```cpp
  KDTree buildKDTree(const Cluster& cluster);
  ```

---

### 3. Nearest Neighbor Query Module

- **Goal:** Query all trees in parallel and collect results in a shared concurrent heap.
- **Strategy:**
  - Use a concurrent max-heap to collect nearest candidates from all trees.
  - Prioritize trees based on root-to-query distance.
- **Function Signature:**
  ```cpp
  std::vector<Point> parallelNearestNeighborQuery(const std::vector<KDTree>& trees, const Point& query, int top_p);
  ```

---

## Optimization Strategies

### 1. Tree Prioritization

- **Problem:** Not all trees are equally useful for a given query.
- **Solution:** Rank trees by the Euclidean distance from the tree's root node to the query point.
- **Implementation:**
  ```cpp
  double rootDistance = distance(query, tree.root->point);
  ```

### 2. Top-P Tree Selection

- Only search the top `p` trees closest to the query point.
- Skip trees where root distance is above a threshold.

### 3. Efficient Heap Updates

- Use fine-grained locks or lock-free designs.
- Partition the heap by thread and merge results if needed.

### 4. Memory and Cache Optimization

- Use contiguous memory (`std::vector`) for better cache performance.
- Consider node pooling to reduce heap fragmentation.

### 5. Task-Based Parallelism

- Use C++17's `std::execution::par` or thread pools to handle:
  - Tree construction
  - Tree querying

---

## Development Roadmap

### Phase 1: Prototyping

- [ ] Implement basic clustering algorithm.
- [ ] Create KDTree class with build and sequential query.
- [ ] Write utilities for Euclidean distance and point operations.

### Phase 2: Parallelization

- [ ] Add parallel construction for trees (1 per cluster).
- [ ] Implement a thread-safe concurrent max-heap.
- [ ] Query each tree in a separate thread using the shared heap.

### Phase 3: Optimization

- [ ] Add root-to-query distance ranking.
- [ ] Implement top-p tree filtering.
- [ ] Profile memory access and improve layout.

### Phase 4: Integration and Testing

- [ ] Integrate all modules (clustering → trees → query).
- [ ] Test with synthetic and real-world datasets.
- [ ] Add performance benchmarks and logs.
