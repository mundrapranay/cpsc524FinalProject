#ifndef DISTANCE_METRICS_H
#define DISTANCE_METRICS_H

#include <cmath>

inline double cosineDistance(const Point& a, const Point& b) {
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (int i = 0; i < a.dimension; ++i) {
        double x = a.coordinates[i];
        double y = b.coordinates[i];
        dot += x * y;
        normA += x * x;
        normB += y * y;
    }
    if (normA == 0.0 || normB == 0.0) return 1.0;
    return 1.0 - (dot / (std::sqrt(normA) * std::sqrt(normB)));
}

#endif