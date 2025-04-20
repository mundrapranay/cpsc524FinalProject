CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -I./include
LDFLAGS = -pthread

SRCS = main.cpp parallel_dbscan.cpp parallel_search.cpp parallel_kdtree.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = dbscan_clustering

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Test program for kD-tree only
# test_kdtree: test_kdtree.o parallel_kdtree.o
# 	$(CXX) $(CXXFLAGS) test_kdtree.o parallel_kdtree.o -o test_kdtree $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET) test_kdtree test_kdtree.o

.PHONY: all clean