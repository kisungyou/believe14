#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include <believe14/core/config.hpp>

#ifdef BELIEVE14_USE_SYSTEM_EIGEN
  #include <Eigen/Sparse>
#else
  #include <third_party/Eigen/Sparse>
#endif

namespace believe14 {
namespace utilities {

inline Eigen::MatrixXd dijkstra_shortest_paths(const Eigen::SparseMatrix<double>& A) {
  if (A.rows() != A.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }

  const Eigen::Index n = A.rows();
  Eigen::MatrixXd D = Eigen::MatrixXd::Constant(n, n, std::numeric_limits<double>::infinity());
  for (Eigen::Index i = 0; i < n; ++i) {
    D(i, i) = 0.0;
  }

  std::vector<std::vector<std::pair<int, double> > > adj(static_cast<std::size_t>(n));
  for (int col = 0; col < A.outerSize(); ++col) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
      const int i = static_cast<int>(it.row());
      const int j = static_cast<int>(it.col());
      const double w = it.value();
      if (!std::isfinite(w) || w < 0.0) {
        throw std::invalid_argument("Adjacency contains invalid edge weight; Dijkstra requires finite nonnegative weights.");
      }
      if (w == 0.0 || i == j) {
        continue;
      }
      adj[static_cast<std::size_t>(i)].push_back(std::make_pair(j, w));
    }
  }

  typedef std::pair<double, int> NodeDist;
  for (int src = 0; src < static_cast<int>(n); ++src) {
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist> > pq;
    std::vector<unsigned char> visited(static_cast<std::size_t>(n), 0);

    D(src, src) = 0.0;
    pq.push(std::make_pair(0.0, src));

    while (!pq.empty()) {
      const NodeDist cur = pq.top();
      pq.pop();
      const int u = cur.second;
      if (visited[static_cast<std::size_t>(u)]) {
        continue;
      }
      visited[static_cast<std::size_t>(u)] = 1;

      const std::vector<std::pair<int, double> >& nbrs = adj[static_cast<std::size_t>(u)];
      for (std::size_t t = 0; t < nbrs.size(); ++t) {
        const int v = nbrs[t].first;
        const double w = nbrs[t].second;
        const double alt = D(src, u) + w;
        if (alt < D(src, v)) {
          D(src, v) = alt;
          pq.push(std::make_pair(alt, v));
        }
      }
    }
  }

  return D;
}

}  // namespace utilities
}  // namespace believe14
