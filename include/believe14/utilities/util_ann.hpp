#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <believe14/core/config.hpp>
#include <believe14/methods/methods_linear.hpp>

namespace believe14 {
namespace utilities {

struct L2AnnIndex {
  Eigen::MatrixXf data;
  std::vector<std::vector<int> > graph;
  int num_neighbors;
};

inline float l2_sqr_row_to_row(const Eigen::Ref<const Eigen::MatrixXf>& X, int i, int j) {
  return (X.row(i) - X.row(j)).squaredNorm();
}

inline float l2_sqr_row_to_vec(const Eigen::Ref<const Eigen::MatrixXf>& X,
                               int i,
                               const Eigen::Ref<const Eigen::VectorXf>& q) {
  return (X.row(i).transpose() - q).squaredNorm();
}

inline L2AnnIndex build_l2_ann_index(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                     Eigen::Index num_neighbors = 16,
                                     Eigen::Index num_candidate_pool = 64,
                                     std::uint32_t seed = 42) {
  if (X.rows() == 0 || X.cols() == 0) {
    throw std::invalid_argument("X must have at least one row and one column.");
  }
  if (num_neighbors <= 0) {
    throw std::invalid_argument("num_neighbors must be positive.");
  }
  if (num_candidate_pool <= 0) {
    throw std::invalid_argument("num_candidate_pool must be positive.");
  }

  const int n = static_cast<int>(X.rows());
  const int k = (n <= 1) ? 0 : std::min(static_cast<int>(num_neighbors), n - 1);
  const int pool = (n <= 1) ? 0 : std::min(static_cast<int>(num_candidate_pool), n - 1);

  L2AnnIndex index;
  index.data = X.cast<float>();
  index.graph.assign(static_cast<std::size_t>(n), std::vector<int>());
  index.num_neighbors = k;

  if (n <= 1) {
    return index;
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> uniform_id(0, n - 1);

  for (int i = 0; i < n; ++i) {
    std::unordered_set<int> candidates;

    // Ring links keep the graph from fragmenting on small sample sizes.
    candidates.insert((i + 1) % n);
    candidates.insert((i + n - 1) % n);

    while (static_cast<int>(candidates.size()) < pool + 2) {
      const int j = uniform_id(rng);
      if (j != i) {
        candidates.insert(j);
      }
    }

    std::vector<std::pair<float, int> > scored;
    scored.reserve(candidates.size());
    for (std::unordered_set<int>::const_iterator it = candidates.begin(); it != candidates.end(); ++it) {
      const int j = *it;
      scored.push_back(std::make_pair(l2_sqr_row_to_row(index.data, i, j), j));
    }

    std::sort(scored.begin(), scored.end());
    const int keep = std::min(k, static_cast<int>(scored.size()));
    index.graph[static_cast<std::size_t>(i)].reserve(static_cast<std::size_t>(keep));
    for (int t = 0; t < keep; ++t) {
      index.graph[static_cast<std::size_t>(i)].push_back(scored[static_cast<std::size_t>(t)].second);
    }
  }

  return index;
}

inline MethodResult l2_ann_search(const L2AnnIndex& index,
                                  const Eigen::Ref<const Eigen::VectorXd>& query,
                                  Eigen::Index k = 10,
                                  Eigen::Index ef_search = 64,
                                  Eigen::Index num_entry_points = 4,
                                  std::uint32_t seed = 42) {
  if (index.data.rows() == 0 || index.data.cols() == 0) {
    throw std::invalid_argument("Index is empty.");
  }
  if (query.size() != index.data.cols()) {
    throw std::invalid_argument("query dimension does not match index dimension.");
  }
  if (k <= 0) {
    throw std::invalid_argument("k must be positive.");
  }
  if (ef_search <= 0) {
    throw std::invalid_argument("ef_search must be positive.");
  }
  if (num_entry_points <= 0) {
    throw std::invalid_argument("num_entry_points must be positive.");
  }

  const int n = index.data.rows();
  const int kk = std::min(static_cast<int>(k), n);
  const int ef = std::max(kk, static_cast<int>(ef_search));
  const int entries = std::min(static_cast<int>(num_entry_points), n);
  const Eigen::VectorXf q = query.cast<float>();

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> uniform_id(0, n - 1);

  std::vector<unsigned char> visited(static_cast<std::size_t>(n), 0);
  std::priority_queue<std::pair<float, int> > result_heap;
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >, std::greater<std::pair<float, int> > >
      candidate_heap;

  for (int i = 0; i < entries; ++i) {
    const int ep = uniform_id(rng);
    if (visited[static_cast<std::size_t>(ep)]) {
      continue;
    }
    visited[static_cast<std::size_t>(ep)] = 1;
    const float d = l2_sqr_row_to_vec(index.data, ep, q);
    candidate_heap.push(std::make_pair(d, ep));
    result_heap.push(std::make_pair(d, ep));
  }

  while (!candidate_heap.empty()) {
    const std::pair<float, int> current = candidate_heap.top();
    candidate_heap.pop();

    const float worst = result_heap.empty() ? std::numeric_limits<float>::infinity() : result_heap.top().first;
    if (static_cast<int>(result_heap.size()) >= ef && current.first > worst) {
      break;
    }

    const std::vector<int>& nbrs = index.graph[static_cast<std::size_t>(current.second)];
    for (std::size_t t = 0; t < nbrs.size(); ++t) {
      const int nb = nbrs[t];
      if (visited[static_cast<std::size_t>(nb)]) {
        continue;
      }
      visited[static_cast<std::size_t>(nb)] = 1;
      const float d = l2_sqr_row_to_vec(index.data, nb, q);

      if (static_cast<int>(result_heap.size()) < ef || d < result_heap.top().first) {
        candidate_heap.push(std::make_pair(d, nb));
        result_heap.push(std::make_pair(d, nb));
        if (static_cast<int>(result_heap.size()) > ef) {
          result_heap.pop();
        }
      }
    }
  }

  std::vector<std::pair<float, int> > out;
  out.reserve(result_heap.size());
  while (!result_heap.empty()) {
    out.push_back(result_heap.top());
    result_heap.pop();
  }
  std::sort(out.begin(), out.end());
  if (static_cast<int>(out.size()) > kk) {
    out.resize(static_cast<std::size_t>(kk));
  }

  Eigen::MatrixXd indices(static_cast<Eigen::Index>(out.size()), 1);
  Eigen::MatrixXd distances(static_cast<Eigen::Index>(out.size()), 1);
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(out.size()); ++i) {
    indices(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i)].second);
    distances(i, 0) = static_cast<double>(out[static_cast<std::size_t>(i)].first);
  }

  MethodResult result;
  result.objects["indices"] = indices;
  result.objects["distances"] = distances;
  return result;
}

}  // namespace utilities
}  // namespace believe14
