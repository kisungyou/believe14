#pragma once

#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <believe14/core/config.hpp>

#ifdef BELIEVE14_USE_SYSTEM_EIGEN
  #include <Eigen/Sparse>
#else
  #include <third_party/Eigen/Sparse>
#endif

namespace believe14 {
namespace utilities {

inline Eigen::SparseMatrix<double> convert_ann2adj(const Eigen::Ref<const Eigen::MatrixXd>& indices) {
  if (indices.rows() == 0) {
    throw std::invalid_argument("indices must have at least one row.");
  }

  const Eigen::Index n = indices.rows();
  std::unordered_set<unsigned long long> edges;
  edges.reserve(static_cast<std::size_t>(indices.rows() * indices.cols() * 2));

  const auto encode = [](Eigen::Index i, Eigen::Index j) -> unsigned long long {
    return (static_cast<unsigned long long>(i) << 32) | static_cast<unsigned long long>(j);
  };

  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index t = 0; t < indices.cols(); ++t) {
      const double raw = indices(i, t);
      if (!std::isfinite(raw)) {
        continue;
      }

      const Eigen::Index j = static_cast<Eigen::Index>(std::llround(raw));
      if (j < 0 || j >= n) {
        continue;
      }
      if (i == j) {
        continue;
      }

      // OR symmetry rule: if i->j exists, ensure both (i,j) and (j,i).
      edges.insert(encode(i, j));
      edges.insert(encode(j, i));
    }
  }

  std::vector<Eigen::Triplet<double> > triplets;
  triplets.reserve(edges.size());
  for (std::unordered_set<unsigned long long>::const_iterator it = edges.begin(); it != edges.end(); ++it) {
    const unsigned long long code = *it;
    const Eigen::Index i = static_cast<Eigen::Index>(code >> 32);
    const Eigen::Index j = static_cast<Eigen::Index>(code & 0xFFFFFFFFull);
    triplets.push_back(Eigen::Triplet<double>(i, j, 1.0));
  }

  Eigen::SparseMatrix<double> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

}  // namespace utilities
}  // namespace believe14
