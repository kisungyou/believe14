#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <believe14/core/config.hpp>

namespace believe14 {

struct MethodResult {
  std::unordered_map<std::string, Eigen::MatrixXd> objects;
};

inline MethodResult pca(const Eigen::Ref<const Eigen::MatrixXd>& X, Eigen::Index ndim) {
  if (X.rows() == 0 || X.cols() == 0) {
    throw std::invalid_argument("X must have at least one row and one column.");
  }
  if (ndim <= 0) {
    throw std::invalid_argument("ndim must be a positive integer.");
  }

  const Eigen::Index max_dim = std::min(X.rows(), X.cols());
  if (ndim > max_dim) {
    throw std::invalid_argument("ndim cannot exceed min(n, p).");
  }

  const Eigen::RowVectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd centered = X.rowwise() - mean;

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

  MethodResult out;
  out.objects["projection"] = svd.matrixV().leftCols(ndim);
  out.objects["embedding"] = centered * out.objects["projection"];
  return out;
}

}  // namespace believe14
