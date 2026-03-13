#pragma once

#include <cmath>

#include <believe14/core/config.hpp>

namespace believe14 {
namespace utilities {

inline bool check_array2d(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  if (X.rows() <= 0 || X.cols() <= 0) {
    return false;
  }

  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    for (Eigen::Index j = 0; j < X.cols(); ++j) {
      if (!std::isfinite(X(i, j))) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace utilities
}  // namespace believe14
