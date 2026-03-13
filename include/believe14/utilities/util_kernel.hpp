#pragma once

#include <algorithm>
#include <cmath>
#include <cctype>
#include <stdexcept>
#include <string>

#include <believe14/core/config.hpp>

namespace believe14 {
namespace utilities {

struct KernelOptions {
  std::string name;
  double gamma;
  double coef0;
  int degree;
  double alpha;
  double nu;
  double omega;

  KernelOptions()
      : name("rbf"),
        gamma(-1.0),
        coef0(1.0),
        degree(3),
        alpha(1.0),
        nu(1.5),
        omega(1.0) {}
};

namespace detail {

enum class KernelType {
  Linear,
  Polynomial,
  RBF,
  Laplacian,
  Exponential,
  Sigmoid,
  Cosine,
  RationalQuadratic,
  Matern,
  Periodic
};

inline std::string canonicalize_kernel_name(const std::string& name) {
  std::string out;
  out.reserve(name.size());
  for (std::size_t i = 0; i < name.size(); ++i) {
    const unsigned char ch = static_cast<unsigned char>(name[i]);
    if (std::isalnum(ch)) {
      out.push_back(static_cast<char>(std::tolower(ch)));
    }
  }
  return out;
}

inline KernelType parse_kernel_name(const std::string& kernel_name) {
  const std::string key = canonicalize_kernel_name(kernel_name);
  if (key == "linear") {
    return KernelType::Linear;
  }
  if (key == "poly" || key == "polynomial") {
    return KernelType::Polynomial;
  }
  if (key == "rbf" || key == "gaussian") {
    return KernelType::RBF;
  }
  if (key == "laplacian") {
    return KernelType::Laplacian;
  }
  if (key == "exponential") {
    return KernelType::Exponential;
  }
  if (key == "sigmoid") {
    return KernelType::Sigmoid;
  }
  if (key == "cosine") {
    return KernelType::Cosine;
  }
  if (key == "rq" || key == "rationalquadratic") {
    return KernelType::RationalQuadratic;
  }
  if (key == "matern") {
    return KernelType::Matern;
  }
  if (key == "periodic") {
    return KernelType::Periodic;
  }

  throw std::invalid_argument(
      "Unsupported kernel: " + kernel_name +
      ". Supported kernels include linear, polynomial, rbf, laplacian, exponential, "
      "sigmoid, cosine, rationalquadratic, matern, periodic.");
}

inline double default_gamma(Eigen::Index p) {
  return 1.0 / static_cast<double>(p);
}

inline void validate_options(const KernelOptions& options, KernelType type) {
  const bool uses_gamma = (type == KernelType::Polynomial || type == KernelType::RBF ||
                           type == KernelType::Laplacian || type == KernelType::Exponential ||
                           type == KernelType::Sigmoid || type == KernelType::RationalQuadratic ||
                           type == KernelType::Matern || type == KernelType::Periodic);
  if (uses_gamma && options.gamma == 0.0) {
    throw std::invalid_argument("gamma must be nonzero for this kernel.");
  }

  if (type == KernelType::Polynomial && options.degree < 1) {
    throw std::invalid_argument("degree must be >= 1 for polynomial kernel.");
  }
  if (type == KernelType::RationalQuadratic && options.alpha <= 0.0) {
    throw std::invalid_argument("alpha must be > 0 for rational quadratic kernel.");
  }
  if (type == KernelType::Matern) {
    const double nu = options.nu;
    const bool supported = (std::abs(nu - 0.5) < 1e-12) || (std::abs(nu - 1.5) < 1e-12) ||
                           (std::abs(nu - 2.5) < 1e-12);
    if (!supported) {
      throw std::invalid_argument("matern kernel currently supports nu in {0.5, 1.5, 2.5}.");
    }
  }
  if (type == KernelType::Periodic && options.omega <= 0.0) {
    throw std::invalid_argument("omega must be > 0 for periodic kernel.");
  }
}

inline double matern_value(double r, double gamma, double nu) {
  if (std::abs(nu - 0.5) < 1e-12) {
    return std::exp(-gamma * r);
  }
  if (std::abs(nu - 1.5) < 1e-12) {
    const double a = std::sqrt(3.0) * gamma * r;
    return (1.0 + a) * std::exp(-a);
  }
  const double a = std::sqrt(5.0) * gamma * r;
  const double b = 5.0 * gamma * gamma * r * r / 3.0;
  return (1.0 + a + b) * std::exp(-a);
}

}  // namespace detail

inline Eigen::MatrixXd pairwise_kernel_matrix(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                              const KernelOptions& options = KernelOptions()) {
  if (X.rows() == 0 || X.cols() == 0) {
    throw std::invalid_argument("X must have at least one row and one column.");
  }
  if (options.name.empty()) {
    throw std::invalid_argument("kernel name must not be empty.");
  }

  const detail::KernelType type = detail::parse_kernel_name(options.name);
  detail::validate_options(options, type);

  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();
  const double gamma = (options.gamma < 0.0) ? detail::default_gamma(p) : options.gamma;

  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, n);

  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = i; j < n; ++j) {
      const Eigen::RowVectorXd xi = X.row(i);
      const Eigen::RowVectorXd xj = X.row(j);
      const Eigen::RowVectorXd diff = xi - xj;
      const double dot = xi.dot(xj);
      const double l2 = diff.norm();
      const double l2sq = diff.squaredNorm();
      const double l1 = diff.array().abs().sum();

      double kij = 0.0;
      if (type == detail::KernelType::Linear) {
        kij = dot;
      } else if (type == detail::KernelType::Polynomial) {
        kij = std::pow(gamma * dot + options.coef0, static_cast<double>(options.degree));
      } else if (type == detail::KernelType::RBF) {
        kij = std::exp(-gamma * l2sq);
      } else if (type == detail::KernelType::Laplacian) {
        kij = std::exp(-gamma * l1);
      } else if (type == detail::KernelType::Exponential) {
        kij = std::exp(-gamma * l2);
      } else if (type == detail::KernelType::Sigmoid) {
        kij = std::tanh(gamma * dot + options.coef0);
      } else if (type == detail::KernelType::Cosine) {
        const double ni = xi.norm();
        const double nj = xj.norm();
        if (ni == 0.0 && nj == 0.0) {
          kij = 1.0;
        } else if (ni == 0.0 || nj == 0.0) {
          kij = 0.0;
        } else {
          kij = dot / (ni * nj);
        }
      } else if (type == detail::KernelType::RationalQuadratic) {
        kij = std::pow(1.0 + gamma * l2sq / options.alpha, -options.alpha);
      } else if (type == detail::KernelType::Matern) {
        kij = detail::matern_value(l2, gamma, options.nu);
      } else if (type == detail::KernelType::Periodic) {
        const double s = std::sin(options.omega * l2);
        kij = std::exp(-2.0 * gamma * s * s);
      }

      K(i, j) = kij;
      K(j, i) = kij;
    }
  }

  return K;
}

inline Eigen::MatrixXd pairwise_kernel_matrix(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                              const std::string& kernel_name) {
  KernelOptions options;
  options.name = kernel_name;
  return pairwise_kernel_matrix(X, options);
}

}  // namespace utilities
}  // namespace believe14
