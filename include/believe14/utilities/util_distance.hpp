#pragma once

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>

#include <believe14/core/config.hpp>

namespace believe14 {
namespace utilities {

namespace detail {

enum class DistanceMetric {
  Euclidean,
  SquaredEuclidean,
  Manhattan,
  Chebyshev,
  Cosine,
  Correlation,
  Canberra,
  BrayCurtis,
  Hamming,
  Jaccard,
  JensenShannon,
  StandardizedEuclidean,
  Mahalanobis
};

inline std::string canonicalize_metric_name(const std::string& name) {
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

inline DistanceMetric parse_metric_name(const std::string& metric_name) {
  const std::string key = canonicalize_metric_name(metric_name);

  if (key == "euclidean" || key == "l2") {
    return DistanceMetric::Euclidean;
  }
  if (key == "sqeuclidean" || key == "squaredeuclidean" || key == "l2squared") {
    return DistanceMetric::SquaredEuclidean;
  }
  if (key == "cityblock" || key == "manhattan" || key == "taxicab" || key == "l1") {
    return DistanceMetric::Manhattan;
  }
  if (key == "chebyshev" || key == "linf" || key == "maximum") {
    return DistanceMetric::Chebyshev;
  }
  if (key == "cosine") {
    return DistanceMetric::Cosine;
  }
  if (key == "correlation" || key == "pearson") {
    return DistanceMetric::Correlation;
  }
  if (key == "canberra") {
    return DistanceMetric::Canberra;
  }
  if (key == "braycurtis" || key == "bray") {
    return DistanceMetric::BrayCurtis;
  }
  if (key == "hamming") {
    return DistanceMetric::Hamming;
  }
  if (key == "jaccard") {
    return DistanceMetric::Jaccard;
  }
  if (key == "jensenshannon" || key == "js") {
    return DistanceMetric::JensenShannon;
  }
  if (key == "seuclidean" || key == "standardizedeuclidean") {
    return DistanceMetric::StandardizedEuclidean;
  }
  if (key == "mahalanobis") {
    return DistanceMetric::Mahalanobis;
  }

  throw std::invalid_argument(
      "Unsupported distance metric: " + metric_name +
      ". Supported metrics include euclidean, sqeuclidean, cityblock, chebyshev, "
      "cosine, correlation, canberra, braycurtis, hamming, jaccard, "
      "jensenshannon, seuclidean, mahalanobis.");
}

inline Eigen::VectorXd sample_variance(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  if (X.rows() < 2) {
    throw std::invalid_argument("At least two rows are required for standardized Euclidean distance.");
  }

  const Eigen::RowVectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd centered = X.rowwise() - mean;
  Eigen::VectorXd var = (centered.array().square().colwise().sum() / static_cast<double>(X.rows() - 1)).transpose();
  if ((var.array() <= 0.0).any()) {
    throw std::invalid_argument(
        "Standardized Euclidean distance requires strictly positive sample variance in every column.");
  }
  return var;
}

inline Eigen::MatrixXd mahalanobis_inverse_covariance(const Eigen::Ref<const Eigen::MatrixXd>& X) {
  if (X.rows() < 2) {
    throw std::invalid_argument("At least two rows are required for Mahalanobis distance.");
  }

  const Eigen::RowVectorXd mean = X.colwise().mean();
  const Eigen::MatrixXd centered = X.rowwise() - mean;
  const Eigen::MatrixXd cov =
      (centered.transpose() * centered) / static_cast<double>(X.rows() - 1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::VectorXd s = svd.singularValues();
  const double tol = std::numeric_limits<double>::epsilon() *
                     static_cast<double>(std::max(cov.rows(), cov.cols())) * s.maxCoeff();

  Eigen::VectorXd s_inv(s.size());
  for (Eigen::Index i = 0; i < s.size(); ++i) {
    s_inv(i) = (s(i) > tol) ? (1.0 / s(i)) : 0.0;
  }
  return svd.matrixV() * s_inv.asDiagonal() * svd.matrixU().transpose();
}

inline Eigen::VectorXd normalize_probability_row(const Eigen::Ref<const Eigen::RowVectorXd>& x) {
  if ((x.array() < 0.0).any()) {
    throw std::invalid_argument("Jensen-Shannon distance requires nonnegative entries.");
  }
  const double s = x.sum();
  if (s <= 0.0) {
    throw std::invalid_argument("Jensen-Shannon distance requires each row to have positive sum.");
  }
  return (x / s).transpose();
}

inline double kl_divergence(const Eigen::Ref<const Eigen::VectorXd>& p,
                            const Eigen::Ref<const Eigen::VectorXd>& q) {
  double out = 0.0;
  for (Eigen::Index k = 0; k < p.size(); ++k) {
    const double pk = p(k);
    if (pk <= 0.0) {
      continue;
    }
    const double qk = q(k);
    if (qk <= 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    out += pk * std::log(pk / qk);
  }
  return out;
}

}  // namespace detail

inline Eigen::MatrixXd pairwise_distance_matrix(const Eigen::Ref<const Eigen::MatrixXd>& X,
                                                const std::string& metric_name) {
  if (X.rows() == 0 || X.cols() == 0) {
    throw std::invalid_argument("X must have at least one row and one column.");
  }

  const detail::DistanceMetric metric = detail::parse_metric_name(metric_name);
  const Eigen::Index n = X.rows();
  const Eigen::Index p = X.cols();

  Eigen::VectorXd var;
  Eigen::MatrixXd inv_cov;
  if (metric == detail::DistanceMetric::StandardizedEuclidean) {
    var = detail::sample_variance(X);
  } else if (metric == detail::DistanceMetric::Mahalanobis) {
    inv_cov = detail::mahalanobis_inverse_covariance(X);
  }

  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = i + 1; j < n; ++j) {
      const Eigen::RowVectorXd xi = X.row(i);
      const Eigen::RowVectorXd xj = X.row(j);
      const Eigen::RowVectorXd diff = xi - xj;
      const Eigen::ArrayXd absdiff = diff.array().abs();

      double d = 0.0;
      if (metric == detail::DistanceMetric::Euclidean) {
        d = diff.norm();
      } else if (metric == detail::DistanceMetric::SquaredEuclidean) {
        d = diff.squaredNorm();
      } else if (metric == detail::DistanceMetric::Manhattan) {
        d = absdiff.sum();
      } else if (metric == detail::DistanceMetric::Chebyshev) {
        d = absdiff.maxCoeff();
      } else if (metric == detail::DistanceMetric::Cosine) {
        const double ni = xi.norm();
        const double nj = xj.norm();
        if (ni == 0.0 && nj == 0.0) {
          d = 0.0;
        } else if (ni == 0.0 || nj == 0.0) {
          d = 1.0;
        } else {
          d = 1.0 - (xi.dot(xj) / (ni * nj));
        }
      } else if (metric == detail::DistanceMetric::Correlation) {
        const double mi = xi.mean();
        const double mj = xj.mean();
        const Eigen::ArrayXd ci = xi.array() - mi;
        const Eigen::ArrayXd cj = xj.array() - mj;
        const double den = std::sqrt(ci.square().sum() * cj.square().sum());
        if (den == 0.0) {
          d = 0.0;
        } else {
          d = 1.0 - (ci * cj).sum() / den;
        }
      } else if (metric == detail::DistanceMetric::Canberra) {
        d = 0.0;
        for (Eigen::Index k = 0; k < p; ++k) {
          const double den = std::abs(xi(k)) + std::abs(xj(k));
          if (den > 0.0) {
            d += std::abs(xi(k) - xj(k)) / den;
          }
        }
      } else if (metric == detail::DistanceMetric::BrayCurtis) {
        const double den = (xi + xj).array().abs().sum();
        d = (den == 0.0) ? 0.0 : (absdiff.sum() / den);
      } else if (metric == detail::DistanceMetric::Hamming) {
        Eigen::Index neq = 0;
        for (Eigen::Index k = 0; k < p; ++k) {
          if (xi(k) != xj(k)) {
            ++neq;
          }
        }
        d = static_cast<double>(neq) / static_cast<double>(p);
      } else if (metric == detail::DistanceMetric::Jaccard) {
        Eigen::Index inter = 0;
        Eigen::Index uni = 0;
        for (Eigen::Index k = 0; k < p; ++k) {
          const bool ai = (xi(k) != 0.0);
          const bool bj = (xj(k) != 0.0);
          if (ai || bj) {
            ++uni;
            if (ai && bj) {
              ++inter;
            }
          }
        }
        d = (uni == 0) ? 0.0 : 1.0 - static_cast<double>(inter) / static_cast<double>(uni);
      } else if (metric == detail::DistanceMetric::JensenShannon) {
        const Eigen::VectorXd pi = detail::normalize_probability_row(xi);
        const Eigen::VectorXd pj = detail::normalize_probability_row(xj);
        const Eigen::VectorXd m = 0.5 * (pi + pj);
        const double js = 0.5 * detail::kl_divergence(pi, m) + 0.5 * detail::kl_divergence(pj, m);
        d = std::sqrt(std::max(0.0, js));
      } else if (metric == detail::DistanceMetric::StandardizedEuclidean) {
        d = std::sqrt((diff.array().square() / var.array()).sum());
      } else if (metric == detail::DistanceMetric::Mahalanobis) {
        const Eigen::VectorXd v = diff.transpose();
        const double d2 = v.dot(inv_cov * v);
        d = std::sqrt(std::max(0.0, d2));
      }

      D(i, j) = d;
      D(j, i) = d;
    }
  }
  return D;
}

}  // namespace utilities
}  // namespace believe14
