#pragma once

#if defined(_MSVC_LANG)
#define BELIEVE14_CPLUSPLUS _MSVC_LANG
#elif defined(__cplusplus)
#define BELIEVE14_CPLUSPLUS __cplusplus
#else
#define BELIEVE14_CPLUSPLUS 0L
#endif

#if BELIEVE14_CPLUSPLUS < 201402L
#error "believe14 requires at least C++14 support."
#endif

// If users define BELIEVE14_USE_SYSTEM_EIGEN before including believe14,
// then we include system Eigen. Otherwise use vendored Eigen.
#ifdef BELIEVE14_USE_SYSTEM_EIGEN
  #include <Eigen/Core>
  #include <Eigen/Dense>
#else
  #include <third_party/Eigen/Core>
  #include <third_party/Eigen/Dense>
#endif

#undef BELIEVE14_CPLUSPLUS
