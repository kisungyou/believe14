cmake_minimum_required(VERSION 3.16)

project(believe14 VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BELIEVE14_USE_SYSTEM_EIGEN "Use system-installed Eigen instead of vendored headers." OFF)
option(BELIEVE14_BUILD_EXAMPLES "Build example programs." ON)

add_library(believe14 INTERFACE)
add_library(believe14::believe14 ALIAS believe14)

target_compile_features(believe14 INTERFACE cxx_std_14)
target_include_directories(
    believe14
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(BELIEVE14_CONFIG_USE_SYSTEM_EIGEN ${BELIEVE14_USE_SYSTEM_EIGEN})

if(BELIEVE14_USE_SYSTEM_EIGEN)
  find_package(Eigen3 REQUIRED CONFIG)
  target_link_libraries(believe14 INTERFACE Eigen3::Eigen)
  target_compile_definitions(believe14 INTERFACE BELIEVE14_USE_SYSTEM_EIGEN=1)
else()
  target_include_directories(
      believe14
      INTERFACE
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/third_party>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/third_party>
  )
endif()

if(BELIEVE14_BUILD_EXAMPLES)
  add_executable(quicky quicky.cpp)
  target_link_libraries(quicky PRIVATE believe14::believe14)
endif()

install(TARGETS believe14 EXPORT believe14Targets)
install(DIRECTORY include/believe14 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES include/believe14.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(NOT BELIEVE14_USE_SYSTEM_EIGEN)
  install(DIRECTORY include/third_party/Eigen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/third_party)
endif()

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/believe14Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/believe14Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/believe14
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/believe14ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/believe14Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/believe14ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/believe14
)

install(
    EXPORT believe14Targets
    FILE believe14Targets.cmake
    NAMESPACE believe14::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/believe14
)

set(CPACK_PACKAGE_NAME "believe14")
set(CPACK_PACKAGE_VENDOR "believe14")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Header-only C++ library for dimensionality reduction and manifold learning")
set(CPACK_GENERATOR "TGZ;DEB")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "all")

if(BELIEVE14_USE_SYSTEM_EIGEN)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "libeigen3-dev")
endif()

include(CPack)
