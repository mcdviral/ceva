cmake_minimum_required(VERSION 3.0.2)
project(ceva)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

add_subdirectory(pybind11)

# Needed non-ros packages
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIR}
)

pybind11_add_module(
  ceva MODULE src/ceva.cpp
)

# Link the module with libraries
target_link_libraries(
  ceva PRIVATE fmt::fmt Eigen3::Eigen ${OpenMP_CXX_FLAGS} ${CERES_LIBRARIES}
)

target_compile_definitions(
  ceva PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
