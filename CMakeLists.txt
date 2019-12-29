project(NLSSolver)

cmake_minimum_required(VERSION 3.4)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")

set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(Solver test/main.cpp src/NLSSolver.cpp)