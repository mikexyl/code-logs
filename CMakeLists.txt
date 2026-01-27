cmake_minimum_required(VERSION 3.10)
project(code_logs_optimizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# Find required packages
find_package(GTSAM REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

# Include directories
include_directories(
  ${GTSAM_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

# Add executable
add_executable(optimize_offline
  src/optimize_offline.cpp
)

# Link libraries
target_link_libraries(optimize_offline
  gtsam
  ${Boost_LIBRARIES}
  stdc++fs
)

# Set compile options
target_compile_options(optimize_offline PRIVATE -fpermissive)

# Installation
install(TARGETS optimize_offline
  RUNTIME DESTINATION bin
)

# Print configuration
message(STATUS "GTSAM version: ${GTSAM_VERSION}")
message(STATUS "GTSAM include: ${GTSAM_INCLUDE_DIR}")
