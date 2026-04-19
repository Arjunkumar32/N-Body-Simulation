cmake_minimum_required(VERSION 3.18)
project(NBodySim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
  add_compile_options(/O2 /openmp /W3)
else()
  add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# ── Find OpenMP ──────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

include_directories(include)

set(COMMON_SOURCES
    src/force.cpp
    src/integrator.cpp
    src/barnes_hut.cpp
)

# ── Target 1: Direct O(N^2) ──────────────────────────────────────────────────
add_executable(nbody_direct src/main.cpp ${COMMON_SOURCES})
target_link_libraries(nbody_direct PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(nbody_direct PRIVATE USE_DIRECT=1)

# ── Target 2: Barnes-Hut O(N log N) ─────────────────────────────────────────
add_executable(nbody_bh src/main.cpp ${COMMON_SOURCES})
target_link_libraries(nbody_bh PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(nbody_bh PRIVATE USE_BARNES_HUT=1)

# Ensure output directory exists after build
add_custom_command(TARGET nbody_direct POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_SOURCE_DIR}/output"
    COMMENT "Creating output directory"
)

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP     : ${OpenMP_CXX_FLAGS}")
