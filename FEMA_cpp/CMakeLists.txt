cmake_minimum_required(VERSION 3.14)
project(FEMA_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra -fsanitize=address")

# ── Find dependencies ────────────────────────────────────────────────────────
# Eigen3 (header-only linear algebra library)
# Try system-installed Eigen first, then fall back to a local copy
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    # Try pkg-config
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(EIGEN3 QUIET eigen3)
    endif()
    if(EIGEN3_FOUND)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIRS}")
    else()
        # Allow user to specify EIGEN3_INCLUDE_DIR manually
        if(DEFINED EIGEN3_INCLUDE_DIR)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
            message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR
                "Eigen3 not found. Install libeigen3-dev or set -DEIGEN3_INCLUDE_DIR=/path/to/eigen")
        endif()
    endif()
endif()

# OpenMP (optional, for parallelism)
find_package(OpenMP)

# ── FEMA library ──────────────────────────────────────────────────────────────
add_library(fema STATIC
    fema_utils.cpp
    fema_parse_family.cpp
    fema_fit.cpp
)

target_include_directories(fema PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fema PUBLIC Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fema PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fema PUBLIC FEMA_USE_OPENMP)
endif()

# ── Example / test binary ────────────────────────────────────────────────────
add_executable(fema_example example_main.cpp)
target_link_libraries(fema_example PRIVATE fema)

# ── Install rules ─────────────────────────────────────────────────────────────
install(TARGETS fema ARCHIVE DESTINATION lib)
install(FILES fema.h DESTINATION include)
install(TARGETS fema_example RUNTIME DESTINATION bin)
