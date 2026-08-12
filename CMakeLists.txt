cmake_minimum_required(VERSION 3.20)

project(gnni_on_gpus LANGUAGES CXX)

option(GNNI_BUILD_TESTS "Build Role A unit tests" ON)

# B/C can provide semicolon-separated CUDA source lists when their backends are
# ready. Empty lists keep the corresponding unavailable stub in the build.
set(GNNI_CUDA_VERTEX_SOURCES "" CACHE STRING "Role B CUDA vertex source files")
set(GNNI_CUDA_EDGE_SOURCES "" CACHE STRING "Role C CUDA edge source files")

add_library(gnni_core STATIC
    src/io/matrix_loader.cpp
    src/io/graph_loader.cpp
    src/io/model_loader.cpp
    src/io/text_token_reader.cpp
    src/core/validation.cpp
    src/core/compare.cpp
    src/backends/cpu_kernels.cpp
    src/backends/cpu_sequential.cpp
    src/backends/cpu_openmp.cpp
    src/backends/dispatch.cpp
    src/backends/cuda_unavailable.cpp
    src/benchmark/runner.cpp
    src/benchmark/report.cpp
)

target_compile_features(gnni_core PUBLIC cxx_std_17)
target_include_directories(gnni_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/io"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends"
)

find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gnni_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gnni_core PUBLIC GNNI_HAS_OPENMP=1)
endif()

if(GNNI_CUDA_VERTEX_SOURCES)
    enable_language(CUDA)
    target_sources(gnni_core PRIVATE ${GNNI_CUDA_VERTEX_SOURCES})
    target_compile_definitions(gnni_core PRIVATE GNNI_WITH_CUDA_VERTEX=1)
endif()

if(GNNI_CUDA_EDGE_SOURCES)
    enable_language(CUDA)
    target_sources(gnni_core PRIVATE ${GNNI_CUDA_EDGE_SOURCES})
    target_compile_definitions(gnni_core PRIVATE GNNI_WITH_CUDA_EDGE=1)
endif()

if(MSVC)
    target_compile_options(gnni_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(gnni_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(gnni
    src/cli/main.cpp
    src/cli/cli_options.cpp
)
target_link_libraries(gnni PRIVATE gnni_core)
target_include_directories(gnni PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cli")

if(GNNI_BUILD_TESTS)
    enable_testing()

    add_executable(test_binary_matrix_loader
        tests/test_binary_matrix_loader.cpp
    )
    target_link_libraries(test_binary_matrix_loader PRIVATE gnni_core)

    add_test(NAME binary_matrix_loader COMMAND test_binary_matrix_loader)
endif()

