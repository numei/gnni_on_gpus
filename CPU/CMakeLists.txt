cmake_minimum_required(VERSION 3.20)

project(gnni_on_gpus LANGUAGES CXX)

if(MSVC)
    add_compile_options(/utf-8 /EHsc)
endif()

option(GNNI_BUILD_TESTS "Build Role A unit tests" ON)

add_library(gnni_core STATIC
    src/io/matrix_loader.cpp
    src/io/graph_loader.cpp
    src/io/model_loader.cpp
    src/core/compare.cpp
    src/backends/cpu_kernels.cpp
    src/backends/cpu_sequential.cpp
    src/backends/cpu_openmp.cpp
    src/backends/dispatch.cpp
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

    add_executable(test_graph_loader
        tests/test_graph_loader.cpp
    )
    target_link_libraries(test_graph_loader PRIVATE gnni_core)

    add_test(NAME graph_loader COMMAND test_graph_loader)

    add_executable(test_cpu_sequential
        tests/test_cpu_sequential.cpp
    )
    target_link_libraries(test_cpu_sequential PRIVATE gnni_core)

    add_test(NAME cpu_sequential COMMAND test_cpu_sequential)

    if(OpenMP_CXX_FOUND)
        add_executable(test_cpu_openmp
            tests/test_cpu_openmp.cpp
        )
        target_link_libraries(test_cpu_openmp PRIVATE gnni_core)

        add_test(NAME cpu_openmp COMMAND test_cpu_openmp)
    endif()

    add_executable(test_cli_options
        tests/test_cli_options.cpp
        src/cli/cli_options.cpp
    )
    target_link_libraries(test_cli_options PRIVATE gnni_core)
    target_include_directories(
        test_cli_options
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cli"
    )

    add_test(NAME cli_options COMMAND test_cli_options)

    add_executable(test_benchmark_report
        tests/test_benchmark_report.cpp
    )
    target_link_libraries(test_benchmark_report PRIVATE gnni_core)

    add_test(NAME benchmark_report COMMAND test_benchmark_report)

    if(OpenMP_CXX_FOUND)
        add_executable(test_cli_end_to_end
            tests/test_cli_end_to_end.cpp
            src/cli/cli_options.cpp
        )
        target_link_libraries(test_cli_end_to_end PRIVATE gnni_core)
        target_include_directories(
            test_cli_end_to_end
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/cli"
        )

        add_test(NAME cli_end_to_end COMMAND test_cli_end_to_end)
    endif()
endif()
