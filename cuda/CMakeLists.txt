cmake_minimum_required(VERSION 3.24)

# Ubuntu 26.04's glibc 2.42 exposes C23 rsqrt declarations through
# _GNU_SOURCE. CUDA 13.1 declares device functions with the same names, so
# disable that feature macro before CMake performs CUDA compiler detection.
string(APPEND CMAKE_CUDA_FLAGS_INIT " -U_GNU_SOURCE")

project(gnni_cuda LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(gnni_dataset STATIC
  src/graph_types.cpp
  src/dataset.cpp
  src/dataset_io.cpp
  src/synthetic_graph.cpp)
target_include_directories(gnni_dataset PUBLIC include)
target_compile_features(gnni_dataset PUBLIC cxx_std_17)

add_library(gnni_cuda STATIC src/gcn_cuda.cu)
target_include_directories(gnni_cuda PUBLIC include)
target_compile_features(gnni_cuda PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(gnni_cuda PUBLIC gnni_dataset)
target_compile_options(gnni_cuda PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-g>
)

add_executable(gcn_cuda_benchmark bench/gcn_benchmark.cu)
target_link_libraries(gcn_cuda_benchmark PRIVATE gnni_cuda)
target_compile_features(gcn_cuda_benchmark PRIVATE cxx_std_17 cuda_std_17)

include(CTest)
if(BUILD_TESTING)
  add_executable(graph_types_test tests/graph_types_test.cpp)
  target_link_libraries(graph_types_test PRIVATE gnni_dataset)
  target_compile_features(graph_types_test PRIVATE cxx_std_17)
  add_test(NAME graph_types COMMAND graph_types_test)

  add_executable(dataset_validation_test tests/dataset_validation_test.cpp)
  target_link_libraries(dataset_validation_test PRIVATE gnni_dataset)
  target_compile_features(dataset_validation_test PRIVATE cxx_std_17)
  add_test(NAME dataset_validation COMMAND dataset_validation_test)

  add_executable(dataset_io_test tests/dataset_io_test.cpp)
  target_link_libraries(dataset_io_test PRIVATE gnni_dataset)
  target_compile_features(dataset_io_test PRIVATE cxx_std_17)
  add_test(NAME dataset_io COMMAND dataset_io_test)

  add_executable(synthetic_graph_test tests/synthetic_graph_test.cpp)
  target_link_libraries(synthetic_graph_test PRIVATE gnni_dataset)
  target_compile_features(synthetic_graph_test PRIVATE cxx_std_17)
  add_test(NAME synthetic_graph COMMAND synthetic_graph_test)

  add_executable(gcn_cuda_test tests/gcn_cuda_test.cu)
  target_link_libraries(gcn_cuda_test PRIVATE gnni_cuda)
  target_compile_features(gcn_cuda_test PRIVATE cxx_std_17 cuda_std_17)
  add_test(NAME gcn_cuda_correctness COMMAND gcn_cuda_test)

  add_test(NAME gcn_cuda_benchmark_smoke COMMAND python3
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_benchmark_output.py
    $<TARGET_FILE:gcn_cuda_benchmark>)
  add_test(NAME cuda_compatibility_check
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cuda_compat.sh)
  add_test(NAME cuda_project_structure_check
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cuda_project.sh)
  add_test(NAME vscode_configuration_check
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_vscode_config.py)
  add_test(NAME obsolete_smoke_artifacts_absent
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_no_obsolete_smoke.py)
endif()

