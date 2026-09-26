cmake_minimum_required(VERSION 3.24)

# Ubuntu 26.04's glibc 2.42 exposes C23 rsqrt declarations through
# _GNU_SOURCE. CUDA 13.1 declares device functions with the same names, so
# disable that feature macro before CMake performs CUDA compiler detection.
if(UNIX)
  string(APPEND CMAKE_CUDA_FLAGS_INIT " -U_GNU_SOURCE")
endif()

project(gnni_cuda LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT TARGET gnni_common)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../common ${CMAKE_BINARY_DIR}/common)
endif()

add_library(gnni_cuda STATIC src/gcn_cuda.cu)
target_include_directories(gnni_cuda PUBLIC include)
target_compile_features(gnni_cuda PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(gnni_cuda PUBLIC gnni::common)
target_compile_options(gnni_cuda PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-g>
)

add_executable(gcn_cuda_benchmark bench/gcn_benchmark.cu)
target_link_libraries(gcn_cuda_benchmark PRIVATE gnni_cuda)

add_executable(gcn_dataset_runner apps/gcn_dataset_runner.cu)
target_link_libraries(gcn_dataset_runner PRIVATE gnni_cuda)
target_compile_features(gcn_dataset_runner PRIVATE cxx_std_17 cuda_std_17)


target_compile_features(gcn_cuda_benchmark PRIVATE cxx_std_17 cuda_std_17)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_executable(gcn_cuda_test tests/gcn_cuda_test.cu)
  target_link_libraries(gcn_cuda_test PRIVATE gnni_cuda)
  target_compile_features(gcn_cuda_test PRIVATE cxx_std_17 cuda_std_17)
  add_test(NAME gcn_cuda_correctness COMMAND gcn_cuda_test)

  add_test(NAME gcn_cuda_benchmark_smoke COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_benchmark_output.py
    $<TARGET_FILE:gcn_cuda_benchmark>)
  if(UNIX)
  add_test(NAME cuda_compatibility_check
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cuda_compat.sh)
  add_test(NAME cuda_project_structure_check
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cuda_project.sh)
  endif()
  add_test(NAME vscode_configuration_check
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_vscode_config.py)
  add_test(NAME obsolete_smoke_artifacts_absent
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_no_obsolete_smoke.py)
  add_test(NAME dataset_runner_smoke COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_dataset_runner.py
    $<TARGET_FILE:gnni_dataset_generator>
    $<TARGET_FILE:gcn_dataset_runner>)

endif()

