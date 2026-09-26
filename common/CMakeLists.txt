cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(gnni_common LANGUAGES CXX)
endif()

add_library(gnni_common STATIC
  src/graph_types.cpp
  src/dataset.cpp
  src/dataset_io.cpp
  src/synthetic_graph.cpp)
add_library(gnni::common ALIAS gnni_common)
target_include_directories(gnni_common PUBLIC include)
target_compile_features(gnni_common PUBLIC cxx_std_17)

add_executable(gnni_dataset_generator apps/dataset_generator.cpp)
target_link_libraries(gnni_dataset_generator PRIVATE gnni_common)
target_compile_features(gnni_dataset_generator PRIVATE cxx_std_17)

include(CTest)
if(BUILD_TESTING)
  foreach(test_name IN ITEMS
      graph_types
      dataset_validation
      dataset_io
      synthetic_graph
      synthetic_models)
    add_executable(${test_name}_test tests/${test_name}_test.cpp)
    target_link_libraries(${test_name}_test PRIVATE gnni_common)
    target_compile_features(${test_name}_test PRIVATE cxx_std_17)
    add_test(NAME ${test_name} COMMAND ${test_name}_test)
  endforeach()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME public_dataset_converter COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_public_converter.py
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_public_dataset.py)
endif()
