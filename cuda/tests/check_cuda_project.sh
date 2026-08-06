#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

test -f "${root_dir}/CMakeLists.txt"
test -f "${root_dir}/src/gcn_cuda.cu"
test -f "${root_dir}/include/gnni/gcn_cuda.hpp"
test -f "${root_dir}/tests/gcn_cuda_test.cu"
test -f "${root_dir}/bench/gcn_benchmark.cu"
test ! -e "${root_dir}/src/main.cu"
grep -Fq 'project(gnni_cuda LANGUAGES CXX CUDA)' "${root_dir}/CMakeLists.txt"
grep -Fq 'add_library(gnni_cuda STATIC src/gcn_cuda.cu)' \
  "${root_dir}/CMakeLists.txt"
grep -Fq 'add_test(NAME gcn_cuda_correctness COMMAND gcn_cuda_test)' \
  "${root_dir}/CMakeLists.txt"

