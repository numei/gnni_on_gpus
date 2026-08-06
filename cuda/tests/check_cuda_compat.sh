#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
grep -Fq 'string(APPEND CMAKE_CUDA_FLAGS_INIT " -U_GNU_SOURCE")' \
  "${root_dir}/CMakeLists.txt"

