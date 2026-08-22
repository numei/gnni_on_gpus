# Role A Scope

This repository records the work owned by Role A: the host-side CPU baseline and the interfaces needed to use it.

## Responsibilities

- Define public graph, dense-matrix, model, configuration, result, and metric types.
- Define and load the versioned binary graph, feature, and model formats.
- Implement the sequential CPU inference path as the numerical reference.
- Implement the OpenMP CPU inference path.
- Compare outputs with tolerances and report correctness metrics.
- Provide CLI parsing, benchmark repetition/statistics, and text/CSV reports.
- Maintain the semantic contract and data-format documentation for this scope.

## Current boundary

The public executable supports only `cpu` and `omp` backends. This repository does not reserve interfaces, source-file slots, build options, or reporting fields for work outside Role A.

## Optional future extensions

The public concepts for softmax, edge features, and attention remain optional. Before implementing any of them, their exact mathematical semantics and file-format changes must be documented and tested against the sequential CPU reference.
