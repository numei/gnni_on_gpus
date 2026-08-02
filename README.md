# Graph Neural Network Inference on GPUs

System and Device Programming Projects - Device Programming Part  
Project Q1 - Prof. Stefano Quer

## Project Summary

Graph Neural Networks (GNNs) extend deep learning models to graph-structured
data. They are used in recommender systems, social network analysis,
cybersecurity, bioinformatics, and industrial recommendation.

The goal of this project is to develop an inference engine for GNNs on
multi-core CPUs using C++ and on GPUs using CUDA. The project evaluates how
different parallelization strategies affect performance and scalability on
large graphs.

The main strategies to explore are:

- Vertex-parallel computation.
- Edge-parallel computation.
- Message-passing batching.
- Sparse versus dense graph representations.

## Problem Definition

Consider a directed or undirected graph:

$$
G = (V, E)
$$

where:

- `V` is the set of vertices.
- `E` is the set of edges.
- Each node has a feature vector.
- Each edge may optionally have edge features.

A typical GNN layer performs two operations for each node `v`:

1. Aggregation of the features of neighboring nodes, for example using sum,
   mean, or attention-based weighted combination.
2. Update of the node representation through a neural function, such as an MLP.

The project requires:

- Implementing one or two basic GNN architectures, such as GCN and/or
  GraphSAGE, for inference only with pre-loaded weights.
- Designing a sequential C/C++ implementation using efficient graph formats,
  such as CSR or CSC.
- Designing a parallel CPU implementation using explicit threads, tasks, or
  OpenMP.
- Designing one or more CUDA implementations using different mappings, such as
  one thread per node, one thread per edge, or one thread per feature.
- Exploring GPU memory optimizations such as shared memory and coalesced
  access on CSR structures.

The implementations should be compared in terms of:

- Throughput, measured in nodes/s or edges/s.
- Memory footprint.
- Scalability with respect to graph size, from tens of thousands to millions of
  nodes.
- Feature vector size.
- Model depth, meaning the number of GNN layers.

## Implementation Details

The project assumes a single large static graph:

$$
G = (V, E)
$$

The graph may be directed or undirected.

Each vertex `v` in `V` has:

- A unique integer id in the range `[0, |V| - 1]`.
- A feature vector $x_v \in \mathbb{R}^F$, representing node attributes.
- Optionally, a label `y_v`, for example a class id, used only for accuracy
  evaluation in a node-classification task.

Each edge `(u, v)` in `E` may optionally have an edge feature vector `e_uv`.
Edge features can be used in extensions such as GraphSAGE with edge features or
attention mechanisms.

## GNN Layer

A GNN layer performs the following operations for every node `v`.

### 1. Aggregation

The layer collects neighbor representations:

$$
h_u^{(l)}, \quad u \in \mathcal{N}(v)
$$

where:

- $h_u^{(l)}$ is the representation of node `u` at layer `l`.
- $\mathcal{N}(v)$ is the neighborhood of node `v`.

Example aggregation operators:

Sum aggregation:

$$
m_v^{(l)} = \sum_{u \in \mathcal{N}(v)} h_u^{(l)}
$$

Mean aggregation:

$$
m_v^{(l)} =
\frac{1}{|\mathcal{N}(v)|}
\sum_{u \in \mathcal{N}(v)} h_u^{(l)}
$$

An attention-weighted sum can be added as an optional extension.

### 2. Update

The layer combines the current node state and the aggregated message through a
fixed neural function.

GCN-style update:

$$
h_v^{(l+1)} = \sigma\left(W^{(l)} m_v^{(l)}\right)
$$

GraphSAGE-style update:

$$
h_v^{(l+1)} =
\sigma\left(W^{(l)} \cdot
\operatorname{concat}\left(h_v^{(l)}, m_v^{(l)}\right)\right)
$$

All weights $W^{(l)}$ and layer parameters are pre-loaded and fixed. The project
focuses on inference only, not training.

## Overall Model

The full model is a stack of `L` GNN layers.

Input:

$$
h_v^{(0)} = x_v
$$

For each layer `l = 0, ..., L - 1`, compute:

$$
h_v^{(l+1)}
$$

for every node `v` by applying one graph-wide message-passing iteration.

Output for node classification:

- Apply a final linear layer or softmax on $h_v^{(L)}$.
- Produce logits for each node.
- Optionally compute accuracy on a given split.

This project focuses on full-batch inference on a single large graph, not
mini-batches across many small graphs.

## Parallelization Dimensions

The main parallelization dimensions are:

- Nodes.
- Edges.
- Feature dimensions.
- Possibly layers.

Each GNN layer should be treated as one graph-wide message-passing iteration.
For performance evaluation, each configuration should be run multiple times to
obtain stable timings.

The configurations may vary by:

- Graph.
- Feature size.
- Model depth.
- Implementation backend.

## Public Datasets and Formats

Large graphs with node features for GNNs are publicly available.

### Open Graph Benchmark

Open Graph Benchmark (OGB) provides large realistic graphs for node and link
prediction, with standardized loaders and splits.

Website:

```text
https://ogb.stanford.edu
```

Example datasets:

- `ogbn-products`
- `ogbn-papers100M`
- `ogbn-arxiv`

These datasets provide:

- A single large graph.
- Node features and labels.
- Standard tasks and splits.

The official PyTorch Geometric and DGL loaders expose the graph in CSR-like
adjacency structures. These can be exported to a custom CSR plus dense-matrix
format.

### Planetoid / Citation Datasets

Classic small-to-medium node-classification benchmarks include:

- Cora
- CiteSeer
- PubMed

These datasets are often distributed in preprocessed NumPy or SciPy formats,
but they are fundamentally sparse adjacency matrices plus dense node feature
matrices. They can be mapped directly to CSR plus dense features.

They are useful for:

- Functional debugging.
- Small benchmark cases.
- Public benchmark comparisons.

### Synthetic Graphs

Synthetic graphs can be generated using standard graph libraries and exported
to CSR.

Useful graph families include:

- Scale-free graphs, such as Barabasi-Albert graphs.
- Erdos-Renyi random graphs.
- Small-world graphs.

## Required Background

The project relies on knowledge from:

- Advanced programming courses, including data structures, graphs, and memory
  management.
- System and Device Programming, including concurrent programming and memory
  models.
- Basic CUDA programming, including thread hierarchy, memory hierarchy, and
  synchronization.

A short introduction to GNNs can be provided through selected survey or tutorial
papers. Students are not required to implement training, only the forward pass.

## Working Environment

Students may work on a laptop or desktop under Unix-like or Windows systems.

The implementation must be done in C/C++ with CUDA extensions for the GPU part.

The following tools are encouraged:

- Lightweight utilities to load graph files, such as edge lists or custom CSR
  files.
- Profiling tools such as `nvprof`, Nsight Systems, Nsight Compute, and `perf`.

The project can naturally be extended into an MSc thesis by adding:

- Mini-batching across multiple graphs.
- Graph partitioning.
- Load-balancing techniques.
- Multi-GPU support.

## Deliverables

The final project delivery must include:

- C/C++ and CUDA source files.
- A plain ASCII README text file with:
  - Compilation instructions.
  - Run instructions.
  - Execution parameters.
  - Dataset formats.
- A documentation file in Word, LaTeX, Markdown, or another suitable format,
  containing:
  - Main design choices.
  - Graph format.
  - Memory layout.
  - CPU and GPU parallelization schemes.
  - Strategies for handling highly skewed degree distributions.
  - Experimental evaluation with tables and plots.
- A set of overhead slides, such as PowerPoint slides, for a 20-minute oral
  project presentation.

The experimental evaluation should compare:

- Sequential CPU implementation versus multi-core CPU implementation versus GPU
  implementation.
- Different parallelization strategies, such as vertex-centric and edge-centric
  approaches.
- Implementations with and without shared memory.
- Implementations with and without feature compression.
- Different graph sizes.
- Different feature sizes.
- Synthetic graphs and public graph benchmarks.

## References

- Course material for System and Device Programming, for general parallel
  programming concepts.
- Introductory tutorials and surveys on Graph Neural Networks and
  message-passing architectures.
- NVIDIA CUDA documentation, including the Programming Guide and Best Practices,
  for memory and parallelism optimizations on GPUs.
- Research papers on efficient GPU implementations of GNNs and optimized sparse
  matrix operations.

## Contact

Prof. Stefano Quer  
stefano.quer@polito.it

## License

This repository is licensed under the MIT License. See `LICENSE` for details.
