import subprocess
import sys


benchmark = sys.argv[1]
completed = subprocess.run(
    [
        benchmark,
        "--vertices",
        "97",
        "--degree",
        "3",
        "--features",
        "7",
        "--outputs",
        "5",
        "--warmup",
        "2",
        "--iterations",
        "3",
    ],
    check=True,
    capture_output=True,
    text=True,
)

metrics = {}
for line in completed.stdout.splitlines():
    key, separator, value = line.partition("=")
    if separator:
        metrics[key] = value

required = {
    "graph_h2d_ms",
    "avg_h2d_ms",
    "avg_kernel_ms",
    "avg_d2h_ms",
    "avg_cuda_pipeline_ms",
    "avg_end_to_end_ms",
    "nodes_per_second",
    "edges_per_second",
    "output_checksum",
}
missing = required - metrics.keys()
assert not missing, f"missing benchmark metrics: {sorted(missing)}"

for name in required:
    assert float(metrics[name]) >= 0.0, f"{name} must be non-negative"

pipeline = float(metrics["avg_cuda_pipeline_ms"])
end_to_end = float(metrics["avg_end_to_end_ms"])
assert end_to_end >= pipeline, (
    "full-call end-to-end time must include the timed CUDA pipeline"
)

