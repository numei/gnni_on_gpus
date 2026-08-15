#!/usr/bin/env python3
import math
import pathlib
import subprocess
import sys
import tempfile


def call(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, command))}\n{result.stderr}")
    return result.stdout


def main():
    generator = pathlib.Path(sys.argv[1])
    runner = pathlib.Path(sys.argv[2])
    with tempfile.TemporaryDirectory(prefix="gnni-runner-test-") as directory:
        dataset = pathlib.Path(directory) / "tiny.gnni"
        call([
            generator, "er", "--vertices", "128", "--average-degree", "8",
            "--features", "16", "--seed", "42", "--self-loops",
            "--output", dataset,
        ])
        output = call([
            runner, "--dataset", dataset, "--layer-widths", "8,4",
            "--warmup", "1", "--iterations", "2",
        ])
        values = {}
        for line in output.splitlines():
            key, separator, value = line.partition("=")
            if not separator or key in values:
                raise RuntimeError(f"invalid runner output line: {line}")
            values[key] = value
        required = {
            "vertices", "csr_entries", "feature_dim", "load_ms", "host_bytes",
            "graph_h2d_ms", "avg_kernel_ms", "avg_end_to_end_ms",
            "nodes_per_second", "csr_entries_per_second", "output_rows",
            "output_cols", "output_checksum",
        }
        if not required.issubset(values):
            raise RuntimeError(f"missing runner keys: {required - values.keys()}")
        if values["vertices"] != "128" or values["output_rows"] != "128" or values["output_cols"] != "4":
            raise RuntimeError("runner output dimensions are wrong")
        for key in ("load_ms", "graph_h2d_ms", "avg_kernel_ms", "avg_end_to_end_ms", "nodes_per_second", "csr_entries_per_second", "output_checksum"):
            if not math.isfinite(float(values[key])):
                raise RuntimeError(f"non-finite {key}")
    print("dataset runner smoke test passed")


if __name__ == "__main__":
    main()
