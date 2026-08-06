import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tasks = json.loads((root / ".vscode/tasks.json").read_text())
launch = json.loads((root / ".vscode/launch.json").read_text())
extensions = json.loads((root / ".vscode/extensions.json").read_text())

labels = {task["label"] for task in tasks["tasks"]}
assert {
    "CMake: configure",
    "CMake: build",
    "GCN CUDA: test",
    "GCN CUDA: benchmark",
} <= labels

config = launch["configurations"][0]
assert config["program"] == "${workspaceFolder}/build/gcn_cuda_test"
assert config["miDebuggerPath"] == "/usr/local/cuda/bin/cuda-gdb"

recommendations = extensions["recommendations"]
assert "ms-vscode.cpptools" in recommendations
assert "nvidia.nsight-vscode-edition" in recommendations

