"""Reproducible Q1 matrix. Raw samples, commands and environment are retained."""
import argparse
import csv
import hashlib
import io
import json
import pathlib
import platform
import random
import subprocess
import time


def capture(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=30)
        return (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", required=True, type=pathlib.Path)
    p.add_argument("--generator", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    p.add_argument("--data-dir", default="tmp/q1-data", type=pathlib.Path)
    p.add_argument("--suite", choices=["smoke", "full", "public"], default="smoke")
    p.add_argument("--backends", default="all")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--public", action="append", type=pathlib.Path, default=[])
    a = p.parse_args()
    if a.repeat < 2 or a.warmup < 0 or a.threads < 1:
        p.error("repeat >= 2, warmup >= 0, threads >= 1 required")
    a.exe = a.exe.resolve(); a.generator = a.generator.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    a.data_dir.mkdir(parents=True, exist_ok=True)
    available = subprocess.check_output([str(a.exe), "--list-backends"], text=True).split()
    backends = available if a.backends == "all" else a.backends.split(",")
    if any(b not in available for b in backends):
        p.error("requested backend unavailable")
    root = pathlib.Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for folder in ("integration", "CPU", "cuda", "common", "experiments"):
        for f in sorted((root / folder).rglob("*")):
            if f.is_file() and f.suffix in (".cpp", ".hpp", ".cu", ".cuh", ".py"):
                source_hash.update(str(f.relative_to(root)).encode()); source_hash.update(f.read_bytes())
    env = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "platform": platform.platform(), "processor": platform.processor(),
           "gpu": capture(["nvidia-smi"]), "commit": capture(["git", "rev-parse", "HEAD"]),
           "git_status": capture(["git", "status", "--short"]),
           "source_sha256": source_hash.hexdigest(),
           "executable_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(),
           "arguments": {k: str(v) for k, v in vars(a).items()}, "backends": backends}
    (a.output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf8")
    configurations = []
    if a.suite == "smoke":
        configurations = [(model, 1000, 16, "16,8", "smoke") for model in ("er", "ba", "ws")]
    elif a.suite == "full":
        configurations += [(model, n, 16, "16,8", "graph-size") for model in ("er", "ba", "ws") for n in (10000, 100000, 1000000)]
        configurations += [("ba", 10000, f, "16,8", "feature-size") for f in (16, 64, 128)]
        configurations += [("ba", 10000, 16, ",".join(["16"] * (depth-1) + ["8"]), "depth") for depth in (1, 2, 4, 8)]
    commands = []
    cases = []
    def run(cmd):
        commands.append([str(x) for x in cmd])
        (a.output / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf8")
        return subprocess.check_output([str(x) for x in cmd], text=True, stderr=subprocess.PIPE)
    for model, n, f, layers, axis in configurations:
        dataset = a.data_dir / f"{model}-n{n}-f{f}-seed42.gnni"
        if not dataset.exists():
            cmd = [a.generator, model, "--vertices", n, "--features", f, "--seed", 42, "--self-loops", "--output", dataset]
            cmd += {"er": ["--average-degree", 16], "ba": ["--edges-per-new-vertex", 8], "ws": ["--neighbors", 16, "--rewire-probability", 0.1]}[model]
            print("Generate", dataset.name, flush=True); run(cmd)
        cases.append((dataset, layers, axis, a.threads, "static", backends))
    for dataset in a.public:
        if not dataset.is_file(): raise FileNotFoundError(dataset)
        cases.append((dataset, "16,8", "public", a.threads, "static", backends))
    if not cases:
        p.error("no configurations selected; --suite public needs --public FILE")
    if a.suite == "full" and "omp" in backends:
        dataset = a.data_dir / "ba-n100000-f16-seed42.gnni"
        for threads in (1, 2, 4, 8):
            for schedule in ("static", "dynamic", "guided"):
                cases.append((dataset, "16,8", "cpu-scaling", threads, schedule, ["omp"]))
    rng = random.Random(42)
    raw = a.output / "raw.csv"
    writer = None
    hashes = {}
    with raw.open("w", newline="", encoding="utf8") as stream:
        for case_id, (dataset, layers, axis, threads, schedule, selected) in enumerate(cases):
            if str(dataset) not in hashes:
                with dataset.open("rb") as f: hashes[str(dataset)] = hashlib.file_digest(f, "sha256").hexdigest()
                (a.output / "datasets.json").write_text(json.dumps(hashes, indent=2), encoding="utf8")
            order = list(selected); rng.shuffle(order)
            for backend in order:
                print(f"[{case_id+1}/{len(cases)}] {dataset.name} L={layers} {backend} threads={threads} {schedule}", flush=True)
                cmd = [a.exe, "--dataset", dataset, "--layers", layers, "--backend", backend,
                       "--aggregation", "mean", "--threads", threads, "--schedule", schedule,
                       "--repeat", a.repeat, "--warmup", a.warmup, "--seed", 42, "--verify"]
                rows = list(csv.DictReader(io.StringIO(run(cmd))))
                if len(rows) != a.repeat or any(r["verified"] != "true" for r in rows):
                    raise RuntimeError("invalid benchmark output")
                for row in rows:
                    row.update(case_id=case_id, axis=axis)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row)
                stream.flush()
    (a.output / "COMPLETE").write_text("All requested configurations completed and verified.\n")


if __name__ == "__main__":
    main()
