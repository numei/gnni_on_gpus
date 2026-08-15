#!/usr/bin/env python3
import csv
import gzip
import json
import pathlib
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def run(converter, *arguments, expect=0):
    result = subprocess.run(
        [sys.executable, str(converter), *map(str, arguments)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != expect:
        raise RuntimeError(
            f"unexpected exit {result.returncode}: {result.stdout}\n{result.stderr}"
        )
    return result


def header(path):
    with path.open("rb") as stream:
        return struct.unpack("<8sIIIIQQQQQ", stream.read(64))


def write_gzip_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_cora(converter, root):
    source = root / "cora-src" / "cora"
    source.mkdir(parents=True)
    (source / "cora.content").write_text(
        "p3\t1\t0\tClassB\n"
        "p1\t0\t1\tClassA\n"
        "p2\t1\t1\tClassB\n",
        encoding="utf-8",
    )
    (source / "cora.cites").write_text(
        "p1\tp2\np1\tp2\np3\tp1\n", encoding="utf-8"
    )
    archive = root / "cora.tgz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname="cora")
    dataset = root / "cora.gnni"
    run(
        converter,
        "cora",
        "--archive",
        archive,
        "--make-undirected",
        "--self-loops",
        "--output",
        dataset,
    )
    values = header(dataset)
    if values[:2] != (b"GNNIDAT1", 1) or values[5:9] != (3, 7, 2, 2):
        raise RuntimeError(f"unexpected Cora header: {values}")
    metadata = json.loads(pathlib.Path(str(dataset) + ".json").read_text())
    if metadata["original_citation_rows"] != 3 or metadata["splits"] != "absent_from_linqs_archive":
        raise RuntimeError("Cora provenance is incomplete")
    run(converter, "cora", "--archive", archive, "--output", dataset, expect=1)


def test_arxiv(converter, root):
    source = root / "arxiv-src" / "ogbn_arxiv"
    write_gzip_csv(source / "raw/node-feat.csv.gz", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    write_gzip_csv(source / "raw/node-label.csv.gz", [[0], [1], [0], [1]])
    write_gzip_csv(source / "raw/edge.csv.gz", [[0, 1], [1, 2], [2, 3]])
    write_gzip_csv(source / "split/time/train.csv.gz", [[0], [1]])
    write_gzip_csv(source / "split/time/valid.csv.gz", [[2]])
    write_gzip_csv(source / "split/time/test.csv.gz", [[3]])
    (source / "RELEASE_v1.txt").write_text("1\n", encoding="utf-8")
    archive = root / "arxiv.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in source.rglob("*"):
            if path.is_file():
                output.write(path, path.relative_to(source.parent))
    dataset = root / "arxiv.gnni"
    run(converter, "ogbn-arxiv", "--archive", archive, "--self-loops", "--output", dataset)
    values = header(dataset)
    if values[:2] != (b"GNNIDAT1", 1) or values[5:9] != (4, 7, 3, 2):
        raise RuntimeError(f"unexpected arxiv header: {values}")
    metadata = json.loads(pathlib.Path(str(dataset) + ".json").read_text())
    if metadata["split_counts"] != {"train": 2, "valid": 1, "test": 1}:
        raise RuntimeError("arxiv split provenance is wrong")


def test_traversal(converter, root):
    archive = root / "evil.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape", "bad")
    result = run(
        converter,
        "ogbn-arxiv",
        "--archive",
        archive,
        "--output",
        root / "evil.gnni",
        expect=1,
    )
    if "unsafe archive path" not in result.stderr:
        raise RuntimeError("path traversal was not rejected descriptively")


def main():
    converter = pathlib.Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="gnni-public-test-") as directory:
        root = pathlib.Path(directory)
        test_cora(converter, root)
        test_arxiv(converter, root)
        test_traversal(converter, root)
    print("public converter tests passed")


if __name__ == "__main__":
    main()
