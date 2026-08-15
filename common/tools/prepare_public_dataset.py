#!/usr/bin/env python3
"""Download and convert Cora or ogbn-arxiv without third-party packages."""

from __future__ import annotations

import argparse
import array
import contextlib
import csv
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

from gnni_format import atomic_write_text, canonical_csr, write_dataset

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
ARXIV_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_destination(root: pathlib.Path, member_name: str) -> pathlib.Path:
    member = pathlib.PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe archive path: {member_name}")
    destination = root.joinpath(*member.parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if resolved_destination != resolved_root and resolved_root not in resolved_destination.parents:
        raise ValueError(f"archive path escapes extraction root: {member_name}")
    return destination


def safe_extract(archive: pathlib.Path, destination: pathlib.Path) -> None:
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as source:
            for member in source.getmembers():
                target = _safe_destination(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"archive contains non-regular entry: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                with extracted, target.open("xb") as output:
                    shutil.copyfileobj(extracted, output, 1024 * 1024)
        return
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = _safe_destination(destination, member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"archive contains symlink: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as extracted, target.open("xb") as output:
                    shutil.copyfileobj(extracted, output, 1024 * 1024)
        return
    raise ValueError(f"unsupported archive type: {archive}")


@contextlib.contextmanager
def acquire_archive(args, default_url: str, archive_name: str):
    if args.archive:
        archive = pathlib.Path(args.archive).resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"archive does not exist: {archive}")
        yield archive, args.url or archive.as_uri(), False
        return
    url = args.url or default_url
    if args.keep_cache or args.cache_dir:
        cache_dir = pathlib.Path(args.cache_dir or args.output.parent / ".cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / archive_name
        if not archive.exists():
            temporary = archive.with_name(archive.name + ".tmp")
            with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
            os.replace(temporary, archive)
        yield archive, url, True
        return
    with tempfile.TemporaryDirectory(prefix="gnni-download-") as directory:
        archive = pathlib.Path(directory) / archive_name
        with urllib.request.urlopen(url) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
        yield archive, url, True


def write_sidecar(output: pathlib.Path, metadata: dict, force: bool) -> None:
    metadata["format_version"] = 1
    metadata["output_bytes"] = output.stat().st_size
    metadata["output_sha256"] = sha256_file(output)
    atomic_write_text(
        pathlib.Path(str(output) + ".json"),
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        force,
    )


def convert_cora(root: pathlib.Path, args, source_url: str, source_hash: str, official: bool) -> None:
    content_paths = list(root.rglob("cora.content"))
    cites_paths = list(root.rglob("cora.cites"))
    if len(content_paths) != 1 or len(cites_paths) != 1:
        raise ValueError("archive must contain exactly one cora.content and cora.cites")
    papers: dict[str, tuple[list[float], str]] = {}
    feature_dim = -1
    with content_paths[0].open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"invalid cora.content row {line_number}")
            paper_id, label = fields[0], fields[-1]
            values = [float(value) for value in fields[1:-1]]
            if any(value not in (0.0, 1.0) for value in values):
                raise ValueError("Cora features must be binary")
            if feature_dim < 0:
                feature_dim = len(values)
            if len(values) != feature_dim or paper_id in papers:
                raise ValueError("inconsistent or duplicate Cora content row")
            papers[paper_id] = (values, label)
    paper_ids = sorted(papers)
    index = {paper_id: position for position, paper_id in enumerate(paper_ids)}
    class_names = sorted({papers[paper_id][1] for paper_id in paper_ids})
    class_index = {name: position for position, name in enumerate(class_names)}
    features = array.array("f")
    labels = array.array("i")
    for paper_id in paper_ids:
        values, label = papers[paper_id]
        features.extend(values)
        labels.append(class_index[label])

    edges: list[tuple[int, int]] = []
    citation_rows = 0
    with cites_paths[0].open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"invalid cora.cites row {line_number}")
            if fields[0] not in index or fields[1] not in index:
                raise ValueError(f"unknown paper id in citation row {line_number}")
            edges.append((index[fields[0]], index[fields[1]]))
            citation_rows += 1
    if official and (len(paper_ids), citation_rows, feature_dim, len(class_names)) != (2708, 5429, 1433, 7):
        raise ValueError("official Cora archive counts do not match LINQS")
    row_offsets, columns = canonical_csr(
        len(paper_ids), edges, args.make_undirected, args.self_loops
    )
    write_dataset(
        args.output,
        directed=not args.make_undirected,
        self_loops=args.self_loops,
        row_offsets=row_offsets,
        columns=columns,
        features=features,
        feature_dim=feature_dim,
        num_classes=len(class_names),
        labels=labels,
        force=args.force,
    )
    write_sidecar(
        args.output,
        {
            "dataset": "cora",
            "source": source_url,
            "source_sha256": source_hash,
            "original_vertices": len(paper_ids),
            "original_citation_rows": citation_rows,
            "stored_csr_entries": len(columns),
            "feature_dim": feature_dim,
            "num_classes": len(class_names),
            "make_undirected": args.make_undirected,
            "self_loops": args.self_loops,
            "splits": "absent_from_linqs_archive",
        },
        args.force,
    )


def _find_one(root: pathlib.Path, suffix: str) -> pathlib.Path:
    matches = [path for path in root.rglob("*") if path.as_posix().endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"archive must contain exactly one {suffix}")
    return matches[0]


def _gzip_rows(path: pathlib.Path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.reader(stream)


def convert_arxiv(root: pathlib.Path, args, source_url: str, source_hash: str, official: bool) -> None:
    release = _find_one(root, "RELEASE_v1.txt")
    if not release.is_file():
        raise ValueError("ogbn-arxiv release marker is missing")
    feature_path = _find_one(root, "raw/node-feat.csv.gz")
    label_path = _find_one(root, "raw/node-label.csv.gz")
    edge_path = _find_one(root, "raw/edge.csv.gz")
    features = array.array("f")
    feature_dim = -1
    num_vertices = 0
    for row in _gzip_rows(feature_path):
        values = [float(value) for value in row]
        if feature_dim < 0:
            feature_dim = len(values)
        if len(values) != feature_dim:
            raise ValueError("inconsistent ogbn-arxiv feature width")
        features.extend(values)
        num_vertices += 1
    labels = array.array("i")
    for row in _gzip_rows(label_path):
        if len(row) != 1:
            raise ValueError("ogbn-arxiv labels must have one column")
        labels.append(int(row[0]))
    if len(labels) != num_vertices:
        raise ValueError("ogbn-arxiv label count does not match features")
    edges: list[tuple[int, int]] = []
    for row in _gzip_rows(edge_path):
        if len(row) != 2:
            raise ValueError("ogbn-arxiv edge row must have two columns")
        edges.append((int(row[0]), int(row[1])))
    splits = bytearray(num_vertices)
    split_counts: dict[str, int] = {}
    for name, value in (("train", 1), ("valid", 2), ("test", 3)):
        count = 0
        split_path = _find_one(root, f"split/time/{name}.csv.gz")
        for row in _gzip_rows(split_path):
            if len(row) != 1:
                raise ValueError("split row must have one node index")
            node = int(row[0])
            if not 0 <= node < num_vertices or splits[node] != 0:
                raise ValueError("split node is invalid or duplicated")
            splits[node] = value
            count += 1
        split_counts[name] = count
    if any(value == 0 for value in splits):
        raise ValueError("official time splits do not cover every node")
    num_classes = max(labels) + 1 if labels else 0
    if official and (num_vertices, len(edges), feature_dim, num_classes) != (169343, 1166243, 128, 40):
        raise ValueError("official ogbn-arxiv release counts do not match OGB v1")
    row_offsets, columns = canonical_csr(
        num_vertices, edges, args.make_undirected, args.self_loops
    )
    write_dataset(
        args.output,
        directed=not args.make_undirected,
        self_loops=args.self_loops,
        row_offsets=row_offsets,
        columns=columns,
        features=features,
        feature_dim=feature_dim,
        num_classes=num_classes,
        labels=labels,
        splits=splits,
        force=args.force,
    )
    write_sidecar(
        args.output,
        {
            "dataset": "ogbn-arxiv",
            "release": 1,
            "source": source_url,
            "source_sha256": source_hash,
            "original_vertices": num_vertices,
            "original_directed_edges": len(edges),
            "stored_csr_entries": len(columns),
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "split_counts": split_counts,
            "make_undirected": args.make_undirected,
            "self_loops": args.self_loops,
        },
        args.force,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    for name in ("cora", "ogbn-arxiv"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--output", type=pathlib.Path, required=True)
        subparser.add_argument("--archive", type=pathlib.Path)
        subparser.add_argument("--url")
        subparser.add_argument("--cache-dir", type=pathlib.Path)
        subparser.add_argument("--keep-cache", action="store_true")
        subparser.add_argument("--make-undirected", action="store_true")
        subparser.add_argument("--self-loops", action="store_true")
        subparser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    sidecar = pathlib.Path(str(args.output) + ".json")
    if not args.force and (os.path.lexists(args.output) or os.path.lexists(sidecar)):
        raise FileExistsError("output dataset or sidecar already exists")
    default_url = CORA_URL if args.dataset == "cora" else ARXIV_URL
    archive_name = "cora.tgz" if args.dataset == "cora" else "arxiv.zip"
    with acquire_archive(args, default_url, archive_name) as (archive, source_url, downloaded):
        source_hash = sha256_file(archive)
        with tempfile.TemporaryDirectory(prefix="gnni-extract-") as directory:
            root = pathlib.Path(directory)
            safe_extract(archive, root)
            official = downloaded and args.url is None
            if args.dataset == "cora":
                convert_cora(root, args, source_url, source_hash, official)
            else:
                convert_arxiv(root, args, source_url, source_hash, official)
    print(f"dataset={args.output}")
    print(f"source_sha256={source_hash}")
    print(f"output_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"public dataset conversion failed: {error}", file=sys.stderr)
        raise SystemExit(1)
