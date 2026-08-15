"""Dependency-free writer for the GNNIDAT1 binary dataset format."""

from __future__ import annotations

import array
import bisect
import math
import os
import pathlib
import struct
import sys
import tempfile
from collections.abc import Iterable, Sequence

MAGIC = b"GNNIDAT1"
VERSION = 1
HEADER_BYTES = 64
DIRECTED = 1 << 0
SELF_LOOPS = 1 << 1
LABELS = 1 << 2
SPLITS = 1 << 3


def canonical_csr(
    num_vertices: int,
    edges: Iterable[tuple[int, int]],
    make_undirected: bool,
    self_loops: bool,
) -> tuple[list[int], array.array[int]]:
    if num_vertices <= 0:
        raise ValueError("num_vertices must be positive")
    rows: list[set[int]] = [set() for _ in range(num_vertices)]
    for source, destination in edges:
        if not 0 <= source < num_vertices or not 0 <= destination < num_vertices:
            raise ValueError("edge endpoint is outside the vertex range")
        rows[destination].add(source)
        if make_undirected:
            rows[source].add(destination)
    if self_loops:
        for vertex, row in enumerate(rows):
            row.add(vertex)
    else:
        for vertex, row in enumerate(rows):
            row.discard(vertex)

    columns = array.array("I")
    row_offsets = [0]
    for row in rows:
        columns.extend(sorted(row))
        row_offsets.append(len(columns))
    return row_offsets, columns


def _write_array(stream, typecode: str, values: Sequence[int] | array.array) -> None:
    data = values if isinstance(values, array.array) and values.typecode == typecode else array.array(typecode, values)
    if sys.byteorder != "little":
        data = array.array(typecode, data)
        data.byteswap()
    data.tofile(stream)


def _validate(
    *,
    directed: bool,
    self_loops: bool,
    row_offsets: Sequence[int],
    columns: Sequence[int],
    features: Sequence[float],
    feature_dim: int,
    num_classes: int,
    labels: Sequence[int] | None,
    splits: Sequence[int] | None,
) -> int:
    if len(row_offsets) < 2:
        raise ValueError("CSR must contain at least one vertex")
    num_vertices = len(row_offsets) - 1
    if feature_dim <= 0 or len(features) != num_vertices * feature_dim:
        raise ValueError("feature payload does not match N*F")
    if row_offsets[0] != 0 or row_offsets[-1] != len(columns):
        raise ValueError("CSR offsets do not span the columns")
    for vertex in range(num_vertices):
        begin, end = row_offsets[vertex], row_offsets[vertex + 1]
        if begin > end:
            raise ValueError("CSR offsets are decreasing")
        previous = -1
        loop_count = 0
        for column in columns[begin:end]:
            if not 0 <= column < num_vertices or column <= previous:
                raise ValueError("CSR rows must be in-range and strictly increasing")
            loop_count += column == vertex
            previous = column
        if loop_count != (1 if self_loops else 0):
            raise ValueError("self-loop flag does not match CSR")
    if not directed:
        for row in range(num_vertices):
            for column in columns[row_offsets[row] : row_offsets[row + 1]]:
                reverse = columns[row_offsets[column] : row_offsets[column + 1]]
                position = bisect.bisect_left(reverse, row)
                if position == len(reverse) or reverse[position] != row:
                    raise ValueError("undirected CSR is not symmetric")
    if any(not math.isfinite(value) for value in features):
        raise ValueError("features must be finite")
    if labels is None:
        if splits is not None or num_classes != 0:
            raise ValueError("splits/classes require labels")
    else:
        if len(labels) != num_vertices or num_classes <= 0:
            raise ValueError("labels require N values and positive classes")
        if any(label < 0 or label >= num_classes for label in labels):
            raise ValueError("label is outside class range")
    if splits is not None:
        if labels is None or len(splits) != num_vertices:
            raise ValueError("splits require one value per labeled vertex")
        if any(value not in (0, 1, 2, 3) for value in splits):
            raise ValueError("split value must be 0, 1, 2, or 3")
    return num_vertices


def write_dataset(
    path: pathlib.Path,
    *,
    directed: bool,
    self_loops: bool,
    row_offsets: Sequence[int],
    columns: Sequence[int],
    features: Sequence[float],
    feature_dim: int,
    num_classes: int = 0,
    labels: Sequence[int] | None = None,
    splits: Sequence[int] | None = None,
    force: bool = False,
) -> int:
    path = pathlib.Path(path)
    num_vertices = _validate(
        directed=directed,
        self_loops=self_loops,
        row_offsets=row_offsets,
        columns=columns,
        features=features,
        feature_dim=feature_dim,
        num_classes=num_classes,
        labels=labels,
        splits=splits,
    )
    if len(columns) > 0x7FFFFFFF or num_vertices > 0x7FFFFFFF:
        raise ValueError("dataset exceeds current signed-int CSR limit")
    flags = (DIRECTED if directed else 0) | (SELF_LOOPS if self_loops else 0)
    if labels is not None:
        flags |= LABELS
    if splits is not None:
        flags |= SPLITS
    payload_bytes = (
        (num_vertices + 1) * 8
        + len(columns) * 4
        + num_vertices * feature_dim * 4
        + (num_vertices * 4 if labels is not None else 0)
        + (num_vertices if splits is not None else 0)
    )
    header = struct.pack(
        "<8sIIIIQQQQQ",
        MAGIC,
        VERSION,
        HEADER_BYTES,
        flags,
        0,
        num_vertices,
        len(columns),
        feature_dim,
        num_classes,
        payload_bytes,
    )
    if len(header) != HEADER_BYTES:
        raise AssertionError("internal header size mismatch")

    path.parent.mkdir(parents=True, exist_ok=True)
    if not force and os.path.lexists(path):
        raise FileExistsError(f"dataset already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".tmp-", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(header)
            _write_array(stream, "Q", row_offsets)
            _write_array(stream, "I", columns)
            _write_array(stream, "f", features)
            if labels is not None:
                _write_array(stream, "i", labels)
            if splits is not None:
                stream.write(bytes(splits))
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path.stat().st_size


def atomic_write_text(path: pathlib.Path, text: str, force: bool) -> None:
    path = pathlib.Path(path)
    if not force and os.path.lexists(path):
        raise FileExistsError(f"file already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".tmp-", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
