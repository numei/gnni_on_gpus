import csv
import io
import pathlib
import subprocess
import struct
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common/tools"))
from gnni_format import write_dataset

exe = sys.argv[1]
with tempfile.TemporaryDirectory() as tmp:
    dataset = pathlib.Path(tmp) / "fixture.gnni"
    write_dataset(dataset, directed=True, self_loops=False, row_offsets=[0, 1, 1],
                  columns=[1], features=[2., 3., 4., 5.], feature_dim=2)
    base = [exe, "--dataset", str(dataset), "--layers", "3,1", "--repeat", "2", "--warmup", "1", "--verify"]
    backends = subprocess.check_output([exe, "--list-backends"], text=True).split()
    for backend in backends:
        rows = list(csv.DictReader(io.StringIO(subprocess.check_output(base + ["--backend", backend, "--threads", "2"], text=True))))
        assert len(rows) == 2 and all(r["verified"] == "true" for r in rows)
        for r in rows:
            assert abs(float(r["nodes_per_second"]) * float(r["end_to_end_ms"]) / 1000 - 2) < 1e-6
            assert int(r["host_payload_bytes"]) > 0
    for bad in [["--layers", "3,"], ["--repeat", "0"], ["--backend", "unknown"], ["--threads", "-1"], ["--aggregation", "attention"]]:
        assert subprocess.run([exe, "--dataset", str(dataset)] + bad, capture_output=True).returncode != 0
    # External model file, independently computed output: 4*2 + 5*3 = 23.
    model=pathlib.Path(tmp)/"weights.gnnimodel"
    model.write_bytes(struct.pack("<8s8IQ",b"GNNIMDL\0",1,48,1,1,1,0,24,0,0)
                     +struct.pack("<QQII2f",2,1,0,0,2.,3.))
    for backend in backends:
        result=subprocess.check_output([exe,"--dataset",str(dataset),"--model",str(model),"--backend",backend,"--threads","2","--repeat","1","--verify"],text=True)
        row=next(csv.DictReader(io.StringIO(result)))
        assert float(row["checksum"])==23 and row["verified"]=="true"
print("Unified CLI and timing contract passed")
