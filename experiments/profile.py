"""Capture Nsight Compute reports; never change driver permissions/settings."""
import argparse
import json
import pathlib
import subprocess

p=argparse.ArgumentParser()
p.add_argument("--exe",required=True,type=pathlib.Path)
p.add_argument("--dataset",required=True,type=pathlib.Path)
p.add_argument("--ncu",default="ncu")
p.add_argument("--output",required=True,type=pathlib.Path)
p.add_argument("--layers",default="16,8")
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
status={}
for backend in ("cuda-vertex","cuda-tiled","cuda-edge","cuda-edge-tiled"):
    cmd=[a.ncu,"--set","basic","--target-processes","all","--launch-count","8","--export",str(a.output/backend),
         str(a.exe.resolve()),"--dataset",str(a.dataset.resolve()),"--backend",backend,"--aggregation","mean",
         "--layers",a.layers,"--warmup","0","--repeat","1"]
    try:
        proc=subprocess.run(cmd,capture_output=True,text=True)
        log=proc.stdout+proc.stderr
        status[backend]={"command":cmd,"exit_code":proc.returncode,"report_created":(a.output/(backend+".ncu-rep")).exists()}
    except OSError as e:
        log=str(e);status[backend]={"command":cmd,"error":log,"report_created":False}
    (a.output/(backend+".log")).write_text(log,encoding="utf8")
    if status[backend].get("report_created"):
        exported=subprocess.run([a.ncu,"--import",str(a.output/(backend+".ncu-rep")),"--csv","--page","raw"],capture_output=True,text=True)
        (a.output/(backend+"-metrics.csv")).write_text(exported.stdout,encoding="utf8")
        status[backend]["csv_exit_code"]=exported.returncode
    print(backend,status[backend],flush=True)
(a.output/"status.json").write_text(json.dumps(status,indent=2),encoding="utf8")
if any(not s["report_created"] or s.get("exit_code",1)!=0 for s in status.values()):
    raise SystemExit("Profiling incomplete; inspect logs. ERR_NVGPUCTRPERM requires the machine owner to enable performance-counter access; this script does not change it.")
