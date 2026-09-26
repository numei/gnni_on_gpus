"""Summarize measured samples and export publication-friendly plots."""
import argparse
import csv
import pathlib
import statistics
from collections import defaultdict

def dataset_path(value):
    # Recorded CSV can be analyzed on a different operating system.
    return pathlib.PurePosixPath(value.replace("\\", "/"))

def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=pathlib.Path);a=p.parse_args()
    if not (a.run/"COMPLETE").is_file():
        raise RuntimeError("Experiment run is incomplete; no COMPLETE marker")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    with (a.run/"raw.csv").open(newline="",encoding="utf8") as f:
        for r in csv.DictReader(f):groups[(r["case_id"],r["backend"])].append(r)
    summaries=[]
    for key,rows in groups.items():
        s={k:v for k,v in rows[0].items() if k not in ("sample","checksum")}
        values=[float(r["end_to_end_ms"]) for r in rows]
        s.update(samples=len(values),median_ms=statistics.median(values),mean_ms=statistics.mean(values),stdev_ms=statistics.stdev(values) if len(values)>1 else 0,min_ms=min(values),max_ms=max(values))
        s["compute_median_ms"]=statistics.median(float(r["compute_ms"]) for r in rows)
        s["h2d_median_ms"]=statistics.median(float(r["h2d_ms"]) for r in rows)
        s["d2h_median_ms"]=statistics.median(float(r["d2h_ms"]) for r in rows)
        summaries.append(s)
    baselines={s["case_id"]:s["median_ms"] for s in summaries if s["backend"]=="cpu"}
    for s in summaries:s["speedup_vs_cpu"]=baselines[s["case_id"]]/s["median_ms"] if s["case_id"] in baselines else ""
    with (a.run/"summary.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    plots=a.run/"plots";plots.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(plots/(name+".png"),dpi=180);fig.savefig(plots/(name+".pdf"));plt.close(fig)
    for axis,xkey,label in [("graph-size","vertices","Vertices"),("feature-size","features","Input features"),("depth","layers","Layers")]:
        subset=[s for s in summaries if s["axis"]==axis]
        models=sorted({dataset_path(s["dataset"]).name.split('-')[0] for s in subset})
        for model in models:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for backend in sorted({s["backend"] for s in subset}):
                points=sorted([s for s in subset if s["backend"]==backend and dataset_path(s["dataset"]).name.split('-')[0]==model],key=lambda s:int(s[xkey]))
                ax.errorbar([int(s[xkey]) for s in points],[s["mean_ms"] for s in points],yerr=[s["stdev_ms"] for s in points],marker='o',capsize=3,label=backend)
            ax.set(xlabel=label,ylabel="End-to-end latency (ms)",title=f"{model.upper()}: {axis} (mean +/- sample SD)",yscale="log")
            if axis=="graph-size":ax.set_xscale("log")
            ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,axis+"-"+model)
    for axis in ("public","smoke"):
        subset=[s for s in summaries if s["axis"]==axis]
        for cid in sorted({s["case_id"] for s in subset},key=int):
            points=[s for s in subset if s["case_id"]==cid]
            fig,ax=plt.subplots(figsize=(8,4.5));ax.bar([s["backend"] for s in points],[s["median_ms"] for s in points]);ax.set(ylabel="Median end-to-end latency (ms)",yscale="log",title=dataset_path(points[0]["dataset"]).stem);ax.tick_params(axis='x',rotation=25);save(fig,axis+"-"+cid)
    subset=[s for s in summaries if s["axis"]=="cpu-scaling"]
    if subset:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for schedule in ("static","dynamic","guided"):
            points=sorted([s for s in subset if s["schedule"]==schedule],key=lambda s:int(s["threads"]))
            ax.plot([int(s["threads"]) for s in points],[s["median_ms"] for s in points],marker='o',label=schedule)
        ax.set(xlabel="OpenMP threads",ylabel="Median end-to-end latency (ms)",title="BA graph: CPU scheduling");ax.legend();save(fig,"cpu-scaling")
    # Plot transfer/compute split and memory for a representative largest graph.
    candidates=[s for s in summaries if s["axis"]=="graph-size"] or summaries
    largest=max(candidates,key=lambda s:int(s["vertices"]))["case_id"]
    points=[s for s in summaries if s["case_id"]==largest]
    gpu=[s for s in points if s["backend"].startswith("cuda")]
    if gpu:
        fig,ax=plt.subplots(figsize=(8,4.5));bottom=[0.]*len(gpu)
        for key,label in [("h2d_median_ms","H2D"),("compute_median_ms","Kernels"),("d2h_median_ms","D2H")]:
            vals=[s[key] for s in gpu];ax.bar([s["backend"] for s in gpu],vals,bottom=bottom,label=label);bottom=[x+y for x,y in zip(bottom,vals)]
        ax.set(ylabel="Stage duration (ms)",title="GPU stage medians (exclude host overhead)");ax.legend();save(fig,"gpu-stages")
    fig,ax=plt.subplots(figsize=(8,4.5));x=list(range(len(points)))
    ax.bar([v-.18 for v in x],[int(s["host_payload_bytes"])/2**20 for s in points],width=.36,label="Host payload estimate")
    ax.bar([v+.18 for v in x],[int(s["device_payload_bytes"])/2**20 for s in points],width=.36,label="Device allocations")
    ax.set_xticks(x,[s["backend"] for s in points],rotation=25);ax.set(ylabel="MiB",title="Largest representative case: array memory");ax.legend();save(fig,"memory")
    lines=["# Measured experiment summary","", "Timing: full inference call, excluding setup/loading and verification. Error bars are sample standard deviation, not confidence intervals. Threads/schedule apply only to OpenMP; other backends ignore those requested settings.", "", "| Dataset | Layers | Backend | Threads | Schedule | Median ms | SD ms | CPU speedup |", "|---|---|---|---:|---|---:|---:|---:|"]
    for s in summaries:
        speed=s["speedup_vs_cpu"]
        thread=s["threads"] if s["backend"]=="omp" else "1" if s["backend"]=="cpu" else "n/a"
        schedule=s["schedule"] if s["backend"]=="omp" else "n/a"
        speed_text=f'{speed:.3f}' if speed!="" else "n/a"
        lines.append(f'| {dataset_path(s["dataset"]).name} | {s["layer_widths"]} | {s["backend"]} | {thread} | {schedule} | {s["median_ms"]:.4f} | {s["stdev_ms"]:.4f} | {speed_text} |')
    (a.run/"REPORT.md").write_text('\n'.join(lines)+'\n',encoding="utf8")
    print(f"Wrote {len(summaries)} summaries and plots to {a.run}")

if __name__=="__main__":main()
