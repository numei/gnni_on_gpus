import csv
import pathlib
import re
import sys

root=pathlib.Path(sys.argv[1])
metrics={"gpu__time_duration.sum":"duration", "sm__throughput.avg.pct_of_peak_sustained_elapsed":"SM throughput",
         "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed":"DRAM throughput",
         "launch__registers_per_thread":"registers/thread", "launch__shared_mem_per_block_static":"static shared/block"}
lines=["# Nsight Compute kernel measurements", "", "BA-100k, F=16, layers 16,8, mean aggregation. Instrumented first inference; not used for application speedup.", "", "| Backend | Kernel | Duration (us) | SM throughput (%) | DRAM throughput (%) | Registers/thread | Static shared (bytes) |", "|---|---|---:|---:|---:|---:|---:|"]
records=[]
for file in sorted(root.glob("*-metrics.csv")):
    with file.open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream);units=next(reader)
        for r in reader:
            if not r.get("ID","").isdigit():continue
            match=re.search(r'(?:aggregate|dense)_\w+',r["Kernel Name"])
            name=match.group(0) if match else r["Kernel Name"]
            vals=[]
            for k in metrics:
                val=float(r[k].replace(',',''));vals.append(val)
            # Nsight raw CSV declares duration units in its first data row.
            unit=units["gpu__time_duration.sum"]
            factor={"nsecond":.001,"usecond":1,"msecond":1000,"second":1e6,"ns":.001,"us":1,"ms":1000,"s":1e6}[unit]
            vals[0]*=factor
            shared_unit=units["launch__shared_mem_per_block_static"].split('/')[0]
            vals[4]*={"byte":1,"Kbyte":1000,"Mbyte":1000000,"B":1,"KB":1000,"MB":1000000}[shared_unit]
            backend=file.name.replace('-metrics.csv','')
            lines.append(f'| {backend} | {name} | '+ ' | '.join(f'{v:.3f}' for v in vals)+' |')
            records.append(dict(backend=backend,kernel=name,duration_us=vals[0],sm_pct=vals[1],dram_pct=vals[2],registers=vals[3],shared_bytes=vals[4]))
(root/"REPORT.md").write_text('\n'.join(lines)+'\n',encoding="utf8")
with (root/"kernel-summary.csv").open('w',newline='',encoding='utf8') as f:
    writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
print('\n'.join(lines))
