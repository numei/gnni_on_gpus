// Build with the supplied Codex presentation runtime (see docs/VALIDATION.md).
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const modules=process.env.RUNTIME_NODE_MODULES;
const skill=process.env.SKILL_DIR;
if(!modules || !skill || !process.env.RUNTIME_PYTHON) throw new Error('Set RUNTIME_NODE_MODULES, SKILL_DIR and RUNTIME_PYTHON');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(modules,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const root=path.resolve('.');const tmp=path.join(root,'tmp/presentation');await fs.mkdir(tmp,{recursive:true});
const font='Arial';
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
function csv(text){
 const rows=[];let row=[],field='',quoted=false;
 for(let i=0;i<text.length;i++){
  const c=text[i];
  if(c==='"'){if(quoted && text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
  else if(c===','&&!quoted){row.push(field);field='';}
  else if(c==='\n'&&!quoted){row.push(field.replace(/\r$/,''));rows.push(row);row=[];field='';}
  else field+=c;
 }
 const header=rows.shift();return rows.filter(r=>r.length===header.length).map(r=>Object.fromEntries(header.map((h,i)=>[h,r[i]])));
}
const data=csv(await fs.readFile('results/rtx3050-full/summary.csv','utf8'));
const publicData=csv(await fs.readFile('results/rtx3050-arxiv/summary.csv','utf8'));
function text(slide,value,x,y,w,h,size=32,color='#172C40',bold=false){
 const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 shape.text=value;shape.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'};return shape;
}
let count=0;
function slide(title,lines,seconds,notes){
 const s=presentation.slides.add();s.background.fill='#FFFFFF';count++;
 text(s,title,64,44,1152,100,44,'#122D42',true);
 lines.forEach((line,i)=>text(s,line,72,175+i*102,1130,92,32));
 text(s,String(count),1160,672,50,28,20,'#516575');
 s.speakerNotes.textFrame.setText(`Suggested time: ${seconds} seconds.\n${notes}\nSources: Project Q1 supplied brief; DOCUMENTATION.md; results/rtx3050-full/raw.csv; results/rtx3050-arxiv/raw.csv.`);
 return s;
}
const cover=slide('Graph Neural Network Inference',[],40,'Introduce Project Q1 and the implementation team. Full-batch forward inference, not training. The results shown were measured on one laptop on 26 September 2026.');
text(cover,'CPU and CUDA',72,212,1100,100,66,'#122D42',true);
text(cover,'Project Q1\nRyzen 9 5900HS / RTX 3050 Laptop GPU\n26 September 2026',76,380,1080,170,32);
slide('Model and graph semantics',[
 'Aggregate incoming neighbor features, then compute A W.',
 'Hidden layers: ReLU. Final layer: Identity.',
 'Sum, mean and symmetric normalization share one API.',
 'Fixed weights. Explicit self-loops. Zero-degree nodes produce zero.'
],90,'Explain H(l+1)=activation(A(l)W(l)). Define CSR row v as incoming source nodes u. Symmetric scaling uses CSR row degrees and gives zero for a zero-degree neighbor. Public experiments are symmetrized. We implement one GCN-style architecture, which satisfies the one-or-two architecture requirement.');
slide('One input contract across backends',[
 '.gnni stores CSR, FP32 features and optional labels/splits.',
 'The adapter converts legacy CPU types once during setup.',
 'All backends receive identical weights and activation rules.',
 '--verify compares the requested backend with the actual CPU reference.'
],90,'Explain why merging directories was insufficient: incompatible C++ types, file formats and normalization. CPU types now use gnni_cpu, while common/CUDA use gnni. The adapter prevents ODR/type collisions. Generated weights support repeatable performance experiments; .gnnimodel loads external weights. No predictive accuracy claim is made for generated weights.');
slide('CPU reference and OpenMP',[
 'Sequential C++ defines the reference result.',
 'OpenMP assigns independent destination rows to workers.',
 'Static, dynamic and guided scheduling expose degree imbalance.',
 'The OpenMP dense loop also improves contiguous weight access.'
],90,'No atomics are needed for CPU row ownership. Discuss why OpenMP speedup is not purely thread count: loop order differs for locality. One-thread OpenMP measurements help interpret it. Thread counts 1,2,4,8 are measured on BA-100k. Default MSVC OpenMP 2 runtime works because scheduling uses loop clauses instead of unsupported runtime functions.');
slide('Four CUDA configurations',[
 'Vertex: one output per vertex-feature thread.',
 'Edge: one contribution per edge-feature thread, atomicAdd.',
 'Baseline dense transform versus 16 x 16 shared-memory tiles.',
 'The 2 x 2 comparison isolates mapping and tiling effects.'
],120,'Explain the vertex thread traversing a CSR row versus edge threads accumulating into row outputs. Edge mapping needs an E-entry destination array and clearing the aggregate, but distributes long rows. Dense tiling reuses two 16x16 arrays, 2048 static shared bytes per block. Partial tiles keep all threads in every barrier. Neither alternative is assumed faster before measurement.');
slide('Memory and transfer boundaries',[
 'CSR uploads once. Intermediate layers remain on device.',
 'Features and packed weights upload per inference call.',
 'Only the final output returns to the CPU.',
 'Memory reports array payloads, not process RSS or driver allocation.'
],90,'Explain ping-pong feature buffers and reusable capacity. Device baseline formula is about 4(N+1+E)+12NFmax+4W bytes, with 4E extra for edge mapping. At million-node scale CSR remains manageable; a dense FP32 adjacency alone would be 4 TB. Host payload estimates include dataset/model, adapted arrays and temporary matrices, but exclude allocator and driver overhead.');
slide('Correctness and memory checks',[
 '20 / 20 combined CTest tests passed on this GPU.',
 'All backends: literal fixtures, three normalizations and multiple layers.',
 'Compute Sanitizer: zero errors and zero leaked bytes.',
 'Racecheck: zero hazards, errors or warnings.'
],80,'The tolerance is absolute 1e-5 plus relative 1e-4 times reference magnitude. Tests include directed asymmetric graphs, self loops, isolated vertices, zero edges, skewed rows, non-tile-aligned widths, context reuse and malformed input. An extra CLI test independently verifies supplied model weights produce checksum 23 across all backends. Linux-specific syscall-hook tests were not executed on Windows. Source: docs/VALIDATION.md and results/q1-memcheck.log, q1-racecheck.log.');
slide('Experimental design',[
 'ER, BA and WS graphs: 10k, 100k and 1m vertices.',
 'Feature widths 16 / 64 / 128. Depths 1 / 2 / 4 / 8.',
 'Public graphs: Cora and ogbn-arxiv.',
 'Two warmups, seven samples, verification before timing.'
],90,'This is a controlled slice matrix, not all combinations. Graph-size sweep uses F16 and widths16,8. The machine is Ryzen 9 5900HS 8 cores/16 threads with a 4GB RTX3050 laptop GPU, CUDA12.6 and driver616.92. Application timing excludes load/setup and includes per-call transfers and host work. Nodes/s is N divided by full-model inference seconds. Error bars in exported scientific plots are sample SD, not confidence intervals.');
function chart(s,categories,series,ylabel){
 series=series.map(item=>({...item,values:item.values.map(v=>Number(v.toFixed(3)))}));
 const c=s.charts.add('bar',{position:{left:84,top:170,width:1112,height:440},categories,series,barOptions:{direction:'column',grouping:'clustered'},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:24,typeface:font}},yAxis:{title:{text:'CPU-relative speedup',textStyle:{fontSize:24,typeface:font}},numberFormatCode:'0',textStyle:{fontSize:24,typeface:font}},xAxis:{textStyle:{fontSize:24,typeface:font}},titlePlacement:'none'});
 applyPresentationChartFont(c,{fontFamily:font});
}
const large=slide('Million-node inference speedup',[],120,'Read CPU-relative median end-to-end speedup, not kernel-only speedup. GPU speedups are roughly 48x on ER/BA for the tiled vertex configuration. WS has better CPU locality and a smaller speedup. Edge mapping is slower than vertex mapping in these measurements. Laptop clocks and the short repeated run protocol limit generalization.');
const colors=['#205E8A','#319B94','#D08739','#99506B'];
const gpuNames=['cuda-vertex','cuda-tiled','cuda-edge','cuda-edge-tiled'];
chart(large,['ER','BA','WS'],gpuNames.map((b,i)=>({name:b,values:['er','ba','ws'].map(g=>Number(data.find(s=>s.axis==='graph-size'&&s.vertices==='1000000'&&s.backend===b&&s.dataset.includes(g+'-n')).speedup_vs_cpu)),fill:colors[i]})),'Speedup versus sequential CPU');
text(large,'Median full-call speedup; F=16, layer widths 16,8',80,623,1110,40,24,'#516575');
const pub=slide('Public graph inference',[],100,'Cora has 2708 nodes and 1433 input features; arxiv has 169343 nodes and 128 features. Both use symmetrized CSR with explicit self loops and the same generated layer widths16,8. Generated weights measure execution performance, not node-classification accuracy. Compare only within each graph.');
chart(pub,['Cora','ogbn-arxiv'],gpuNames.map((b,i)=>({name:b,values:[Number(data.find(s=>s.axis==='public'&&s.backend===b).speedup_vs_cpu),Number(publicData.find(s=>s.backend===b).speedup_vs_cpu)],fill:colors[i]})),'Speedup versus sequential CPU');
text(pub,'Cora: best 23.9x. ogbn-arxiv: best 29.3x.',80,623,1110,40,24,'#516575');
slide('Kernel profiling explains tradeoffs',[
 'BA-100k aggregation: vertex ~858 us, edge ~3321 us.',
 'Dense 16 -> 16: baseline 332 us, tiled 270 us.',
 'Dense 16 -> 8: baseline 169 us, tiled 270 us.',
 'A narrow output tile wastes work despite shared-memory reuse.'
],100,'Nsight Compute 2024.3.2 basic metrics captured on first inference with kernel replay. These instrumented kernel times are not application benchmark values. Edge DRAM throughput about19% vs vertex65% of sustained peak is consistent with substantial atomic/instruction overhead; it does not by itself prove a unique stall cause. Tiled dense uses2048 bytes static shared memory and40 registers/thread versus35 baseline registers/thread. Source: results/rtx3050-profile/REPORT.md and .ncu-rep files.');
slide('Interpretation and limits',[
 'Vertex mapping wins over this atomic edge implementation.',
 'Shared-memory tiling helps some sizes and hurts others.',
 'More CPU threads help, but scheduling overhead matters.',
 'Results cover one GPU and one fixed-weight model family.'
],100,'The negative optimization results are useful: do not claim shared memory or edge parallelism is universally faster. BA100k static schedule improves from202ms at1thread to53ms at8threads, but dynamic overhead remains high. Full results contain raw samples and measured variance. Transfer and host overhead prevent kernel speedups from translating one-for-one to application speedups. Compression, dense adjacency, attention and multiGPU are not evaluated.');
slide('Reproduction and next experiments',[
 'README.txt documents build and run commands.',
 'Raw CSV, hashes, plots and Nsight reports are included.',
 'The root CLI loads generated or supplied fixed weights.',
 'Next: wider output tiles, row-adaptive aggregation and more GPUs.'
],90,'Conclude with the reproducible workflow: prepare dataset; build Release; ctest; run_experiments.py; analyze.py. Preserve dataset semantics and timing definitions for new kernels. All code changes remain on feat/q1-integration. Do not infer training accuracy from throughput. Allow questions after this20minute presentation.');
const candidate=path.join(tmp,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
for(let i=0;i<presentation.slides.items.length;i++){
 const s=presentation.slides.items[i];const png=await presentation.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const finalPath=path.join(root,'docs/Q1-presentation-v2.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 explicitTotalSlideCount:13,requiredNativeChartOwnerSlides:[9,10],requiredNativeTableOwnerSlides:[],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(tmp,'validation-v2.json')});
console.log(JSON.stringify(result));
