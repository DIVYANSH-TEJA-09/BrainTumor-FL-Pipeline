/** NeuroAI Dashboard — Pipeline (Classify, Segment, Progression) */

// ═══ CLASSIFICATION ═══
async function startClassification(){
  goToPage('classify');$('#classifyEmpty').classList.add('hidden');
  $('#classifyLoading').classList.remove('hidden');$('#classifyResults').classList.add('hidden');
  try{
    const fd=new FormData();fd.append('session_id',State.sessionId);
    const r=await fetch('/api/classify',{method:'POST',body:fd});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Failed')}
    const d=await r.json();State.classificationResult=d;State.isGlioma=d.is_glioma;
    $('#classifyLoading').classList.add('hidden');$('#classifyResults').classList.remove('hidden');
    renderClassification(d);showToast(`Classified: ${d.consensus.class_name}`,'success');
  }catch(e){showToast(e.message,'error');$('#classifyLoading').innerHTML=`<div style="color:#EF4444;text-align:center;padding:2rem"><p style="font-size:1.1rem;font-weight:600">❌ Classification Failed</p><p style="color:#64748B;margin-top:.5rem">${escapeHtml(e.message)}</p><button class="btn btn-secondary" style="margin-top:1rem" onclick="location.reload()">🔄 Try Again</button></div>`}
}

function renderClassification(d){
  // Protocol banner
  const pb=$('#protocolBanner');pb.classList.add('hidden');
  if(d.brats_override){pb.classList.remove('hidden');pb.innerHTML='<strong>BraTS protocol detected:</strong> Dataset is glioma-only, tumor-type classification was safely bypassed.'}

  // Input preview
  if(d.slice_image_b64){$('#inputPreview').classList.remove('hidden');$('#classifySliceImg').src=`data:image/png;base64,${d.slice_image_b64}`;$('#classifySliceMeta').textContent=`${d.source_modality?.toUpperCase()||'MRI'} · Slice ${d.slice_index||'?'} / ${d.total_slices||'?'}`}
  else{$('#inputPreview').classList.add('hidden')}

  // Consensus
  const cs=d.consensus;
  if(cs&&!cs.error){
  const ci=CLASS_NAMES.indexOf(cs.class_name);
  $('#consensusIcon').textContent='';
    $('#consensusClass').textContent=cs.class_name;
    // Confidence bar
    const conf=parseFloat((d.ensemble||{}).confidence||0);
    const cc=getConfidenceColor(conf);
    const pct=(conf*100).toFixed(1);
    $('#confidenceBarWrap').innerHTML=`<div class="confidence-bar"><div class="confidence-fill" style="width:${pct}%;background:${cc}">${pct}%</div></div><div style="font-size:.78rem;color:#64748B;text-align:center">${conf>=0.85?'High':conf>=0.65?'Moderate':'Low'} Confidence</div>`;
    let detail=cs.unanimous?`All ${cs.total_models} models agree`:`${cs.vote_count}/${cs.total_models} models agree`;
    if(d.brats_override)detail='Protocol-based clinical safeguard applied';
    else if(Object.keys(d.results||{}).length===1)detail='Single-model inference (QPSO-FL)';
    $('#consensusDetail').textContent=detail;
  }

  // Model cards
  const mg=$('#modelResultsGrid');mg.innerHTML='';
  for(const[name,res]of Object.entries(d.results||{})){
    if(res.error)continue;
    const card=document.createElement('div');card.className='model-result-card';
    card.style.borderTop=`3px solid ${res.color||'#3B82F6'}`;
    let bars='';
    for(const[cls,prob]of Object.entries(res.probabilities||{})){
      const ci=CLASS_NAMES.indexOf(cls);const col=CLASS_COLORS[ci]||'#666';const p=(prob*100).toFixed(1);
      bars+=`<div class="prob-row"><span class="prob-label">${cls}</span><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${p}%;background:${col}"><span class="prob-bar-text">${p}%</span></div></div></div>`}
    card.innerHTML=`<div class="model-header"><span class="model-name" style="color:${res.color||'#3B82F6'}">${name}</span><span class="model-badge" style="background:${res.color}15;color:${res.color}">${res.class_name}</span></div><div class="prob-bars">${bars}</div>`;
    mg.appendChild(card);
    requestAnimationFrame(()=>{card.querySelectorAll('.prob-bar-fill').forEach(b=>{const w=b.style.width;b.style.width='0%';requestAnimationFrame(()=>{b.style.width=w})})});
  }

  // Ensemble chart
  if(d.ensemble&&!d.ensemble.error)renderEnsembleChart(d.ensemble);

  // Clinical context
  const ctx=$('#clinicalContextBody');const cn=cs?.class_name;
  if(ctx&&cn&&CLINICAL_CONTEXT[cn]){ctx.innerHTML=`<div class="clinical-context-body">${CLINICAL_CONTEXT[cn]}</div>`;$('#clinicalContextSection').classList.remove('hidden')}

  // Explainability
  renderExplainability(d);
  // Risk
  renderRisk(d);

  // Actions
  const act=$('#classifyActions');act.innerHTML='';
  if(d.is_glioma){
    const hasAll=State.modalities.t1&&State.modalities.t1ce&&State.modalities.t2&&State.modalities.flair;
    if(hasAll){
      const segBtn=document.createElement('button');segBtn.className='btn btn-success btn-lg';segBtn.id='btnGoSegment';segBtn.innerHTML='🎯 Continue to Segmentation';
      segBtn.addEventListener('click',startSegmentation);act.appendChild(segBtn);
      const hint=document.createElement('p');hint.style.cssText='color:#64748B;font-size:.82rem;margin-top:.5rem';hint.textContent='Glioma detected — proceeding to 3D segmentation';act.appendChild(hint);
    } else {
      const warn=document.createElement('div');warn.style.cssText='color:#F59E0B;font-size:.88rem;margin-top:1rem';
      warn.textContent=`⚠️ Glioma detected but all 4 modalities required for segmentation. Missing: ${['t1','t1ce','t2','flair'].filter(m=>!State.modalities[m]).join(', ').toUpperCase()}`;
      act.appendChild(warn);
    }
    const dlBtn=document.createElement('button');dlBtn.className='btn btn-secondary';dlBtn.style.marginTop='.8rem';dlBtn.id='btnDlReportClassify';dlBtn.innerHTML='📄 Download Report';
    dlBtn.addEventListener('click',()=>downloadReport());act.appendChild(dlBtn);
  }else{
    // Show non-glioma + clinical context, then allow report download
    const ngi=$('#nonGliomaIcon'),ngc=$('#nonGliomaClass'),ngctx=$('#nonGliomaClinicalContext');
    if(ngi)ngi.textContent='';if(ngc)ngc.textContent=cn;
    if(ngctx&&CLINICAL_CONTEXT[cn])ngctx.innerHTML=CLINICAL_CONTEXT[cn];
    setTimeout(()=>goToPage('nonglioma'),600);
  }
}

function renderExplainability(d){
  const sec=$('#classifyExplainSection'),ctrl=$('#classifyExplainControls'),img=$('#classifyExplainImg'),note=$('#classifyExplainNote');
  if(!sec)return;const exp=d.explainability;
  if(!exp||!exp.heatmap_b64||!exp.blend_b64||!d.slice_image_b64){sec.classList.add('hidden');return}
  sec.classList.remove('hidden');
  const modes={original:d.slice_image_b64,heatmap:exp.heatmap_b64,blend:exp.blend_b64};
  if(!modes[State.classifyExplainMode])State.classifyExplainMode='blend';
  ctrl.innerHTML='';
  [['original','Original'],['heatmap','Heatmap'],['blend','Blend']].forEach(([k,l])=>{
    const b=document.createElement('button');b.type='button';b.className=`overlay-chip ${State.classifyExplainMode===k?'active':''}`;b.textContent=l;
    b.addEventListener('click',()=>{State.classifyExplainMode=k;renderExplainability(d)});ctrl.appendChild(b)});
  img.src=`data:image/png;base64,${modes[State.classifyExplainMode]}`;
  note.textContent=State.classifyExplainMode==='blend'?'Blend overlays attribution heatmap on the original slice.':State.classifyExplainMode==='heatmap'?'Heatmap highlights regions influencing the prediction.':'Original classification slice.';
}

function renderRisk(d){
  const sec=$('#classifyRiskSection'),ul=$('#classifyInsightsList');if(!sec)return;
  const unc=d.uncertainty||{},q=d.quality_check||d.quality_overview||{},ins=d.insights||[];
  if(!unc.level&&!q.status&&!ins.length){sec.classList.add('hidden');return}
  sec.classList.remove('hidden');
  const lvl=unc.level||'unknown',ent=Number(unc.entropy_normalized||0),mar=Number(unc.margin_top1_top2||0);
  $('#uncertaintyLine').textContent=`Uncertainty: ${lvl.toUpperCase()} (entropy=${ent.toFixed(3)}, margin=${mar.toFixed(3)})${unc.review_recommended?' · manual review suggested':''}`;
  const qs=String(q.status||'review').toUpperCase(),qw=(q.warnings||[]).slice(0,2);
  $('#qualityLine').textContent=qw.length?`Quality: ${qs} · ${qw.join(' | ')}`:`Quality: ${qs}`;
  ul.innerHTML='';ins.slice(0,4).forEach(m=>{const li=document.createElement('li');li.textContent=m;ul.appendChild(li)});
}

function renderEnsembleChart(ens){
  const probs=CLASS_NAMES.map(c=>(ens.probabilities[c]||0)*100);
  Plotly.newPlot('ensembleChart',[{x:probs,y:CLASS_NAMES,type:'bar',orientation:'h',marker:{color:CLASS_COLORS},text:probs.map(p=>p.toFixed(1)+'%'),textposition:'auto',textfont:{color:'white',size:13,family:'Inter'}}],{
    height:200,margin:{l:100,r:30,t:10,b:30},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#F8FAFC',
    xaxis:{range:[0,100],title:{text:'Probability (%)',font:{color:'#64748B',size:12}},gridcolor:'#E2E8F0',color:'#64748B'},
    yaxis:{color:'#1E293B',tickfont:{size:13}},font:{family:'Inter',color:'#1E293B'}
  },{displayModeBar:false,responsive:true});
}

// ═══ SEGMENTATION ═══
async function startSegmentation(){
  goToPage('segment');$('#segmentEmpty').classList.add('hidden');
  $('#segmentLoading').classList.remove('hidden');$('#segmentResults').classList.add('hidden');
  try{
    const fd=new FormData();fd.append('session_id',State.sessionId);
    const r=await fetch('/api/segment',{method:'POST',body:fd});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Failed')}
    const d=await r.json();State.segmentationResult=d;
    $('#segmentLoading').classList.add('hidden');$('#segmentResults').classList.remove('hidden');
    renderSegmentation(d);showToast('Segmentation complete!','success');
  }catch(e){showToast(e.message,'error');$('#segmentLoading').innerHTML=`<div style="color:#EF4444;text-align:center;padding:2rem"><p style="font-weight:600">❌ Segmentation Failed</p><p style="color:#64748B;margin-top:.5rem">${escapeHtml(e.message)}</p><button class="btn btn-secondary" style="margin-top:1rem" onclick="location.reload()">🔄 Try Again</button></div>`}
}

function renderSegmentation(d){
  const sr=$('#segStatsRow');sr.innerHTML='';
  [{key:'TC',name:'Tumor Core',color:'#E74C3C'},{key:'WT',name:'Whole Tumor',color:'#F59E0B'},{key:'ET',name:'Enhancing Tumor',color:'#F97316'}].forEach(r=>{
    const v=d.volumes[r.key]||0;const p=((v/d.total_voxels)*100).toFixed(2);
    sr.innerHTML+=`<div class="stat-card" style="border-top:3px solid ${r.color}"><div class="stat-label">${r.name}</div><div class="stat-value">${v.toLocaleString()} mm³</div><div class="stat-delta neutral">${p}% of volume</div></div>`});
  // Uncertainty
  const unc=d.uncertainty_summary;
  if(unc){$('#segUncertaintyCard').classList.remove('hidden');const lvl=String(unc.level||'unknown').toUpperCase();$('#segUncertaintyText').textContent=`Level: ${lvl} (mean=${Number(unc.mean||0).toFixed(3)}, high-unc=${(Number(unc.high_uncertainty_ratio||0)*100).toFixed(1)}%)${unc.review_recommended?' · review suggested':''}`}
  // 3D + Slices
  initOverlayChips('segOverlayControls',[{key:'BRAIN',label:'Brain',dot:'⚪'},{key:'WT',label:'Whole Tumor',dot:'🟡'},{key:'TC',label:'Tumor Core',dot:'🔴'},{key:'ET',label:'Enhancing',dot:'🟠'},{key:'UNC',label:'Uncertainty',dot:'🟣'}],State.segOverlayState,()=>render3dSeg(d.mesh_data));
  render3dSeg(d.mesh_data);
  initOverlayChips('sliceOverlayControls',[{key:'WT',label:'Whole Tumor',dot:'🟡'},{key:'TC',label:'Tumor Core',dot:'🔴'},{key:'ET',label:'Enhancing',dot:'🟠'},{key:'UNC',label:'Uncertainty',dot:'🟣'}],State.sliceOverlayState,()=>renderSlices(d.slices));
  renderSlices(d.slices);
  const sl=$('#segSliceSlider');if(sl){sl.max=String((d.slices||[]).length-1);sl.value=String(Math.floor(((d.slices||[]).length-1)/2));sl.oninput=()=>renderSlices(d.slices)}
  const usl=$('#segUncertaintyOpacity');if(usl){usl.value=String(State.segUncertaintyOpacity);usl.oninput=()=>{State.segUncertaintyOpacity=Number(usl.value);render3dSeg(d.mesh_data);renderSlices(d.slices)}}
  $('#btnStartProgression').onclick=()=>startProgression();
}

function initOverlayChips(id,items,state,cb){
  const el=$(`#${id}`);if(!el)return;el.innerHTML='';
  items.forEach(it=>{const b=document.createElement('button');b.type='button';b.className=`overlay-chip ${state[it.key]?'active':''}`;b.textContent=`${it.dot} ${it.label}`;
    b.addEventListener('click',()=>{state[it.key]=!state[it.key];b.classList.toggle('active',state[it.key]);cb()});el.appendChild(b)});
}

function render3dSeg(md){
  if(!md){$('#seg3dViewer').innerHTML='<div style="padding:2rem;text-align:center;color:#94A3B8">No mesh data</div>';return}
  const traces=[];const cfg=[['WT','Whole Tumor'],['TC','Tumor Core'],['ET','Enhancing'],['BRAIN','Brain'],['UNC','Uncertainty']];
  cfg.forEach(([k,n])=>{if(!md[k]||!State.segOverlayState[k])return;const m=md[k];
    traces.push({type:'mesh3d',x:m.vertices.map(v=>v[0]),y:m.vertices.map(v=>v[1]),z:m.vertices.map(v=>v[2]),i:m.faces.map(f=>f[0]),j:m.faces.map(f=>f[1]),k:m.faces.map(f=>f[2]),color:m.color,opacity:m.opacity||(k==='UNC'?State.segUncertaintyOpacity:0.7),name:n,flatshading:true})});
  Plotly.newPlot('seg3dViewer',traces,{scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},bgcolor:'#F8FAFC'},margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'rgba(0,0,0,0)',showlegend:true,legend:{font:{family:'Inter',size:11,color:'#1E293B'}}},{displayModeBar:false,responsive:true});
}

function renderSlices(slices){
  if(!slices||!slices.length)return;
  const sl=$('#segSliceSlider');const idx=sl?parseInt(sl.value):0;const s=slices[Math.min(idx,slices.length-1)];
  const info=$('#segSliceInfo');if(info)info.textContent=`Slice ${idx+1} / ${slices.length} (axial index: ${s.index})`;
  const viewer=$('#segSliceViewer');const W=s.flair[0]?.length||0,H=s.flair.length||0;
  let canvas=viewer.querySelector('canvas');
  if(!canvas){canvas=document.createElement('canvas');viewer.innerHTML='';viewer.appendChild(canvas)}
  canvas.width=W;canvas.height=H;canvas.style.width='100%';canvas.style.height='100%';canvas.style.objectFit='contain';
  const ctx=canvas.getContext('2d');const img=ctx.createImageData(W,H);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){const i=(y*W+x)*4;const v=s.flair[y][x];img.data[i]=v;img.data[i+1]=v;img.data[i+2]=v;img.data[i+3]=255;
    // Overlays: WT=yellow, TC=red, ET=orange
    if(State.sliceOverlayState.WT&&s.wt[y]?.[x]){img.data[i]=255;img.data[i+1]=213;img.data[i+2]=0;img.data[i+3]=200}
    if(State.sliceOverlayState.TC&&s.tc[y]?.[x]){img.data[i]=255;img.data[i+1]=0;img.data[i+2]=0;img.data[i+3]=220}
    if(State.sliceOverlayState.ET&&s.et[y]?.[x]){img.data[i]=249;img.data[i+1]=115;img.data[i+2]=22;img.data[i+3]=220}
    if(State.sliceOverlayState.UNC&&s.uncertainty){const u=s.uncertainty[y]?.[x]||0;if(u>150){img.data[i]=192;img.data[i+1]=132;img.data[i+2]=252;img.data[i+3]=Math.min(255,u+80)}}
  }
  ctx.putImageData(img,0,0);
}

// ═══ PROGRESSION ═══
async function startProgression(grade='HGG'){
  goToPage('progression');$('#progressionEmpty').classList.add('hidden');
  $('#progressionLoading').classList.remove('hidden');$('#progressionResults').classList.add('hidden');
  try{
    const fd=new FormData();fd.append('session_id',State.sessionId);fd.append('grade',grade);
    const r=await fetch('/api/progression',{method:'POST',body:fd});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Failed')}
    const d=await r.json();State.progressionResult=d;
    $('#progressionLoading').classList.add('hidden');$('#progressionResults').classList.remove('hidden');
    renderProgression(d);showToast('Progression analysis complete!','success');
  }catch(e){showToast(e.message,'error');$('#progressionLoading').innerHTML=`<div style="color:#EF4444;text-align:center;padding:2rem"><p style="font-weight:600">❌ Progression Failed</p><p style="color:#64748B;margin-top:.5rem">${escapeHtml(e.message)}</p></div>`}
}

function renderProgression(d){
  // Stats
  const sr=$('#progStatsRow');sr.innerHTML='';
  sr.innerHTML=`<div class="stat-card" style="border-top:3px solid #3B82F6"><div class="stat-label">Current Volume</div><div class="stat-value">${Math.round(d.current_volume).toLocaleString()} mm³</div><div class="stat-delta neutral">${d.grade} assumption</div></div>`;

  // RANO status from 90-day projection
  const proj=d.logistic?.projections||[];
  const p90=proj.find(p=>p.day===90);
  if(p90){
    const rs=$('#ranoSection');rs.classList.remove('hidden');
    const[code,color,label]=getRano(p90.growth_pct);
    $('#ranoBadge').textContent=`${code} — ${label}`;$('#ranoBadge').style.background=color;
    const ra=$('#riskAlert');
    if(p90.growth_pct>25){ra.classList.remove('hidden');ra.textContent=`⚠️ Projected growth of ${p90.growth_pct.toFixed(1)}% in 3 months exceeds 25% threshold — qualifies as Progressive Disease (PD) under RANO criteria.`}
    else{ra.classList.add('hidden')}
  }

  // Growth curve
  const curve=d.logistic?.curve;
  if(curve){
    Plotly.newPlot('growthCurveChart',[{x:curve.days,y:curve.volumes,type:'scatter',mode:'lines',line:{color:'#3B82F6',width:2.5},name:'Projected',fill:'tozeroy',fillcolor:'rgba(59,130,246,0.08)'},{x:[0],y:[d.current_volume],type:'scatter',mode:'markers',marker:{color:'#EF4444',size:10,symbol:'circle'},name:'Current'}],{
      margin:{l:70,r:30,t:20,b:50},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#FAFBFC',
      xaxis:{title:{text:'Days',font:{color:'#64748B'}},gridcolor:'#E2E8F0',color:'#64748B'},
      yaxis:{title:{text:'Volume (mm³)',font:{color:'#64748B'}},gridcolor:'#E2E8F0',color:'#64748B'},
      font:{family:'Inter',color:'#1E293B'},showlegend:true,legend:{font:{size:11}}
    },{displayModeBar:false,responsive:true});
  }

  // Projections table
  const tb=$('#projectionsBody');tb.innerHTML='';
  proj.forEach(p=>{
    const[code,color]=getRano(p.growth_pct);
    const dayLabel=p.day<=30?`${p.day} days (1 mo)`:p.day<=90?`${p.day} days (3 mo)`:p.day<=180?`${p.day} days (6 mo)`:`${p.day} days (1 yr)`;
    tb.innerHTML+=`<tr><td>${dayLabel}</td><td>${Math.round(p.volume).toLocaleString()} mm³</td><td style="color:${p.growth_pct>=0?'#EF4444':'#22C55E'}">${p.growth_pct>=0?'+':''}${p.growth_pct.toFixed(1)}%</td><td><span class="rano-badge" style="background:${color};font-size:.72rem;padding:2px 8px">${code}</span></td></tr>`});

  // Spatial
  if(d.spatial&&d.spatial.mesh_data){
    const ss=$('#spatialSection');ss.classList.remove('hidden');
    const st=d.spatial.stats||{};
    $('#spatialStats').innerHTML=`<div class="stat-card" style="border-top:3px solid #3B82F6"><div class="stat-label">Stable</div><div class="stat-value">${(st.stable_voxels||0).toLocaleString()}</div></div><div class="stat-card" style="border-top:3px solid #EF4444"><div class="stat-label">Growth</div><div class="stat-value">${(st.growth_voxels||0).toLocaleString()}</div></div><div class="stat-card" style="border-top:3px solid #22C55E"><div class="stat-label">Regression</div><div class="stat-value">${(st.regression_voxels||0).toLocaleString()}</div></div>`;
    initOverlayChips('spatialOverlayControls',[{key:'stable',label:'Stable',dot:'🔵'},{key:'growth',label:'Growth',dot:'🔴'},{key:'regression',label:'Regression',dot:'🟢'},{key:'envelope',label:'Envelope',dot:'⚪'},{key:'brain',label:'Brain',dot:'⚪'}],State.spatialOverlayState,()=>renderSpatial3d(d.spatial.mesh_data));
    renderSpatial3d(d.spatial.mesh_data);
  }

  // Insights
  const el=d.explainability||{};const dr=el.drivers||{};
  $('#progDriverLine').textContent=dr.growth_rate_r_per_day?`Growth rate: ${dr.growth_rate_r_per_day.toFixed(4)}/day · Carrying capacity: ${Math.round(dr.carrying_capacity_mm3||0).toLocaleString()} mm³`:'';
  const sb=el.spatial_balance||{};
  $('#progSpatialLine').textContent=sb.volume_change_pct!==undefined?`Spatial volume change: ${sb.volume_change_pct.toFixed(1)}%`:'';
  const il=$('#progInsightsList');il.innerHTML='';(d.insights||[]).slice(0,4).forEach(m=>{const li=document.createElement('li');li.textContent=m;il.appendChild(li)});

  // Report download
  $('#btnDownloadReport').onclick=()=>downloadReport();
}

function renderSpatial3d(md){
  if(!md)return;const traces=[];
  Object.entries(md).forEach(([k,m])=>{if(!m||!State.spatialOverlayState[k])return;
    traces.push({type:'mesh3d',x:m.vertices.map(v=>v[0]),y:m.vertices.map(v=>v[1]),z:m.vertices.map(v=>v[2]),i:m.faces.map(f=>f[0]),j:m.faces.map(f=>f[1]),k:m.faces.map(f=>f[2]),color:m.color,opacity:m.opacity||0.7,name:k,flatshading:true})});
  Plotly.newPlot('spatial3dViewer',traces,{scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},bgcolor:'#F8FAFC'},margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'rgba(0,0,0,0)',showlegend:true,legend:{font:{family:'Inter',size:11,color:'#1E293B'}}},{displayModeBar:false,responsive:true});
}

// ═══ GRADE SELECTOR ═══
function initGradeSelector(){
  $$('.grade-btn').forEach(b=>{b.addEventListener('click',()=>{$$('.grade-btn').forEach(g=>g.classList.remove('active'));b.classList.add('active');startProgression(b.dataset.grade)})});
}

// ═══ REPORT DOWNLOAD ═══
async function downloadReport(){
  if(!State.sessionId)return showToast('No session','error');
  try{
    showToast('Generating report...','info');
    const meta=getPatientMetadata();
    const qs=new URLSearchParams();
    if(meta.patient_id)qs.set('patient_id',meta.patient_id);
    if(meta.age)qs.set('age',meta.age);
    if(meta.sex)qs.set('sex',meta.sex);
    if(meta.scan_date)qs.set('scan_date',meta.scan_date);
    if(meta.notes)qs.set('notes',meta.notes);
    const url=`/api/report/${State.sessionId}${qs.toString()?`?${qs.toString()}`:''}`;
    const res=await fetch(url);
    if(!res.ok)throw new Error('Report generation failed');
    const blob=await res.blob();
    const blobUrl=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=blobUrl;
    a.download=`NeuroAI_Clinical_Report_${State.sessionId.slice(0,8)}.pdf`;
    document.body.appendChild(a);a.click();
    setTimeout(()=>{document.body.removeChild(a);URL.revokeObjectURL(blobUrl)},100);
    showToast('Report downloaded!','success');
  }catch(e){showToast(e.message,'error')}
}
