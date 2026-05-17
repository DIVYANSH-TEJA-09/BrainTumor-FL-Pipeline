/** NeuroAI Dashboard — Core (Navigation, Upload, Utils) */
const $=s=>document.querySelector(s),$$=s=>document.querySelectorAll(s);
const State={currentPage:'upload',sessionId:null,modalities:{},classificationResult:null,segmentationResult:null,progressionResult:null,isGlioma:false,classifyExplainMode:'blend',segOverlayState:{BRAIN:true,WT:true,TC:true,ET:true,UNC:true},sliceOverlayState:{WT:true,TC:true,ET:true,UNC:false},spatialOverlayState:{envelope:true,brain:true,stable:true,growth:true,regression:true},segUncertaintyOpacity:0.55};
const CLINICAL_CONTEXT={Glioma:"High-grade gliomas are aggressive. Immediate neurosurgery referral, MRI contrast enhancement, molecular profiling (IDH1, MGMT) recommended. RANO criteria apply for follow-up.",Meningioma:"Usually benign. Watchful waiting or surgical resection depending on size and symptoms. Annual MRI follow-up standard.",Pituitary:"Evaluate for hormonal dysfunction (prolactin, GH, ACTH). Ophthalmology referral if visual field defects. Transsphenoidal surgery if indicated.","No Tumor":"No tumor detected. If clinical suspicion persists, consider repeat imaging with contrast enhancement or alternative modalities."};
const CLASS_NAMES=['Glioma','Meningioma','No Tumor','Pituitary'];
const CLASS_COLORS=['#E74C3C','#3498DB','#2ECC71','#9B59B6'];
const CLASS_ICONS=['🔴','🔵','🟢','🟣'];
function escapeHtml(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML}
function getConfidenceColor(c){return c>=0.85?'#22C55E':c>=0.65?'#F59E0B':'#EF4444'}
function getRano(g){if(g<=-100)return['CR','#22C55E','Complete Response'];if(g<=-25)return['PR','#3B82F6','Partial Response'];if(g<=25)return['SD','#F59E0B','Stable Disease'];return['PD','#EF4444','Progressive Disease']}

// ═══ INIT ═══
document.addEventListener('DOMContentLoaded',()=>{initNav();initUpload();initHealth();initGradeSelector();initSamples()});

// ═══ NAVIGATION ═══
function initNav(){
  $$('.nav-item').forEach(btn=>{btn.addEventListener('click',()=>goToPage(btn.dataset.page))});
  const r=$('#btnRestart');if(r)r.addEventListener('click',async()=>{await cleanupSession();location.reload()});
}
function goToPage(page){
  State.currentPage=page;
  $$('.page').forEach(p=>p.classList.remove('page-active'));
  const el=$(`.page[data-page="${page}"]`);if(el)el.classList.add('page-active');
  $$('.nav-item').forEach(n=>{n.classList.remove('active');if(n.dataset.page===page)n.classList.add('active')});
  window.scrollTo({top:0,behavior:'smooth'});
}

// ═══ TOAST ═══
function showToast(msg,type='info'){
  const c=$('#toastContainer'),t=document.createElement('div');
  const labels={success:'Success',error:'Error',info:'Info',warning:'Warning'};
  t.className=`toast toast-${type}`;
  t.innerHTML=`<span class="toast-dot"></span><span class="toast-label">${labels[type]||'Info'}</span><span>${msg}</span>`;
  c.appendChild(t);setTimeout(()=>t.remove(),4000);
}

// ═══ HEALTH ═══
async function initHealth(){
  const el=$('#healthItems');if(!el)return;
  try{const r=await fetch('/api/health');const d=await r.json();const m=d.models||{};
    el.innerHTML=[['classification/QPSO-FL','Classification'],['segmentation','Segmentation'],['spatial_unet','Spatial U-Net']].map(([k,l])=>{
      const ok=!!m[k];return`<span class="health-item ${ok?'ok':'fail'}"><span class="status-dot"></span>${l}</span>`}).join('');
  }catch(e){el.innerHTML='<span class="health-item fail">❌ Unavailable</span>'}
}

// ═══ SAMPLE PATIENTS ═══
async function initSamples(){
  const grid=$('#sampleGrid');if(!grid)return;
  try{
    const r=await fetch('/api/samples');const d=await r.json();
    const patients=d.patients||[];
    if(!patients.length){$('#samplePatientsCard').classList.add('hidden');$('#orDivider').classList.add('hidden');return}
    grid.innerHTML='';
    patients.forEach(p=>{
      const card=document.createElement('div');
      card.className='sample-card';
      card.id=`sample-${p.id}`;
      const gradeClass=p.grade==='LGG'?'lgg':'hgg';
      card.innerHTML=`
        <div class="sample-card-icon">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
        </div>
        <div class="sample-card-body">
          <div class="sample-card-name">${escapeHtml(p.patient_id)} — ${escapeHtml(p.notes?.split('.')[0]||'Glioma')}</div>
          <div class="sample-card-meta">
            <span>${p.age||'?'}y ${p.sex||''}</span>
            <span>${p.scan_date||''}</span>
            <span>${p.file_count||0} modalities</span>
          </div>
        </div>
        <span class="sample-card-badge ${gradeClass}">${p.grade||'HGG'}</span>
      `;
      card.addEventListener('click',()=>loadSample(p));
      grid.appendChild(card);
    });
  }catch(e){
    grid.innerHTML='<div class="sample-loading">Database unavailable</div>';
  }
}

async function loadSample(patient){
  const card=$(`#sample-${patient.id}`);
  if(card)card.classList.add('loading');
  // Hide the upload zone + sample section, show progress
  $('#uploadZone').classList.add('hidden');
  $('#orDivider').classList.add('hidden');
  $('#samplePatientsCard').classList.add('hidden');
  $('#uploadProgress').classList.remove('hidden');
  $('#uploadProgressFill').style.width='20%';
  $('#uploadProgressText').textContent=`Loading ${patient.patient_id} from database...`;

  try{
    const fd=new FormData();fd.append('sample_id',patient.id);
    $('#uploadProgressFill').style.width='50%';
    $('#uploadProgressText').textContent='Querying patient record...';
    const r=await fetch('/api/samples/load',{method:'POST',body:fd});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Failed to load sample')}
    const d=await r.json();
    $('#uploadProgressFill').style.width='100%';
    $('#uploadProgressText').textContent='Record loaded!';
    State.sessionId=d.session_id;State.modalities=d.modalities||{};

    // Auto-fill patient metadata
    if(patient.patient_id)$('#patientId').value=patient.patient_id;
    if(patient.age)$('#patientAge').value=patient.age;
    if(patient.sex)$('#patientSex').value=patient.sex;
    if(patient.scan_date)$('#scanDate').value=patient.scan_date;
    if(patient.notes)$('#clinicalNotes').value=patient.notes;

    setTimeout(()=>{$('#uploadProgress').classList.add('hidden');showFileAnalysis(d);},400);
    showToast(`Loaded ${patient.patient_id} from database`,'success');
  }catch(e){
    showToast(e.message,'error');
    $('#uploadProgress').classList.add('hidden');
    $('#uploadZone').classList.remove('hidden');
    $('#orDivider').classList.remove('hidden');
    $('#samplePatientsCard').classList.remove('hidden');
    if(card)card.classList.remove('loading');
  }
}

// ═══ UPLOAD ═══
function initUpload(){
  const z=$('#uploadZone'),inp=$('#fileInput');
  z.addEventListener('dragover',e=>{e.preventDefault();z.classList.add('dragover')});
  z.addEventListener('dragleave',()=>z.classList.remove('dragover'));
  z.addEventListener('drop',e=>{e.preventDefault();z.classList.remove('dragover');handleFiles(e.dataTransfer.files)});
  z.addEventListener('click',()=>inp.click());
  inp.addEventListener('change',e=>handleFiles(e.target.files));
  $('#btnAnalyze').addEventListener('click',startClassification);
}
async function handleFiles(fl){
  if(!fl||!fl.length)return;
  const fd=new FormData();for(const f of fl)fd.append('files',f);
  $('#uploadZone').classList.add('hidden');$('#orDivider').classList.add('hidden');$('#samplePatientsCard').classList.add('hidden');$('#uploadProgress').classList.remove('hidden');
  try{
    $('#uploadProgressFill').style.width='30%';$('#uploadProgressText').textContent='Uploading...';
    const r=await fetch('/api/upload',{method:'POST',body:fd});
    $('#uploadProgressFill').style.width='80%';$('#uploadProgressText').textContent='Analyzing...';
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Upload failed')}
    const d=await r.json();$('#uploadProgressFill').style.width='100%';$('#uploadProgressText').textContent='Done!';
    State.sessionId=d.session_id;State.modalities=d.modalities||{};
    setTimeout(()=>{$('#uploadProgress').classList.add('hidden');showFileAnalysis(d)},400);
    showToast('Files uploaded','success');
  }catch(e){showToast(e.message,'error');$('#uploadProgress').classList.add('hidden');$('#uploadZone').classList.remove('hidden');$('#orDivider').classList.remove('hidden');$('#samplePatientsCard').classList.remove('hidden')}
}
function showFileAnalysis(d){
  $('#fileAnalysis').classList.remove('hidden');
  const g=$('#modalityGrid');g.innerHTML='';
  // Show source tag if loaded from database
  if(d.source==='sample_database'){
    const tag=document.createElement('div');tag.className='sample-source-tag';
    tag.innerHTML=`<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M20 5v6c0 1.66-3.58 3-8 3s-8-1.34-8-3V5"/><path d="M20 11v6c0 1.66-3.58 3-8 3s-8-1.34-8-3v-6"/></svg> Loaded from Clinical Database`;
    g.parentElement.insertBefore(tag,g);
  }
  [{key:'t1',name:'T1',icon:'🔵'},{key:'t1ce',name:'T1ce',icon:'🟣'},{key:'t2',name:'T2',icon:'🔷'},{key:'flair',name:'FLAIR',icon:'🟢'},{key:'seg',name:'Seg',icon:'🎯'}].forEach(m=>{
    const det=d.modalities&&d.modalities[m.key];
    const c=document.createElement('div');c.className=`modality-card ${det?'detected':(m.key==='seg'?'':'missing')}`;
    c.innerHTML=`<div class="modality-status">${det?'✅':(m.key==='seg'?'◽':'❌')}</div><div class="modality-name">${m.icon} ${m.name}</div><div class="modality-desc">${det?'Detected':(m.key==='seg'?'Optional':'Missing')}</div>`;
    g.appendChild(c)});
  const hasImg=d.files&&d.files.some(f=>f.match(/\.(jpg|jpeg|png)$/i));
  const btn=$('#btnAnalyze');btn.disabled=!(d.has_all_required||d.modalities?.flair||d.modalities?.t1ce||hasImg);
  renderUploadQuality(d);
}
function renderUploadQuality(d){
  const p=$('#uploadQualitySummary'),l=$('#uploadQualityLine'),ul=$('#uploadQualityWarnings');
  const o=d.quality_overview;if(!o){p.classList.add('hidden');return}
  p.classList.remove('hidden');l.textContent=`Status: ${String(o.status||'review').toUpperCase()} · Modalities: ${o.modalities_checked||0}`;
  ul.innerHTML='';(o.warnings||[]).slice(0,6).forEach(w=>{const li=document.createElement('li');li.textContent=w;ul.appendChild(li)});
}
function getPatientMetadata(){
  return{patient_id:$('#patientId')?.value||'',age:$('#patientAge')?.value||'',sex:$('#patientSex')?.value||'',scan_date:$('#scanDate')?.value||'',notes:$('#clinicalNotes')?.value||''}
}
async function cleanupSession(){if(State.sessionId)try{await fetch(`/api/session/${State.sessionId}`,{method:'DELETE'})}catch(e){}}

