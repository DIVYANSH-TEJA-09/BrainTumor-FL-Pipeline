/**
 * NeuroAI Dashboard — Main Application Controller
 * ==================================================
 * SPA controller managing the full pipeline:
 * Upload → Classify → Segment → Progression
 */

// ═══════════════ STATE ═══════════════
const State = {
    currentStep: 0,
    sessionId: null,
    modalities: {},
    classificationResult: null,
    segmentationResult: null,
    progressionResult: null,
    isGlioma: false,
    classificationExplainMode: 'blend',
    segOverlayState: { BRAIN: true, WT: true, TC: true, ET: true, UNC: true },
    sliceOverlayState: { WT: true, TC: true, ET: true, UNC: false },
    spatialOverlayState: { envelope: true, brain: true, stable: true, growth: true, regression: true },
    segUncertaintyOpacity: 0.55,
};

// ═══════════════ DOM REFS ═══════════════
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ═══════════════ INIT ═══════════════
document.addEventListener('DOMContentLoaded', () => {
    initHealthPanel();
    initUpload();
    initNavigation();
    initGradeSelector();
});

async function initHealthPanel() {
    const items = $('#healthItems');
    if (!items) return;

    items.innerHTML = '<span class="health-item warn">⏳ Checking services...</span>';
    try {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error('Health check failed');
        const data = await res.json();
        const models = data.models || {};

        const entries = [
            ['classification/QPSO-FL', 'Classification'],
            ['segmentation', 'Segmentation'],
            ['spatial_unet', 'Spatial U-Net'],
        ];

        items.innerHTML = entries.map(([key, label]) => {
            const ok = !!models[key];
            const cls = ok ? 'ok' : 'fail';
            const icon = ok ? '✅' : '❌';
            return `<span class="health-item ${cls}">${icon} ${label}</span>`;
        }).join('');
    } catch (err) {
        items.innerHTML = '<span class="health-item fail">❌ Health check unavailable</span>';
    }
}

// ═══════════════ NAVIGATION ═══════════════
function initNavigation() {
    // Restart button
    const btnRestart = $('#btnRestart');
    if (btnRestart) {
        btnRestart.addEventListener('click', async () => {
            await cleanupSession();
            location.reload();
        });
    }
}

function goToStep(step) {
    State.currentStep = step;

    // Update panels
    $$('.panel').forEach(p => p.classList.remove('panel-active'));

    if (step === 99) {
        // Non-glioma result
        $('#panelNonGlioma').classList.add('panel-active');
    } else {
        const panels = ['panelUpload', 'panelClassify', 'panelSegment', 'panelProgression'];
        const panel = $(`#${panels[step]}`);
        if (panel) panel.classList.add('panel-active');
    }

    // Update nav steps
    $$('.step-item').forEach((item, idx) => {
        item.classList.remove('active', 'completed');
        if (idx < step) item.classList.add('completed');
        if (idx === step) item.classList.add('active');
    });

    $$('.step-connector').forEach((conn, idx) => {
        conn.classList.toggle('active', idx < step);
    });

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setStatus(text, type = 'normal') {
    const dot = $('.status-dot');
    const txt = $('.status-text');
    txt.textContent = text;
    dot.style.background = type === 'error' ? 'var(--accent-red)' :
                           type === 'working' ? 'var(--accent-orange)' :
                           'var(--accent-green)';
}

// ═══════════════ TOAST NOTIFICATIONS ═══════════════
function showToast(message, type = 'info') {
    const container = $('#toastContainer');
    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || 'ℹ️'}</span>
        <span class="toast-msg">${message}</span>
    `;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ═══════════════ UPLOAD ═══════════════
function initUpload() {
    const zone = $('#uploadZone');
    const input = $('#fileInput');

    // Drag & Drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    // Click to browse
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => handleFiles(e.target.files));

    // Classification button
    $('#btnStartClassification').addEventListener('click', startClassification);
}

async function handleFiles(fileList) {
    if (!fileList || fileList.length === 0) return;

    const formData = new FormData();
    for (const file of fileList) {
        formData.append('files', file);
    }

    // Show progress
    $('#uploadZone').classList.add('hidden');
    $('#uploadProgress').classList.remove('hidden');
    setStatus('Uploading...', 'working');

    try {
        const fill = $('#uploadProgressFill');
        const text = $('#uploadProgressText');

        fill.style.width = '30%';
        text.textContent = 'Uploading files...';

        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        fill.style.width = '80%';
        text.textContent = 'Analyzing files...';

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await res.json();
        fill.style.width = '100%';
        text.textContent = 'Upload complete!';

        State.sessionId = data.session_id;
        State.modalities = data.modalities || {};

        // Show analysis
        setTimeout(() => {
            $('#uploadProgress').classList.add('hidden');
            showFileAnalysis(data);
        }, 500);

        setStatus('Files uploaded', 'normal');
        showToast('Files uploaded successfully', 'success');

    } catch (err) {
        setStatus('Upload failed', 'error');
        showToast(err.message, 'error');
        $('#uploadProgress').classList.add('hidden');
        $('#uploadZone').classList.remove('hidden');
    }
}

function showFileAnalysis(data) {
    $('#fileAnalysis').classList.remove('hidden');

    const grid = $('#modalityGrid');
    grid.innerHTML = '';

    const allMods = [
        { key: 't1', name: 'T1', desc: 'Pre-contrast structural', icon: '🔵' },
        { key: 't1ce', name: 'T1ce', desc: 'Contrast-enhanced', icon: '🟣' },
        { key: 't2', name: 'T2', desc: 'Edema visualization', icon: '🔷' },
        { key: 'flair', name: 'FLAIR', desc: 'Fluid attenuated', icon: '🟢' },
        { key: 'seg', name: 'Seg Mask', desc: 'Ground truth (optional)', icon: '🎯' },
    ];

    allMods.forEach(mod => {
        const detected = data.modalities && data.modalities[mod.key];
        const card = document.createElement('div');
        card.className = `modality-card ${detected ? 'detected' : (mod.key === 'seg' ? '' : 'missing')}`;
        card.innerHTML = `
            <div class="modality-status">${detected ? '✅' : (mod.key === 'seg' ? '◽' : '❌')}</div>
            <div class="modality-name">${mod.icon} ${mod.name}</div>
            <div class="modality-desc">${detected ? 'Detected' : (mod.key === 'seg' ? 'Optional' : 'Missing')}</div>
        `;
        grid.appendChild(card);
    });

    // Also check for regular image files in the analysis
    const hasImages = data.files && data.files.some(f =>
        f.match(/\.(jpg|jpeg|png)$/i)
    );

    // Enable button if we have at least FLAIR or any image
    const btn = $('#btnStartClassification');
    const canProceed = data.has_all_required || data.modalities?.flair || data.modalities?.t1ce || hasImages;
    btn.disabled = !canProceed;

    if (!canProceed) {
        const msg = document.createElement('p');
        msg.style.cssText = 'color: var(--accent-red); font-size: 0.85rem; margin-top: 1rem; text-align: center;';
        msg.textContent = 'Please upload at least one MRI modality (NIfTI format) or a brain MRI image (JPG/PNG).';
        grid.after(msg);
    }

    renderUploadQualitySummary(data);
}

function renderUploadQualitySummary(data) {
    const panel = $('#uploadQualitySummary');
    const line = $('#uploadQualityLine');
    const list = $('#uploadQualityWarnings');
    if (!panel || !line || !list) return;

    const overview = data.quality_overview;
    if (!overview) {
        panel.classList.add('hidden');
        return;
    }

    panel.classList.remove('hidden');
    const status = String(overview.status || 'review').toUpperCase();
    const checked = Number(overview.modalities_checked || 0);
    line.textContent = `Status: ${status} · Modalities checked: ${checked}`;

    list.innerHTML = '';
    const warnings = overview.warnings || [];
    if (warnings.length === 0) {
        const li = document.createElement('li');
        li.textContent = 'No quality warnings detected.';
        list.appendChild(li);
        return;
    }

    for (const w of warnings.slice(0, 6)) {
        const li = document.createElement('li');
        li.textContent = w;
        list.appendChild(li);
    }
}

// ═══════════════ CLASSIFICATION ═══════════════
async function startClassification() {
    goToStep(1);
    setStatus('Classifying...', 'working');

    // Show loading
    $('#classifyLoading').classList.remove('hidden');
    $('#classifyResults').classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('session_id', State.sessionId);

        const res = await fetch('/api/classify', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Classification failed');
        }

        const data = await res.json();
        State.classificationResult = data;
        State.isGlioma = data.is_glioma;

        // Hide loading, show results
        $('#classifyLoading').classList.add('hidden');
        $('#classifyResults').classList.remove('hidden');

        renderClassificationResults(data);
        setStatus('Classification complete');
        showToast(`Classified as: ${data.consensus.class_name}`, 'success');

    } catch (err) {
        setStatus('Classification failed', 'error');
        showToast(err.message, 'error');
        $('#classifyLoading').innerHTML = `
            <div style="color: var(--accent-red); text-align: center; padding: 2rem;">
                <p style="font-size: 1.2rem; font-weight: 600;">❌ Classification Failed</p>
                <p style="color: var(--text-secondary); margin-top: 0.5rem;">${escapeHtml(err.message)}</p>
                <button class="btn btn-secondary" style="margin-top: 1rem;" onclick="location.reload()">
                    🔄 Try Again
                </button>
            </div>
        `;
    }
}

function renderClassificationResults(data) {
    const protocolBanner = $('#protocolBanner');
    protocolBanner.classList.add('hidden');
    protocolBanner.innerHTML = '';

    if (data.brats_override) {
        protocolBanner.classList.remove('hidden');
        protocolBanner.innerHTML = '<strong>BraTS protocol detected:</strong> Dataset is glioma-only, so tumor-type classification was safely bypassed.';
    }

    // Input preview
    if (data.slice_image_b64) {
        $('#inputPreview').classList.remove('hidden');
        const img = $('#classifySliceImg');
        img.src = `data:image/png;base64,${data.slice_image_b64}`;
        const meta = $('#classifySliceMeta');
        meta.textContent = `${data.source_modality?.toUpperCase() || 'MRI'} · Slice ${data.slice_index || '?'} / ${data.total_slices || '?'}`;
    } else {
        $('#inputPreview').classList.add('hidden');
    }

    renderClassificationExplainability(data);
    renderClassificationRisk(data);

    // Consensus
    const consensus = data.consensus;
    if (consensus && !consensus.error) {
        const classColors = { 'Glioma': '#E74C3C', 'Meningioma': '#3498DB', 'No Tumor': '#2ECC71', 'Pituitary': '#9B59B6' };
        const classIcons = { 'Glioma': '🔴', 'Meningioma': '🔵', 'No Tumor': '🟢', 'Pituitary': '🟣' };

        $('#consensusIcon').textContent = classIcons[consensus.class_name] || '🧠';
        $('#consensusClass').textContent = consensus.class_name;
        $('#consensusClass').style.background = `linear-gradient(135deg, ${classColors[consensus.class_name] || '#667eea'}, #764ba2)`;
        $('#consensusClass').style.webkitBackgroundClip = 'text';
        $('#consensusDetail').textContent = consensus.unanimous
            ? `All ${consensus.total_models} models agree`
            : `${consensus.vote_count}/${consensus.total_models} models agree (majority vote)`;
    }

    // Model cards
    const grid = $('#modelResultsGrid');
    grid.innerHTML = '';
    const modelColors = { 'QPSO-FL': '#2ca02c', 'FedAvg': '#1f77b4', 'FedProx': '#ff7f0e' };
    const classNames = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'];
    const classColors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'];

    for (const [name, result] of Object.entries(data.results)) {
        if (result.error) continue;

        const card = document.createElement('div');
        card.className = 'model-result-card';
        card.style.borderTop = `3px solid ${modelColors[name] || '#666'}`;

        let probBars = '';
        for (const [cls, prob] of Object.entries(result.probabilities)) {
            const idx = classNames.indexOf(cls);
            const color = classColors[idx] || '#666';
            const pct = (prob * 100).toFixed(1);
            probBars += `
                <div class="prob-row">
                    <span class="prob-label">${cls}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:${pct}%;background:${color};">
                            <span class="prob-bar-text">${pct}%</span>
                        </div>
                    </div>
                </div>
            `;
        }

        card.innerHTML = `
            <div class="model-header">
                <span class="model-name" style="color:${modelColors[name]}">${name}</span>
                <span class="model-badge" style="background:${result.color}20;color:${result.color}">
                    ${result.icon} ${result.class_name}
                </span>
            </div>
            <div class="prob-bars">${probBars}</div>
        `;
        grid.appendChild(card);

        // Animate bars
        requestAnimationFrame(() => {
            card.querySelectorAll('.prob-bar-fill').forEach(bar => {
                const w = bar.style.width;
                bar.style.width = '0%';
                requestAnimationFrame(() => { bar.style.width = w; });
            });
        });
    }

    // Ensemble chart
    if (data.ensemble && !data.ensemble.error) {
        renderEnsembleChart(data.ensemble);
    }

    // Action buttons
    const actions = $('#classifyActions');
    actions.innerHTML = '';

    const classCount = Object.keys(data.results || {}).length;
    let detailText = consensus.unanimous
        ? `All ${consensus.total_models} models agree`
        : `${consensus.vote_count}/${consensus.total_models} models agree (majority vote)`;
    if (classCount === 1 && !data.brats_override) {
        detailText = 'Single-model inference (QPSO-FL)';
    }
    if (data.brats_override) {
        detailText = 'Protocol-based clinical safeguard applied';
    }
    $('#consensusDetail').textContent = detailText;

    if (data.is_glioma) {
        // Check if we have all modalities for segmentation
        const hasAllMods = State.modalities.t1 && State.modalities.t1ce &&
                          State.modalities.t2 && State.modalities.flair;
        if (hasAllMods) {
            actions.innerHTML = `
                <button class="btn btn-success btn-lg" id="btnGoSegment">
                    <span class="btn-icon">🎯</span>
                    Continue to Segmentation
                </button>
                <p style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.5rem;">
                    Glioma detected — proceeding to 3D tumor segmentation
                </p>
            `;
            $('#btnGoSegment').addEventListener('click', startSegmentation);
        } else {
            actions.innerHTML = `
                <div style="color:var(--accent-orange);font-size:0.9rem;margin-top:1rem;">
                    ⚠️ Glioma detected, but all 4 modalities (T1, T1ce, T2, FLAIR) are required for segmentation.
                    <br>Missing: ${['t1','t1ce','t2','flair'].filter(m => !State.modalities[m]).join(', ').toUpperCase()}
                </div>
            `;
        }
    } else {
        // Non-glioma: Show final result
        setTimeout(() => showNonGliomaResult(data), 800);
    }
}

function renderClassificationExplainability(data) {
    const section = $('#classifyExplainSection');
    const controls = $('#classifyExplainControls');
    const img = $('#classifyExplainImg');
    const note = $('#classifyExplainNote');
    if (!section || !controls || !img || !note) return;

    const exp = data.explainability;
    const hasExplain = exp && exp.heatmap_b64 && exp.blend_b64;
    if (!hasExplain || !data.slice_image_b64) {
        section.classList.add('hidden');
        controls.innerHTML = '';
        return;
    }

    const modes = {
        original: data.slice_image_b64,
        heatmap: exp.heatmap_b64,
        blend: exp.blend_b64,
    };

    if (!modes[State.classificationExplainMode]) {
        State.classificationExplainMode = 'blend';
    }

    section.classList.remove('hidden');
    controls.innerHTML = '';

    const labels = [
        ['original', 'Original'],
        ['heatmap', 'Heatmap'],
        ['blend', 'Blend'],
    ];

    for (const [key, label] of labels) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `overlay-chip ${State.classificationExplainMode === key ? 'active' : ''}`;
        btn.textContent = label;
        btn.addEventListener('click', () => {
            State.classificationExplainMode = key;
            renderClassificationExplainability(data);
        });
        controls.appendChild(btn);
    }

    img.src = `data:image/png;base64,${modes[State.classificationExplainMode]}`;
    note.textContent = State.classificationExplainMode === 'original'
        ? 'Original classification slice.'
        : State.classificationExplainMode === 'heatmap'
            ? 'Heatmap highlights regions that most influenced the prediction.'
            : 'Blend overlays attribution heatmap on the original slice.';
}

function renderClassificationRisk(data) {
    const section = $('#classifyRiskSection');
    const uncLine = $('#uncertaintyLine');
    const qualityLine = $('#qualityLine');
    const insightsList = $('#classifyInsightsList');
    if (!section || !uncLine || !qualityLine || !insightsList) return;

    const uncertainty = data.uncertainty || {};
    const quality = data.quality_check || data.quality_overview || {};
    const insights = data.insights || [];

    if (!uncertainty.level && !quality.status && insights.length === 0) {
        section.classList.add('hidden');
        return;
    }

    section.classList.remove('hidden');

    const level = uncertainty.level || 'unknown';
    const ent = Number(uncertainty.entropy_normalized || 0);
    const margin = Number(uncertainty.margin_top1_top2 || 0);
    const reviewTag = uncertainty.review_recommended ? ' · manual review suggested' : '';
    uncLine.textContent = `Uncertainty: ${level.toUpperCase()} (entropy=${ent.toFixed(3)}, margin=${margin.toFixed(3)})${reviewTag}`;

    const qStatus = String(quality.status || 'review').toUpperCase();
    const qWarnings = (quality.warnings || []).slice(0, 2);
    qualityLine.textContent = qWarnings.length > 0
        ? `Data quality: ${qStatus} · ${qWarnings.join(' | ')}`
        : `Data quality: ${qStatus}`;

    insightsList.innerHTML = '';
    for (const msg of insights.slice(0, 4)) {
        const li = document.createElement('li');
        li.textContent = msg;
        insightsList.appendChild(li);
    }
}

function renderEnsembleChart(ensemble) {
    const classNames = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'];
    const classColors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6'];

    const probs = classNames.map(c => ((ensemble.probabilities[c] || 0) * 100));

    const trace = {
        x: probs,
        y: classNames,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: classColors,
            line: { width: 0 }
        },
        text: probs.map(p => `${p.toFixed(1)}%`),
        textposition: 'auto',
        textfont: { color: 'white', size: 13, family: 'Inter' },
    };

    const layout = {
        height: 200,
        margin: { l: 100, r: 30, t: 10, b: 30 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(13,17,23,0.5)',
        xaxis: {
            range: [0, 100],
            title: { text: 'Probability (%)', font: { color: '#8b949e', size: 12 } },
            gridcolor: 'rgba(48,54,61,0.3)',
            color: '#8b949e',
        },
        yaxis: { color: '#e6edf3', tickfont: { size: 13 } },
        font: { family: 'Inter', color: '#e6edf3' },
    };

    Plotly.newPlot('ensembleChart', [trace], layout, {
        displayModeBar: false,
        responsive: true,
    });
}

function showNonGliomaResult(data) {
    const classIcons = { 'Glioma': '🔴', 'Meningioma': '🔵', 'No Tumor': '🟢', 'Pituitary': '🟣' };
    const cls = data.consensus.class_name;
    $('#nonGliomaIcon').textContent = classIcons[cls] || '🧠';
    $('#nonGliomaClass').textContent = cls;
    goToStep(99);
}

// ═══════════════ SEGMENTATION ═══════════════
async function startSegmentation() {
    goToStep(2);
    setStatus('Segmenting...', 'working');

    $('#segmentLoading').classList.remove('hidden');
    $('#segmentResults').classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('session_id', State.sessionId);

        const res = await fetch('/api/segment', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Segmentation failed');
        }

        const data = await res.json();
        State.segmentationResult = data;

        $('#segmentLoading').classList.add('hidden');
        $('#segmentResults').classList.remove('hidden');

        renderSegmentationResults(data);
        setStatus('Segmentation complete');
        showToast('3D Segmentation complete!', 'success');

    } catch (err) {
        setStatus('Segmentation failed', 'error');
        showToast(err.message, 'error');
        $('#segmentLoading').innerHTML = `
            <div style="color: var(--accent-red); text-align: center; padding: 2rem;">
                <p style="font-size: 1.2rem; font-weight: 600;">❌ Segmentation Failed</p>
                <p style="color: var(--text-secondary); margin-top: 0.5rem;">${escapeHtml(err.message)}</p>
                <button class="btn btn-secondary" style="margin-top: 1rem;" onclick="location.reload()">
                    🔄 Try Again
                </button>
            </div>
        `;
    }
}

function renderSegmentationResults(data) {
    // Volume stats
    const statsRow = $('#segStatsRow');
    statsRow.innerHTML = '';

    const regions = [
        { key: 'TC', name: 'Tumor Core', color: '#E74C3C' },
        { key: 'WT', name: 'Whole Tumor', color: '#3498DB' },
        { key: 'ET', name: 'Enhancing Tumor', color: '#F39C12' },
    ];

    for (const r of regions) {
        const vol = data.volumes[r.key] || 0;
        const pct = ((vol / data.total_voxels) * 100).toFixed(2);
        statsRow.innerHTML += `
            <div class="stat-card" style="border-top: 3px solid ${r.color};">
                <div class="stat-label">${r.name}</div>
                <div class="stat-value">${vol.toLocaleString()} mm³</div>
                <div class="stat-delta neutral">${pct}% of brain volume</div>
            </div>
        `;
    }

    // Total
    const totalVol = Object.values(data.volumes).reduce((a, b) => a + b, 0);
    statsRow.innerHTML += `
        <div class="stat-card" style="border-top: 3px solid var(--accent-purple);">
            <div class="stat-label">Total Volume</div>
            <div class="stat-value">${Math.round(totalVol).toLocaleString()}</div>
            <div class="stat-delta neutral">${data.total_voxels.toLocaleString()} total voxels</div>
        </div>
    `;

    renderSegmentationUncertainty(data);
    initSegUncertaintyControls();

    // 3D Visualization
    initSegOverlayControls();
    render3dSegmentation(data.mesh_data);

    // Slice viewer
    initSliceOverlayControls();
    renderSliceViewer(data.slices);

    // Progression button
    const progressionBtn = $('#btnStartProgression');
    if (progressionBtn) {
        progressionBtn.onclick = () => startProgression();
    }
}

function createOverlayChips(containerId, items, state, onChange) {
    const root = $(`#${containerId}`);
    if (!root) return;
    root.innerHTML = '';

    for (const item of items) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `overlay-chip ${state[item.key] ? 'active' : ''}`;
        btn.textContent = `${item.dot} ${item.label}`;
        btn.addEventListener('click', () => {
            state[item.key] = !state[item.key];
            btn.classList.toggle('active', state[item.key]);
            onChange();
        });
        root.appendChild(btn);
    }
}

function initSegOverlayControls() {
    createOverlayChips(
        'segOverlayControls',
        [
            { key: 'BRAIN', label: 'Brain', dot: '⚪' },
            { key: 'WT', label: 'WT', dot: '🔵' },
            { key: 'TC', label: 'TC', dot: '🔴' },
            { key: 'ET', label: 'ET', dot: '🟠' },
            { key: 'UNC', label: 'Uncertainty', dot: '🟣' },
        ],
        State.segOverlayState,
        () => render3dSegmentation(State.segmentationResult?.mesh_data)
    );
}

function initSliceOverlayControls() {
    createOverlayChips(
        'sliceOverlayControls',
        [
            { key: 'WT', label: 'WT', dot: '🔵' },
            { key: 'TC', label: 'TC', dot: '🔴' },
            { key: 'ET', label: 'ET', dot: '🟠' },
            { key: 'UNC', label: 'Uncertainty', dot: '🟣' },
        ],
        State.sliceOverlayState,
        () => renderSliceViewer(State.segmentationResult?.slices)
    );
}

function renderSegmentationUncertainty(data) {
    const card = $('#segUncertaintyCard');
    const text = $('#segUncertaintyText');
    if (!card || !text) return;

    const unc = data.uncertainty_summary;
    if (!unc) {
        card.classList.add('hidden');
        text.textContent = '';
        return;
    }

    card.classList.remove('hidden');
    const level = String(unc.level || 'unknown').toUpperCase();
    const mean = Number(unc.mean || 0).toFixed(3);
    const p95 = Number(unc.p95 || 0).toFixed(3);
    const ratio = (Number(unc.high_uncertainty_ratio || 0) * 100).toFixed(1);
    const review = unc.review_recommended ? ' · manual review suggested' : '';
    text.textContent = `Level: ${level} (mean=${mean}, p95=${p95}, high-unc voxels=${ratio}%)${review}`;
}

function initSegUncertaintyControls() {
    const slider = $('#segUncertaintyOpacity');
    if (!slider) return;
    slider.value = String(State.segUncertaintyOpacity);
    slider.oninput = () => {
        State.segUncertaintyOpacity = Number(slider.value || 0.55);
        render3dSegmentation(State.segmentationResult?.mesh_data);
        renderSliceViewer(State.segmentationResult?.slices);
    };
}

function render3dSegmentation(meshData) {
    if (!meshData) {
        $('#seg3dViewer').innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:3rem;">3D mesh data unavailable</p>';
        return;
    }

    const traces = [];
    const colorMap = { TC: '#E74C3C', WT: '#3498DB', ET: '#F39C12', BRAIN: '#94A3B8', UNC: '#C084FC' };
    const opacityMap = { TC: 0.8, WT: 0.3, ET: 0.9, BRAIN: 0.12, UNC: State.segUncertaintyOpacity };

    for (const [name, mesh] of Object.entries(meshData)) {
        if (!mesh || !mesh.vertices) continue;
        if (State.segOverlayState[name] === false) continue;
        const v = mesh.vertices;
        const f = mesh.faces;

        traces.push({
            type: 'mesh3d',
            x: v.map(p => p[0]),
            y: v.map(p => p[1]),
            z: v.map(p => p[2]),
            i: f.map(t => t[0]),
            j: f.map(t => t[1]),
            k: f.map(t => t[2]),
            color: colorMap[name] || mesh.color,
            opacity: name === 'UNC'
                ? State.segUncertaintyOpacity
                : (mesh.opacity ?? opacityMap[name] ?? 0.5),
            name: name,
            flatshading: true,
            lighting: { ambient: 0.65, diffuse: 0.7, specular: 0.2 },
        });
    }

    if (traces.length === 0) {
        $('#seg3dViewer').innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:3rem;">No tumor regions detected</p>';
        return;
    }

    const layout = {
        scene: {
            xaxis: { visible: false },
            yaxis: { visible: false },
            zaxis: { visible: false },
            bgcolor: 'rgba(0,0,0,0)',
            aspectmode: 'data',
            camera: { eye: { x: 1.5, y: 0.8, z: 0.6 } },
        },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        height: 400,
        paper_bgcolor: 'rgba(0,0,0,0)',
        showlegend: true,
        legend: { font: { color: '#e6edf3', size: 12 } },
    };

    Plotly.newPlot('seg3dViewer', traces, layout, {
        displayModeBar: true,
        displaylogo: false,
        responsive: true,
    });
}

function renderSliceViewer(slices) {
    if (!slices || slices.length === 0) {
        $('#segSliceViewer').innerHTML = '<p style="color:var(--text-secondary);">No slice data</p>';
        return;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const viewer = $('#segSliceViewer');
    viewer.innerHTML = '';
    viewer.appendChild(canvas);

    const slider = $('#segSliceSlider');
    slider.max = slices.length - 1;
    slider.value = Math.floor(slices.length / 2);

    function drawSlice(idx) {
        const slice = slices[idx];
        const h = slice.flair.length;
        const w = slice.flair[0].length;
        canvas.width = w;
        canvas.height = h;
        canvas.style.maxWidth = '100%';
        canvas.style.height = 'auto';

        const imgData = ctx.createImageData(w, h);

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const i = (y * w + x) * 4;
                const g = slice.flair[y][x];

                // Base grayscale
                let r = g, green = g, b = g;

                // Overlay segmentation
                const showET = State.sliceOverlayState.ET !== false;
                const showTC = State.sliceOverlayState.TC !== false;
                const showWT = State.sliceOverlayState.WT !== false;
                const showUNC = State.sliceOverlayState.UNC !== false;
                if (showET && slice.et[y] && slice.et[y][x]) { r = 243; green = 156; b = 18; }
                else if (showTC && slice.tc[y] && slice.tc[y][x]) { r = 231; green = 76; b = 60; }
                else if (showWT && slice.wt[y] && slice.wt[y][x]) { r = 52; green = 152; b = 219; }

                if (showUNC && slice.uncertainty && slice.uncertainty[y]) {
                    const u = Number(slice.uncertainty[y][x] || 0) / 255;
                    if (u > 0) {
                        const alpha = Math.min(1, u * State.segUncertaintyOpacity);
                        r = Math.round(r * (1 - alpha) + 192 * alpha);
                        green = Math.round(green * (1 - alpha) + 132 * alpha);
                        b = Math.round(b * (1 - alpha) + 252 * alpha);
                    }
                }

                imgData.data[i] = r;
                imgData.data[i + 1] = green;
                imgData.data[i + 2] = b;
                imgData.data[i + 3] = 255;
            }
        }

        ctx.putImageData(imgData, 0, 0);
        $('#segSliceInfo').textContent = `Slice ${idx + 1} / ${slices.length} (axial index ${slice.index})`;
    }

    slider.oninput = () => drawSlice(parseInt(slider.value));
    drawSlice(parseInt(slider.value));
}

// ═══════════════ PROGRESSION ═══════════════
function initGradeSelector() {
    $$('.grade-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.grade-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            // Re-run progression with new grade if we have results
            if (State.sessionId && State.segmentationResult) {
                startProgression(btn.dataset.grade);
            }
        });
    });
}

async function startProgression(grade) {
    goToStep(3);
    setStatus('Analyzing progression...', 'working');

    $('#progressionLoading').classList.remove('hidden');
    $('#progressionResults').classList.add('hidden');

    const selectedGrade = grade || 'HGG';

    try {
        const formData = new FormData();
        formData.append('session_id', State.sessionId);
        formData.append('grade', selectedGrade);

        const res = await fetch('/api/progression', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Progression analysis failed');
        }

        const data = await res.json();
        State.progressionResult = data;

        $('#progressionLoading').classList.add('hidden');
        $('#progressionResults').classList.remove('hidden');

        renderProgressionResults(data);
        setStatus('Analysis complete');
        showToast('Progression analysis complete!', 'success');

    } catch (err) {
        setStatus('Analysis failed', 'error');
        showToast(err.message, 'error');
        $('#progressionLoading').innerHTML = `
            <div style="color: var(--accent-red); text-align: center; padding: 2rem;">
                <p style="font-size: 1.2rem; font-weight: 600;">❌ Progression Failed</p>
                <p style="color: var(--text-secondary); margin-top: 0.5rem;">${escapeHtml(err.message)}</p>
                <button class="btn btn-secondary" style="margin-top:1rem;" onclick="startProgression()">🔄 Retry</button>
            </div>
        `;
    }
}

async function cleanupSession() {
    if (!State.sessionId) return;
    try {
        await fetch(`/api/session/${State.sessionId}`, { method: 'DELETE' });
        State.sessionId = null;
    } catch (err) {
        // no-op
    }
}

window.addEventListener('beforeunload', () => {
    if (State.sessionId) {
        fetch(`/api/session/${State.sessionId}`, { method: 'DELETE', keepalive: true });
    }
});

function renderProgressionResults(data) {
    const logistic = data.logistic;
    const params = logistic.params;

    // Stats row
    const statsRow = $('#progStatsRow');
    statsRow.innerHTML = `
        <div class="stat-card" style="border-top: 3px solid var(--accent-blue);">
            <div class="stat-label">Current Volume</div>
            <div class="stat-value">${Math.round(data.current_volume).toLocaleString()} mm³</div>
            <div class="stat-delta neutral">From segmentation</div>
        </div>
        <div class="stat-card" style="border-top: 3px solid var(--accent-purple);">
            <div class="stat-label">Tumor Grade</div>
            <div class="stat-value">${data.grade}</div>
            <div class="stat-delta neutral">${data.grade === 'HGG' ? 'High-Grade Glioma' : 'Low-Grade Glioma'}</div>
        </div>
        <div class="stat-card" style="border-top: 3px solid var(--accent-orange);">
            <div class="stat-label">Growth Rate</div>
            <div class="stat-value">${params.r.toFixed(4)}/day</div>
            <div class="stat-delta neutral">Logistic parameter r</div>
        </div>
        <div class="stat-card" style="border-top: 3px solid var(--accent-red);">
            <div class="stat-label">Carrying Capacity</div>
            <div class="stat-value">${Math.round(params.k).toLocaleString()}</div>
            <div class="stat-delta neutral">Max predicted volume (mm³)</div>
        </div>
    `;

    // Growth curve chart
    renderGrowthCurve(logistic, data.current_volume);

    // Projections table
    renderProjectionsTable(logistic.projections);

    renderProgressionInsights(data);
    initReportDownload();

    // Spatial prediction
    if (data.spatial && data.spatial.mesh_data) {
        initSpatialOverlayControls();
        renderSpatialPrediction(data.spatial);
    } else {
        $('#spatialSection')?.classList.add('hidden');
    }
}

function initSpatialOverlayControls() {
    createOverlayChips(
        'spatialOverlayControls',
        [
            { key: 'envelope', label: 'Envelope', dot: '⚪' },
            { key: 'brain', label: 'Brain', dot: '⚪' },
            { key: 'stable', label: 'Stable', dot: '🔵' },
            { key: 'growth', label: 'Growth', dot: '🔴' },
            { key: 'regression', label: 'Regression', dot: '🟢' },
        ],
        State.spatialOverlayState,
        () => {
            if (State.progressionResult?.spatial) {
                renderSpatialPrediction(State.progressionResult.spatial);
            }
        }
    );
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderGrowthCurve(logistic, currentVol) {
    const curve = logistic.curve;
    const projections = logistic.projections;

    const traces = [
        // Smooth curve
        {
            x: curve.days,
            y: curve.volumes,
            mode: 'lines',
            name: 'Logistic Fit',
            line: { color: '#d97706', width: 2, dash: 'dot' },
        },
        // Current volume
        {
            x: [0],
            y: [currentVol],
            mode: 'markers',
            name: 'Current',
            marker: { size: 14, color: '#3b82f6', symbol: 'circle', line: { width: 2, color: '#fff' } },
        },
        // Projected points
        {
            x: projections.map(p => p.day),
            y: projections.map(p => p.volume),
            mode: 'markers+text',
            name: 'Projections',
            marker: { size: 10, color: '#ef4444', symbol: 'diamond' },
            text: projections.map(p => `${p.day}d`),
            textposition: 'top center',
            textfont: { color: '#8b949e', size: 11 },
        },
    ];

    const layout = {
        height: 400,
        margin: { l: 70, r: 30, t: 20, b: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(13,17,23,0.5)',
        xaxis: {
            title: { text: 'Days from Current Scan', font: { color: '#8b949e' } },
            gridcolor: 'rgba(48,54,61,0.3)',
            color: '#8b949e',
        },
        yaxis: {
            title: { text: 'Volume (mm³)', font: { color: '#8b949e' } },
            gridcolor: 'rgba(48,54,61,0.3)',
            color: '#8b949e',
        },
        font: { family: 'Inter', color: '#e6edf3' },
        legend: { font: { color: '#e6edf3' } },
        showlegend: true,
    };

    Plotly.newPlot('growthCurveChart', traces, layout, {
        displayModeBar: false,
        responsive: true,
    });
}

function renderProjectionsTable(projections) {
    const tbody = $('#projectionsBody');
    tbody.innerHTML = '';

    const labels = { 30: '1 Month', 60: '2 Months', 90: '3 Months', 180: '6 Months', 365: '1 Year' };

    for (const p of projections) {
        const growth = p.growth_pct;
        const badgeClass = growth > 5 ? 'up' : growth < -5 ? 'down' : 'stable';
        const statusText = growth > 5 ? '⚠️ Growth' : growth < -5 ? '✅ Regression' : '● Stable';

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${labels[p.day] || p.day + ' days'}</strong><br><span style="color:var(--text-muted);font-size:0.8rem;">Day ${p.day}</span></td>
            <td style="font-weight:600;">${Math.round(p.volume).toLocaleString()} mm³</td>
            <td><span class="growth-badge ${badgeClass}">${growth > 0 ? '+' : ''}${growth.toFixed(1)}%</span></td>
            <td style="font-size:0.85rem;">${statusText}</td>
        `;
        tbody.appendChild(row);
    }
}

function renderProgressionInsights(data) {
    const driverLine = $('#progDriverLine');
    const spatialLine = $('#progSpatialLine');
    const list = $('#progInsightsList');
    if (!driverLine || !spatialLine || !list) return;

    const exp = data.explainability || {};
    const drivers = exp.drivers || {};
    const gradeCtx = exp.grade_context || data.grade || 'N/A';
    const r = Number(drivers.growth_rate_r_per_day || 0);
    const k = Math.round(Number(drivers.carrying_capacity_mm3 || 0));
    driverLine.textContent = `Primary drivers: grade context ${gradeCtx}, r=${r.toFixed(4)}/day, carrying capacity=${k.toLocaleString()} mm³.`;

    const spatial = exp.spatial_balance;
    if (spatial) {
        const growth = Number(spatial.growth_voxels || 0).toLocaleString();
        const stable = Number(spatial.stable_voxels || 0).toLocaleString();
        const reg = Number(spatial.regression_voxels || 0).toLocaleString();
        const d = Number(spatial.volume_change_pct || 0).toFixed(1);
        spatialLine.textContent = `Spatial balance: growth ${growth}, stable ${stable}, regression ${reg}, net Δ ${d}%`;
    } else {
        spatialLine.textContent = 'Spatial explainability unavailable (spatial model output missing).';
    }

    list.innerHTML = '';
    const insights = data.insights || [];
    if (insights.length === 0) {
        const li = document.createElement('li');
        li.textContent = 'No progression insights were generated.';
        list.appendChild(li);
        return;
    }
    for (const msg of insights.slice(0, 6)) {
        const li = document.createElement('li');
        li.textContent = msg;
        list.appendChild(li);
    }
}

function initReportDownload() {
    const btn = $('#btnDownloadReport');
    if (!btn) return;

    btn.onclick = async () => {
        if (!State.sessionId) {
            showToast('No active session for report generation', 'warning');
            return;
        }

        btn.disabled = true;
        const original = btn.innerHTML;
        btn.innerHTML = '<span class="btn-icon">⏳</span> Preparing report...';
        try {
            const res = await fetch(`/api/report/${State.sessionId}`);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || 'Report generation failed');
            }

            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `neuroai_report_${State.sessionId}.pdf`;
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            showToast('Clinical report downloaded', 'success');
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = original;
        }
    };
}

function renderSpatialPrediction(spatial) {
    const section = $('#spatialSection');
    section.classList.remove('hidden');

    // Stats
    const stats = spatial.stats;
    const spatialStats = $('#spatialStats');
    spatialStats.innerHTML = `
        <div class="spatial-stat" style="border-color: #ef4444;">
            <div style="color:#ef4444;font-size:1.3rem;font-weight:700;">+${stats.growth_voxels.toLocaleString()}</div>
            <div style="color:var(--text-secondary);font-size:0.8rem;">Growth Voxels</div>
        </div>
        <div class="spatial-stat" style="border-color: #3b82f6;">
            <div style="color:#3b82f6;font-size:1.3rem;font-weight:700;">${stats.stable_voxels.toLocaleString()}</div>
            <div style="color:var(--text-secondary);font-size:0.8rem;">Stable Voxels</div>
        </div>
        <div class="spatial-stat" style="border-color: #22c55e;">
            <div style="color:#22c55e;font-size:1.3rem;font-weight:700;">-${stats.regression_voxels.toLocaleString()}</div>
            <div style="color:var(--text-secondary);font-size:0.8rem;">Regression Voxels</div>
        </div>
    `;

    // 3D mesh
    if (spatial.mesh_data) {
        const traces = [];
        const configs = {
            brain: { color: '#94A3B8', opacity: 0.10, name: 'Brain' },
            stable: { color: '#3b82f6', opacity: 0.5, name: 'Stable' },
            growth: { color: '#ef4444', opacity: 0.9, name: 'Growth' },
            regression: { color: '#22c55e', opacity: 0.6, name: 'Regression' },
            envelope: { color: '#94A3B8', opacity: 0.10, name: 'Envelope' },
        };

        for (const [key, mesh] of Object.entries(spatial.mesh_data)) {
            if (!mesh || !mesh.vertices) continue;
            if (State.spatialOverlayState[key] === false) continue;
            const cfg = configs[key] || {};
            traces.push({
                type: 'mesh3d',
                x: mesh.vertices.map(p => p[0]),
                y: mesh.vertices.map(p => p[1]),
                z: mesh.vertices.map(p => p[2]),
                i: mesh.faces.map(t => t[0]),
                j: mesh.faces.map(t => t[1]),
                k: mesh.faces.map(t => t[2]),
                color: cfg.color || mesh.color,
                opacity: cfg.opacity || mesh.opacity || 0.5,
                name: cfg.name || key,
                flatshading: true,
            });
        }

        if (traces.length === 0) {
            $('#spatial3dViewer').innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:3rem;">No spatial overlays selected</p>';
            return;
        }

        const layout = {
            scene: {
                xaxis: { visible: false },
                yaxis: { visible: false },
                zaxis: { visible: false },
                bgcolor: 'rgba(0,0,0,0)',
                aspectmode: 'data',
                camera: { eye: { x: 1.5, y: 0.8, z: 0.6 } },
            },
            margin: { l: 0, r: 0, t: 0, b: 0 },
            height: 400,
            paper_bgcolor: 'rgba(0,0,0,0)',
            showlegend: true,
            legend: { font: { color: '#e6edf3' } },
        };
        Plotly.newPlot('spatial3dViewer', traces, layout, {
            displayModeBar: true,
            displaylogo: false,
            responsive: true,
        });
    }
}
