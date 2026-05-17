"""Build presentation HTML from slide data."""
import json, textwrap
from pathlib import Path

OUT = Path(__file__).parent / "index.html"

HEAD = '''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroAI — Brain Tumor Analysis</title><link rel="stylesheet" href="style.css">
</head><body>
<nav><div class="nav-inner"><div class="nav-logo">Neuro<span>AI</span></div>
<div class="nav-dots" id="navDots"></div><span class="nav-counter" id="navCounter">01</span></div></nav>
'''
FOOT = '<script src="script.js"></script></body></html>'

SLIDES = []

def S(id, label, title, body):
    SLIDES.append((id, label, title, body))

# ── 1. Title ──
S("title","Project Title","","")

# ── 2. Introduction ──
S("intro","Introduction","Introduction","""
<p class="body-text">Brain tumours require highly accurate detection, segmentation, and continuous monitoring to support timely clinical decisions. Traditional manual analysis of MRI scans is slow, subjective, and limited by human fatigue.</p>
<p class="body-text" style="margin-top:1rem">This project builds an integrated AI pipeline: <strong>3D multimodal tumour segmentation</strong>, <strong>federated classification</strong>, and <strong>longitudinal progression forecasting</strong> — motivated by precision, explainability, scalability, and privacy.</p>
""")

# ── 3. Abstract ──
S("abstract","Abstract","Abstract","""
<p class="body-text">This project proposes a multimodal AI framework for brain tumour diagnosis and monitoring. The system integrates 3D Attention-U-Net segmentation, federated tumour classification with QPSO aggregation, and longitudinal tumour progression forecasting using mathematical models and LSTM networks.</p>
<div class="stat-grid reveal stagger" style="margin-top:1.5rem">
<div class="stat-block"><div class="stat-number" data-count="3">0</div><div class="stat-label">AI Modules</div></div>
<div class="stat-block"><div class="stat-number" data-count="5.9" data-suffix="M" data-decimals="1">0</div><div class="stat-label">U-Net Params</div></div>
<div class="stat-block"><div class="stat-number" data-count="1251" data-suffix="+">0</div><div class="stat-label">Patients</div></div>
<div class="stat-block"><div class="stat-number" data-count="4">0</div><div class="stat-label">MRI Modalities</div></div>
</div>""")

# ── 4. Problem Statement ──
S("problem","Problem Statement","Problem Statement","""
<div class="card" style="border-left:4px solid var(--gold);margin-bottom:1.5rem">
<h3>A Federated Multimodal AI Framework with QPSO</h3>
<p class="body-text">For Brain Tumor Classification, 3D Glioma Segmentation, and Tumor Progression Analysis</p></div>
<div class="card-grid reveal stagger">
<div class="card card-sm"><span class="tag tag-coral">Gap 1</span><h3 style="margin-top:.5rem">Isolated Systems</h3><p style="font-size:.88rem;color:var(--slate)">Existing AI performs basic classification OR 2D segmentation — never integrated end-to-end</p></div>
<div class="card card-sm"><span class="tag tag-coral">Gap 2</span><h3 style="margin-top:.5rem">No Progression</h3><p style="font-size:.88rem;color:var(--slate)">No system predicts tumour evolution over time using longitudinal data</p></div>
<div class="card card-sm"><span class="tag tag-coral">Gap 3</span><h3 style="margin-top:.5rem">Privacy Risk</h3><p style="font-size:.88rem;color:var(--slate)">Centralized training requires sharing sensitive patient data across hospitals</p></div>
<div class="card card-sm"><span class="tag tag-coral">Gap 4</span><h3 style="margin-top:.5rem">Fairness Gap</h3><p style="font-size:.88rem;color:var(--slate)">Standard FL algorithms (FedAvg) leave smaller hospitals with poor accuracy</p></div>
</div>""")

# ── 5. Literature Survey ──
S("literature","Literature Survey","Literature Survey","""
<div class="table-wrap reveal"><table>
<tr><th>#</th><th>Author</th><th>Method</th><th>Architecture</th><th>Gap</th></tr>
<tr><td>1</td><td>Edla &amp; Indhumathi, 2025</td><td>QPSO on IID MNIST</td><td>CNN + FedAvg/QPSO</td><td>IID only; no FedProx comparison</td></tr>
<tr><td>2</td><td>Pati et al., 2022</td><td>FL across 71 institutions</td><td>U-Net + FedAvg</td><td>No fairness analysis; no QPSO</td></tr>
<tr><td>3</td><td>Isensee et al., 2021</td><td>nnU-Net self-configuring</td><td>3D U-Net</td><td>No attention mechanism</td></tr>
<tr><td>4</td><td>Subramanian et al., 2021</td><td>ST-ConvLSTM 4D MRI</td><td>Spatio-Temporal LSTM</td><td>Small dataset; no math models</td></tr>
<tr><td>5</td><td>Zhang et al., 2023</td><td>Self-supervised temporal</td><td>ResNet + Temporal</td><td>Binary only; no volume forecast</td></tr>
</table></div>""")

# ── 6. Proposed System ──
S("proposed","Proposed System","Proposed System","""
<div class="card-grid reveal stagger">
<div class="card"><span class="tag tag-gold">Module 01</span><h3 style="margin-top:.8rem">Federated Classification</h3>
<p style="font-size:.88rem;color:var(--slate)">FedAvg, FedProx, and novel Layer-by-Layer QPSO across 3 simulated hospitals. ResNet-18, 4-class, 100 FL rounds.</p>
<p style="font-size:.82rem;font-weight:600;color:var(--teal);margin-top:.5rem">QPSO achieves 3.5x better fairness than FedAvg</p></div>
<div class="card"><span class="tag tag-teal">Module 02</span><h3 style="margin-top:.8rem">3D Brain Tumor Segmentation</h3>
<p style="font-size:.88rem;color:var(--slate)">3D Attention U-Net (5.9M params) on BraTS 2021, 4 MRI modalities, 3 tumor regions (TC, WT, ET).</p>
<p style="font-size:.82rem;font-weight:600;color:var(--teal);margin-top:.5rem">Mean Dice Score: 0.76 on BraTS 2021</p></div>
<div class="card"><span class="tag tag-accent">Module 03</span><h3 style="margin-top:.8rem">Progression Forecasting</h3>
<p style="font-size:.88rem;color:var(--slate)">Math models (Logistic, Gompertz) + Attention-LSTM hybrid. 3D U-Net spatial prediction for WHERE growth occurs.</p>
<p style="font-size:.82rem;font-weight:600;color:var(--teal);margin-top:.5rem">MAE improved 7.88% for HGG patients</p></div>
</div>""")

# ── 7. Objectives ──
S("objectives","Objectives","Objectives","""
<ul class="obj-list reveal stagger">
<li>Develop 3D multimodal tumor segmentation using Attention U-Net on BraTS 2021</li>
<li>Implement federated tumor classification across 3 simulated hospital nodes</li>
<li>Design QPSO-based aggregation for fair, privacy-preserving FL</li>
<li>Predict tumor progression using hybrid Math + LSTM models</li>
<li>Build 3D U-Net spatial predictor for growth/regression mapping</li>
<li>Integrate all modules into a clinical decision support dashboard</li>
</ul>""")

# ── 8. System Architecture ──
S("architecture","Architecture","System Architecture","""
<div class="reveal" style="text-align:center"><img src="assets/system architecture.png" alt="System Architecture" style="max-width:100%;border-radius:var(--radius);border:1px solid var(--border);box-shadow:var(--shadow)"></div>
<div class="card-grid reveal stagger" style="margin-top:1.5rem">
<div class="card card-sm" style="text-align:center"><img src="assets/use case.png" alt="Use Case" style="max-width:100%;border-radius:8px"></div>
<div class="card card-sm" style="text-align:center"><img src="assets/activity uml.png" alt="Activity" style="max-width:100%;border-radius:8px"></div>
</div>""")

# ── 9. Technologies Used ──
S("tech","Technologies","Technologies Used","""
<div class="split">
<div>
<h3>Classification Module</h3>
<div class="tech-pills reveal stagger"><span class="tech-pill">PyTorch 2.x</span><span class="tech-pill">torchvision</span><span class="tech-pill">ResNet-18</span><span class="tech-pill">PIL / OpenCV</span><span class="tech-pill">NumPy / SciPy</span><span class="tech-pill">Kaggle P100</span></div>
<h3 style="margin-top:1.5rem">Segmentation Module</h3>
<div class="tech-pills reveal stagger"><span class="tech-pill">MONAI</span><span class="tech-pill">Attention U-Net</span><span class="tech-pill">nibabel</span><span class="tech-pill">3D Sliding Window</span><span class="tech-pill">BraTS 2021</span></div>
</div><div>
<h3>Progression Module</h3>
<div class="tech-pills reveal stagger"><span class="tech-pill">SciPy curve_fit</span><span class="tech-pill">Attention-LSTM</span><span class="tech-pill">3D U-Net</span><span class="tech-pill">Pandas</span><span class="tech-pill">Plotly</span></div>
<h3 style="margin-top:1.5rem">Dashboard &amp; Deployment</h3>
<div class="tech-pills reveal stagger"><span class="tech-pill">FastAPI</span><span class="tech-pill">Vanilla JS</span><span class="tech-pill">Plotly.js</span><span class="tech-pill">Docker</span><span class="tech-pill">HF Spaces</span><span class="tech-pill">fpdf2</span></div>
</div></div>""")

# ── 10. Implementation ──
S("implementation","Implementation","Implementation","""
<div class="card-grid">
<div class="card reveal"><span class="tag tag-gold">FedQPSO</span>
<div class="code-block" style="margin-top:.8rem;font-size:.72rem"><span class="code-label">Python</span>
<span class="cm"># QPSO Aggregation Step</span>
<span class="kw">for</span> param_name <span class="kw">in</span> global_model.state_dict():
  phi = random.uniform(<span class="num">0</span>, <span class="num">1</span>)
  u = random.uniform(<span class="num">0.3</span>, <span class="num">1.0</span>)
  <span class="cm"># Attraction point</span>
  p = phi * mbest[param] + (<span class="num">1</span>-phi) * gbest[param]
  <span class="cm"># Quantum perturbation</span>
  perturb = beta * <span class="fn">abs</span>(mbest-gbest) * <span class="fn">ln</span>(<span class="num">1</span>/u)
  perturb = <span class="fn">clip</span>(perturb, <span class="num">-0.1</span>, <span class="num">0.1</span>)
  w_new[param] = p + choice([-<span class="num">1</span>,<span class="num">1</span>]) * perturb</div></div>
<div class="card reveal"><span class="tag tag-teal">3D Segmentation</span>
<div class="code-block" style="margin-top:.8rem;font-size:.72rem"><span class="code-label">Python</span>
<span class="cm"># Sliding window inference</span>
outputs = <span class="fn">sliding_window_inference</span>(
  inputs=image,  <span class="cm"># (1, 4, H, W, D)</span>
  roi_size=(<span class="num">96</span>, <span class="num">96</span>, <span class="num">96</span>),
  sw_batch_size=<span class="num">4</span>,
  predictor=model,
  overlap=<span class="num">0.5</span>
)
predictions = (outputs.<span class="fn">sigmoid</span>() > <span class="num">0.5</span>).<span class="fn">float</span>()
<span class="cm"># Output: (1, 3, H, W, D) — TC, WT, ET</span></div></div>
</div>""")

# ── 11. FL Animation ──
S("fl-demo","Federated Learning","Federated Learning in Action","""
<div class="split">
<div>
<p class="body-text">Three hospitals train locally, then send weights to the central server where QPSO aggregation finds the optimal global model through quantum-inspired position updates.</p>
<p id="flStatus" style="font-size:.85rem;font-weight:600;color:var(--gold);margin-top:1rem;font-family:var(--font-mono)">Waiting...</p>
<div class="stat-grid" style="margin-top:1rem">
<div class="stat-block"><div class="stat-number">100</div><div class="stat-label">FL Rounds</div></div>
<div class="stat-block"><div class="stat-number">3</div><div class="stat-label">Hospitals</div></div>
</div></div>
<div class="fl-viz" id="flViz">
<div class="fl-client" style="left:15%;top:15%">Hospital<br>A</div>
<div class="fl-client" style="right:15%;top:15%">Hospital<br>B</div>
<div class="fl-client" style="left:35%;bottom:8%">Hospital<br>C</div>
<div class="fl-server">QPSO<br>Server</div>
</div></div>""")

# ── 12. Results ──
S("results","Results","Results &amp; Output","""
<div class="card-grid reveal stagger">
<div class="card card-sm" style="text-align:center"><h3>Upload &amp; Analysis</h3><img src="assets/upload files ui.jpeg" style="max-width:100%;border-radius:8px;margin-top:.5rem"></div>
<div class="card card-sm" style="text-align:center"><h3>Classification</h3><img src="assets/classification ui.jpeg" style="max-width:100%;border-radius:8px;margin-top:.5rem"></div>
<div class="card card-sm" style="text-align:center"><h3>Segmentation</h3><img src="assets/segmentation ui.jpeg" style="max-width:100%;border-radius:8px;margin-top:.5rem"></div>
<div class="card card-sm" style="text-align:center"><h3>Progression</h3><img src="assets/progression ui 1.jpeg" style="max-width:100%;border-radius:8px;margin-top:.5rem"></div>
</div>""")

# ── 13. Comparative Analysis ──
S("comparison","Comparison","Comparative Analysis","""
<h3 class="reveal">Setup 2 — Label Skew (70/10/10/10) — The Hardest Test</h3>
<div class="table-wrap reveal" style="margin:1rem 0"><table>
<tr><th>Metric</th><th>FedAvg</th><th>FedProx</th><th>L-b-L QPSO</th></tr>
<tr><td>Best Global Accuracy</td><td>90.56%</td><td class="best-cell">93.02%</td><td>92.09%</td></tr>
<tr><td>Client 1 (smallest)</td><td>60.42%</td><td>77.50%</td><td class="best-cell">80.00%</td></tr>
<tr><td>Client 2</td><td>85.73%</td><td>86.80%</td><td class="best-cell">87.33%</td></tr>
<tr><td>Client 3 (largest)</td><td class="best-cell">89.17%</td><td>92.14%</td><td>88.10%</td></tr>
<tr><td>Client Std Dev</td><td>12.82%</td><td>6.05%</td><td class="best-cell">3.65%</td></tr>
<tr><td>Fairness Improvement</td><td>Baseline</td><td>2.1x fairer</td><td class="best-cell">3.5x fairer</td></tr>
</table></div>
<div class="card-grid reveal stagger" style="margin-top:1rem">
<div class="card card-sm" style="text-align:center"><img src="assets/s2_fairness.png" style="max-width:100%;border-radius:8px"><p style="font-size:.75rem;color:var(--muted);margin-top:.3rem">Per-Client Fairness</p></div>
<div class="card card-sm" style="text-align:center"><img src="assets/s2_roc_auc.png" style="max-width:100%;border-radius:8px"><p style="font-size:.75rem;color:var(--muted);margin-top:.3rem">ROC-AUC Curves</p></div>
</div>""")

# ── 14. Progression Demo ──
S("progression","Progression","Tumor Progression Forecasting","""
<div class="split"><div>
<p class="body-text">Watch the tumor grow over time. Our hybrid Math + LSTM model predicts growth trajectory and raises clinical alerts when growth exceeds RANO thresholds.</p>
<div class="stat-grid reveal stagger" style="margin-top:1rem">
<div class="stat-block"><div class="stat-number" data-count="7.88" data-suffix="%" data-decimals="2">0</div><div class="stat-label">MAE Improvement (HGG)</div></div>
<div class="stat-block"><div class="stat-number" data-count="203">0</div><div class="stat-label">Patients Analyzed</div></div>
</div></div>
<div><div class="tumor-viz">
<div class="brain-circle"><div class="tumor-dot" id="tumorDot"></div></div>
<div class="tumor-alert" id="tumorAlert">&#9888; Progressive Disease<br><span style="font-size:.75rem;font-weight:400">Growth exceeds 25% RANO threshold</span></div>
</div></div></div>""")

# ── 15. Conclusion ──
S("conclusion","Conclusion","Conclusion","""
<div class="card-grid reveal stagger">
<div class="card card-sm" style="border-left:3px solid var(--gold)"><h3>End-to-End Pipeline</h3><p style="font-size:.88rem;color:var(--slate)">Integrated classification + segmentation + progression forecasting</p></div>
<div class="card card-sm" style="border-left:3px solid var(--teal)"><h3>FedQPSO</h3><p style="font-size:.88rem;color:var(--slate)">Reduces inter-hospital accuracy gap by 81% vs FedAvg. Only method keeping weakest hospital at 80%+</p></div>
<div class="card card-sm" style="border-left:3px solid var(--accent)"><h3>Mean Dice 0.76</h3><p style="font-size:.88rem;color:var(--slate)">3D Attention U-Net on BraTS 2021 segmentation benchmark</p></div>
<div class="card card-sm" style="border-left:3px solid var(--coral)"><h3>Key Insight</h3><p style="font-size:.88rem;color:var(--slate)">Global accuracy hides inequity. Per-client fairness metrics are essential in clinical FL (Cohen's d: 0.35 to 1.26)</p></div>
</div>""")

# ── 16. Future Scope ──
S("future","Future Scope","Future Scope","""
<div class="card-grid reveal stagger">
<div class="card card-sm"><span class="tag tag-gold">Short-Term (6-12 mo)</span><ul style="margin-top:.5rem;padding-left:1.2rem;font-size:.85rem;color:var(--slate)"><li>Differential Privacy (DP-FedAvg &amp; DP-QPSO)</li><li>QPSO hyperparameter ablation studies</li><li>Hospital pilot deployment (3-5 sites)</li></ul></div>
<div class="card card-sm"><span class="tag tag-teal">Medium-Term (1-2 yr)</span><ul style="margin-top:.5rem;padding-left:1.2rem;font-size:.85rem;color:var(--slate)"><li>Federated Segmentation &amp; Progression across 10+ hospitals</li><li>Gradient compression (50% bandwidth reduction)</li><li>Survival forecasting (Cox regression)</li></ul></div>
<div class="card card-sm"><span class="tag tag-accent">Long-Term (2-5 yr)</span><ul style="margin-top:.5rem;padding-left:1.2rem;font-size:.85rem;color:var(--slate)"><li>Multi-modal fusion (genomics + radiomics + MRI)</li><li>Federated vision transformers</li><li>FDA 510(k) clinical validation</li><li>Scale to 20+ hospitals</li></ul></div>
</div>""")

# ── 17. References ──
S("references","References","References","""
<ol class="ref-list reveal">
<li>McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.</li>
<li>Li et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys.</li>
<li>Sun, Feng &amp; Xu (2004). "Particle Swarm Optimization with Particles Having Quantum Behavior." IEEE CEC.</li>
<li>He et al. (2016). "Deep Residual Learning for Image Recognition." IEEE CVPR.</li>
<li>Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.</li>
<li>Schlemper et al. (2019). "Attention Gated Networks." Medical Image Analysis, 53, 197-207.</li>
<li>Hochreiter &amp; Schmidhuber (1997). "Long Short-Term Memory." Neural Computation, 9(8).</li>
<li>Baid et al. (2021). "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark." arXiv:2107.02314.</li>
<li>Masoud Nickparvar (2021). "Brain Tumor MRI Dataset." Kaggle.</li>
<li>BRISC (2025). "Brain Tumor MRI Classification Dataset 2025." Kaggle.</li>
</ol>""")

# ── 18. Thank You ──
S("thanks","","","")

# ── BUILD HTML ──
html = HEAD

# Nav dots
dots_html = ""
for sid, label, _, _ in SLIDES:
    dots_html += f'<button class="nav-dot" data-target="{sid}" title="{label}"></button>'

html = html.replace('<div class="nav-dots" id="navDots"></div>',
                     f'<div class="nav-dots" id="navDots">{dots_html}</div>')

for i, (sid, label, title, body) in enumerate(SLIDES):
    if sid == "title":
        html += f'''<section id="{sid}" class="hero">
<div class="section-inner hero-content">
<div class="hero-badge">Final Year Major Project</div>
<h1>Automated Multimodal 3D Brain Tumor Analysis Using <span>Federated Learning</span> &amp; Progression Forecasting</h1>
<p class="subtitle">An integrated AI pipeline for privacy-preserving classification, volumetric segmentation, and longitudinal growth prediction.</p>
<div class="hero-meta">
<div class="hero-meta-item"><strong>Guide:</strong> Mrs. P. Uma Maheshwari</div>
<div class="hero-meta-item"><strong>Team:</strong> Edla Divyansh Teja &middot; Nimmakayala Vishnu &middot; Vangepuram Stacy</div>
</div></div></section>\n'''
    elif sid == "thanks":
        html += f'''<section id="{sid}" style="text-align:center">
<div class="section-inner">
<h1 style="font-size:clamp(2rem,4vw,3.5rem)">Thank You</h1>
<p class="subtitle" style="margin:1rem auto;text-align:center">Questions &amp; Discussion</p>
<div style="margin-top:2rem;font-size:.9rem;color:var(--muted)">
<p><strong style="color:var(--ink)">Live Demo:</strong> <a href="https://huggingface.co/spaces/Divs0910/neuroai-dashboard" target="_blank" style="color:var(--gold)">huggingface.co/spaces/Divs0910/neuroai-dashboard</a></p>
</div></div></section>\n'''
    else:
        html += f'''<section id="{sid}">
<div class="section-inner">
<div class="section-label">{label}</div>
<h2 class="reveal">{title}</h2>
{body}
</div></section>\n'''

html += FOOT
OUT.write_text(html, encoding="utf-8")
print(f"Built {OUT} ({len(html):,} bytes, {len(SLIDES)} slides)")
