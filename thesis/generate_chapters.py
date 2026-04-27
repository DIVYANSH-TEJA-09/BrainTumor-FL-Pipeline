"""Generate all chapter .tex files from v2.docx content + real repo code."""
import os, re
from docx import Document

BASE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(os.path.dirname(BASE), "documentation", "FL_QPSO_FedAvg_FINAL_COMPLETE_v2.docx")
OUT = os.path.join(BASE, "chapters")
SRC = os.path.join(os.path.dirname(BASE), "federated_learning", "src")
PROG = os.path.join(os.path.dirname(BASE), "progression", "src")
os.makedirs(OUT, exist_ok=True)

def tex_escape(s):
    """Escape special LaTeX characters."""
    s = s.replace('\\', '\\textbackslash{}')
    for c in ['&', '%', '$', '#', '_', '{', '}', '~', '^']:
        s = s.replace(c, '\\' + c)
    s = s.replace('\\textbackslash\\{\\}', '\\textbackslash{}')
    return s

def read_code_file(path, start=None, end=None):
    """Read a source file and return as LaTeX listing."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if start and end:
        lines = lines[start-1:end]
    return ''.join(lines)

# Parse v2.docx into chapter groups
doc = Document(V2)
chapters = {}
current_ch = None
current_paras = []

for para in doc.paragraphs:
    sn = para.style.name if para.style else ''
    text = para.text.strip()
    if not text or text == '---':
        continue
    
    if sn == 'Heading 1' and text.startswith('Chapter'):
        if current_ch:
            chapters[current_ch] = current_paras
        current_ch = text
        current_paras = []
    elif sn == 'Heading 1' and 'APPENDIX' in text:
        if current_ch:
            chapters[current_ch] = current_paras
        current_ch = text
        current_paras = []
    elif current_ch:
        current_paras.append((sn, text))

if current_ch:
    chapters[current_ch] = current_paras

print(f"Parsed {len(chapters)} chapters from v2.docx")

# Chapter figure mappings
CH_FIGS = {
    1: [("01_system_architecture", "System Architecture Overview", 0.85)],
    2: [("10_federated_vs_centralized", "Federated vs Centralized Learning Comparison", 0.8)],
    3: [("02_data_flow", "Data Flow Diagram", 0.8),
        ("18_dfd_level0", "DFD Level 0 --- Context Diagram", 0.85),
        ("09_integration_architecture", "End-to-End Integration Architecture", 0.85)],
    4: [("01_system_architecture", "High-Level System Architecture", 0.85),
        ("12_class_diagram", "UML Class Diagram", 0.9),
        ("13_use_case_diagram", "UML Use Case Diagram", 0.85),
        ("14_activity_diagram", "UML Activity Diagram", 0.9),
        ("15_component_diagram", "UML Component Diagram", 0.85),
        ("16_deployment_diagram", "UML Deployment Diagram", 0.85),
        ("17_state_chart", "UML State Chart Diagram --- FL Training Lifecycle", 0.9),
        ("04_fl_sequence", "FL Communication Sequence Diagram", 0.8),
        ("05_aggregation_strategies", "Aggregation Strategies Comparison", 0.8),
        ("03_unet_architecture", "3D Attention U-Net Architecture", 0.85),
        ("08_lstm_architecture", "LSTM Hybrid Architecture", 0.7),
        ("07_progression_pipeline", "Progression Forecasting Pipeline", 0.85)],
    5: [("04_fl_sequence", "Federated Learning Training Sequence", 0.8),
        ("07_progression_pipeline", "Progression Module Workflow", 0.85)],
    6: [("06_experimental_setups", "Experimental Setup Configuration", 0.8)],
}

RESULT_FIGS = {
    7: [
        ("s1_comparison", "Setup 1: Accuracy and Loss Curves Over 100 FL Rounds", 0.95),
        ("s1_fairness", "Setup 1: Per-Client Fairness Analysis", 0.8),
        ("s1_roc_auc", "Setup 1: ROC-AUC Curves", 0.8),
        ("s2_comparison", "Setup 2: Accuracy and Loss Curves Over 100 FL Rounds", 0.95),
        ("s2_fairness", "Setup 2: Per-Client Fairness Analysis", 0.8),
        ("s2_roc_auc", "Setup 2: ROC-AUC Curves", 0.8),
    ]
}

def convert_para_to_latex(sn, text):
    """Convert a paragraph to LaTeX."""
    # Skip ASCII art
    special = sum(1 for c in text if 0x2500 <= ord(c) <= 0x259F)
    if special >= 3:
        return None
    if text.startswith('Next: Proceed to'):
        return None
    
    safe = tex_escape(text)
    
    if sn == 'Heading 2':
        # Extract number if present
        return f"\n\\section{{{safe}}}\n"
    elif sn == 'Heading 3':
        return f"\n\\subsection{{{safe}}}\n"
    elif sn == 'Heading 4':
        return f"\n\\subsubsection{{{safe}}}\n"
    elif sn == 'List Bullet':
        return f"  \\item {safe}\n"
    elif sn == 'List Number':
        return f"  \\item {safe}\n"
    else:
        return f"{safe}\n\n"

def write_chapter(ch_num, title, paras, filename):
    """Write a chapter .tex file."""
    lines = []
    lines.append(f"\\chapter{{{title}}}\n")
    lines.append(f"\\label{{ch:{ch_num}}}\n\n")
    
    in_bullet_list = False
    in_number_list = False
    fig_idx = 0
    figs = CH_FIGS.get(ch_num, [])
    result_figs_list = RESULT_FIGS.get(ch_num, [])
    result_idx = 0
    
    for i, (sn, text) in enumerate(paras):
        # Close open lists if style changes
        if sn != 'List Bullet' and in_bullet_list:
            lines.append("\\end{itemize}\n\n")
            in_bullet_list = False
        if sn != 'List Number' and in_number_list:
            lines.append("\\end{enumerate}\n\n")
            in_number_list = False
        
        # Open lists
        if sn == 'List Bullet' and not in_bullet_list:
            lines.append("\\begin{itemize}\n")
            in_bullet_list = True
        if sn == 'List Number' and not in_number_list:
            lines.append("\\begin{enumerate}\n")
            in_number_list = True
        
        # Insert figures at section boundaries
        if sn == 'Heading 2' and fig_idx < len(figs):
            fname, caption, width = figs[fig_idx]
            lines.append(f"\n\\begin{{figure}}[H]\n\\centering\n")
            lines.append(f"\\includegraphics[width={width}\\textwidth]{{{fname}.png}}\n")
            lines.append(f"\\caption{{{caption}}}\n")
            lines.append(f"\\label{{fig:{fname}}}\n\\end{{figure}}\n\n")
            fig_idx += 1
        
        # Insert result figures for chapter 7
        if ch_num == 7 and sn == 'Heading 3' and result_idx < len(result_figs_list):
            if any(kw in text for kw in ['Accuracy', 'Fairness', 'ROC']):
                fname, caption, width = result_figs_list[result_idx]
                lines.append(f"\n\\begin{{figure}}[H]\n\\centering\n")
                lines.append(f"\\includegraphics[width={width}\\textwidth]{{{fname}.png}}\n")
                lines.append(f"\\caption{{{caption}}}\n")
                lines.append(f"\\label{{fig:{fname}}}\n\\end{{figure}}\n\n")
                result_idx += 1
        
        converted = convert_para_to_latex(sn, text)
        if converted:
            lines.append(converted)
    
    # Close any open lists
    if in_bullet_list:
        lines.append("\\end{itemize}\n")
    if in_number_list:
        lines.append("\\end{enumerate}\n")
    
    # Add chapter summary if not chapter 9
    if ch_num <= 8:
        summaries = {
            1: "This chapter introduced the motivation behind privacy-preserving brain tumor analysis, outlined the three-module pipeline, and established the project scope.",
            2: "This chapter reviewed existing literature on federated learning, brain tumor analysis, and identified key research gaps in fairness-aware FL for healthcare.",
            3: "This chapter analyzed system requirements, proposed the three-module architecture, and confirmed technical, operational, and economic feasibility.",
            4: "This chapter detailed the system design with UML diagrams, data flow diagrams, and mathematical formulations for all three aggregation strategies.",
            5: "This chapter described implementation details of all three modules using Python, PyTorch, MONAI, and SciPy with key code components.",
            6: "This chapter documented the testing methodology including unit, integration, and performance testing under IID and non-IID distributions.",
            7: "This chapter presented comprehensive results: QPSO-FL fairness superiority, clinical-grade segmentation Dice scores, and LSTM progression improvement.",
            8: "This chapter concluded the project, summarizing contributions and outlining future work including differential privacy and real-world deployment.",
        }
        if ch_num in summaries:
            lines.append(f"\n\\section{{Chapter Summary}}\n{summaries[ch_num]}\n")
    
    filepath = os.path.join(OUT, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))
    print(f"  Written {filename} ({len(lines)} lines)")

# Map chapter keys to numbers and titles
ch_map = {}
for key in chapters:
    if key.startswith('Chapter'):
        parts = key.split(':', 1)
        num = int(parts[0].replace('Chapter', '').strip())
        title = parts[1].strip() if len(parts) > 1 else f"Chapter {num}"
        ch_map[num] = (title, chapters[key])

# Write chapter files
filenames = {
    1: "ch1_introduction.tex",
    2: "ch2_literature.tex",
    3: "ch3_system_analysis.tex",
    4: "ch4_system_design.tex",
    5: "ch5_implementation.tex",
    6: "ch6_testing.tex",
    7: "ch7_results.tex",
    8: "ch8_conclusion.tex",
}

print("\nGenerating chapter files...")
for num in sorted(ch_map.keys()):
    if num in filenames and num <= 8:
        title, paras = ch_map[num]
        write_chapter(num, title, paras, filenames[num])

# ---- Chapter 5: Add real code listings ----
print("\nAdding real code to Chapter 5...")
ch5_path = os.path.join(OUT, "ch5_implementation.tex")
code_appendix = """

\\section{Key Code Components}

\\subsection{BrainTumorResNet --- Classification Model}
\\begin{lstlisting}[caption={ResNet-18 Brain Tumor Classifier (model.py)}, label={lst:model}]
""" + read_code_file(os.path.join(SRC, "model.py")) + """\\end{lstlisting}

\\subsection{FederatedClient --- Local Training}
\\begin{lstlisting}[caption={Federated Client (client.py)}, label={lst:client}]
""" + read_code_file(os.path.join(SRC, "client.py")) + """\\end{lstlisting}

\\subsection{QPSOServer --- Novel Aggregation}
\\begin{lstlisting}[caption={QPSO Server Aggregation (server\\_qpso.py)}, label={lst:qpso}]
""" + read_code_file(os.path.join(SRC, "server_qpso.py")) + """\\end{lstlisting}

\\subsection{FedAvgServer --- Baseline Aggregation}
\\begin{lstlisting}[caption={FedAvg Server (server\\_fedavg.py)}, label={lst:fedavg}]
""" + read_code_file(os.path.join(SRC, "server_fedavg.py")) + """\\end{lstlisting}

\\subsection{ResidualLSTM --- Progression Model}
\\begin{lstlisting}[caption={LSTM Hybrid Residual Model (infrastructure\\_lstm.py, lines 273--349)}, label={lst:lstm}]
""" + read_code_file(os.path.join(PROG, "06_hybrid_lstm_infrastructure.py"), 273, 349) + """\\end{lstlisting}
"""

with open(ch5_path, 'a', encoding='utf-8') as f:
    f.write(code_appendix)
print("  Appended code listings to ch5_implementation.tex")

# ---- Appendices ----
print("\nGenerating appendix files...")
appendix_map = {
    'APPENDIX A': ("Federated Learning Comparative Analysis", "appendix_a.tex"),
    'APPENDIX B': ("Segmentation Techniques Comparative Analysis", "appendix_b.tex"),
    'APPENDIX C': ("Progression Forecasting Comparative Analysis", "appendix_c.tex"),
}

for key, paras in chapters.items():
    for app_key, (title, fname) in appendix_map.items():
        if app_key in key:
            lines = [f"\\chapter{{{title}}}\n\\label{{app:{app_key[-1].lower()}}}\n\n"]
            in_bl = False
            in_nl = False
            for sn, text in paras:
                if sn != 'List Bullet' and in_bl:
                    lines.append("\\end{itemize}\n\n")
                    in_bl = False
                if sn != 'List Number' and in_nl:
                    lines.append("\\end{enumerate}\n\n")
                    in_nl = False
                if sn == 'List Bullet' and not in_bl:
                    lines.append("\\begin{itemize}\n")
                    in_bl = True
                if sn == 'List Number' and not in_nl:
                    lines.append("\\begin{enumerate}\n")
                    in_nl = True
                converted = convert_para_to_latex(sn, text)
                if converted:
                    # Skip duplicate Comparative Analysis H1 headings
                    if sn == 'Heading 1' and 'Comparative Analysis' in text:
                        continue
                    lines.append(converted)
            if in_bl: lines.append("\\end{itemize}\n")
            if in_nl: lines.append("\\end{enumerate}\n")
            
            with open(os.path.join(OUT, fname), 'w', encoding='utf-8') as f:
                f.write(''.join(lines))
            print(f"  Written {fname}")
            break

# ---- References .bib ----
print("\nGenerating references.bib...")
bib = r"""@article{mcmahan2017,
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and Arcas, Blaise Ag{\"u}era y},
  title={Communication-Efficient Learning of Deep Networks from Decentralized Data},
  journal={Proc. AISTATS},
  year={2017}
}
@article{li2020fedprox,
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  title={Federated Optimization in Heterogeneous Networks},
  journal={Proc. MLSys},
  year={2020}
}
@article{sun2004qpso,
  author={Sun, Jun and Feng, Bin and Xu, Wenbo},
  title={Particle Swarm Optimization with Particles Having Quantum Behavior},
  journal={Proc. CEC},
  year={2004}
}
@article{zhao2018,
  author={Zhao, Yue and Li, Meng and Lai, Liangzhen and Suda, Naveen and Civin, Damon and Chandra, Vikas},
  title={Federated Learning with Non-IID Data},
  journal={arXiv:1806.00582},
  year={2018}
}
@article{sheller2018,
  author={Sheller, Micah J and Reina, G Anthony and Edwards, Brandon and Martin, Jason and Bakas, Spyridon},
  title={Multi-institutional Deep Learning Modeling Without Sharing Patient Data},
  journal={Scientific Reports},
  volume={10},
  year={2020}
}
@article{ronneberger2015,
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  journal={Proc. MICCAI},
  year={2015}
}
@article{cicek2016,
  author={{\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n and Abdulkadir, Ahmed and Lienkamp, Soeren S and Brox, Thomas and Ronneberger, Olaf},
  title={3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation},
  journal={Proc. MICCAI},
  year={2016}
}
@article{isensee2021,
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  journal={Nature Methods},
  volume={18},
  pages={203--211},
  year={2021}
}
@article{schlemper2019,
  author={Schlemper, Jo and Oktay, Ozan and Schaap, Michiel and Heinrich, Mattias and Kainz, Bernhard and Glocker, Ben and Rueckert, Daniel},
  title={Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images},
  journal={Medical Image Analysis},
  volume={53},
  pages={197--207},
  year={2019}
}
@article{hochreiter1997,
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  title={Long Short-Term Memory},
  journal={Neural Computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997}
}
@article{he2016resnet,
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  title={Deep Residual Learning for Image Recognition},
  journal={Proc. CVPR},
  year={2016}
}
@article{brats2021,
  author={Baid, Ujjwal and others},
  title={The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification},
  journal={arXiv:2107.02314},
  year={2021}
}
@misc{masoud,
  author={Masoud Nickparvar},
  title={Brain Tumor MRI Dataset},
  howpublished={Kaggle},
  year={2021}
}
@misc{brisc,
  author={BRISC},
  title={Brain Tumor MRI Classification Dataset 2025},
  howpublished={Kaggle},
  year={2025}
}
@article{edla2025mnist,
  author={Edla, Divyansh Teja and Indhumathi, L. K.},
  title={Enhancing Federated Learning with Quantum-Inspired PSO: An IID MNIST Study},
  journal={In preparation},
  year={2025}
}
@article{ostrom2021,
  author={Ostrom, Quinn T and others},
  title={CBTRUS Statistical Report: Primary Brain and Other CNS Tumors Diagnosed in the US},
  journal={Neuro-Oncology},
  volume={23},
  year={2021}
}
@article{sultan2019,
  author={Sultan, Hossam H and Salem, Nancy M and Al-Atabany, Walid},
  title={Multi-Classification of Brain Tumor Images Using Deep Neural Network},
  journal={IEEE Access},
  volume={7},
  year={2019}
}
@misc{hipaa,
  title={Health Insurance Portability and Accountability Act},
  year={1996}
}
@article{kennedy1995,
  author={Kennedy, James and Eberhart, Russell},
  title={Particle Swarm Optimization},
  journal={Proc. ICNN},
  year={1995}
}
@article{karimireddy2020scaffold,
  author={Karimireddy, Sai Praneeth and Kale, Satyen and Mohri, Mehryar and Reddi, Sashank and Stich, Sebastian and Suresh, Ananda Theertha},
  title={SCAFFOLD: Stochastic Controlled Averaging for Federated Learning},
  journal={Proc. ICML},
  year={2020}
}
@article{kulkarni2020pfl,
  author={Kulkarni, Viraj and Kulkarni, Milind and Pant, Aniruddha},
  title={Survey of Personalization Techniques for Federated Learning},
  journal={arXiv:2003.08673},
  year={2020}
}
@article{fedpso2021,
  author={Qolomany, Basheer and others},
  title={Particle Swarm Optimized Federated Learning for Industrial IoT and Smart City},
  journal={Computing},
  year={2022}
}
@article{rieke2020,
  author={Rieke, Nicola and others},
  title={The Future of Digital Health with Federated Learning},
  journal={NPJ Digital Medicine},
  volume={3},
  year={2020}
}
"""
with open(os.path.join(BASE, "references.bib"), 'w', encoding='utf-8') as f:
    f.write(bib)
print("  Written references.bib")

# Summary
print("\n" + "="*60)
print("DONE! Thesis LaTeX project ready at:")
print(f"  {BASE}")
print(f"\nFiles created:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  chapters/{f} ({size//1024}KB)")
print(f"  main.tex")
print(f"  references.bib")
print(f"  figures/ ({len(os.listdir(os.path.join(BASE, 'figures')))} images)")
print(f"\nUpload the entire thesis_latex/ folder to Overleaf to compile.")
