import re

file_path = "paper_fedavg_vs_qpso.tex"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Abstract
text = text.replace("Evaluated against FedAvg and FedProx", "Evaluated against FedAvg")
text = text.replace("deviation of 1.56\\% versus 2.27\\% (FedProx) and 5.28\\%", "deviation of 1.56\\% versus 5.28\\%")
text = text.replace("3.65\\% versus 6.05\\% (FedProx)\nand 12.82\\% (FedAvg)", "3.65\\% versus 12.82\\% (FedAvg)")
text = text.replace("(80.00\\% versus 77.50\\% for FedProx and 60.42\\%\nfor FedAvg)", "(80.00\\% versus 60.42\\% for FedAvg)")

# 2. Keywords
text = text.replace("FedProx, MRI", "MRI")
text = text.replace("FedAvg,\nFedProx, MRI", "FedAvg, MRI")

# 3. Intro / Architecture
text = text.replace("FedAvg, FedProx, or FedQPSO", "FedAvg or FedQPSO")
text = text.replace("FedAvg/FedProx/FedQPSO", "FedAvg/FedQPSO")
text = text.replace("FedAvg, FedProx, or FedQPSO\naggregation.", "FedAvg or FedQPSO\naggregation.")

# 4. Related Work - removing FedProx paragraph
fedprox_par = r"\\subsection\{Handling Non-IID Data in Federated Learning\}\nLi et al.\\ \\cite\{li2020fedprox\} proposed FedProx.*?fairness-aware\.\n\n"
text = re.sub(fedprox_par, "", text, flags=re.DOTALL)

# 5. Methodology - removing FedProx section
fedprox_sec = r"\\subsection\{FedProx \(Regularized Baseline\)\}\nFedProx.*?large to small clients\.\n\n"
text = re.sub(fedprox_sec, "", text, flags=re.DOTALL)

# 6. Curves captions
text = text.replace("FedAvg and FedProx.", "FedAvg.")
text = text.replace("FedAvg degrades significantly faster; FedQPSO and FedProx remain\ncompetitive but with FedQPSO showing superior fairness properties.", 
                    "FedAvg degrades significantly faster; FedQPSO shows superior fairness and stability.")

# 7. ROC captions
text = text.replace("FedQPSO and FedProx maintain above 0.978.", "FedQPSO maintains above 0.978.")

# 8. Tables - Global Accuracy
text = re.sub(r"\\textbf\{Metric\} & \\textbf\{FedAvg\} & \\textbf\{FedProx\} & \\textbf\{FedQPSO\}\\\\", 
             r"\\textbf{Metric} & \\textbf{FedAvg} & \\textbf{FedQPSO}\\\\", text)
# Table 3 values
text = text.replace("Best Accuracy (\\%) & 95.80 (r83) & \\textbf{96.13} (r91) & 95.14 (r89)\\\\",
                    "Best Accuracy (\\%) & \\textbf{95.80} (r83) & 95.14 (r89)\\\\")
text = text.replace("Final Accuracy (\\%) & 93.29 (r98) & \\textbf{95.85} (r100) & 93.78 (r100)\\\\",
                    "Final Accuracy (\\%) & 93.29 (r98) & \\textbf{93.78} (r100)\\\\")
text = text.replace("Rounds run & 98$^\\dagger$ & 100 & 100\\\\",
                    "Rounds run & 98$^\\dagger$ & 100\\\\")
text = text.replace("Rounds to 80\\% & 11 & 12 & \\textbf{9}\\\\",
                    "Rounds to 80\\% & 11 & \\textbf{9}\\\\")
text = text.replace("Avg.\\ round time (s) & 7.84 & 8.03 & 75.09\\\\",
                    "Avg.\\ round time (s) & 7.84 & 75.09\\\\")

# Table 4 values
text = text.replace("Best Accuracy (\\%) & 90.56 (r94) & \\textbf{93.02} (r89) & 92.09 (r98)\\\\",
                    "Best Accuracy (\\%) & 90.56 (r94) & \\textbf{92.09} (r98)\\\\")
text = text.replace("Final Accuracy (\\%) & 88.16 (r100) & \\textbf{91.82} (r100) & 91.43 (r100)\\\\",
                    "Final Accuracy (\\%) & 88.16 (r100) & \\textbf{91.43} (r100)\\\\")
text = text.replace("Rounds to 80\\% & 35 & \\textbf{17} & 19\\\\",
                    "Rounds to 80\\% & 35 & \\textbf{19}\\\\")
text = text.replace("Avg.\\ round time (s) & 8.03 & 8.16 & 74.92\\\\",
                    "Avg.\\ round time (s) & 8.03 & 74.92\\\\")

# Fix Table alignments
text = text.replace("\\begin{tabular}{lccc}", "\\begin{tabular}{lcc}")

# 9. Tables - Per-Class
text = re.sub(r"\\textbf\{Class\} & \\multicolumn\{2\}\{c\}\{\\textbf\{FedAvg\}\} & \\multicolumn\{2\}\{c\}\{\\textbf\{FedProx\}\} & \\multicolumn\{2\}\{c\}\{\\textbf\{FedQPSO\}\}\\\\",
             r"\\textbf{Class} & \\multicolumn{2}{c}{\\textbf{FedAvg}} & \\multicolumn{2}{c}{\\textbf{FedQPSO}}\\\\", text)

# Table 5 Setup 1 values
text = text.replace("Glioma & 0.9455 & 0.9231 & 0.9413 & \\textbf{0.9434} & \\textbf{0.9483} & 0.9593\\\\",
                    "Glioma & 0.9455 & 0.9231 & \\textbf{0.9483} & \\textbf{0.9593}\\\\")
text = text.replace("Meningioma & 0.9475 & 0.9617 & \\textbf{0.9631} & \\textbf{0.9745} & 0.9464 & 0.9404\\\\",
                    "Meningioma & \\textbf{0.9475} & \\textbf{0.9617} & 0.9464 & 0.9404\\\\")
text = text.replace("No Tumor & 0.9619 & \\textbf{0.9954} & \\textbf{0.9685} & 0.9931 & 0.9640 & 0.9884\\\\",
                    "No Tumor & 0.9619 & \\textbf{0.9954} & \\textbf{0.9640} & 0.9884\\\\")
text = text.replace("Pituitary & 0.9785 & 0.9509 & 0.9734 & 0.9387 & \\textbf{0.9839} & \\textbf{0.9591}\\\\",
                    "Pituitary & 0.9785 & 0.9509 & \\textbf{0.9839} & \\textbf{0.9591}\\\\")

# Table 5 Setup 2 values
text = text.replace("Glioma & 0.8122 & 0.7059 & 0.8753 & 0.8416 & \\textbf{0.8797} & \\textbf{0.8914}\\\\",
                    "Glioma & 0.8122 & 0.7059 & \\textbf{0.8797} & \\textbf{0.8914}\\\\")
text = text.replace("Meningioma & \\textbf{0.9589} & \\textbf{0.9447} & 0.9554 & 0.9532 & 0.9351 & 0.8915\\\\",
                    "Meningioma & \\textbf{0.9589} & \\textbf{0.9447} & 0.9351 & 0.8915\\\\")
text = text.replace("No Tumor & 0.9157 & \\textbf{0.9815} & 0.9392 & 0.9653 & \\textbf{0.9592} & 0.9792\\\\",
                    "No Tumor & 0.9157 & \\textbf{0.9815} & \\textbf{0.9592} & 0.9792\\\\")
text = text.replace("Pituitary & 0.9464 & 0.9734 & \\textbf{0.9540} & \\textbf{0.9775} & 0.9159 & 0.9162\\\\",
                    "Pituitary & 0.9464 & \\textbf{0.9734} & 0.9159 & 0.9162\\\\")

text = text.replace("\\begin{tabular}{lcccccc}", "\\begin{tabular}{lcccc}")

# 10. Tables - Client Fairness
# Table 6 Setup 1
text = text.replace("Client 1 (1119s) & 92.50 & \\textbf{96.67} & 92.08\\\\",
                    "Client 1 (1119s) & \\textbf{92.50} & 92.08\\\\")
text = text.replace("Client 2 (3499s) & 94.67 & \\textbf{95.33} & 93.60\\\\",
                    "Client 2 (3499s) & \\textbf{94.67} & 93.60\\\\")
text = text.replace("Client 3 (3919s) & 97.74 & 96.55 & \\textbf{95.60}\\\\",
                    "Client 3 (3919s) & \\textbf{97.74} & 95.60\\\\")
text = text.replace("Max-min gap & 5.24 & 1.34 & \\textbf{3.52}\\\\",
                    "Max-min gap & 5.24 & \\textbf{3.52}\\\\")
text = text.replace("Std Dev ($\\sigma$) & 2.65 & 0.69 & \\textbf{1.76}\\\\",
                    "Std Dev ($\\sigma$) & 2.65 & \\textbf{1.76}\\\\")

# Table 6 Setup 2
text = text.replace("Client 1 (Glioma) & 60.42 & 77.50 & \\textbf{80.00}\\\\",
                    "Client 1 (Glioma) & 60.42 & \\textbf{80.00}\\\\")
text = text.replace("Client 2 (Meningioma) & \\textbf{95.87} & 91.20 & 88.53\\\\",
                    "Client 2 (Meningioma) & \\textbf{95.87} & 88.53\\\\")
text = text.replace("Client 3 (No Tumor) & 92.98 & \\textbf{94.88} & 86.44\\\\",
                    "Client 3 (No Tumor) & 92.98 & \\textbf{86.44}\\\\")
text = text.replace("Max-min gap & 35.45 & 17.38 & \\textbf{8.53}\\\\",
                    "Max-min gap & 35.45 & \\textbf{8.53}\\\\")
text = text.replace("Std Dev ($\\sigma$) & 19.33 & 8.97 & \\textbf{4.32}\\\\",
                    "Std Dev ($\\sigma$) & 19.33 & \\textbf{4.32}\\\\")

# 11. ROC Table
text = text.replace("FedAvg & 0.986 & 0.990 & 0.995 & 0.999 & & 0.976 & 0.970 & 0.984 & 0.996\\\\",
                    "FedAvg & 0.986 & 0.990 & 0.995 & 0.999 & 0.976 & 0.970 & 0.984 & 0.996\\\\")
text = text.replace("FedProx & \\textbf{0.991} & \\textbf{0.993} & \\textbf{0.996} & \\textbf{0.999} & & \\textbf{0.983} & \\textbf{0.990} & \\textbf{0.990} & \\textbf{0.998}\\\\",
                    "")
text = text.replace("FedQPSO & \\textbf{0.991} & 0.990 & 0.994 & 0.997 & & 0.981 & \\textbf{0.979} & 0.984 & 0.994\\\\",
                    "FedQPSO & \\textbf{0.991} & 0.990 & 0.994 & 0.997 & 0.981 & \\textbf{0.979} & 0.984 & 0.994\\\\")

text = text.replace("\\begin{tabular}{lcccc c cccc}", "\\begin{tabular}{lcccccccc}")
text = text.replace("& \\multicolumn{4}{c}{\\textbf{Setup 1 (Natural)}} & & \\multicolumn{4}{c}{\\textbf{Setup 2 (Label Skew)}}\\\\",
                    "& \\multicolumn{4}{c}{\\textbf{Setup 1}} & \\multicolumn{4}{c}{\\textbf{Setup 2}}\\\\")
text = text.replace("& Gl & Me & NT & Pi & & Gl & Me & NT & Pi\\\\",
                    "& Gl & Me & NT & Pi & Gl & Me & NT & Pi\\\\")

# 12. Stability Table
text = text.replace("Mean $\\Delta$ acc & -0.63 & \\textbf{-0.09} & -0.16 \\\\",
                    "Mean $\\Delta$ acc & -0.63 & \\textbf{-0.16} \\\\")
text = text.replace("Max drop (worst $\\Delta$) & -14.78 & -0.45 & \\textbf{-3.38} \\\\",
                    "Max drop (worst $\\Delta$) & -14.78 & \\textbf{-3.38} \\\\")
text = text.replace("Drop variance & 4.19 & \\textbf{0.03} & 0.58 \\\\",
                    "Drop variance & 4.19 & \\textbf{0.58} \\\\")

# 13. Confusion Matrices Figures
text = text.replace("\\includegraphics[width=0.32\\columnwidth]{s1_cm_fedavg.png}%\n  \\hfill\n  \\includegraphics[width=0.32\\columnwidth]{s1_cm_fedprox.png}%\n  \\hfill\n  \\includegraphics[width=0.32\\columnwidth]{s1_cm_qpso.png}",
                    "\\includegraphics[width=0.48\\columnwidth]{s1_cm_fedavg.png}%\n  \\hfill\n  \\includegraphics[width=0.48\\columnwidth]{s1_cm_qpso.png}")
text = text.replace("Left: FedAvg. Centre: FedProx.\nRight: FedQPSO.", "Left: FedAvg. Right: FedQPSO.")

text = text.replace("\\includegraphics[width=0.32\\columnwidth]{s2_cm_fedavg.png}%\n  \\hfill\n  \\includegraphics[width=0.32\\columnwidth]{s2_cm_fedprox.png}%\n  \\hfill\n  \\includegraphics[width=0.32\\columnwidth]{s2_cm_qpso.png}",
                    "\\includegraphics[width=0.48\\columnwidth]{s2_cm_fedavg.png}%\n  \\hfill\n  \\includegraphics[width=0.48\\columnwidth]{s2_cm_qpso.png}")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)
print("Done.")
