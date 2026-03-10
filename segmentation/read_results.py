import json, os

base = r"d:\Major_Project\FL_QPSO_FedAvg\federated_learning\results"

setups = [
    ("Case 1: Transfer Learning (ResNet-18 Pretrained)", "results_transfer_learning",
     [("Natural", "Natural Setup"), ("Moderate Skew", "Moderate Skew")]),
    ("Case 2: From Scratch (ResNet-18)", "results_no_pretrained",
     [("Natural", "natural setup"), ("Moderate Skew", "moderate skew")]),
    ("Case 3: Naive QPSO (SimpleCNN)", "results_lightweight_model",
     [("Natural", "natural setup"), ("Moderate Skew", "moderate skew")]),
    ("Case 4: Layer-by-Layer QPSO (SimpleCNN)", "results_layer_by_layer_QPSO",
     [("Natural", "Setup_1"), ("Moderate Skew", "Setup_2")]),
]

for case_name, folder, configs in setups:
    print(f"\n===== {case_name} =====")
    for setup_label, subfolder in configs:
        for subpath in [
            f"{subfolder}/results/executive_summary.json",
            f"{subfolder}/results_phase4/executive_summary.json",
        ]:
            p = os.path.join(base, folder, subpath)
            if os.path.exists(p):
                with open(p) as f:
                    d = json.load(f)
                print(f"  [{setup_label}]")
                for algo in ["fedavg", "fedprox", "qpso"]:
                    if algo in d:
                        info = d[algo]
                        best = info["best_acc"]
                        final = info["final_acc"]
                        r80 = info.get("round_to_target", "N/A")
                        std = info.get("client_std", "N/A")
                        print(f"    {algo:8s}: best={best}%  final={final}%  round_to_80={r80}  client_std={std}")
                if "stats_fa_qp" in d:
                    s = d["stats_fa_qp"]
                    print(f"    [Stats FedAvg vs QPSO] p={s['p_value']:.6f}  cohen_d={s['cohens_d']}  significant={s['significant']}")
                break
