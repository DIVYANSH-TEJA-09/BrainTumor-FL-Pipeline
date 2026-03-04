"""
Results analysis — convergence metrics, statistical tests, LaTeX export.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats


# ---------- convergence -------------------------------------------------------

def calculate_convergence_metrics(df, target_accuracy=80.0):
    """
    Compute key convergence metrics from a training history DataFrame.

    Returns dict with: final_acc, best_acc, round_to_target,
                       avg_improvement_per_round, total_time_min.
    """
    final = df["global_test_acc"].iloc[-1]
    best  = df["global_test_acc"].max()

    reached = df[df["global_test_acc"] >= target_accuracy]
    r2t = int(reached["round"].min()) if len(reached) else None

    avg_imp = (best - df["global_test_acc"].iloc[0]) / len(df)

    return {
        "final_acc": round(final, 2),
        "best_acc": round(best, 2),
        "round_to_target": r2t,
        "avg_improvement_per_round": round(avg_imp, 4),
        "avg_round_time_s": round(df["round_time"].mean(), 2),
        "total_time_min": round(df["round_time"].sum() / 60, 2),
    }


# ---------- statistical comparison --------------------------------------------

def perform_statistical_analysis(df_fedavg, df_qpso):
    """
    Paired t-test + Cohen's d on per-round global accuracies.
    Returns dict with p_value, effect_size, is_significant.
    """
    a = df_fedavg["global_test_acc"].values
    b = df_qpso["global_test_acc"].values
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    t_stat, p_val = stats.ttest_rel(b, a)
    diff = b - a
    d = diff.mean() / (diff.std(ddof=1) + 1e-8)   # Cohen's d

    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "cohens_d": round(float(d), 4),
        "is_significant": bool(p_val < 0.05),
        "mean_improvement": round(float(diff.mean()), 4),
    }


# ---------- LaTeX table -------------------------------------------------------

def generate_latex_table(comparison_df, caption="FedAvg vs QPSO-FL"):
    """Convert a comparison DataFrame to a LaTeX table string."""
    latex = comparison_df.to_latex(index=False, float_format="%.2f",
                                  caption=caption, label="tab:comparison")
    return latex


# ---------- executive summary -------------------------------------------------

def create_executive_summary(df_fedavg, df_qpso, save_path=None):
    """Build a concise results summary dict (and optionally save as JSON)."""
    m_fa = calculate_convergence_metrics(df_fedavg)
    m_qp = calculate_convergence_metrics(df_qpso)
    st   = perform_statistical_analysis(df_fedavg, df_qpso)

    # client fairness
    fa_std = np.std([df_fedavg[f"client{i}_val_acc"].iloc[-1]
                     for i in range(1, 4)])
    qp_std = np.std([df_qpso[f"client{i}_val_acc"].iloc[-1]
                     for i in range(1, 4)])

    summary = {
        "fedavg": m_fa,
        "qpso":   m_qp,
        "accuracy_improvement_pct":
            round(m_qp["best_acc"] - m_fa["best_acc"], 2),
        "fedavg_client_std": round(float(fa_std), 2),
        "qpso_client_std":   round(float(qp_std), 2),
        "statistical_test": st,
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary saved → {save_path}")

    return summary


# ---------- comparison DataFrame builder --------------------------------------

def build_comparison_df(df_fedavg, df_qpso):
    """Create a tidy comparison table suitable for display or LaTeX export."""
    m_fa = calculate_convergence_metrics(df_fedavg)
    m_qp = calculate_convergence_metrics(df_qpso)

    fa_std = np.std([df_fedavg[f"client{i}_val_acc"].iloc[-1]
                     for i in range(1, 4)])
    qp_std = np.std([df_qpso[f"client{i}_val_acc"].iloc[-1]
                     for i in range(1, 4)])

    rows = [
        ("Final Global Accuracy (%)", m_fa["final_acc"], m_qp["final_acc"]),
        ("Best Global Accuracy (%)",  m_fa["best_acc"],  m_qp["best_acc"]),
        ("Rounds to 80 %",
         m_fa["round_to_target"] or "N/A",
         m_qp["round_to_target"] or "N/A"),
        ("Avg Round Time (s)",        m_fa["avg_round_time_s"],
                                      m_qp["avg_round_time_s"]),
        ("Total Training Time (min)", m_fa["total_time_min"],
                                      m_qp["total_time_min"]),
        ("Client Acc Std Dev",        round(float(fa_std), 2),
                                      round(float(qp_std), 2)),
    ]

    return pd.DataFrame(rows, columns=["Metric", "FedAvg", "QPSO-FL"])
