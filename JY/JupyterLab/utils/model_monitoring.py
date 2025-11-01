"""
model_monitoring.py
-------------------
Lagged monitoring of model performance and data drift (PSI, CSI).

Triggered via Airflow DAG once new labels become available.
Evaluates:
  1. Performance metrics (AUC, Gini, KS, etc.)
  2. PSI & CSI drift for key variables (vs. training baseline)
  3. Breach logic for redeployment triggers (AUC/Gini immediate, KS requires 2 consecutive breaches)

Alerts are logged only — no DAG-breaking exceptions.
"""

import os, sys, json, yaml, logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def resolve_relative_path(path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.abspath(os.path.join(base_dir, path))

def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def convert_to_serializable(obj):
    """Recursively convert numpy and pandas types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    else:
        return obj

def load_training_reference_snapshot():
    """Read deployed model metadata and return the training reference month (baseline snapshot)."""
    deployed_meta_path = resolve_relative_path("model_store/deployed_model/metadata.json")
    if not os.path.exists(deployed_meta_path):
        logger.warning("⚠️ No deployed model metadata found. PSI baseline cannot be determined.")
        return None
    with open(deployed_meta_path, "r") as f:
        meta = json.load(f)
    training_window = meta.get("training_window", [])
    if len(training_window) >= 2:
        ref_snapshot = training_window[1]  # second element = latest training month
        logger.info(f"📘 PSI reference snapshot (from deployed model): {ref_snapshot}")
        return ref_snapshot
    logger.warning("⚠️ training_window missing or malformed in metadata.json.")
    return None

# ---------------------------------------------------------------------
# 1️⃣ Performance Evaluation
# ---------------------------------------------------------------------
def evaluate_performance(pred_path, label_path, thresholds):
    preds = pd.read_parquet(pred_path)
    labels = pd.read_parquet(label_path)
    df = preds.merge(labels, on="Customer_ID", how="inner")

    y_true = df["label"]
    y_prob = df["prediction_score"]
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_true, y_prob),
        "gini": 2 * roc_auc_score(y_true, y_prob) - 1,
        "ks": ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0]).statistic,
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred)
    }

    logger.info("📊 Performance metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    breach_flags = {
        "auc_threshold_breached": metrics["auc"] < thresholds["auc"],
        "gini_threshold_breached": metrics["gini"] < thresholds["gini"],
        "ks_threshold_breached": metrics["ks"] < thresholds["ks"]
    }
    for metric, breached in breach_flags.items():
        if breached:
            logger.warning(f"⚠️ {metric.replace('_', ' ').upper()} detected breach.")
    return metrics, breach_flags

# ---------------------------------------------------------------------
# 2️⃣ Stability Metrics (PSI + CSI)
# ---------------------------------------------------------------------
def compute_psi(expected, actual, buckets=10):
    eps = 1e-6
    quantiles = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
    expected_bins = np.clip(np.digitize(expected, quantiles, right=True), 1, buckets)
    actual_bins = np.clip(np.digitize(actual, quantiles, right=True), 1, buckets)
    expected_dist = np.bincount(expected_bins, minlength=buckets + 1)[1:] / len(expected_bins)
    actual_dist = np.bincount(actual_bins, minlength=buckets + 1)[1:] / len(actual_bins)
    psi = np.sum((actual_dist - expected_dist) * np.log((actual_dist + eps) / (expected_dist + eps)))
    return psi

def psi_for_categorical(expected, actual):
    exp_dist = expected.value_counts(normalize=True)
    act_dist = actual.value_counts(normalize=True)
    all_cats = exp_dist.index.union(act_dist.index)
    psi = np.sum([
        (act_dist.get(cat, 0) - exp_dist.get(cat, 0)) *
        np.log((act_dist.get(cat, 1e-6)) / (exp_dist.get(cat, 1e-6)))
        for cat in all_cats
    ])
    return psi

def compute_csi(ref_df, curr_df, score_col, var, bins=10):
    """Compute CSI (Characteristic Stability Index) for numeric features."""
    eps = 1e-6
    if score_col not in ref_df.columns or score_col not in curr_df.columns:
        return np.nan
    quantiles = np.nanpercentile(ref_df[score_col], np.linspace(0, 100, bins + 1))
    ref_df = ref_df.copy()
    curr_df = curr_df.copy()
    ref_df["score_bin"] = np.clip(np.digitize(ref_df[score_col], quantiles, right=True), 1, bins)
    curr_df["score_bin"] = np.clip(np.digitize(curr_df[score_col], quantiles, right=True), 1, bins)
    ref_means = ref_df.groupby("score_bin")[var].mean()
    curr_means = curr_df.groupby("score_bin")[var].mean()
    ref_prop = ref_means / ref_means.sum()
    curr_prop = curr_means / curr_means.sum()
    csi = np.sum((curr_prop - ref_prop) * np.log((curr_prop + eps) / (ref_prop + eps)))
    return csi

# ---------------------------------------------------------------------
# 🔍 Combined PSI + CSI Stability Evaluation with Visual Dashboard
# ---------------------------------------------------------------------
def evaluate_stability(feature_dir, ref_snapshot, curr_snapshot,
                       variables_cfg, psi_thresholds, csi_thresholds,
                       save_plots=False, plot_dir=None):
    """
    Evaluate both PSI and CSI for numeric and categorical variables,
    comparing the current snapshot against the baseline (training month).
    Also generates a combined visual dashboard grid for all variables.
    """
    results = []
    ref_df = pd.read_parquet(os.path.join(feature_dir, f"{ref_snapshot.replace('-', '_')}.parquet"))
    curr_df = pd.read_parquet(os.path.join(feature_dir, f"{curr_snapshot.replace('-', '_')}.parquet"))

    logger.info(f"📊 Evaluating stability between {ref_snapshot} (baseline) and {curr_snapshot} (current).")

    # Determine total plots
    total_vars = len(variables_cfg["numeric"]) + len(variables_cfg["categorical"])
    ncols = 2
    nrows = int(np.ceil(total_vars / ncols))

    if save_plots and plot_dir:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3))
        axes = axes.flatten()

    plot_idx = 0

    # ---- Helper: categorize drift severity ----
    def categorize_drift(value, thresholds):
        if value < thresholds["stable"]:
            return "Stable", "black"
        elif value < thresholds["moderate"]:
            return "Moderate", "orange"
        else:
            return "Significant", "red"

    # ---- Numeric Variables ----
    for var in variables_cfg["numeric"]:
        psi_val = compute_psi(ref_df[var], curr_df[var])
        csi_val = abs(np.nanmean(curr_df[var]) - np.nanmean(ref_df[var])) / (np.nanstd(ref_df[var]) + 1e-6)

        psi_cat, psi_color = categorize_drift(psi_val, psi_thresholds)
        csi_cat, csi_color = categorize_drift(csi_val, csi_thresholds)

        results.append({
            "variable": var,
            "type": "numeric",
            "curr_snapshot": curr_snapshot,
            "ref_snapshot": ref_snapshot,
            "psi": float(psi_val),
            "csi": float(csi_val),
            "psi_category": psi_cat,
            "csi_category": csi_cat
        })

        if save_plots and plot_dir:
            ax = axes[plot_idx]
            plot_idx += 1

            ref_vals = ref_df[var].dropna()
            curr_vals = curr_df[var].dropna()

            # Handle zero-variance gracefully
            if ref_vals.nunique() > 1:
                sns.kdeplot(ref_vals, label=f"{ref_snapshot} (Baseline)", fill=True, ax=ax)
            else:
                ax.axvline(ref_vals.mean(), color="blue", linestyle="--", label=f"{ref_snapshot} (Baseline)")
            if curr_vals.nunique() > 1:
                sns.kdeplot(curr_vals, label=f"{curr_snapshot} (Current)", fill=True, ax=ax)
            else:
                ax.axvline(curr_vals.mean(), color="orange", linestyle="--", label=f"{curr_snapshot} (Current)")

            csi_display = f"{csi_val:.3f}" if csi_val is not None else "0.000"
            ax.set_title(
                f"{var}\nPSI={psi_val:.3f} ({psi_cat}) | CSI={csi_display} ({csi_cat})",
                color="red" if psi_cat == "Significant" or csi_cat == "Significant"
                      else "orange" if psi_cat == "Moderate" or csi_cat == "Moderate"
                      else "black",
                fontsize=10
            )
            ax.legend()

    # ---- Categorical Variables ----
    for var in variables_cfg["categorical"]:
        psi_val = psi_for_categorical(ref_df[var], curr_df[var])
        csi_val = abs(psi_val)  # proxy, since CSI isn't typically defined for categorical

        psi_cat, psi_color = categorize_drift(psi_val, psi_thresholds)
        csi_cat, csi_color = categorize_drift(csi_val, csi_thresholds)

        results.append({
            "variable": var,
            "type": "categorical",
            "curr_snapshot": curr_snapshot,
            "ref_snapshot": ref_snapshot,
            "psi": float(psi_val),
            "csi": float(csi_val),
            "psi_category": psi_cat,
            "csi_category": csi_cat
        })

        if save_plots and plot_dir:
            ax = axes[plot_idx]
            plot_idx += 1

            ref_freq = ref_df[var].value_counts(normalize=True)
            curr_freq = curr_df[var].value_counts(normalize=True)
            freq_df = pd.DataFrame({f"{ref_snapshot} (Baseline)": ref_freq,
                                    f"{curr_snapshot} (Current)": curr_freq}).fillna(0)

            freq_df.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
            ax.set_title(
                f"{var}\nPSI={psi_val:.3f} ({psi_cat}) | CSI={csi_val:.3f} ({csi_cat})",
                color="red" if psi_cat == "Significant" or csi_cat == "Significant"
                      else "orange" if psi_cat == "Moderate" or csi_cat == "Moderate"
                      else "black",
                fontsize=10
            )
            ax.set_xlabel("Category")
            ax.set_ylabel("Proportion")

    # ---- Adjust layout and save combined dashboard ----
    if save_plots and plot_dir:
        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
        combined_path = os.path.join(plot_dir, f"stability_dashboard_{curr_snapshot}.png")
        plt.savefig(combined_path, dpi=200)
        plt.close()
        logger.info(f"📊 Combined PSI+CSI dashboard saved to: {combined_path}")

    return pd.DataFrame(results)

# ---------------------------------------------------------------------
# 📈 Candidate Model AUC Trend Visualization (Simplified — AUC only)
# ---------------------------------------------------------------------
def plot_candidate_auc_trend(model_store_dir, deployed_meta_path, output_dir):
    """
    Generate a clean line plot of candidate model AUC scores over time,
    highlighting the currently deployed model.
    """
    import glob

    candidate_paths = glob.glob(os.path.join(model_store_dir, "candidate_models", "**", "metadata.json"), recursive=True)
    auc_records = []

    for path in candidate_paths:
        try:
            with open(path, "r") as f:
                meta = json.load(f)

            # ---- Extract AUC (priority: OOT > VAL > TEST > CV) ----
            auc_val = (
                meta.get("results", {}).get("oot", {}).get("auc")
                or meta.get("results", {}).get("val", {}).get("auc")
                or meta.get("results", {}).get("test", {}).get("auc")
                or meta.get("cv_auc")
            )

            # ---- Derive training end date ----
            if "training_window" in meta and len(meta["training_window"]) >= 2:
                train_end = meta["training_window"][1]
            elif "model_version" in meta:
                train_end = meta["model_version"].split("_")[-1]
            else:
                train_end = os.path.basename(os.path.dirname(path))

            if auc_val and train_end:
                parsed_date = pd.to_datetime(str(train_end).replace("_", "-"), errors="coerce")
                auc_records.append({
                    "model_name": os.path.basename(os.path.dirname(path)),
                    "training_end": parsed_date,
                    "auc": float(auc_val)
                })
        except Exception as e:
            logger.warning(f"⚠️ Could not read {path}: {e}")

    if not auc_records:
        logger.warning("⚠️ No candidate model AUC data found for plotting.")
        return

    df_auc = pd.DataFrame(auc_records).dropna(subset=["training_end"]).sort_values("training_end")

    # ---- Load deployed model info ----
    deployed_auc, deployed_train_end = None, None
    if os.path.exists(deployed_meta_path):
        with open(deployed_meta_path, "r") as f:
            deployed_meta = json.load(f)
        deployed_auc = (
            deployed_meta.get("results", {}).get("oot", {}).get("auc")
            or deployed_meta.get("results", {}).get("val", {}).get("auc")
            or deployed_meta.get("results", {}).get("test", {}).get("auc")
            or deployed_meta.get("cv_auc")
        )
        if "training_window" in deployed_meta and len(deployed_meta["training_window"]) >= 2:
            deployed_train_end = pd.to_datetime(deployed_meta["training_window"][1].replace("_", "-"), errors="coerce")

    # ---- Plot AUC Trend ----
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.lineplot(
        x="training_end", y="auc", data=df_auc,
        marker="o", linewidth=1.5, color="#1f77b4", label="Candidate AUC", ax=ax
    )

    # Label formatting
    ax.set_xlabel("Model Training End Date", fontsize=10)
    ax.set_ylabel("AUC", fontsize=10, color="#1f77b4")
    ax.set_title("📈 Candidate Model AUC Trend Over Time", fontsize=11, pad=12)
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=min(5, len(df_auc))))
    ax.set_xticks(df_auc["training_end"])
    ax.set_xticklabels(
        [d.strftime("%Y-%m-%d") for d in df_auc["training_end"]],
        rotation=20, ha="right"
    )

    # Highlight deployed model
    if deployed_train_end is not None and deployed_auc is not None:
        ax.axvline(deployed_train_end, color="red", linestyle="--", label="Deployed Model")
        ax.text(
            deployed_train_end, deployed_auc + 0.002,
            f"Deployed\nAUC={deployed_auc:.3f}",
            color="red", fontsize=8, ha="left", va="bottom"
        )

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "candidate_auc_trend.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"📊 Candidate model AUC trend saved to: {out_path}")

# ---------------------------------------------------------------------
# 4️⃣ Main Entry
# ---------------------------------------------------------------------
def main(snapshot_date):
    # ---------------------------------------------------------------
    # EARLY EXIT: Skip monitoring if snapshot_date < first valid month
    # ---------------------------------------------------------------
    deployment_date = datetime(2024, 6, 1)   # First deployed model
    first_monitoring_date = deployment_date + pd.DateOffset(months=4)  # 4-month lag
    curr_date = datetime.strptime(snapshot_date, "%Y-%m-%d")

    if curr_date < first_monitoring_date:
        logger.info(
            f"⏭️ Monitoring skipped for {snapshot_date} "
            f"(first monitoring only starts on {first_monitoring_date.strftime('%Y-%m-%d')})."
        )
        return    
    config_path = resolve_relative_path("config/monitoring_config.yaml")
    cfg = load_yaml(config_path)
    gold_dir = resolve_relative_path("datamart/gold")
    feature_dir = os.path.join(gold_dir, "feature_store")
    label_dir = os.path.join(gold_dir, "label_store")
    pred_dir = os.path.join(gold_dir, "model_predictions")
    out_dir = resolve_relative_path(cfg["reporting"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    label_lag_months = 4
    label_snapshot = snapshot_date
    pred_snapshot = (datetime.strptime(snapshot_date, "%Y-%m-%d") -
                     pd.DateOffset(months=label_lag_months)).strftime("%Y-%m-%d")
    pred_path = os.path.join(pred_dir, f"{pred_snapshot.replace('-', '_')}.parquet")
    label_path = os.path.join(label_dir, f"{label_snapshot.replace('-', '_')}.parquet")

    logger.info("🧮 Monitoring alignment check:")
    logger.info(f"  → Label snapshot (ground truth available): {label_snapshot}")
    logger.info(f"  → Prediction snapshot (inference month): {pred_snapshot}")
    logger.info(f"  → Evaluating model performance for predictions made in {pred_snapshot} using labels from {label_snapshot}")

    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        logger.warning("⚠️ Missing prediction or label file. Skipping performance evaluation.")
        metrics, perf_flags = {}, {}
    else:
        metrics, perf_flags = evaluate_performance(
            pred_path, label_path,
            cfg["performance_monitoring"]["performance_thresholds"]
        )

    ref_snapshot = load_training_reference_snapshot()
    curr_snapshot = pred_snapshot
    logger.info("🧮 PSI alignment check:")
    logger.info(f"  → Baseline (training end): {ref_snapshot}")
    logger.info(f"  → Current (inference month): {curr_snapshot}")
    logger.info(f"  → Monitoring month (label availability): {snapshot_date}")

    psi_df = pd.DataFrame()
    if ref_snapshot:
        psi_df = evaluate_stability(
            feature_dir, ref_snapshot, curr_snapshot,
            cfg["psi_monitoring"]["variables"],
            cfg["psi_monitoring"]["psi_thresholds"],
            cfg.get("csi_monitoring", {}).get("csi_thresholds", {"stable":0.1,"moderate":0.25,"significant":0.5}),
            save_plots=cfg["reporting"]["save_plots"],
            plot_dir=out_dir
        )

    auc_breach = perf_flags.get("auc_threshold_breached", False)
    gini_breach = perf_flags.get("gini_threshold_breached", False)
    ks_breach = perf_flags.get("ks_threshold_breached", False)
    breach_flag = False
    if auc_breach or gini_breach:
        breach_flag = True
        logger.warning("🚨 AUC or Gini threshold breached — immediate redeployment recommended.")
    elif ks_breach:
        prev_files = [f for f in os.listdir(out_dir) if f.startswith("monitoring_report_")]
        if prev_files:
            prev_latest = sorted(prev_files)[-1]
            with open(os.path.join(out_dir, prev_latest), "r") as f:
                prev_report = json.load(f)
            prev_ks_breach = prev_report.get("ks_threshold_breached", False)
            if prev_ks_breach:
                breach_flag = True
                logger.warning("🚨 KS breached for two consecutive months — triggering redeployment.")
            else:
                logger.warning("⚠️ KS breached this month (first occurrence).")
        else:
            logger.info("ℹ️ No previous report found — cannot check KS breach history.")
    else:
        logger.info("✅ All monitored metrics within acceptable thresholds.")

    # Use prediction month (inference month) for report tracking
    report = {
        "prediction_snapshot": pred_snapshot,
        "label_snapshot": snapshot_date,
        "performance_metrics": metrics,
        "auc_threshold_breached": auc_breach,
        "gini_threshold_breached": gini_breach,
        "ks_threshold_breached": ks_breach,
        "breach_flag": breach_flag,
        "psi_summary": psi_df.to_dict(orient="records")
    }
    
    # Save file under prediction month instead of label month
    report_filename = f"monitoring_report_{pred_snapshot.replace('-', '_')}.json"
    report_path = os.path.join(out_dir, report_filename)
    
    with open(report_path, "w") as f:
        json.dump(convert_to_serializable(report), f, indent=2)
    
    logger.info(f"💾 Saved monitoring report to {report_path}")
    logger.info(f"📅 Report corresponds to inference run on {pred_snapshot} (evaluated with labels from {snapshot_date})")

    # -----------------------------------------------------------------
    # Candidate Model AUC Trend Dashboard
    # -----------------------------------------------------------------
    model_store_dir = resolve_relative_path("model_store")
    deployed_meta_path = os.path.join(model_store_dir, "deployed_model", "metadata.json")
    plot_candidate_auc_trend(model_store_dir, deployed_meta_path, out_dir)

    logger.info("✅ Monitoring run completed successfully")

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Model Performance")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()

    main(snapshot_date=args.snapshot_date)