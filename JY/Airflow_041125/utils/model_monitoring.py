"""
model_monitoring.py
-------------------
Monitors monthly performance for a deployed regression model.

Rules:
- deployment_date = from monitoring_config.yaml
- first_monitoring_date = deployment_date + 1 month
- If snapshot_date < first_monitoring_date → skip monitoring
- Else:
  1) Load labels for <snapshot_date> from datamart/gold/label_store/
  2) Load predictions for <prediction_snapshot> = (snapshot_date - 1 month)
     from datamart/gold/predictions/
  3) Evaluate RMSE / MAE / R2 against ground truth
  4) Compare vs. deployed model’s baseline (OOT metrics) with thresholds
  5) Compute PSI for configured variables:
     between <prediction_snapshot> features and <deployed_model.training_window_end> features
  6) Compute CSI on predicted_sales between same two periods
  7) Generate:
      - 10-chart drift dashboard (PSI only)
      - candidate models performance chart (optional)
      - monitoring_report_<snapshot_date>.json with metrics, flags, PSI/CSI summary

Outputs:
- datamart/gold/model_monitoring/monitoring_report_<YYYY-MM-DD>.json
- datamart/gold/model_monitoring/drift_dashboard_<YYYY-MM-DD>.png
- datamart/gold/model_monitoring/candidate_models_perf_<YYYY-MM-DD>.png
"""

import os, json, logging, argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_monitoring")


# ---------------------------
# Path helpers
# ---------------------------
def resolve_relative_path(rel_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.join(base_dir, rel_path)


# ---------------------------
# IO helpers
# ---------------------------
def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_parquet_exact(base_dir: str, snapshot_date: str) -> pd.DataFrame:
    """Load a parquet file strictly matching the given snapshot date."""
    candidates = [
        os.path.join(base_dir, f"{snapshot_date}.parquet"),
        os.path.join(base_dir, f"{snapshot_date.replace('-', '_')}.parquet"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_parquet(p)
    raise FileNotFoundError(f"No parquet found for {snapshot_date} under {base_dir}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Metrics
# ---------------------------
def eval_regression(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


# ---------------------------
# PSI / CSI
# ---------------------------
def _make_bins(ref_series: pd.Series, n_bins: int, method: str):
    ser = ref_series.dropna()
    if ser.empty:
        return np.array([-np.inf, np.inf])
    if method == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(ser, qs))
        if len(edges) < 2:
            edges = np.array([ser.min() - 1e-9, ser.max() + 1e-9])
        return edges
    elif method == "uniform":
        lo, hi = ser.min(), ser.max()
        if lo == hi:
            lo, hi = lo - 1e-9, hi + 1e-9
        return np.linspace(lo, hi, n_bins + 1)
    else:
        raise ValueError("bin_method must be 'quantile' or 'uniform'")

def _proportions(x: pd.Series, bins: np.ndarray):
    counts, _ = np.histogram(x.dropna(), bins=bins)
    counts = counts.astype(float)
    counts += 1e-6
    return counts / counts.sum()

def compute_psi(ref: pd.Series, cur: pd.Series, n_bins: int = 10, bin_method: str = "quantile") -> float:
    bins = _make_bins(ref, n_bins, bin_method)
    P = _proportions(ref, bins)
    Q = _proportions(cur, bins)
    return float(np.sum((P - Q) * np.log(P / Q)))

def compute_csi(ref: pd.Series, cur: pd.Series, n_bins: int = 10, bin_method: str = "quantile") -> float:
    bins = _make_bins(ref, n_bins, bin_method)
    P = _proportions(ref, bins)
    Q = _proportions(cur, bins)
    return float(np.sum(np.abs(P - Q)))


# ---------------------------
# Severity helpers
# ---------------------------
def severity_from_thresholds(value: float, thresholds: dict) -> str:
    stable_t = thresholds["stable"]
    moderate_t = thresholds["moderate"]
    if value < stable_t:
        return "stable"
    elif value < moderate_t:
        return "moderate"
    else:
        return "significant"

def color_for_severity(sev: str) -> str:
    return {"stable": "tab:green", "moderate": "tab:orange", "significant": "tab:red"}.get(sev, "tab:gray")


# ---------------------------
# PSI-only Drift Dashboard
# ---------------------------
def plot_drift_dashboard(ref_df: pd.DataFrame,
                         cur_df: pd.DataFrame,
                         variables: list,
                         psi_thr: dict,
                         save_path: str,
                         n_bins: int = 10,
                         bin_method: str = "quantile",
                         title_suffix: str = ""):
    """
    Improved PSI-only drift dashboard:
    - Handles numeric vs categorical variables robustly
    - Automatically clips long tails for readability
    - Retains correct PSI computation on full data
    """
    n = len(variables)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, var in enumerate(variables):
        ax = axes[i]
        if var not in ref_df.columns or var not in cur_df.columns:
            ax.axis("off")
            ax.set_title(f"{var}\n(Missing in one dataset)", color="gray")
            continue

        ref = ref_df[var].dropna()
        cur = cur_df[var].dropna()
        if ref.empty and cur.empty:
            ax.axis("off")
            ax.set_title(f"{var}\n(No data in both periods)", color="gray")
            continue

        # Compute PSI on full data
        psi_val = compute_psi(ref, cur, n_bins, bin_method)
        psi_sev = severity_from_thresholds(abs(psi_val), psi_thr)
        title_color = color_for_severity(psi_sev)

        # Detect numeric vs categorical feature
        is_numeric = np.issubdtype(ref.dtype, np.number) and np.issubdtype(cur.dtype, np.number)

        if is_numeric and ref.nunique() > 2:
            # --- Numeric variable ---
            # Visual clipping for readability (does NOT affect PSI)
            ref_clip = ref.clip(upper=ref.quantile(0.99))
            cur_clip = cur.clip(upper=cur.quantile(0.99))
            bins = _make_bins(ref, n_bins, bin_method)

            ax.hist(ref_clip, bins=bins, alpha=0.5, density=True, label="Reference")
            ax.hist(cur_clip, bins=bins, alpha=0.5, density=True, label="Current")

            # Log scale for skewed variables
            if (ref >= 0).all() and ref.max() / (ref[ref>0].min() if (ref>0).any() else 1) > 1000:
                ax.set_xscale("log")

            ax.set_ylabel("Density")

        else:
            # --- Categorical variable ---
            unique_vals = sorted(list(set(ref.unique()) | set(cur.unique())))
            ref_counts = ref.value_counts(normalize=True).reindex(unique_vals, fill_value=0)
            cur_counts = cur.value_counts(normalize=True).reindex(unique_vals, fill_value=0)
            x = np.arange(len(unique_vals))
            width = 0.4

            ax.bar(x - width/2, ref_counts.values, width=width, alpha=0.6, label="Reference")
            ax.bar(x + width/2, cur_counts.values, width=width, alpha=0.6, label="Current")

            # Annotate percentage differences
            for j, val in enumerate(unique_vals):
                diff = cur_counts[val] - ref_counts[val]
                if abs(diff) > 0.001:
                    ax.text(x[j], max(ref_counts[val], cur_counts[val]) + 0.002,
                            f"{diff:+.1%}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in unique_vals], rotation=45, ha="right")
            ax.set_ylabel("Proportion")

        # Title formatting
        ax.set_title(f"{var}\nPSI={psi_val:.3f}", color=title_color, fontsize=11)
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.2)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Feature Drift Dashboard (PSI Only) {title_suffix}", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=160)
    plt.close(fig)

# ---------------------------
# Candidate models chart
# ---------------------------
def plot_candidate_models_performance(candidate_dir: str, deployed_version: str, save_path: str):
    if not os.path.exists(candidate_dir):
        return False
    rows = []
    for d in os.listdir(candidate_dir):
        full = os.path.join(candidate_dir, d)
        if not os.path.isdir(full):
            continue
        meta_path = os.path.join(full, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            r2 = meta.get("metrics", {}).get("oot", {}).get("r2")
            if r2 is not None:
                rows.append((d, r2))
        except Exception:
            continue

    if not rows:
        return False

    rows = sorted(rows, key=lambda x: x[0])
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = ["tab:blue" if lab != deployed_version else "tab:green" for lab in labels]

    plt.figure(figsize=(12, 4))
    plt.bar(labels, values, color=colors)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("OOT R²")
    plt.title("Candidate Models - OOT R² (green = deployed)")
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=160)
    plt.close()
    return True


# ---------------------------
# Main monitoring logic
# ---------------------------
def main(snapshot_date: str):
    logger.info(f"🚀 Starting monitoring for snapshot_date={snapshot_date}")

    cfg_path     = resolve_relative_path("config/monitoring_config.yaml")
    deployed_dir = resolve_relative_path("model_store/deployed_model")
    label_dir    = resolve_relative_path("datamart/gold/label_store")
    pred_dir     = resolve_relative_path("datamart/gold/predictions")
    feat_dir     = resolve_relative_path("datamart/gold/feature_store")
    out_dir      = resolve_relative_path("datamart/gold/model_monitoring")
    ensure_dir(out_dir)

    cfg = load_yaml(cfg_path)
    dep_date = datetime.strptime(cfg["deployment_date"], "%Y-%m-%d").date()
    first_monitoring_date = (datetime.combine(dep_date, datetime.min.time()) + relativedelta(months=1)).date()
    snap_dt = datetime.strptime(snapshot_date, "%Y-%m-%d").date()

    if snap_dt < first_monitoring_date:
        logger.info(f"⏭️ Skipped: first monitoring date is {first_monitoring_date}")
        return

    meta_path = os.path.join(deployed_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Deployed model metadata.json not found.")
    deployed_meta = json.load(open(meta_path, "r", encoding="utf-8"))

    join_keys        = deployed_meta.get("join_keys", ["store_nbr", "family"])
    target_col       = deployed_meta.get("target_col", "sales")
    baseline_metrics = deployed_meta.get("metrics", {}).get("oot", {})
    baseline_r2      = baseline_metrics.get("r2")
    baseline_rmse    = baseline_metrics.get("rmse")
    baseline_mae     = baseline_metrics.get("mae")
    training_window_end = deployed_meta.get("training_window_end")

    pred_snapshot_dt = (datetime.combine(snap_dt, datetime.min.time()) - relativedelta(months=1)).date()
    pred_snapshot = pred_snapshot_dt.strftime("%Y-%m-%d")
    ref_snapshot  = training_window_end

    logger.info(f"📅 Labels={snapshot_date}, Predictions={pred_snapshot}")
    logger.info(f"📌 Drift reference (training window end): {ref_snapshot}")

    # --- Load data
    labels_df = load_parquet_exact(label_dir, snapshot_date)
    preds_df  = load_parquet_exact(pred_dir, pred_snapshot)

    if target_col not in labels_df.columns:
        raise KeyError(f"Target '{target_col}' not in labels.")
    if "predicted_sales" not in preds_df.columns:
        raise KeyError("'predicted_sales' not in predictions.")

    eval_df = labels_df[join_keys + [target_col]].merge(
        preds_df[join_keys + ["predicted_sales"]],
        on=join_keys,
        how="inner"
    )
    if eval_df.empty:
        raise ValueError("No overlapping rows for evaluation.")

    metrics_curr = eval_regression(eval_df[target_col], eval_df["predicted_sales"])

    # --- Performance breach evaluation with persistence logic ----
    thr = cfg["performance_monitoring"]["thresholds"]

    breaches = {
        "r2_breach":  (baseline_r2 is not None  and (baseline_r2 - metrics_curr["r2"])  > thr["r2_drop_allowed"]),
        "rmse_breach": (baseline_rmse is not None and (metrics_curr["rmse"] / baseline_rmse) > thr["rmse_increase_ratio"]),
        "mae_breach":  (baseline_mae  is not None and (metrics_curr["mae"]  / baseline_mae)  > thr["mae_increase_ratio"]),
    }

    # Default: only R² breach triggers immediately
    breach_flag = breaches["r2_breach"]

    # For RMSE/MAE, require persistent breach (previous month also breached)
    if not breach_flag and (breaches["rmse_breach"] or breaches["mae_breach"]):
        prev_snapshot_dt = (datetime.combine(snap_dt, datetime.min.time()) - relativedelta(months=1)).date()
        prev_report_path = os.path.join(out_dir, f"monitoring_report_{prev_snapshot_dt.strftime('%Y-%m-%d')}.json")

        if os.path.exists(prev_report_path):
            try:
                prev_report = json.load(open(prev_report_path, "r", encoding="utf-8"))
                prev_breaches = prev_report.get("breaches", {})
                # Only if same metric breached in both months
                persistent_rmse = breaches["rmse_breach"] and prev_breaches.get("rmse_breach", False)
                persistent_mae  = breaches["mae_breach"]  and prev_breaches.get("mae_breach", False)
                if persistent_rmse or persistent_mae:
                    breach_flag = True
                    logger.info("⚠️ Persistent RMSE/MAE breach detected (2 consecutive months).")
            except Exception as e:
                logger.warning(f"⚠️ Could not read previous monitoring report: {e}")
        else:
            logger.info("ℹ️ No previous monitoring report found; ignoring transient RMSE/MAE breach.")


    # --- PSI and CSI
    vars_to_track = cfg["drift_monitoring"]["variables"]
    n_bins     = cfg["drift_monitoring"].get("n_bins", 10)
    bin_method = cfg["drift_monitoring"].get("bin_method", "quantile")
    psi_thr    = cfg["psi_monitoring"]["psi_thresholds"]
    csi_thr    = cfg["csi_monitoring"]["csi_thresholds"]

    ref_feat_df = load_parquet_exact(feat_dir, ref_snapshot)
    cur_feat_df = load_parquet_exact(feat_dir, pred_snapshot)

    missing_in_ref = [v for v in vars_to_track if v not in ref_feat_df.columns]
    missing_in_cur = [v for v in vars_to_track if v not in cur_feat_df.columns]
    available_vars = [v for v in vars_to_track if v in ref_feat_df.columns and v in cur_feat_df.columns]

    psi_summary = []
    for v in available_vars:
        psi_val = compute_psi(ref_feat_df[v], cur_feat_df[v], n_bins=n_bins, bin_method=bin_method)
        psi_sev = severity_from_thresholds(abs(psi_val), psi_thr)
        psi_summary.append({"variable": v, "psi": round(psi_val, 6), "psi_severity": psi_sev})

    # --- True CSI on predicted_sales
    ref_pred_path = os.path.join(pred_dir, f"{ref_snapshot}.parquet")
    cur_pred_path = os.path.join(pred_dir, f"{pred_snapshot}.parquet")
    prediction_csi = None
    prediction_csi_sev = None

    if os.path.exists(ref_pred_path) and os.path.exists(cur_pred_path):
        ref_pred_df = pd.read_parquet(ref_pred_path)
        cur_pred_df = pd.read_parquet(cur_pred_path)
        if all(col in ref_pred_df.columns for col in join_keys + ["predicted_sales"]) and \
           all(col in cur_pred_df.columns for col in join_keys + ["predicted_sales"]):
            merged_preds = ref_pred_df[join_keys + ["predicted_sales"]].merge(
                cur_pred_df[join_keys + ["predicted_sales"]],
                on=join_keys, suffixes=("_ref", "_cur"), how="inner"
            )
            if not merged_preds.empty:
                prediction_csi = compute_csi(merged_preds["predicted_sales_ref"],
                                             merged_preds["predicted_sales_cur"],
                                             n_bins=n_bins, bin_method=bin_method)
                prediction_csi_sev = severity_from_thresholds(abs(prediction_csi), csi_thr)
                logger.info(f"📉 Prediction CSI={prediction_csi:.3f} ({prediction_csi_sev})")
        else:
            logger.warning("⚠️ Missing required columns in prediction files for CSI.")
    else:
        logger.warning("⚠️ Missing prediction parquet(s) for CSI computation.")

    # --- PSI dashboard (no CSI)
    dash_path = os.path.join(out_dir, f"drift_dashboard_{snapshot_date}.png")
    if available_vars:
        plot_drift_dashboard(ref_df=ref_feat_df, cur_df=cur_feat_df,
                             variables=available_vars, psi_thr=psi_thr,
                             save_path=dash_path, n_bins=n_bins,
                             bin_method=bin_method,
                             title_suffix=f"(Ref={ref_snapshot} vs Curr={pred_snapshot})")
        logger.info(f"🖼️ Saved PSI drift dashboard → {dash_path}")
    else:
        dash_path = None
        logger.warning("⚠️ No common variables for drift plots; dashboard not generated.")

    # --- Candidate performance chart
    cand_dir = resolve_relative_path("model_store/candidate_models")
    deployed_version = deployed_meta.get("model_version")
    perf_chart_path = os.path.join(out_dir, f"candidate_models_perf_{snapshot_date}.png")
    perf_chart_done = plot_candidate_models_performance(cand_dir, deployed_version, perf_chart_path)
    if perf_chart_done:
        logger.info(f"📈 Saved candidate performance chart → {perf_chart_path}")

    # --- Report
    report = {
        "snapshot_date": snapshot_date,
        "prediction_snapshot": pred_snapshot,
        "reference_snapshot_for_drift": ref_snapshot,
        "metrics": metrics_curr,
        "baseline_metrics": {"oot": baseline_metrics},
        "breaches": breaches,
        "breach_flag": breach_flag,
        "thresholds": thr,
        "psi_summary": psi_summary,
        "prediction_csi": {
            "value": prediction_csi,
            "severity": prediction_csi_sev,
            "reference_snapshot": ref_snapshot,
            "current_snapshot": pred_snapshot
        },
        "missing_in_reference": missing_in_ref,
        "missing_in_current": missing_in_cur,
        "artifacts": {
            "drift_dashboard": dash_path,
            "candidate_models_performance": perf_chart_path if perf_chart_done else None
        }
    }

    report_path = os.path.join(out_dir, f"monitoring_report_{snapshot_date}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"📝 Saved monitoring report → {report_path}")

    logger.info(f"🔔 breach_flag={breach_flag} | r2_breach={breaches.get('r2_breach')} "
                f"| rmse_breach={breaches.get('rmse_breach')} | mae_breach={breaches.get('mae_breach')}")
    logger.info("✅ Monitoring complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run monthly model monitoring")
    parser.add_argument("--snapshot_date", required=True, help="Label month in YYYY-MM-DD (e.g., 2015-07-01)")
    args = parser.parse_args()
    main(snapshot_date=args.snapshot_date)