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
  5) Compute PSI/CSI for configured variables:
     between <prediction_snapshot> features and <deployed_model.training_window_end> features
  6) Generate:
      - 10-chart drift dashboard (overlay two periods + PSI/CSI with severity color)
      - candidate models performance chart (optional; if candidate metadata found)
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
    """
    Load a parquet file strictly matching the given snapshot date.
    Tries YYYY-MM-DD.parquet and YYYY_MM_DD.parquet.
    """
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
    """Create bin edges from the reference series."""
    ser = ref_series.dropna()
    if ser.empty:
        # fallback single bin if completely empty (should not happen in practice)
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
    """Histogram proportions with Laplace smoothing to avoid zeros."""
    counts, _ = np.histogram(x.dropna(), bins=bins)
    counts = counts.astype(float)
    counts += 1e-6  # smoothing
    return counts / counts.sum()

def compute_psi(ref: pd.Series, cur: pd.Series, n_bins: int = 10, bin_method: str = "quantile") -> float:
    """
    Population Stability Index:
    PSI = sum( (Pi - Qi) * ln(Pi / Qi) )
    where P = ref proportions, Q = current proportions.
    """
    bins = _make_bins(ref, n_bins, bin_method)
    P = _proportions(ref, bins)
    Q = _proportions(cur, bins)
    return float(np.sum((P - Q) * np.log(P / Q)))

def compute_csi(ref: pd.Series, cur: pd.Series, n_bins: int = 10, bin_method: str = "quantile") -> float:
    """
    Characteristic Stability Index (simple variant used in practice):
    CSI = sum( |Pi - Qi| )
    """
    bins = _make_bins(ref, n_bins, bin_method)
    P = _proportions(ref, bins)
    Q = _proportions(cur, bins)
    return float(np.sum(np.abs(P - Q)))


# ---------------------------
# Severity helpers
# ---------------------------
def severity_from_thresholds(value: float, thresholds: dict) -> str:
    """
    thresholds = {stable: 0.10, moderate: 0.25, significant: 0.25}
    Interprets as:
      < stable  -> "stable"
      < moderate -> "moderate"
      else -> "significant"
    """
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
# Plotting
# ---------------------------
def plot_drift_dashboard(ref_df: pd.DataFrame,
                         cur_df: pd.DataFrame,
                         variables: list,
                         psi_thr: dict,
                         csi_thr: dict,
                         save_path: str,
                         n_bins: int = 10,
                         bin_method: str = "quantile",
                         title_suffix: str = ""):
    """
    Creates a 2-column grid of variable distributions:
    - Overlay reference vs current histograms or bar charts
    - Title shows PSI & CSI with color by severity
    - Skips constant variables automatically
    """
    n = len(variables)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    skipped_vars = []

    for i, var in enumerate(variables):
        ax = axes[i]
        if var not in ref_df.columns or var not in cur_df.columns:
            ax.axis("off")
            ax.set_title(f"{var}\n(Missing in one dataset)", color="gray")
            continue

        ref = ref_df[var].dropna()
        cur = cur_df[var].dropna()

        # Skip constant variables
        if ref.nunique() <= 1 and cur.nunique() <= 1:
            ax.axis("off")
            ax.set_title(f"{var}\n(Constant across periods)", color="gray")
            skipped_vars.append(var)
            continue

        # Decide plotting mode
        if ref.nunique() <= 10 and cur.nunique() <= 10:
            # Treat as categorical
            combined_cats = sorted(list(set(ref.unique()) | set(cur.unique())))
            ref_counts = ref.value_counts(normalize=True).reindex(combined_cats, fill_value=0)
            cur_counts = cur.value_counts(normalize=True).reindex(combined_cats, fill_value=0)

            x = np.arange(len(combined_cats))
            width = 0.4
            ax.bar(x - width / 2, ref_counts.values, width=width, alpha=0.6, label="Reference")
            ax.bar(x + width / 2, cur_counts.values, width=width, alpha=0.6, label="Current")
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in combined_cats], rotation=45, ha="right")
        else:
            # Numeric — create bins based on reference
            bins = _make_bins(ref, n_bins, bin_method)
            ax.hist(ref, bins=bins, alpha=0.5, density=False, label="Reference")
            ax.hist(cur, bins=bins, alpha=0.5, density=False, label="Current")

            # Auto log-scale for highly skewed variables
            if ref.min() > 0 and ref.max() / max(ref.min(), 1e-9) > 1000:
                ax.set_xscale("log")

        # Compute drift metrics
        psi = compute_psi(ref, cur, n_bins, bin_method)
        csi = compute_csi(ref, cur, n_bins, bin_method)
        psi_sev = severity_from_thresholds(abs(psi), psi_thr)
        csi_sev = severity_from_thresholds(abs(csi), csi_thr)
        combined_sev = max(
            [psi_sev, csi_sev],
            key=lambda s: ["stable", "moderate", "significant"].index(s)
        )

        ax.set_title(
            f"{var}\nPSI={psi:.3f} | CSI={csi:.3f}",
            color=color_for_severity(combined_sev),
            fontsize=11
        )
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.2)

    # Hide any extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Feature Drift Dashboard {title_suffix}", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=160)
    plt.close(fig)

    if skipped_vars:
        logger.info(f"ℹ️ Skipped {len(skipped_vars)} constant variables: {skipped_vars}")

def plot_candidate_models_performance(candidate_dir: str, deployed_version: str, save_path: str):
    """
    Creates a bar chart of candidate models' OOT R² (if available),
    highlighting the deployed model.
    """
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

    rows = sorted(rows, key=lambda x: x[0])  # sort by version/date label
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
# Main
# ---------------------------
def main(snapshot_date: str):
    """
    snapshot_date: the label month (e.g., 2015-07-01)
    We compare labels @ snapshot_date vs predictions @ (snapshot_date - 1 month).
    PSI/CSI: between features @ pred_snapshot and features @ deployed_model.training_window_end
    """
    logger.info(f"🚀 Starting monitoring for snapshot_date={snapshot_date}")

    # --- Paths
    cfg_path     = resolve_relative_path("config/monitoring_config.yaml")
    deployed_dir = resolve_relative_path("model_store/deployed_model")
    label_dir    = resolve_relative_path("datamart/gold/label_store")
    pred_dir     = resolve_relative_path("datamart/gold/predictions")
    feat_dir     = resolve_relative_path("datamart/gold/feature_store")
    out_dir      = resolve_relative_path("datamart/gold/model_monitoring")
    ensure_dir(out_dir)

    # --- Load config & deployed metadata
    cfg = load_yaml(cfg_path)
    dep_date = datetime.strptime(cfg["deployment_date"], "%Y-%m-%d").date()
    first_monitoring_date = (datetime.combine(dep_date, datetime.min.time()) + relativedelta(months=1)).date()

    snap_dt = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
    if snap_dt < first_monitoring_date:
        logger.info(f"⏭️ Monitoring skipped for {snapshot_date}: first monitoring date is {first_monitoring_date}.")
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
    training_window_end = deployed_meta.get("training_window_end")  # e.g., "2015-04-01"

    # --- Determine prediction snapshot (previous month)
    pred_snapshot_dt = (datetime.combine(snap_dt, datetime.min.time()) - relativedelta(months=1)).date()
    pred_snapshot = pred_snapshot_dt.strftime("%Y-%m-%d")
    ref_snapshot  = training_window_end  # reference for PSI/CSI

    logger.info(f"📅 Monitoring window → labels: {snapshot_date}, predictions: {pred_snapshot} (prev month)")
    logger.info(f"📌 Drift reference for PSI/CSI: {ref_snapshot} (deployed model training window end)")

    # --- Load labels (current) and predictions (previous)
    labels_df = load_parquet_exact(label_dir, snapshot_date)
    preds_df  = load_parquet_exact(pred_dir,  pred_snapshot)

    # Ensure target col exists
    if target_col not in labels_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in label store {snapshot_date}.")

    # Ensure prediction column exists
    if "predicted_sales" not in preds_df.columns:
        raise KeyError(f"'predicted_sales' not found in predictions {pred_snapshot}.")

    # Merge on join keys
    eval_df = labels_df[join_keys + [target_col]].merge(
        preds_df[join_keys + ["predicted_sales"]],
        on=join_keys,
        how="inner"
    )

    if eval_df.empty:
        raise ValueError("No overlapping rows between labels and predictions on join keys.")

    # --- Compute performance metrics
    metrics_curr = eval_regression(eval_df[target_col].values, eval_df["predicted_sales"].values)

    # --- Threshold logic
    thr = cfg["performance_monitoring"]["thresholds"]
    r2_drop_allowed      = thr["r2_drop_allowed"]          # absolute drop allowed
    rmse_increase_ratio  = thr["rmse_increase_ratio"]       # e.g., 1.25 = +25%
    mae_increase_ratio   = thr["mae_increase_ratio"]        # e.g., 1.25

    # Compare vs baseline if available
    breaches = {}
    if baseline_r2 is not None:
        breaches["r2_breach"] = (baseline_r2 - metrics_curr["r2"]) > r2_drop_allowed
    if baseline_rmse is not None:
        breaches["rmse_breach"] = (metrics_curr["rmse"] / baseline_rmse) > rmse_increase_ratio
    if baseline_mae is not None:
        breaches["mae_breach"]  = (metrics_curr["mae"]  / baseline_mae)  > mae_increase_ratio

    breach_flag = any(breaches.values())

    # --- PSI/CSI calculations
    vars_to_track = cfg["drift_monitoring"]["variables"]
    n_bins        = cfg["drift_monitoring"].get("n_bins", 10)
    bin_method    = cfg["drift_monitoring"].get("bin_method", "quantile")
    psi_thr       = cfg["psi_monitoring"]["psi_thresholds"]
    csi_thr       = cfg["csi_monitoring"]["csi_thresholds"]

    # Load reference and current feature snapshots (for drift)
    ref_feat_df = load_parquet_exact(feat_dir, ref_snapshot)
    cur_feat_df = load_parquet_exact(feat_dir, pred_snapshot)

    # Subset to the variables (silently drop missing but record them)
    missing_in_ref = [v for v in vars_to_track if v not in ref_feat_df.columns]
    missing_in_cur = [v for v in vars_to_track if v not in cur_feat_df.columns]
    available_vars = [v for v in vars_to_track if v in ref_feat_df.columns and v in cur_feat_df.columns]

    psi_csi_summary = []
    for v in available_vars:
        psi = compute_psi(ref_feat_df[v], cur_feat_df[v], n_bins=n_bins, bin_method=bin_method)
        csi = compute_csi(ref_feat_df[v], cur_feat_df[v], n_bins=n_bins, bin_method=bin_method)
        psi_sev = severity_from_thresholds(abs(psi), psi_thr)
        csi_sev = severity_from_thresholds(abs(csi), csi_thr)
        psi_csi_summary.append({
            "variable": v,
            "psi": round(float(psi), 6),
            "psi_severity": psi_sev,
            "csi": round(float(csi), 6),
            "csi_severity": csi_sev
        })

    # --- Visual dashboards
    dash_path = os.path.join(out_dir, f"drift_dashboard_{snapshot_date}.png")
    if available_vars:
        plot_drift_dashboard(
            ref_df=ref_feat_df,
            cur_df=cur_feat_df,
            variables=available_vars,
            psi_thr=psi_thr,
            csi_thr=csi_thr,
            save_path=dash_path,
            n_bins=n_bins,
            bin_method=bin_method,
            title_suffix=f"(Ref={ref_snapshot} vs Curr={pred_snapshot})"
        )
        logger.info(f"🖼️ Saved drift dashboard → {dash_path}")
    else:
        dash_path = None
        logger.warning("⚠️ No common variables found for drift plots; dashboard not generated.")

    # Candidate models performance chart (optional)
    cand_dir  = resolve_relative_path("model_store/candidate_models")
    deployed_version = deployed_meta.get("model_version")
    perf_chart_path = os.path.join(out_dir, f"candidate_models_perf_{snapshot_date}.png")
    perf_chart_done = plot_candidate_models_performance(cand_dir, deployed_version, perf_chart_path)
    if perf_chart_done:
        logger.info(f"📈 Saved candidate performance chart → {perf_chart_path}")
    else:
        perf_chart_path = None
        logger.info("ℹ️ Candidate performance chart not created (no candidate metadata found).")

    # --- Assemble monitoring report
    report = {
        "snapshot_date": snapshot_date,
        "prediction_snapshot": pred_snapshot,
        "reference_snapshot_for_drift": ref_snapshot,
        "metrics": metrics_curr,
        "baseline_metrics": {"oot": {"r2": baseline_r2, "rmse": baseline_rmse, "mae": baseline_mae}},
        "breaches": breaches,
        "breach_flag": breach_flag,
        "thresholds": thr,
        "psi_csi_summary": psi_csi_summary,
        "missing_in_reference": missing_in_ref,
        "missing_in_current": missing_in_cur,
        "artifacts": {
            "drift_dashboard": dash_path,
            "candidate_models_performance": perf_chart_path
        }
    }

    report_path = os.path.join(out_dir, f"monitoring_report_{snapshot_date}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"📝 Saved monitoring report → {report_path}")

    # Summary log for promote_best_model.py
    logger.info(f"🔔 breach_flag={breach_flag} | r2_breach={breaches.get('r2_breach')} "
                f"| rmse_breach={breaches.get('rmse_breach')} | mae_breach={breaches.get('mae_breach')}")
    logger.info("✅ Monitoring complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run monthly model monitoring")
    parser.add_argument("--snapshot_date", required=True, help="Label month in YYYY-MM-DD (e.g., 2015-07-01)")
    args = parser.parse_args()
    main(snapshot_date=args.snapshot_date)
