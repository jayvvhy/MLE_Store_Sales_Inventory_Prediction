"""
promote_best_model.py
---------------------
Selects the best-performing regression model from candidate_models
based on OOT R² and promotes it to model_store/deployed_model
IF a monitoring breach was detected in the most recent monitoring report.
"""

import os, json, shutil, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_relative_path(path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.join(base_dir, path)


def latest_monitoring_report(report_dir: str):
    """Return the most recent monitoring report path."""
    if not os.path.exists(report_dir):
        return None
    files = [f for f in os.listdir(report_dir)
             if f.startswith("monitoring_report_") and f.endswith(".json")]
    if not files:
        return None
    latest = sorted(files)[-1]
    return os.path.join(report_dir, latest)


def load_oot_r2(candidate_dir: str):
    """Return list of (folder_path, oot_r2) pairs."""
    models = []
    for d in os.listdir(candidate_dir):
        full = os.path.join(candidate_dir, d)
        if not os.path.isdir(full):
            continue
        meta_path = os.path.join(full, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        oot_r2 = meta.get("metrics", {}).get("oot", {}).get("r2")
        if oot_r2 is not None:
            models.append((full, oot_r2))
    return models


def promote_model(best_path: str, best_r2: float, deployed_dir: str):
    """Copy the best model to deployed_model folder."""
    if os.path.exists(deployed_dir):
        shutil.rmtree(deployed_dir)
    shutil.copytree(best_path, deployed_dir)
    logger.info(f"🚀 Promoted {os.path.basename(best_path)} (OOT R²={best_r2:.3f}) → deployed_model/")


def main(snapshot_date: str):
    DEPLOYMENT_DATE = datetime(2015, 6, 1)   # first deployment date
    current_date = datetime.strptime(snapshot_date, "%Y-%m-%d")

    candidate_dir = resolve_relative_path("model_store/candidate_models")
    deployed_dir  = resolve_relative_path("model_store/deployed_model")
    monitor_dir   = resolve_relative_path("datamart/gold/model_monitoring")

    os.makedirs(os.path.dirname(deployed_dir), exist_ok=True)

    # Skip if before first deployment
    if current_date < DEPLOYMENT_DATE:
        logger.info(f"⏭️ Promotion skipped for {snapshot_date}: deployment starts on {DEPLOYMENT_DATE.date()}.")
        return

    # ----------------------------
    # 1️⃣ Initial deployment
    # ----------------------------
    if current_date == DEPLOYMENT_DATE:
        logger.info(f"🚀 Initial deployment date reached ({snapshot_date}). Selecting best candidate by R².")
        models = load_oot_r2(candidate_dir)
        if not models:
            logger.warning("⚠️ No candidate models found for initial deployment.")
            return
        best_path, best_r2 = max(models, key=lambda x: x[1])
        promote_model(best_path, best_r2, deployed_dir)
        logger.info(f"✅ Initial model deployed: {os.path.basename(best_path)} (OOT R²={best_r2:.3f}).")
        return

    # ----------------------------
    # 2️⃣ Subsequent months
    # ----------------------------
    report_path = latest_monitoring_report(monitor_dir)
    if not report_path:
        logger.info("ℹ️ No monitoring report found yet. Skipping promotion.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    breach_flag = report.get("breach_flag", False)
    snap = report.get("snapshot_date")
    logger.info(f"📊 Latest monitoring report: {snap}, breach_flag={breach_flag}")

    if not breach_flag:
        logger.info("✅ No performance breaches detected. Skipping promotion.")
        return

    # ----------------------------
    # 3️⃣ Breach detected → promote best
    # ----------------------------
    models = load_oot_r2(candidate_dir)
    if not models:
        logger.warning("⚠️ No candidate models with valid metadata found.")
        return
    best_path, best_r2 = max(models, key=lambda x: x[1])
    logger.info(f"🏆 Selected best candidate: {os.path.basename(best_path)} (OOT R²={best_r2:.3f})")
    promote_model(best_path, best_r2, deployed_dir)

    # ----------------------------
    # 4️⃣ Log promotion event
    # ----------------------------
    log_path = os.path.join(candidate_dir, "model_promotion_log.json")
    record = {
        "promoted_model": os.path.basename(best_path),
        "oot_r2": best_r2,
        "triggered_by": snap,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "monitoring_breach"
    }

    existing = []
    if os.path.exists(log_path):
        try:
            existing = json.load(open(log_path))
        except Exception:
            existing = []
    existing.append(record)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"📝 Logged promotion event to {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Promote Best Performing Regression Model (R²-based)")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    main(snapshot_date=args.snapshot_date)
