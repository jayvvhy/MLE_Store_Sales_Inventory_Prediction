"""
promote_best_model.py
---------------------
Selects best-performing model from candidate_models based on OOT AUC
and copies it to model_store/deployed_model IF a monitoring breach
was detected in the most recent monitoring report.
"""

import os, json, shutil, logging, sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def resolve_relative_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.join(base_dir, path)

def latest_monitoring_report(report_dir):
    files = [f for f in os.listdir(report_dir) if f.startswith("monitoring_report_") and f.endswith(".json")]
    if not files:
        return None
    latest = sorted(files)[-1]
    return os.path.join(report_dir, latest)

def load_oot_auc(candidate_dir):
    """Return list of (folder_path, oot_auc) pairs."""
    models = []
    for d in os.listdir(candidate_dir):
        full = os.path.join(candidate_dir, d)
        if not os.path.isdir(full): 
            continue
        meta_path = os.path.join(full, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            oot_auc = meta["results"].get("oot", {}).get("auc")
            if oot_auc is not None:
                models.append((full, oot_auc))
    return models

def promote_model(best_path, best_auc, deployed_dir):
    # Remove deployed_dir if it already exists (since dirs_exist_ok not supported in py3.7)
    if os.path.exists(deployed_dir):
        shutil.rmtree(deployed_dir)

    # Copy model directory
    shutil.copytree(best_path, deployed_dir)

    logger.info(f"🚀 Promoted {os.path.basename(best_path)} (OOT AUC={best_auc:.3f}) → deployed_model/")


def main(snapshot_date):
    DEPLOYMENT_DATE = datetime(2024, 6, 1)
    current_date = datetime.strptime(snapshot_date, "%Y-%m-%d")
    # -----------------------------------------------------------------
    # 0️⃣ Skip if before deployment date
    # -----------------------------------------------------------------
    if current_date < DEPLOYMENT_DATE:
        logger.info(f"⏭️ Promotion skipped for {snapshot_date}: "
                    f"deployment starts on {DEPLOYMENT_DATE.date()}.")
        return

    candidate_dir = resolve_relative_path("model_store/candidate_models")
    deployed_dir  = resolve_relative_path("model_store/deployed_model")
    monitor_dir   = resolve_relative_path("datamart/gold/model_monitoring")
    os.makedirs(os.path.dirname(deployed_dir), exist_ok=True)
    os.makedirs(deployed_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # 1️⃣  Deployment date — promote best baseline model
    # -----------------------------------------------------------------
    if current_date == DEPLOYMENT_DATE:
        logger.info(f"🚀 Initial deployment date reached ({snapshot_date}). "
                    "Selecting best candidate for first deployment.")
        models = load_oot_auc(candidate_dir)
        if not models:
            logger.warning("⚠️ No candidate models found for initial deployment.")
            return
        best_path, best_auc = max(models, key=lambda x: x[1])
        promote_model(best_path, best_auc, deployed_dir)
        logger.info(f"✅ Initial model deployed from {os.path.basename(best_path)} "
                    f"(OOT AUC = {best_auc:.3f}).")
        return

    # -----------------------------------------------------------------
    # 2️⃣  Subsequent months — check monitoring reports
    # -----------------------------------------------------------------
    if not os.path.exists(monitor_dir):
        logger.info(f"ℹ️ Monitoring folder not found yet ({monitor_dir}). "
                    "Skipping promotion for this month.")
        return

    report_path = latest_monitoring_report(monitor_dir)
    if not report_path:
        logger.info("ℹ️ No monitoring report found yet. Skipping promotion.")
        return

    with open(report_path, "r") as f:
        report = json.load(f)
    breach_flag = report.get("breach_flag", False)
    snap = report.get("snapshot_date")
    logger.info(f"📊 Latest monitoring report: {snap}, breach_flag={breach_flag}")

    if not breach_flag:
        logger.info("✅ No performance breaches detected. Skipping promotion.")
        return

    # -----------------------------------------------------------------
    # 3️⃣  Breach detected → promote best candidate
    # -----------------------------------------------------------------
    models = load_oot_auc(candidate_dir)
    if not models:
        logger.warning("⚠️ No candidate models with valid metadata found.")
        return

    best_path, best_auc = max(models, key=lambda x: x[1])
    logger.info(f"🏆 Selected best candidate for redeployment: "
                f"{os.path.basename(best_path)} (OOT AUC = {best_auc:.3f})")

    promote_model(best_path, best_auc, deployed_dir)

    # -----------------------------------------------------------------
    # 4️⃣  Log promotion event
    # -----------------------------------------------------------------
    log_path = os.path.join(candidate_dir, "model_promotion_log.json")
    record = {
        "promoted_model": os.path.basename(best_path),
        "oot_auc": best_auc,
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
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"📝 Logged promotion event to {log_path}")

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote Best Performing Model")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()

    main(snapshot_date=args.snapshot_date)
