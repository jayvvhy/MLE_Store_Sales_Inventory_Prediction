"""
update_model_registry.py
------------------------
Maintains a model registry (JSON) summarizing all candidate models
and their key OOT metrics. Also flags which model is currently deployed.
"""

import os, json, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def resolve_relative_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.join(base_dir, path)

def update_registry():
    candidate_dir = resolve_relative_path("model_store/candidate_models")
    deployed_dir  = resolve_relative_path("model_store/deployed_model")
    registry_path = resolve_relative_path("model_store/model_registry.json")

    # -----------------------------------------------------------------
    # Load existing registry (if any)
    # -----------------------------------------------------------------
    entries = []
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                entries = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ Could not parse existing registry ({e}). Rebuilding fresh.")
            entries = []

    # -----------------------------------------------------------------
    # Identify currently deployed model (if any)
    # -----------------------------------------------------------------
    deployed_model_version = None
    deployed_meta_path = os.path.join(deployed_dir, "metadata.json")
    if os.path.exists(deployed_meta_path):
        try:
            with open(deployed_meta_path, "r") as f:
                deployed_meta = json.load(f)
                deployed_model_version = deployed_meta.get("model_version")
                logger.info(f"📦 Currently deployed model: {deployed_model_version}")
        except Exception as e:
            logger.warning(f"⚠️ Unable to read deployed model metadata: {e}")

    # -----------------------------------------------------------------
    # Iterate through all candidate models
    # -----------------------------------------------------------------
    for d in sorted(os.listdir(candidate_dir)):
        meta_path = os.path.join(candidate_dir, d, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            logger.warning(f"⚠️ Skipping corrupt metadata in {d}")
            continue

        results_oot = meta["results"].get("oot", {})
        entry = {
            "model_version": meta.get("model_version", d),
            "snapshot_date": d.replace("_", "-"),
            "oot_auc": results_oot.get("auc"),
            "oot_gini": results_oot.get("gini"),
            "oot_ks": results_oot.get("ks"),
            "oot_f1": results_oot.get("f1"),
            "oot_accuracy": results_oot.get("accuracy"),
            "oot_precision": results_oot.get("precision"),
            "oot_recall": results_oot.get("recall"),
            "training_window": meta.get("training_window", []),
            "timestamp": meta.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "currently_deployed": (
                "Yes" if meta.get("model_version") == deployed_model_version else "No"
            )
        }

        # Avoid duplicates
        existing_versions = [e["model_version"] for e in entries]
        if entry["model_version"] not in existing_versions:
            entries.append(entry)
        else:
            # Update existing record if deployed flag changed
            for e in entries:
                if e["model_version"] == entry["model_version"]:
                    e.update(entry)

    # -----------------------------------------------------------------
    # Save updated registry
    # -----------------------------------------------------------------
    with open(registry_path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info(f"✅ Model registry updated → {registry_path}")
    logger.info(f"📊 {len(entries)} models tracked.")

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    update_registry()
