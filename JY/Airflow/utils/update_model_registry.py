"""
update_model_registry.py
------------------------
Aggregates metadata from all candidate models and updates
a central registry file (model_store/model_registry.json).

Each record includes:
- model_version
- training_window_end
- key metrics (val/test/oot)
- split stats
- timestamp
- flag for deployed model
"""

import os, json, logging
import pandas as pd
from datetime import datetime

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("update_model_registry")


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
def load_metadata(meta_path: str) -> dict:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ Failed to read {meta_path}: {e}")
        return {}


def extract_summary_from_metadata(meta: dict, is_deployed: bool) -> dict:
    """Flatten and extract key fields for registry."""
    mv = meta.get("model_version", "unknown")
    ts = meta.get("timestamp_utc")
    train_end = meta.get("training_window_end")

    # metrics
    metrics = meta.get("metrics", {})
    val_m = metrics.get("val", {})
    test_m = metrics.get("test", {})
    oot_m = metrics.get("oot", {})

    split_stats = meta.get("split_stats", {})
    train_rows = split_stats.get("train", {}).get("rows")
    val_rows = split_stats.get("val", {}).get("rows")
    test_rows = split_stats.get("test", {}).get("rows")
    oot_rows = split_stats.get("oot", {}).get("rows")

    return {
        "model_version": mv,
        "training_window_end": train_end,
        "timestamp_utc": ts,
        "is_deployed": is_deployed,

        # metrics
        "val_rmse": val_m.get("rmse"),
        "val_mae": val_m.get("mae"),
        "val_r2": val_m.get("r2"),

        "test_rmse": test_m.get("rmse"),
        "test_mae": test_m.get("mae"),
        "test_r2": test_m.get("r2"),

        "oot_rmse": oot_m.get("rmse"),
        "oot_mae": oot_m.get("mae"),
        "oot_r2": oot_m.get("r2"),

        # split info
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "oot_rows": oot_rows
    }


# ---------------------------
# Main logic
# ---------------------------
def main():
    logger.info("🚀 Updating model registry...")

    # --- Paths
    candidate_dir = resolve_relative_path("model_store/candidate_models")
    deployed_dir  = resolve_relative_path("model_store/deployed_model")
    registry_json = resolve_relative_path("model_store/model_registry.json")
    registry_csv  = resolve_relative_path("model_store/model_registry.csv")

    # --- Load deployed model metadata (to flag)
    deployed_meta = {}
    deployed_version = None
    dep_meta_path = os.path.join(deployed_dir, "metadata.json")
    if os.path.exists(dep_meta_path):
        deployed_meta = load_metadata(dep_meta_path)
        deployed_version = deployed_meta.get("model_version")
        logger.info(f"✅ Found deployed model: {deployed_version}")
    else:
        logger.warning("⚠️ No deployed model metadata found.")

    # --- Collect all candidate metadata
    records = []
    if not os.path.exists(candidate_dir):
        logger.warning(f"⚠️ Candidate directory not found: {candidate_dir}")
        return

    for sub in sorted(os.listdir(candidate_dir)):
        full = os.path.join(candidate_dir, sub)
        if not os.path.isdir(full):
            continue
        meta_path = os.path.join(full, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        meta = load_metadata(meta_path)
        if not meta:
            continue
        is_deployed = (meta.get("model_version") == deployed_version)
        rec = extract_summary_from_metadata(meta, is_deployed)
        records.append(rec)

    if not records:
        logger.warning("⚠️ No candidate models with metadata found.")
        return

    # --- Convert to DataFrame
    df = pd.DataFrame(records)
    df = df.sort_values(by="training_window_end", ascending=True)

    # --- Save registry files
    df.to_json(registry_json, orient="records", indent=2)
    df.to_csv(registry_csv, index=False)
    logger.info(f"💾 Updated registry → {registry_json}")
    logger.info(f"📄 Also saved CSV copy → {registry_csv}")

    # --- Log summary
    logger.info(f"📊 Registry now contains {len(df)} models "
                f"({df['is_deployed'].sum()} deployed).")

    # --- Optional: print short table
    cols_show = ["model_version", "training_window_end", "oot_r2", "oot_rmse", "is_deployed"]
    logger.info("\n" + df[cols_show].to_string(index=False))

    return df


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    main()
