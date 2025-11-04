"""
model_inference.py
------------------
Run inference for a given snapshot_date.

Steps:
1️⃣ Check if deployment and required files exist (skip early if not)
2️⃣ Load deployed model from model_store/deployed_model/
3️⃣ Load feature snapshot datamart/gold/feature_store/<snapshot_date>.parquet
4️⃣ Preprocess (numeric→0, categorical→"Unknown")
5️⃣ Predict and output (store_nbr, family, predicted_sales)
6️⃣ Save results to datamart/gold/predictions/<snapshot_date>.parquet
"""

import os, sys, json, pickle, logging, yaml, warnings
from datetime import datetime
import pandas as pd

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_inference")

# ---------------------------------------------------------------------
# Utility: resolve paths relative to project root
# ---------------------------------------------------------------------
def resolve_relative_path(rel_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.join(base_dir, rel_path)

# ---------------------------------------------------------------------
# Load deployment configuration
# ---------------------------------------------------------------------
def load_deployment_date() -> datetime:
    """Load deployment_date from monitoring_config.yaml."""
    config_path = resolve_relative_path("config/monitoring_config.yaml")
    if not os.path.exists(config_path):
        logger.warning(f"⚠️ monitoring_config.yaml not found at {config_path}. Using default 2099-12-31.")
        return datetime(2099, 12, 31)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    date_str = cfg.get("deployment_date", "2099-12-31")
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception as e:
        logger.warning(f"⚠️ Invalid deployment_date in YAML ({date_str}). Using default 2099-12-31. Error: {e}")
        return datetime(2099, 12, 31)

# ---------------------------------------------------------------------
# Load feature snapshot
# ---------------------------------------------------------------------
def load_feature_snapshot(base_path: str, snapshot_date: str) -> pd.DataFrame:
    """Load the feature snapshot exactly matching the given date."""
    for fmt in ("%Y-%m-%d", "%Y_%m_%d"):
        path = os.path.join(base_path, f"{snapshot_date.replace('-', '_')}.parquet") \
            if fmt == "%Y_%m_%d" else os.path.join(base_path, f"{snapshot_date}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            logger.info(f"📦 Loaded feature snapshot: {os.path.basename(path)} ({len(df):,} rows)")
            return df
    logger.warning(f"⚠️ No parquet file found for snapshot_date={snapshot_date} in {base_path}")
    return pd.DataFrame()

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
def preprocess(df, numeric_cols, categorical_cols):
    """Simple preprocessing consistent with training."""
    df = df.copy()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0)
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Unknown").astype("category")
    return df

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(snapshot_date: str):
    logger.info(f"🚀 Starting inference for snapshot_date={snapshot_date}")

    # -----------------------------------------------------------------
    # Load deployment date
    # -----------------------------------------------------------------
    DEPLOYMENT_DATE = load_deployment_date()
    logger.info(f"📅 Deployment start date: {DEPLOYMENT_DATE.date()}")

    current_date = datetime.strptime(snapshot_date, "%Y-%m-%d")
    if current_date < DEPLOYMENT_DATE:
        logger.info(f"⏭️ Skipping inference for {snapshot_date}: deployment starts on {DEPLOYMENT_DATE.date()}.")
        return

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------
    feature_store_dir = resolve_relative_path("datamart/gold/feature_store")
    prediction_dir    = resolve_relative_path("datamart/gold/predictions")
    deployed_dir      = resolve_relative_path("model_store/deployed_model")
    os.makedirs(prediction_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Check model directory
    # -----------------------------------------------------------------
    model_path = os.path.join(deployed_dir, "model.pkl")
    meta_path  = os.path.join(deployed_dir, "metadata.json")

    if not os.path.exists(deployed_dir):
        logger.warning(f"⏭️ Skipping inference: deployed_model directory not found → {deployed_dir}")
        return

    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        logger.warning("⏭️ Skipping inference: no deployed model found. "
                       "Please run promote_best_model.py first.")
        return

    # -----------------------------------------------------------------
    # Load model + metadata
    # -----------------------------------------------------------------
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(model_path, "rb") as f:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                model = pickle.load(f)
        logger.info(f"✅ Loaded deployed model from {deployed_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Skipping inference — failed to load model: {e}")
        return

    numeric_cols = meta.get("numerical_features", [])
    categorical_cols = meta.get("categorical_features", [])
    join_keys = meta.get("join_keys", ["store_nbr", "family"])

    # -----------------------------------------------------------------
    # Load features
    # -----------------------------------------------------------------
    df_feat = load_feature_snapshot(feature_store_dir, snapshot_date)
    if df_feat.empty:
        logger.warning(f"⏭️ Skipping inference: no feature data for {snapshot_date}")
        return

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------
    feature_cols = [c for c in numeric_cols + categorical_cols if c in df_feat.columns]
    if not feature_cols:
        logger.warning(f"⏭️ Skipping inference: no matching feature columns found.")
        return

    X = preprocess(df_feat, numeric_cols, categorical_cols)
    X = X[feature_cols]

    try:
        preds = model.predict(X)
    except Exception as e:
        logger.warning(f"⚠️ Model prediction failed: {e}")
        return

    df_pred = df_feat[join_keys].copy()
    df_pred["snapshot_date"] = snapshot_date
    df_pred["predicted_sales"] = preds

    # -----------------------------------------------------------------
    # Save predictions
    # -----------------------------------------------------------------
    out_path_parquet = os.path.join(prediction_dir, f"{snapshot_date}.parquet")
    out_path_csv     = os.path.join(prediction_dir, f"{snapshot_date}.csv")

    try:
        # Save to Parquet
        df_pred.to_parquet(out_path_parquet, index=False)
        logger.info(f"💾 Saved predictions → {out_path_parquet}")

        # Also save to CSV (lightweight, easy inspection)
        df_pred.to_csv(out_path_csv, index=False)
        logger.info(f"📄 Saved CSV copy → {out_path_csv}")

        logger.info(f"✅ Inference complete: {len(df_pred):,} rows predicted.")
    except Exception as e:
        logger.warning(f"⚠️ Skipping save due to error: {e}")

    return df_pred

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run model inference for a given snapshot date")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    main(snapshot_date=args.snapshot_date)
