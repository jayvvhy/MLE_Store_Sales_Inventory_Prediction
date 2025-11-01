"""
model_inference.py
------------------
Run inference for a given snapshot_date.

Steps:
1️⃣ Load deployed model from model_store/deployed_model/
2️⃣ Load feature snapshot datamart/gold/feature_store/<snapshot_date>.parquet
3️⃣ Preprocess (numeric→0, categorical→"Unknown")
4️⃣ Predict and output (store_nbr, family, predicted_sales)
5️⃣ Save results to datamart/gold/predictions/<snapshot_date>.parquet
"""

import os, sys, json, pickle, logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_inference")


def resolve_relative_path(rel_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.join(base_dir, rel_path)


def load_feature_snapshot(base_path: str, snapshot_date: str) -> pd.DataFrame:
    """Load the feature snapshot exactly matching the given date."""
    for fmt in ("%Y-%m-%d", "%Y_%m_%d"):
        path = os.path.join(base_path, f"{snapshot_date.replace('-', '_')}.parquet") \
            if fmt == "%Y_%m_%d" else os.path.join(base_path, f"{snapshot_date}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            logger.info(f"📦 Loaded feature snapshot: {os.path.basename(path)} ({len(df):,} rows)")
            return df
    raise FileNotFoundError(f"No parquet file found for snapshot_date={snapshot_date} in {base_path}")


def preprocess(df, numeric_cols, categorical_cols):
    """Simple preprocessing consistent with training."""
    df = df.copy()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0)
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("Unknown").astype("category")
    return df


def main(snapshot_date: str):
    logger.info(f"🚀 Starting inference for snapshot_date={snapshot_date}")

    # --- Paths
    feature_store_dir = resolve_relative_path("datamart/gold/feature_store")
    prediction_dir    = resolve_relative_path("datamart/gold/predictions")
    deployed_dir      = resolve_relative_path("model_store/deployed_model")
    os.makedirs(prediction_dir, exist_ok=True)

    # --- Load deployed model + metadata
    model_path = os.path.join(deployed_dir, "model.pkl")
    meta_path  = os.path.join(deployed_dir, "metadata.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("No deployed model found. Please run promote_best_model.py first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    numeric_cols = meta["numerical_features"]
    categorical_cols = meta["categorical_features"]
    join_keys = meta.get("join_keys", ["store_nbr", "family"])

    # --- Load correct feature snapshot
    df_feat = load_feature_snapshot(feature_store_dir, snapshot_date)
    X = preprocess(df_feat, numeric_cols, categorical_cols)

    # --- Predict
    feature_cols = numeric_cols + categorical_cols
    X = X[feature_cols]  # keep only model features
    preds = model.predict(X)

    df_pred = df_feat[join_keys].copy()
    df_pred["snapshot_date"] = snapshot_date
    df_pred["predicted_sales"] = preds

    # --- Save predictions
    out_path = os.path.join(prediction_dir, f"{snapshot_date}.parquet")
    df_pred.to_parquet(out_path, index=False)
    logger.info(f"💾 Saved predictions → {out_path}")
    logger.info(f"✅ Inference complete: {len(df_pred):,} rows predicted.")

    return df_pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run model inference for a given snapshot date")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()
    main(snapshot_date=args.snapshot_date)
