"""
model_inference.py
------------------
Batch inference script for Credit Default model.

Steps:
1. Parse input snapshot_date
2. Load deployed model and metadata
3. Load preprocessing artefacts (imputers, encoder, column schema)
4. Load feature_store/<snapshot_date>.parquet and filter loans to score
5. Apply preprocessing (impute + OHE with training encoders)
6. Predict probabilities and save results to
   datamart/gold/model_predictions/<snapshot_date>.parquet
"""

import os
import sys
import json
import joblib
import logging
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the deployment start date (first inference allowed)
DEPLOYMENT_DATE = datetime(2024, 6, 1)
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
# Utility: resolve paths relative to project root
# ---------------------------------------------------------------------
def resolve_relative_path(path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return os.path.abspath(os.path.join(base_dir, path))

# ---------------------------------------------------------------------
# Load deployed model and preprocessing artefacts
# ---------------------------------------------------------------------
def load_deployed_model(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    meta_path = os.path.join(model_dir, "metadata.json")
    preproc_dir = os.path.join(model_dir, "preprocessing")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not os.path.exists(preproc_dir):
        raise FileNotFoundError(f"Preprocessing artefacts missing: {preproc_dir}")

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # load preprocessing components
    num_imputer = joblib.load(os.path.join(preproc_dir, "num_imputer.pkl"))
    cat_imputer = joblib.load(os.path.join(preproc_dir, "cat_imputer.pkl"))
    encoder = joblib.load(os.path.join(preproc_dir, "ohe_encoder.pkl"))
    with open(os.path.join(preproc_dir, "training_columns.json"), "r") as f:
        train_meta = json.load(f)

    logger.info(f"✅ Loaded deployed model from {model_dir}")
    logger.info(f"Model version: {metadata['model_version']} | trained: {metadata['timestamp']}")
    return model, metadata, num_imputer, cat_imputer, encoder, train_meta

# ---------------------------------------------------------------------
# Load feature data for given snapshot
# ---------------------------------------------------------------------
def load_feature_data(snapshot_date):
    feature_dir = resolve_relative_path("datamart/gold/feature_store")
    feature_file = os.path.join(feature_dir, f"{snapshot_date.replace('-', '_')}.parquet")
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    df = pd.read_parquet(feature_file)

    # Filter for loans that start on the snapshot_date
    df["loan_start_date"] = pd.to_datetime(df["loan_start_date"]).dt.date
    snapshot_dt = pd.to_datetime(snapshot_date).date()
    df = df[df["loan_start_date"] == snapshot_dt]

    logger.info(f"📦 Loaded feature data: {len(df)} records to score.")
    return df

# ---------------------------------------------------------------------
# Apply preprocessing using saved artefacts
# ---------------------------------------------------------------------
def preprocess_features(df, num_imputer, cat_imputer, encoder, train_meta):
    num_feats = train_meta["numeric_features"]
    cat_feats = train_meta["categorical_features"]
    transformed_feature_names = train_meta["transformed_feature_names"]

    # Ensure missing columns exist (if schema drift)
    for col in num_feats + cat_feats:
        if col not in df.columns:
            df[col] = np.nan

    # Split
    X_num = df[num_feats].copy()
    X_cat = df[cat_feats].copy()

    # --- Align features with training schema before imputation ---
    num_order = getattr(num_imputer, "feature_names_in_", num_feats)
    cat_order = getattr(cat_imputer, "feature_names_in_", cat_feats)
    
    # Add missing columns if any (set to NaN)
    for col in num_order:
        if col not in X_num.columns:
            X_num[col] = np.nan
    for col in cat_order:
        if col not in X_cat.columns:
            X_cat[col] = np.nan
    
    # Reorder columns to match imputer training
    X_num = X_num[num_order]
    X_cat = X_cat[cat_order]

    logger.info(f"Categorical imputer trained on: {list(cat_imputer.feature_names_in_)}")
    logger.info(f"Categorical columns in current batch: {list(X_cat.columns)}")
    
    # Impute
    X_num_imp = pd.DataFrame(num_imputer.transform(X_num), columns=num_order, index=df.index)
    X_cat_imp = pd.DataFrame(cat_imputer.transform(X_cat), columns=cat_order, index=df.index)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        X_cat_enc = pd.DataFrame(
            encoder.transform(X_cat_imp),
            columns=encoder.get_feature_names_out(cat_order),
            index=df.index
        )

    # Encode categorical
    X_cat_enc = pd.DataFrame(
        encoder.transform(X_cat_imp),
        columns=encoder.get_feature_names_out(cat_feats),
        index=df.index
    )

    # Combine numeric + encoded categorical + passthroughs (if any)
    passthrough_cols = [
        c for c in df.columns if c not in num_feats + cat_feats
    ]
    X_passthrough = df[passthrough_cols].copy()

    X_final = pd.concat([X_num_imp, X_cat_enc, X_passthrough], axis=1)

    # Keep only features that existed during training (safe alignment)
    X_final = X_final.reindex(columns=transformed_feature_names, fill_value=0)

    logger.info(f"🔧 Preprocessed features with {X_final.shape[1]} columns.")
    return X_final

# ---------------------------------------------------------------------
# Perform inference
# ---------------------------------------------------------------------
def run_inference(model, X, customer_ids, snapshot_date, model_meta):
    y_pred_proba = model.predict_proba(X)[:, 1]
    preds = pd.DataFrame({
        "Customer_ID": customer_ids,
        "prediction_score": y_pred_proba,
        "model_version": model_meta["model_version"],
        "snapshot_date": snapshot_date
    })
    return preds

# ---------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------
def save_predictions(preds, snapshot_date):
    output_dir = resolve_relative_path(f"datamart/gold/model_predictions")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{snapshot_date.replace('-', '_')}.parquet")
    preds.to_parquet(output_path, index=False)
    logger.info(f"💾 Saved predictions to {output_path}")
    return output_path

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(snapshot_date: str):

    # Early exit if before deployment_date
    current_date = datetime.strptime(snapshot_date, "%Y-%m-%d")
    
    if current_date < DEPLOYMENT_DATE:
        logger.info(f"⏭️ Inference skipped for {snapshot_date}: "
                    f"deployment starts on {DEPLOYMENT_DATE.date()}.")
        return
    
    # Load model + artefacts
    model_dir = resolve_relative_path("model_store/deployed_model")
    model, meta, num_imputer, cat_imputer, encoder, train_meta = load_deployed_model(model_dir)

    # Load feature data
    features_df = load_feature_data(snapshot_date)
    if features_df.empty:
        logger.warning(f"No loans with loan_start_date = {snapshot_date} found. Exiting.")
        return

    # Preprocess
    X_preprocessed = preprocess_features(features_df, num_imputer, cat_imputer, encoder, train_meta)

    # Run inference
    preds = run_inference(
        model,
        X_preprocessed,
        features_df["Customer_ID"],
        snapshot_date,
        meta
    )

    # Save
    save_predictions(preds, snapshot_date)
    logger.info(f"🎉 Inference completed successfully for snapshot={snapshot_date}")

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model Inference")
    parser.add_argument("--snapshot_date", required=True, help="Snapshot date in YYYY-MM-DD format")
    args = parser.parse_args()

    main(snapshot_date=args.snapshot_date)