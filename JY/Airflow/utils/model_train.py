import os, sys, json, glob, pickle, logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
try:
    from lightgbm import early_stopping, log_evaluation
except Exception:
    from lightgbm.callback import early_stopping, log_evaluation

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_train")

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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_snapshot_dates(base_path: str) -> List[datetime]:
    """List YYYY-MM-DD or YYYY_MM_DD parquet partitions under a directory."""
    if not os.path.exists(base_path):
        return []
    dates = []
    for p in glob.glob(os.path.join(base_path, "*.parquet")):
        name = os.path.basename(p).replace(".parquet", "")
        for fmt in ("%Y-%m-%d", "%Y_%m_%d"):
            try:
                dt = datetime.strptime(name, fmt)
                dates.append(dt)
                break
            except Exception:
                continue
    return sorted(dates)

def load_partition(base_path: str, snap_dt: datetime) -> pd.DataFrame:
    for fmt in ("%Y_%m_%d", "%Y-%m-%d"):
        path = os.path.join(base_path, f"{snap_dt.strftime(fmt)}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError(f"No parquet file found for {snap_dt.strftime('%Y-%m-%d')} in {base_path}")

# ---------------------------
# Dataset assembly
# ---------------------------
def assemble_aligned_dataset(feature_base: str, label_base: str,
                             snap_dates: List[datetime],
                             join_keys: List[str],
                             target_col: str) -> pd.DataFrame:
    frames = []
    for dt in snap_dates:
        feat = load_partition(feature_base, dt)
        lbl = load_partition(label_base, dt)
        merged = feat.merge(lbl[[*join_keys, target_col]], on=join_keys, how="inner")
        merged["snapshot_date"] = dt
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)

# ---------------------------
# Split logic
# ---------------------------
def build_splits(all_dates: List[datetime],
                 train_m: int, val_m: int, test_m: int, oot_m: int,
                 latest_anchor: datetime = None) -> Dict[str, List[datetime]]:
    total = train_m + val_m + test_m + oot_m
    if len(all_dates) < total:
        return {}

    all_dates_sorted = sorted(all_dates)
    if latest_anchor is None:
        latest_anchor = all_dates_sorted[-1]

    anchor_idx = max(i for i, d in enumerate(all_dates_sorted) if d <= latest_anchor)
    start_idx = anchor_idx - total + 1
    if start_idx < 0:
        return {}

    window = all_dates_sorted[start_idx: anchor_idx + 1]
    assert len(window) == total

    return {
        "train": window[:train_m],
        "val":   window[train_m: train_m + val_m],
        "test":  window[train_m + val_m: train_m + val_m + test_m],
        "oot":   window[-oot_m:],
        "latest_training_window_date": window[-1].strftime("%Y-%m-%d"),
    }

# ---------------------------
# Preprocessing (imputation + dtype)
# ---------------------------
def preprocess_for_lgbm(df: pd.DataFrame,
                        numeric_cols: List[str],
                        categorical_cols: List[str],
                        target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0)
    for c in categorical_cols:
        if c not in df.columns:
            continue
        df[c] = df[c].astype("object").fillna("Unknown").astype("category")

    X = df[numeric_cols + categorical_cols].copy()
    y = df[target_col].copy()
    return X, y

# ---------------------------
# Metrics helpers
# ---------------------------
def eval_regression(y_true, y_pred) -> Dict[str, float]:
    # Older sklearn versions do not support squared=False
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_true, y_pred) ** 0.5  # manual sqrt for old versions
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def filter_metrics(metrics_dict, keep_list):
    return {k: v for k, v in metrics_dict.items() if k in keep_list}

# ---------------------------
# Feature importance helpers
# ---------------------------
def extract_lgbm_importance(model: LGBMRegressor, feature_names: List[str]) -> Dict[str, float]:
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    importance = {n: float(g) for n, g in zip(names, gains)}
    for f in feature_names:
        importance.setdefault(f, 0.0)
    importance_sorted = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
    return importance_sorted

# ---------------------------
# Main training
# ---------------------------
def main(latest_snapshot_date: str = None):
    logger.info("🚀 Starting model training job")

    # --- Paths
    feature_store_dir = resolve_relative_path("datamart/gold/feature_store")
    label_store_dir   = resolve_relative_path("datamart/gold/label_store")
    model_store_dir   = resolve_relative_path("model_store/candidate_models")
    os.makedirs(model_store_dir, exist_ok=True)

    model_cfg_path = resolve_relative_path("config/ML_config.yaml")
    cfg = load_yaml(model_cfg_path)

    # --- Configs
    splits = cfg["splits"]
    train_m = int(splits["train_months"])
    val_m   = int(splits["val_months"])
    test_m  = int(splits["test_months"])
    oot_m   = int(splits["oot_months"])
    total_m = train_m + val_m + test_m + oot_m

    target_col        = cfg["target_col"]
    join_keys         = cfg.get("join_keys", ["store_nbr", "family"])
    categorical_cols  = cfg["categorical_features"]
    numeric_cols      = cfg["numerical_features"]
    eval_metrics_cfg  = cfg.get("evaluation_metrics", ["rmse", "mae", "r2"])

    # --- Available aligned snapshots
    feat_dates  = set(list_snapshot_dates(feature_store_dir))
    label_dates = set(list_snapshot_dates(label_store_dir))
    common_dates = sorted(list(feat_dates & label_dates))
    if not common_dates:
        logger.info("ℹ️ No aligned feature/label snapshots found yet. Skipping.")
        return

    latest_anchor = None
    if latest_snapshot_date:
        latest_anchor = datetime.strptime(latest_snapshot_date, "%Y-%m-%d")

    split_dates = build_splits(common_dates, train_m, val_m, test_m, oot_m, latest_anchor)
    if not split_dates:
        logger.info(f"ℹ️ Insufficient data. Need {total_m} aligned months; found {len(common_dates)}. Skipping.")
        return

    latest_training_window_date = split_dates["latest_training_window_date"]
    out_dir = os.path.join(model_store_dir, latest_training_window_date)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"🗓️ Using latest training window ending: {latest_training_window_date}")
    logger.info("📦 Window sizes → train=%d, val=%d, test=%d, oot=%d",
                len(split_dates["train"]), len(split_dates["val"]),
                len(split_dates["test"]), len(split_dates["oot"]))

    # --- Assemble datasets
    train_df = assemble_aligned_dataset(feature_store_dir, label_store_dir, split_dates["train"], join_keys, target_col)
    val_df   = assemble_aligned_dataset(feature_store_dir, label_store_dir, split_dates["val"],   join_keys, target_col)
    test_df  = assemble_aligned_dataset(feature_store_dir, label_store_dir, split_dates["test"],  join_keys, target_col)
    oot_df   = assemble_aligned_dataset(feature_store_dir, label_store_dir, split_dates["oot"],   join_keys, target_col)

    # --- Preprocess
    X_train, y_train = preprocess_for_lgbm(train_df, numeric_cols, categorical_cols, target_col)
    X_val,   y_val   = preprocess_for_lgbm(val_df,   numeric_cols, categorical_cols, target_col)
    X_test,  y_test  = preprocess_for_lgbm(test_df,  numeric_cols, categorical_cols, target_col)
    X_oot,   y_oot   = preprocess_for_lgbm(oot_df,   numeric_cols, categorical_cols, target_col)

    # --- Default LightGBM params (no tuning)
    lgbm_params = cfg.get("lgbm_base_params", {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 500,
        "num_leaves": 63,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    })

    logger.info("🧠 Training LightGBM model (no hyperparameter search)...")
    model = LGBMRegressor(**lgbm_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        categorical_feature=categorical_cols,
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=0),
        ],
    )

    best_model = model
    best_iter = getattr(model, "best_iteration_", None)
    logger.info("✅ Training complete. Best iteration: %s", best_iter or "N/A")

    # --- Evaluate
    val_pred  = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    oot_pred  = best_model.predict(X_oot)

    val_metrics  = filter_metrics(eval_regression(y_val,  val_pred),  eval_metrics_cfg)
    test_metrics = filter_metrics(eval_regression(y_test, test_pred), eval_metrics_cfg)
    oot_metrics  = filter_metrics(eval_regression(y_oot,  oot_pred),  eval_metrics_cfg)

    logger.info(f"📊 VAL  → {val_metrics}")
    logger.info(f"📊 TEST → {test_metrics}")
    logger.info(f"📊 OOT  → {oot_metrics}")

    # --- Feature importance
    feature_order = X_train.columns.tolist()
    importance_full = extract_lgbm_importance(best_model, feature_order)
    importance_top20 = [{"feature": k, "gain": v} for k, v in list(importance_full.items())[:20]]

    # --- Split stats
    split_stats = {
        "train": {"start": split_dates["train"][0].strftime("%Y-%m-%d"),
                  "end": split_dates["train"][-1].strftime("%Y-%m-%d"),
                  "rows": len(train_df)},
        "val": {"start": split_dates["val"][0].strftime("%Y-%m-%d"),
                "end": split_dates["val"][-1].strftime("%Y-%m-%d"),
                "rows": len(val_df)},
        "test": {"start": split_dates["test"][0].strftime("%Y-%m-%d"),
                 "end": split_dates["test"][-1].strftime("%Y-%m-%d"),
                 "rows": len(test_df)},
        "oot": {"start": split_dates["oot"][0].strftime("%Y-%m-%d"),
                "end": split_dates["oot"][-1].strftime("%Y-%m-%d"),
                "rows": len(oot_df)},
    }

    # --- Save model and metadata
    model_path = os.path.join(out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    metadata = {
        "model_version": latest_training_window_date,
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_window_end": latest_training_window_date,
        "join_keys": join_keys,
        "target_col": target_col,
        "categorical_features": categorical_cols,
        "numerical_features": numeric_cols,
        "best_params": lgbm_params,
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
            "oot": oot_metrics,
        },
        "feature_importance_top20_gain": importance_top20,
        "feature_importance_full_gain": importance_full,
        "split_stats": split_stats,
        "artifact_paths": {"model": model_path},
    }

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"💾 Saved model → {model_path}")
    logger.info(f"📝 Saved metadata → {meta_path}")
    logger.info("🎉 Training complete.")


if __name__ == "__main__":
    latest = sys.argv[1] if len(sys.argv) > 1 else None
    main(latest_snapshot_date=latest)
