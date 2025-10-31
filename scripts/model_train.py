import argparse
import os
import json
import glob
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def main(snapshotdate: str):
    print("\n\n--- Starting Linear Regression Training ---\n")

    spark = SparkSession.builder.master("local[*]").appName("lr_training_multi").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # ---------------- Paths ----------------
    base_dir = "/opt/airflow/scripts/datamart/train_test/"
    model_bank_dir = "/opt/airflow/scripts/datamart/model_bank/"
    os.makedirs(model_bank_dir, exist_ok=True)

    stamp = snapshotdate.replace("-", "_")

    # ---------------- Find all train/test/oot files ----------------
    train_files = glob.glob(os.path.join(base_dir, f"train_store*_{stamp}.parquet"))
    if not train_files:
        raise FileNotFoundError(f"No train files found for {snapshotdate} in {base_dir}")

    print(f"📦 Found {len(train_files)} train/test/oot file sets for {snapshotdate}")

    evaluator_rmse = RegressionEvaluator(labelCol="weekly_sales_sum", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="weekly_sales_sum", predictionCol="prediction", metricName="r2")

    summary_records = []

    # ---------------- Loop over each combination ----------------
    for train_path in train_files:
        base_name = os.path.basename(train_path)
        combo_name = base_name.replace("train_", "").replace(f"_{stamp}.parquet", "")
        test_path = os.path.join(base_dir, f"test_{combo_name}_{stamp}.parquet")
        oot_path = os.path.join(base_dir, f"oot_{combo_name}_{stamp}.parquet")

        if not os.path.exists(test_path) or not os.path.exists(oot_path):
            print(f"⚠️ Missing test/oot for {combo_name}, skipping...")
            continue

        print(f"\n🚀 Training model for {combo_name}")

        # ---------------- Load Data ----------------
        train_df = spark.read.parquet(train_path)
        test_df = spark.read.parquet(test_path)
        oot_df = spark.read.parquet(oot_path)
        print(f"✅ Loaded: Train={train_df.count()}, Test={test_df.count()}, OOT={oot_df.count()}")

        # ---------------- Train Model ----------------
        target_col = "weekly_sales_sum"
        lr = LinearRegression(
            featuresCol="features",
            labelCol=target_col,
            predictionCol="prediction",
            regParam=0.1,
            elasticNetParam=0.0,
        )

        lr_model = lr.fit(train_df)
        print("✅ Model training complete")

        # ---------------- Evaluate ----------------
        metrics = {}
        for name, df_ in [("Train", train_df), ("Test", test_df), ("OOT", oot_df)]:
            preds = lr_model.transform(df_)
            rmse = evaluator_rmse.evaluate(preds)
            r2 = evaluator_r2.evaluate(preds)
            metrics[name.lower()] = {"rmse": rmse, "r2": r2}
            print(f"📊 {combo_name} [{name}] RMSE={rmse:.4f} | R²={r2:.4f}")

        # ---------------- Save Model ----------------
        model_path = os.path.join(model_bank_dir, f"lr_model_{combo_name}_{stamp}")
        if os.path.exists(model_path):
            import shutil
            shutil.rmtree(model_path)
        lr_model.save(model_path)
        print(f"💾 Model saved to {model_path}")

        # ---------------- Save Metadata ----------------
        metadata = {
            "combo": combo_name,
            "snapshot_date": snapshotdate,
            "model_type": "LinearRegression",
            "regParam": lr.getOrDefault("regParam"),
            "elasticNetParam": lr.getOrDefault("elasticNetParam"),
            "train_rmse": metrics["train"]["rmse"],
            "test_rmse": metrics["test"]["rmse"],
            "oot_rmse": metrics["oot"]["rmse"],
            "train_r2": metrics["train"]["r2"],
            "test_r2": metrics["test"]["r2"],
            "oot_r2": metrics["oot"]["r2"],
        }
        meta_path = os.path.join(model_bank_dir, f"lr_model_{combo_name}_{stamp}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"📝 Metadata saved: {meta_path}")

        # ---------------- Append to Summary ----------------
        summary_records.append(metadata)

    # ---------------- Save Combined Summary CSV ----------------
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(model_bank_dir, f"training_summary_{stamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📈 Training summary saved to: {summary_path}")

    spark.stop()
    print("\n--- Completed Multi-Combination Linear Regression Training ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Linear Regression models per store-family")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
