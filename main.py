import argparse
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    OneHotEncoder,
    StandardScaler,
    Imputer,
    PCA,
    SQLTransformer,
)
from pyspark.ml.classification import (
    RandomForestClassifier,
    DecisionTreeClassifier,
    LogisticRegression,
    GBTClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def create_spark_session():
    """Creates and configures a SparkSession."""
    spark = (
        SparkSession.builder.appName("WildfireAnalysisExperiments")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, data_path):
    """Loads the preprocessed, non-leaking dataset."""
    print("Loading non-leaking data...")
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(f"{data_path}/wildfire_processed_no_leakage.csv")
    )
    print(f"Dataset shape: {df.count()} rows, {len(df.columns)} columns")
    return df


def prepare_feature_lists(df):
    """Defines and returns lists of categorical and numerical features."""
    print("Preparing feature lists (non-leaking)...")
    cats = [
        "season",
        "geographic_region",
        "SOURCE_SYSTEM_TYPE",
        "NWCG_REPORTING_AGENCY",
        "STAT_CAUSE_DESCR",
        "OWNER_DESCR",
        "STATE",
        "SOURCE_SYSTEM",
    ]
    nums = [
        "discovery_month",
        "discovery_year",
        "discovery_day_of_week",
        "discovery_day_of_year",
        "decade",
        "fire_duration_days",
        "discovery_hour",
        "LATITUDE",
        "LONGITUDE",
        "lat_bin",
        "lon_bin",
        "location_fire_density",
        "human_caused",
        "STAT_CAUSE_CODE",
        "FIPS_CODE",
        "OWNER_CODE",
        "state_fire_frequency",
        "county_fire_frequency",
        "cause_frequency",
        "agency_fire_frequency",
        "fire_season",
        "is_weekend",
        "high_risk_month",
        "drought_season",
    ]
    available_cats = [c for c in cats if c in df.columns]
    available_nums = [n for n in nums if n in df.columns]
    print(
        f"Using {len(available_cats)} categorical and {len(available_nums)} numerical features."
    )
    return df, available_cats, available_nums


# =============================================================================
# HELPER FUNCTIONS FOR EXPERIMENTS
# =============================================================================


def get_detailed_metrics(predictions, label_map):
    """Calculates and prints the confusion matrix and per-class metrics."""
    print("\n--- Detailed Metrics ---")
    # Confusion Matrix
    print("Confusion Matrix (Predicted vs. True):")
    cm = predictions.groupBy("label", "prediction").count().toPandas()
    cm["label"] = cm["label"].apply(lambda l: label_map.get(l, "N/A"))
    cm["prediction"] = cm["prediction"].apply(lambda p: label_map.get(p, "N/A"))
    cm_pivot = cm.pivot_table(
        index="label", columns="prediction", values="count", fill_value=0
    )
    print(cm_pivot)

    # Per-Class Metrics
    print("\nPer-Class Metrics:")
    metrics_data = []
    for label_numeric, label_str in label_map.items():
        tp = predictions.filter(
            (col("label") == label_numeric)
            & (col("prediction") == label_numeric)
        ).count()
        fp = predictions.filter(
            (col("label") != label_numeric)
            & (col("prediction") == label_numeric)
        ).count()
        fn = predictions.filter(
            (col("label") == label_numeric)
            & (col("prediction") != label_numeric)
        ).count()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        metrics_data.append(
            {
                "Class": label_str,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )

    metrics_df = pd.DataFrame(metrics_data)
    print(metrics_df.to_string(index=False))


def get_feature_importances(model, pipeline_model):
    """Extracts and prints feature importances from a tree-based model."""
    print("\n--- Feature Importances (from RandomForest) ---")
    # The VectorAssembler is the second to last stage in this pipeline
    assembler_stage = pipeline_model.stages[-2]
    feature_list = assembler_stage.getInputCols()
    importances = model.featureImportances.toArray()

    feature_importance_df = pd.DataFrame(
        list(zip(feature_list, importances)),
        columns=["feature", "importance"],
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="importance", ascending=False
    )

    print(feature_importance_df.head(15).to_string(index=False))


def create_binary_labels_for_gbt(df):
    """Creates a binary label for the GBT model (Large vs. Small fires)."""
    return df.withColumn(
        "binary_label",
        when(col("FIRE_SIZE_CLASS").isin(["C", "D", "E", "F", "G"]), 1.0).otherwise(
            0.0
        ),
    )


def train_all_models(train_data, test_data):
    """Trains and evaluates all models for scaling and PCA experiments."""
    # --- Multiclass Models ---
    models = {
        "RandomForest": RandomForestClassifier(
            featuresCol="features", labelCol="label", seed=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            featuresCol="features", labelCol="label", seed=42
        ),
        "LogisticRegression": LogisticRegression(
            featuresCol="features", labelCol="label"
        ),
    }
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )

    for name, model in models.items():
        print(f"  Training {name}...")
        start_time = time.time()
        fitted_model = model.fit(train_data)
        predictions = fitted_model.transform(test_data)
        accuracy = evaluator.evaluate(
            predictions, {evaluator.metricName: "accuracy"}
        )
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        duration = time.time() - start_time
        print(
            f"    -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {duration:.2f}s"
        )

    # --- GBT Binary Model ---
    print("  Training GBT (Binary)...")
    train_bin = create_binary_labels_for_gbt(train_data)
    test_bin = create_binary_labels_for_gbt(test_data)
    gbt = GBTClassifier(featuresCol="features", labelCol="binary_label", seed=42)
    start_time = time.time()
    gbt_model = gbt.fit(train_bin)
    gbt_predictions = gbt_model.transform(test_bin)
    binary_evaluator = MulticlassClassificationEvaluator(
        labelCol="binary_label", predictionCol="prediction"
    )
    gbt_accuracy = binary_evaluator.evaluate(
        gbt_predictions, {binary_evaluator.metricName: "accuracy"}
    )
    gbt_f1 = binary_evaluator.evaluate(
        gbt_predictions, {binary_evaluator.metricName: "f1"}
    )
    duration = time.time() - start_time
    print(
        f"    -> Accuracy: {gbt_accuracy:.4f}, F1: {gbt_f1:.4f}, Time: {duration:.2f}s"
    )


# =============================================================================
# EXPERIMENT 1: CLASS WEIGHTING
# =============================================================================


def build_class_weight_pipeline(categorical_cols, numerical_cols):
    """Builds the preprocessing pipeline for the class weighting experiment."""
    stages = []
    imputed_num_cols = [f"{c}_imputed" for c in numerical_cols]
    imputer = Imputer(
        inputCols=numerical_cols, outputCols=imputed_num_cols, strategy="mean"
    )
    stages.append(imputer)

    encoded_cat_cols = []
    for c in categorical_cols:
        indexer = StringIndexer(
            inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"
        )
        encoder = OneHotEncoder(
            inputCol=indexer.getOutputCol(), outputCol=f"{c}_enc"
        )
        stages.extend([indexer, encoder])
        encoded_cat_cols.append(encoder.getOutputCol())

    label_indexer = StringIndexer(
        inputCol="FIRE_SIZE_CLASS",
        outputCol="label",
        stringOrderType="alphabetAsc",
    )
    stages.append(label_indexer)

    assembler_inputs = imputed_num_cols + encoded_cat_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="raw_features",
        handleInvalid="keep",
    )
    stages.append(assembler)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False,
    )
    stages.append(scaler)
    return Pipeline(stages=stages)


def run_class_weight_experiment(df, cats, nums):
    """Runs a full training and evaluation experiment for class weighting."""
    print("\n\n" + "=" * 40)
    print("  RUNNING CLASS WEIGHTING EXPERIMENT  ")
    print("=" * 40)

    pipeline = build_class_weight_pipeline(cats, nums)
    pipeline_model = pipeline.fit(df)
    transformed_df = pipeline_model.transform(df)

    # --- Sub-Experiment 1: Baseline (No Weights) ---
    print("\n--- Sub-Experiment 1: Baseline (No Weights) ---")
    train_data, test_data = transformed_df.randomSplit([0.8, 0.2], seed=42)
    train_data.cache()
    test_data.cache()
    print(
        f"Training samples: {train_data.count()}, Test samples: {test_data.count()}"
    )

    models = {
        "RandomForest": RandomForestClassifier(
            featuresCol="features", labelCol="label", seed=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            featuresCol="features", labelCol="label", seed=42
        ),
        "LogisticRegression": LogisticRegression(
            featuresCol="features", labelCol="label"
        ),
    }
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )
    rf_model_for_importance = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        fitted_model = model.fit(train_data)
        predictions = fitted_model.transform(test_data)
        accuracy = evaluator.evaluate(
            predictions, {evaluator.metricName: "accuracy"}
        )
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        duration = time.time() - start_time
        print(
            f"  -> Accuracy: {accuracy:.4f}, F1 (Weighted): {f1:.4f}, Time: {duration:.2f}s"
        )

        if name == "RandomForest":
            rf_model_for_importance = fitted_model
            labels = pipeline_model.stages[-3].labels
            label_map = {float(i): label for i, label in enumerate(labels)}
            get_detailed_metrics(predictions, label_map)

    if rf_model_for_importance:
        get_feature_importances(rf_model_for_importance, pipeline_model)

    # --- Sub-Experiment 2: With Class Weights ---
    print("\n--- Sub-Experiment 2: With Class Weighting ---")
    balance_df = train_data.groupBy("label").count()
    total_samples = train_data.count()
    num_classes = balance_df.count()
    balance_df = balance_df.withColumn(
        "weight", total_samples / (num_classes * col("count"))
    )
    train_data_weighted = train_data.join(
        balance_df.select("label", "weight"), "label", "left"
    )
    print("Class weights calculated and applied.")

    weighted_models = {
        "RandomForest": RandomForestClassifier(
            featuresCol="features", labelCol="label", seed=42, weightCol="weight"
        ),
        "DecisionTree": DecisionTreeClassifier(
            featuresCol="features", labelCol="label", seed=42, weightCol="weight"
        ),
        "LogisticRegression": LogisticRegression(
            featuresCol="features", labelCol="label", weightCol="weight"
        ),
    }

    for name, model in weighted_models.items():
        print(f"\nTraining {name} with weights...")
        start_time = time.time()
        fitted_model = model.fit(train_data_weighted)
        predictions = fitted_model.transform(test_data)
        accuracy = evaluator.evaluate(
            predictions, {evaluator.metricName: "accuracy"}
        )
        f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
        duration = time.time() - start_time
        print(
            f"  -> Accuracy: {accuracy:.4f}, F1 (Weighted): {f1:.4f}, Time: {duration:.2f}s"
        )

    train_data.unpersist()
    test_data.unpersist()


# =============================================================================
# EXPERIMENT 2 & 3: SCALING AND PCA
# =============================================================================


def build_scaling_pca_pipeline(
    categorical_cols, numerical_cols, use_scaling=True
):
    """Builds a preprocessing pipeline for scaling/PCA experiments."""
    print(f"Building pipeline (Scaling: {use_scaling})...")
    stages = []

    imputed_num_cols = [f"{c}_imputed" for c in numerical_cols]
    imputer = Imputer(
        inputCols=numerical_cols, outputCols=imputed_num_cols, strategy="mean"
    )
    stages.append(imputer)

    encoded_cat_cols = []
    for c in categorical_cols:
        indexer = StringIndexer(
            inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"
        )
        encoder = OneHotEncoder(
            inputCol=indexer.getOutputCol(), outputCol=f"{c}_enc"
        )
        stages.extend([indexer, encoder])
        encoded_cat_cols.append(encoder.getOutputCol())

    label_indexer = StringIndexer(inputCol="FIRE_SIZE_CLASS", outputCol="label")
    stages.append(label_indexer)

    assembler_inputs = imputed_num_cols + encoded_cat_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs, outputCol="raw_features"
    )
    stages.append(assembler)

    if use_scaling:
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=False,
        )
        stages.append(scaler)
    else:
        # If not scaling, just rename raw_features to features
        renamer = SQLTransformer(
            statement="SELECT *, raw_features AS features FROM __THIS__"
        )
        stages.append(renamer)

    return Pipeline(stages=stages)


def run_scaling_experiment(df, cats, nums):
    """Runs the experiment comparing performance with and without scaling."""
    print("\n\n" + "=" * 40)
    print("  RUNNING FEATURE SCALING EXPERIMENT  ")
    print("=" * 40)

    # --- With Scaling ---
    print("\n--- Sub-Experiment 1: With Feature Scaling ---")
    pipeline_scaled = build_scaling_pca_pipeline(cats, nums, use_scaling=True)
    model_scaled = pipeline_scaled.fit(df)
    df_scaled = model_scaled.transform(df)
    train_scaled, test_scaled = df_scaled.randomSplit([0.8, 0.2], seed=42)
    train_scaled.cache()
    test_scaled.cache()
    print(
        f"Training samples: {train_scaled.count()}, Test samples: {test_scaled.count()}"
    )
    train_all_models(train_scaled, test_scaled)
    train_scaled.unpersist()
    test_scaled.unpersist()

    # --- Without Scaling ---
    print("\n--- Sub-Experiment 2: Without Feature Scaling ---")
    pipeline_unscaled = build_scaling_pca_pipeline(
        cats, nums, use_scaling=False
    )
    model_unscaled = pipeline_unscaled.fit(df)
    df_unscaled = model_unscaled.transform(df)
    train_unscaled, test_unscaled = df_unscaled.randomSplit(
        [0.8, 0.2], seed=42
    )
    train_unscaled.cache()
    test_unscaled.cache()
    print(
        f"Training samples: {train_unscaled.count()}, Test samples: {test_unscaled.count()}"
    )
    train_all_models(train_unscaled, test_unscaled)
    train_unscaled.unpersist()
    test_unscaled.unpersist()


def run_pca_experiment(df, cats, nums):
    """Runs experiments using PCA with various numbers of components."""
    print("\n\n" + "=" * 40)
    print("    RUNNING PCA EXPERIMENT    ")
    print("=" * 40)

    # Build a base pipeline that does everything *before* PCA
    base_pipeline = build_scaling_pca_pipeline(cats, nums, use_scaling=True)
    base_stages = base_pipeline.getStages()[:-1]  # All stages except final
    base_stages.append(
        StandardScaler(
            inputCol="raw_features", outputCol="scaled_features"
        )
    )

    base_pipeline_for_pca = Pipeline(stages=base_stages)
    base_model = base_pipeline_for_pca.fit(df)
    processed_df = base_model.transform(df).cache()
    print("Base data for PCA has been processed and cached.")

    k_values = [10, 20, 30, 40, 50]
    for k in k_values:
        print(f"\n--- PCA Experiment with k={k} ---")
        pca = PCA(k=k, inputCol="scaled_features", outputCol="features")
        pca_model = pca.fit(processed_df)

        explained_variance = sum(pca_model.explainedVariance.toArray())
        print(f"  Explained Variance for k={k}: {explained_variance:.4f}")

        df_pca = pca_model.transform(processed_df)
        train_pca, test_pca = df_pca.randomSplit([0.8, 0.2], seed=42)
        train_pca.cache()
        test_pca.cache()
        print(
            f"  Training samples: {train_pca.count()}, Test samples: {test_pca.count()}"
        )
        train_all_models(train_pca, test_pca)
        train_pca.unpersist()
        test_pca.unpersist()

    processed_df.unpersist()


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================


def main():
    """Main function to orchestrate the experiments."""
    parser = argparse.ArgumentParser(
        description="Run PySpark ML experiments on wildfire data."
    )
    parser.add_argument(
        "--dataset_directory",
        required=True,
        help="HDFS path to the directory containing the input CSV.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["class_weight", "scaling", "pca"],
        help="The experiment to run.",
    )
    args = parser.parse_args()

    spark = create_spark_session()
    df = load_data(spark, args.dataset_directory)
    df_prep, cats, nums = prepare_feature_lists(df)

    if args.experiment == "class_weight":
        run_class_weight_experiment(df_prep, cats, nums)
    elif args.experiment == "scaling":
        run_scaling_experiment(df_prep, cats, nums)
    elif args.experiment == "pca":
        run_pca_experiment(df_prep, cats, nums)

    spark.stop()


if __name__ == "__main__":
    main()
