# Wildfire Analysis with PySpark

This project uses Apache Spark to analyse the "1.88 Million US Wildfires" dataset. It provides pipelines for feature engineering, model training, and evaluation on a Hadoop YARN cluster.

## Prerequisites

-   SSH access to the cluster node `co246a-1`.
-   `curl` and `unzip` installed on your local machine.

## How to Run

The process is two steps: first, prepare the dataset locally, and second, run the analysis on the cluster.

### Step 1: Prepare the Dataset

From your local machine, run the automated download and processing script. This will download the source data, generate the required CSV, and clean up intermediate files.

```bash
# Make the script executable (only needs to be done once)
chmod +x download_and_process.sh

# Run the script
./download_and_process.sh
```

This will create the final dataset at `data/wildfire_processed_no_leakage.csv`.

> **Note**: If you already have the processed CSV, simply ensure it is in the correct location and skip this step.

### Step 2: Run the Spark Analysis

1.  **Connect to the Cluster**:
    ```bash
    ssh co246a-1
    ```

2.  **Edit the Run Script**:
    The `main.py` script can perform several different analyses. Open `run_spark.sh` and set the `--experiment` flag at the bottom of the file to choose which one to run.

    ```bash
    # In run-spark.sh, find the spark-submit command:
    spark-submit \
        ...
        main.py \
        --dataset_directory /user/$USER/input/data \
        --experiment <CHOOSE_ONE>
    ```

    Your choices for `<CHOOSE_ONE>` are:
    -   `class_weight`: Compares model performance with and without class weighting.
    -   `scaling`: Compares model performance with and without feature scaling.
    -   `pca`: Runs the analysis using PCA for dimensionality reduction.

3.  **Execute the Job**:
    Navigate to your project directory on the cluster and run the script.

    ```bash
    cd /path/to/your/project/
    ./run-spark.sh
    ```

The script will handle uploading the data to HDFS and submitting the selected experiment to the Spark cluster.
