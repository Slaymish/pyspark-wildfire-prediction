#!/bin/bash

export HADOOP_VERSION=3.3.6
export HADOOP_HOME=/local/Hadoop/hadoop-$HADOOP_VERSION
export SPARK_HOME=/local/spark
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop

# need java8
export JAVA_HOME="/usr/pkg/java/sun-8"
export PATH=${PATH}:$JAVA_HOME:$HADOOP_HOME/bin:$SPARK_HOME/bin
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$JAVA_HOME/jre/lib/amd64/server
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH

pip install pyspark
pip install numpy

# 1. Run the new preprocessing script to generate the non-leaking data
#echo "Running preprocessing script to generate non-leaking data..."
#python3 /home/$USER/Documents/AIML/Solo/preprocess_data.py

# 2. Create HDFS directories if they don't exist
hdfs dfs -mkdir -p /user/$USER/input
hdfs dfs -mkdir -p /user/$USER/output

# 3. Put file into HDFS
echo "Uploading corrected dataset to HDFS..."
hdfs dfs -put -f data/wildfire_processed_no_leakage.csv /user/$USER/input/

chmod +x main.py

# 4. Submit the Spark job, pointing to the correct input file path in HDFS
echo "Submitting Spark job..."

spark-submit \
    --master yarn \
    --deploy-mode client \
    --driver-memory 2g \
    --executor-memory 2g \
    --executor-cores 1 \
    --conf spark.yarn.executor.memoryOverhead=512m \
    main.py \
    --dataset_directory /user/$USER/input/data \
    --experiment class_weight
