#!/bin/bash
#
# This script performs the following operations:
# 1. Optimizes the provided SavedModel or frozen graph
# 2. Benchmarks performance of the optimized frozen graph after optimization

# Where the dataset is saved to.
DATASET_DIR=../tmp/datasets/imagenet

# Where the SavedModel is saved to.
MODEL_DIR=../tmp/models/inception_v3

# Where the frozen graph is saved to (or will be saved to after optimization if the SavedModel is provided).
GRAPH_DIR=../tmp/models/inception_v3/frozen_graph.pb

# Where the optimized frozen graph will be saved to.
OPTIMIZED_GRAPH_DIR=../tmp/model/inception_v3/optimized_graph.pb

# Optimize the model
python ../src/optimize.py \
    --savedmodel_dir=${MODEL_DIR} \

# Benchmark the optimized model
python ../src/simple_benchmark.py \
    --graph_path=${GRAPH_DIR} \
    --optimized_graph_path=${OPTIMIZED_GRAPH_DIR} \
    --input_nodes="input:0" \
    --output_nodes="ToInt64:0, TopKV2:0" \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=32 \
    --max_batches=30 \
    --image_size=299