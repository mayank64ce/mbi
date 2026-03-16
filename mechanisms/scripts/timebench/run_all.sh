#!/bin/bash

# Timebench script to run aim_HE_simulate_l1 and aim_HE_simulate_l2
# on all 3 datasets with seed=42

echo "=== Starting Timebench ==="
echo "Seed: 42"
echo ""

# COMPAS - L1
echo "=== Running FHAIM L1 on COMPAS ==="
python aim_HE_simulate_l1.py \
    --dataset "../data/compas_train.csv" \
    --domain "../data/compass-domain.json" \
    --seed 42

# COMPAS - L2
echo "=== Running FHAIM L2 on COMPAS ==="
python aim_HE_simulate_l2.py \
    --dataset "../data/compas_train.csv" \
    --domain "../data/compass-domain.json" \
    --seed 42

# Breast Cancer - L1
echo "=== Running FHAIM L1 on Breast Cancer ==="
python aim_HE_simulate_l1.py \
    --dataset "../data/breast_train.csv" \
    --domain "../data/breast-domain.json" \
    --seed 42

# Breast Cancer - L2
echo "=== Running FHAIM L2 on Breast Cancer ==="
python aim_HE_simulate_l2.py \
    --dataset "../data/breast_train.csv" \
    --domain "../data/breast-domain.json" \
    --seed 42

# Diabetes - L1
echo "=== Running FHAIM L1 on Diabetes ==="
python aim_HE_simulate_l1.py \
    --dataset "../data/diabetes_train.csv" \
    --domain "../data/diabetes-domain.json" \
    --seed 42

# Diabetes - L2
echo "=== Running FHAIM L2 on Diabetes ==="
python aim_HE_simulate_l2.py \
    --dataset "../data/diabetes_train.csv" \
    --domain "../data/diabetes-domain.json" \
    --seed 42

echo ""
echo "=== Timebench Complete ==="
