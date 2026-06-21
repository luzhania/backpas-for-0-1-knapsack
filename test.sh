#!/bin/bash

./backpas/src/4_create_trust_region.py --dataset_path ./backpas/dataset/14_bounded_strongly_correlated --dataset_wkdir_path ./backpas/wkdir/14_bounded_strongly_correlated --method thresholded_expected_error --model_path ./backpas/wkdir/14_bounded_strongly_correlated/ml_training/graph_with_literals_3_GTR/best_model.pth --run_purpose test --use_cuda

find backpas/wkdir/14_bounded_strongly_correlated/test/graph_with_literals_3_GTR/trust_region_thresholded_expected_error backpas/wkdir/14_bounded_strongly_correlated/test/baseline -name "*.opb" > instance_list.txt

./backpas/src/5_run_gurobi.sh instance_list.txt