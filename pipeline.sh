#!/bin/bash
./backpas/src/0_run_backbone_extraction.sh ./guroback/guroback ./backpas/dataset/14_bounded_strongly_correlated "*.opb"
./backpas/src/1_create_ml_dataset.py --dataset_path ./backpas/dataset/14_bounded_strongly_correlated --graph_type literals
./backpas/src/2_create_partitions.py --dataset_path ./backpas/dataset/14_bounded_strongly_correlated --ml_dataset_path ./backpas/dataset/14_bounded_strongly_correlated/ml_dataset_literals --wkdir_path ./backpas/wkdir
./backpas/src/3_train.py --ml_dataset_path ./backpas/dataset/14_bounded_strongly_correlated/ml_dataset_literals --dataset_wkdir_path ./backpas/wkdir/14_bounded_strongly_correlated  --use_cuda