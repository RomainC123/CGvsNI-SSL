#!/bin/bash

#OAR -n MNIST-1000-test-unsup-loss
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=1:00:00
#OAR --stdout scripts_logs/MNIST-1000-test-unsup-loss.out
#OAR --stderr scripts_logs/MNIST-1000-test-unsup-loss.err
#OAR --project pr-cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --data MNIST --dataset_name MNIST-1000 --img_mode L --method TemporalEnsembling --epochs 150
