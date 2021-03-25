#!/bin/bash

#OAR -n Test_Gricad
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=1:00:00
#OAR --stdout scripts_logs/test_gricad.out
#OAR --stderr scripts_logs/test_gricad.err
#OAR --project pr-cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --data CIFAR10 --dataset_name CIFAR10-full-10 --img_mode RGB --method TemporalEnsembling --epochs 150 --no-verbose
