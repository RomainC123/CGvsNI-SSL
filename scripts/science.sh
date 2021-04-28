#!/bin/bash

#OAR -n MNIST-science
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=1:00:00
#OAR --stdout scripts_logs/science.out
#OAR --stderr scripts_logs/science.err
#OAR --project pr-cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --data MNIST --nb_samples_test 10000 --nb_samples_labeled 0 --img_mode L --model CNN --method TemporalEnsembling --epochs 300 --no-verbose
