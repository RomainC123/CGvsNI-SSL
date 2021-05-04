#!/bin/bash

#OAR -n CIFAR10-fullsup
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=4:00:00
#OAR --stdout scripts_logs/CIFAR10-fullsup.out
#OAR --stderr scripts_logs/CIFAR10-fullsup.err
#OAR --project pr-cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --data CIFAR10 --nb_samples_test 10000 --nb_samples_labeled -1 --img_mode RGB --model VGG --method TemporalEnsembling --epochs 300 --no-verbose
