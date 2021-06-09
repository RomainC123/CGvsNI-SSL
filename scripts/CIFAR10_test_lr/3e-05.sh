#!/bin/bash

#OAR -n CIFAR10_test_lr_3e-05
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=18:00:00
#OAR --stdout CIFAR10_test_lr/scripts_logs/CIFAR10_test_lr_3e-05.out
#OAR --stderr CIFAR10_test_lr/scripts_logs/CIFAR10_test_lr_3e-05.err
#OAR --project cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --folder CIFAR10_test_lr --name 3e-05 --data CIFAR10 --nb_samples_test 10000 --nb_samples_labeled 1000 --img_mode RGB --model SimpleNet --max_lr 3e-05 --method OnlySup --epochs 300 --no-verbose