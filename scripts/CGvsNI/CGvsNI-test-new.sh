#!/bin/bash

#OAR -n CGvsNI-test-new
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=8:00:00
#OAR --stdout scripts_logs/CGvsNI-test-new.out
#OAR --stderr scripts_logs/CGvsNI-test-new.err
#OAR --project cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train --data CGvsNI --datasets_to_use Artlantis --label_mode Biclass --nb_samples_total 2048 --nb_samples_test 512 --nb_samples_labeled 256 --img_mode RGB --model ENet --method TemporalEnsemblingNewLoss --epochs 300
