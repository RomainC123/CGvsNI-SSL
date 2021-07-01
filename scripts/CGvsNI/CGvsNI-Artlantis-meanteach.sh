#!/bin/bash

#OAR -n CGvsNI-Artlantis-meanteach
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=12:00:00
#OAR --stdout scripts_logs/CGvsNI-Artlantis-meanteach.out
#OAR --stderr scripts_logs/CGvsNI-Artlantis-meanteach.err
#OAR --project cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --data CGvsNI --datasets_to_use Artlantis --label_mode Biclass --nb_samples_total 10080 --nb_samples_test 720 --nb_samples_labeled 1008 --img_mode RGB --model ENet --method MeanTeacher --epochs 300 --no-verbose
