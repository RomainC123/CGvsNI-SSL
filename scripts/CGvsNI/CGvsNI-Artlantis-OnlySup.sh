#!/bin/bash

#OAR -n CGvsNI-Artlantis-onlysup
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=12:00:00
#OAR --stdout scripts_logs/CGvsNI-Artlantis-onlysup.out
#OAR --stderr scripts_logs/CGvsNI-Artlantis-onlysup.err
#OAR --project cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train --data CGvsNI --datasets_to_use Artlantis --label_mode Biclass --nb_samples_total 10800 --nb_samples_test 720 --nb_samples_labeled 10080 --img_mode RGB --model ENet --method OnlySup --epochs 300 --no-verbose
