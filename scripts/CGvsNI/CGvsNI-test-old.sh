#!/bin/bash

#OAR -n CGvsNI-test-old
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=8:00:00
#OAR --stdout scripts_logs/CGvsNI-test-old.out
#OAR --stderr scripts_logs/CGvsNI-test-old.err
#OAR --project cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train --data CGvsNI --datasets_to_use Artlantis --label_mode Biclass --nb_samples_total 4000 --nb_samples_test 500 --nb_samples_labeled 500 --img_mode RGB --model ENet --method TemporalEnsembling --epochs 300
