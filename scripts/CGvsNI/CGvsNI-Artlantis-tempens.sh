#!/bin/bash

#OAR -n CGvsNI-Artlantis-tempens
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=12:00:00
#OAR --stdout scripts_logs/CGvsNI-Artlantis-tempens.out
#OAR --stderr scripts_logs/CGvsNI-Artlantis-tempens.err
#OAR --project cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --data CGvsNI --datasets_to_use Artlantis --label_mode Biclass --nb_samples_total 10800 --nb_samples_test 718 --nb_samples_labeled 1080 --img_mode RGB --model ENet --method TemporalEnsembling --epochs 300 --no-verbose