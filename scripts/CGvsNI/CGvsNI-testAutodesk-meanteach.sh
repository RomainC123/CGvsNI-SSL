#!/bin/bash

#OAR -n CGvsNI-testAutodesk-meanteach
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=16:00:00
#OAR --stdout scripts_logs/CGvsNI-testAutodesk-meanteach.out
#OAR --stderr scripts_logs/CGvsNI-testAutodesk-meanteach.err
#OAR --project cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./test_cgvsni.py --datasets_to_use Autodesk --label_mode Biclass --img_mode RGB --nb_samples_train 10080 --nb_samples_test 720 --nb_samples_labeled 1008 --max_lr 0.001 --method MeanTeacher --epochs 300 --no-verbose