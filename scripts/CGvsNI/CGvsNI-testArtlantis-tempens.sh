#!/bin/bash

#OAR -n CGvsNI-testArtlantis-tempens
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=12:00:00
#OAR --stdout scripts_logs/CGvsNI-testArtlantis-tempens.out
#OAR --stderr scripts_logs/CGvsNI-testArtlantis-tempens.err
#OAR --project cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./test_cgvsni.py --datasets_to_use Artlantis --label_mode Biclass --img_mode RGB --nb_samples_train 10800 --nb_samples_test 720 --nb_samples_labeled 1080 --max_lr 0.0002 --method TemporalEnsembling --epochs 300 --no-verbose
