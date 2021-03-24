#OAR -n Test_Gricad
#OAR -t gpu 
#OAR -l /nodes=1/gpudevice=1,walltime=0:01:00
#OAR --stdout test_gricad.out
#OAR --stderr test_gricad.err
#OAR --project cg4n6

source applis/environements/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI/src
python main.py --train-test --data CIFAR10 --dataset_name CIFAR10-full-10 --img_mode RGB --method TemporalEnsembling --epochs 150
