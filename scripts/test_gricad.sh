#OAR -t gpu -l /nodes=1/gpunodes=1,walltime=00:01:00
#OAR --stdout test_gricad.out
#OAR --stderr test_gricad.err
#OAR --project cg4n6

cd code/CGvsNI/src
python main.py --train-test --data CIFAR10 --dataset_name CIFAR10-full-10 --img_mode RGB --method TemporalEnsembling --epochs 150
