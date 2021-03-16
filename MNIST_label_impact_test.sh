cd src
python main.py --data MNIST --dataset_name MNIST_10 --img_mode L --method TemporalEnsembling --epochs 30 --no-train
python main.py --data MNIST --dataset_name MNIST_1 --img_mode L --method TemporalEnsembling --epochs 30 --no-train
python main.py --data MNIST --dataset_name MNIST_0.1 --img_mode L --method TemporalEnsembling --epochs 30 --no-train
