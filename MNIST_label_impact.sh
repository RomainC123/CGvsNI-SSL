cd utils
cd MNIST
python make_dataset.py --dataset_name MNIST_10 --nb_labels 0.1
python make_dataset.py --dataset_name MNIST_1 --nb_labels 0.01
python make_dataset.py --dataset_name MNIST_0.1 --nb_labels 0.001

cd ..
cd ..
cd src
python main.py --data MNIST --dataset_name MNIST_10 --img_mode L --method TemporalEnsembling --epochs 30 --no-graph --no-test
python main.py --data MNIST --dataset_name MNIST_1 --img_mode L --method TemporalEnsembling --epochs 30 --no-graph --no-test
python main.py --data MNIST --dataset_name MNIST_0.1 --img_mode L --method TemporalEnsembling --epochs 30 --no-graph --no-test
