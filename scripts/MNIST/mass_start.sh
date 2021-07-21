oarsub -S ./MNIST-fullsup.sh

oarsub -S ./MNIST-onlysup-20.sh
oarsub -S ./MNIST-onlysup-50.sh
oarsub -S ./MNIST-onlysup-100.sh

oarsub -S ./MNIST-tempens-20.sh
oarsub -S ./MNIST-tempens-50.sh
oarsub -S ./MNIST-tempens-100.sh

oarsub -S ./MNIST-meanteach-20.sh
oarsub -S ./MNIST-meanteach-50.sh
oarsub -S ./MNIST-meanteach-100.sh
