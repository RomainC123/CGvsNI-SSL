import os

def make_script(folder, name, walltime, nb_samples_labeled, max_lr):

    cmd = f"#!/bin/bash\n\
\n\
#OAR -n {folder + '_' + name}\n\
#OAR -t gpu\n\
#OAR -l /nodes=1/gpudevice=1,walltime={walltime}\n\
#OAR --stdout {folder}/scripts_logs/{folder + '_' + name}.out\n\
#OAR --stderr {folder}/scripts_logs/{folder + '_' + name}.err\n\
#OAR --project cg4n6\n\
\n\
source /applis/environments/conda.sh\n\
conda activate CGDetection\n\
\n\
cd ~/code/CGvsNI-SSL/src\n\
python ./main.py --train-test --folder {folder} --name {name} --data CIFAR10 --nb_samples_test 10000 --nb_samples_labeled {nb_samples_labeled} --img_mode RGB --model SimpleNet --max_lr {max_lr} --method TemporalEnsemblingNewLoss --epochs 300 --no-verbose"

    with open(os.path.join(folder, f'{name}.sh'), 'w+') as f:
        f.write(cmd)


lr = 0.0003
nb_samples_labeled = [100, 300, 1000, 3000, 10000, 30000]
walltime = '6:00:00'
folder = 'CIFAR10_test_nb_labels'

if os.path.exists(folder):
    raise RuntimeError(f'Folder {folder} already exists!')

if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(os.path.join(folder, 'scripts_logs')):
    os.makedirs(os.path.join(folder, 'scripts_logs'))

for nb_samples in nb_samples_labeled:
    make_script(folder, str(nb_samples_labeled), walltime, nb_samples_labeled, lr)

for script in os.listdir(folder):
    os.system(f'chmod +x {os.path.join(folder, script)}')
    os.system(f'oarsub -S ./{os.path.join(folder, script)}')
