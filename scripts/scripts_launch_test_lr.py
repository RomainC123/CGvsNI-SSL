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
python ./main.py --train-test --folder {folder} --name {name} --data CIFAR10 --nb_samples_test 10000 --nb_samples_labeled {nb_samples_labeled} --img_mode RGB --model SimpleNet --max_lr {max_lr} --method OnlySup --epochs 300 --no-verbose"

    with open(os.path.join(folder, f'{name}.sh'), 'w+') as f:
        f.write(cmd)


lr_to_test = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
nb_samples_labeled = 1000
walltime = '18:00:00'
data = 'CIFAR10'
folder = 'test_lr'

path = os.path.join(data, folder)

if os.path.exists(path):
    raise RuntimeError(f'Folder {folder} in {data} already exists!')

if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(os.path.join(path, 'scripts_logs')):
    os.makedirs(os.path.join(path, 'scripts_logs'))

for lr in lr_to_test:
    make_script(path, str(lr), walltime, nb_samples_labeled, lr)

for script in os.listdir(path):
    os.system(f'chmod +x {os.path.join(path, script)}')
    os.system(f'oarsub -S ./{os.path.join(path, script)}')
