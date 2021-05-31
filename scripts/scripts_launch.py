import os

def make_script(folder, name, walltime, nb_samples_labeled, max_lr):

    cmd = f"#!/bin/bash\n\
    \n\
    #OAR -n {name}\n\
    #OAR -t gpu\n\
    #OAR -l /nodes=1/gpudevice=1,walltime={walltime}\n\
    #OAR --stdout scripts_logs/{name}.out\n\
    #OAR --stderr scripts_logs/{name}.err\n\
    #OAR --project cg4n6\n\
    \n\
    source /applis/environments/conda.sh\n\
    conda activate CGDetection\n\
    \n\
    cd ~/code/CGvsNI-SSL/src\n\
    python ./main.py --train-test --name {name} --data CIFAR10 --nb_samples_test 10000 --nb_samples_labeled {nb_samples_labeled} --img_mode RGB --model SimpleNet --max_lr {max_lr} --method TemporalEnsemblingNewLoss --epochs 300 --no-verbose"

    with open(os.path.join(folder, f'{name}.sh'), 'w+') as f:
        f.write(cmd)


lr_to_test = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
nb_samples_labeled = 1000
walltime = '6:00:00'
folder = 'CIFAR10_test_lr'

if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists(os.path.join(folder, 'scripts_logs')):
    os.makedirs(os.path.join(folder, 'scripts_logs'))

for lr in lr_to_test:
    make_script(folder, folder + '_' + str(lr), walltime, nb_samples_labeled, lr)

for script in os.listdir(folder):
    os.system(f'chmod +x {os.path.join(folder, script)}')
    os.system(f'oarsub -S ./{os.path.join(folder, script)}')
