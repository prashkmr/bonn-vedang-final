#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8      # workers for gpu   #no of workers+1
#SBATCH --ntasks-per-node=1    # numer of total tasks equal no of gpus
#SBATCH --mem=48G              # memory per node
#SBATCH --output=test.txt
#SBATCH --error=test-err.txt
#SBATCH --time=00:30:00

cd
source dslr/bin/activate
module load StdEnv/2020
module load gcc/8.4.0
module load cuda/10.2

module load yaml/2.3.6
module load Cython/0.29.27
module load numpy/1.22.2
module load nixpkgs/16.09
module load ninja/1.9.0
cd /home/eqdong/scratch/diff-lidar-slam-dslr*/seg/bonnetal_vedang/train/tasks/semantic/
# sh gndnet.sh

python train.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/ -ac config/arch/darknet53.yaml -dc config/labels/semantic-kitti.yaml -l log/test  -p pre_trained_models/darknet53





# ./infer.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/  -m log/2/  -l  log/infer --dc config/labels/semantic-kitti.yaml
# ./evaluate_iou.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/  -p log/dummy --split valid  --dc config/labels/semantic-kitti.yaml

#for infer of part4 on the test set
# ./infer.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/  -m   -l  log/test-set-pretrained-11-21 --dc config/labels/semantic-kitti.yaml

