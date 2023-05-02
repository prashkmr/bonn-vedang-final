
#!/bin/bash
#SBATCH --job-name=para-gndv2-1-loss
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --cpus-per-task=8           ## workers for gpu   #no of workers+1
#SBATCH --ntasks-per-node=4         ## numer of total tasks equal no of gpus
#SBATCH --mem=96G                   ## memory per node
#SBATCH --output=parallel-gndv2-1-loss.txt
#SBATCH --error=parallel-gndv2-1-loss-err.txt
#SBATCH --time=03:00:00

cd
source dslr/bin/activate
cd /home/eqdong/scratch/diff-lidar-slam-dslr*/seg/bonnetal_vedang/train/tasks/semantic/
# sh gndnet.sh
module load nccl
export NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"
echo "r$SLURM_NTASKS 
echo "r$MASTER_ADDR












## first one latest
# srun python train.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/ -ac config/arch/darknet53.yaml -dc config/labels/semantic-kitti.yaml -l log/above-gnd-v2-both-loss/both_loss/1  -p pre_trained_models/darknet53 --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS


##second one
# srun python train.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/ -ac config/arch/darknet53.yaml -dc config/labels/semantic-kitti.yaml -l log/4  -p log/3 --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS

# echo "Now runnign with  commands as  python -m torch.distributed.launch train.py   "
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch train.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/ -ac config/arch/darknet53.yaml -dc config/labels/semantic-kitti.yaml -l log/para-scr-1  -p pre_trained_models/darknet53



# ./infer.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/  -m log/bs8-pretrained-loaded-fintune-with-slam-part2/  -l  log/predict-bs8-pretrained-loaded-fintune-with-slam-part3 --dc config/labels/semantic-kitti.yaml
# ./evaluate_iou.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/  -p log/dummy --split valid  --dc config/labels/semantic-kitti.yaml

#for infer of part4 on the test set
# ./infer.py -d /home/eqdong/scratch/diff-lidar-slam-dslr\*/seg/kitti-data/dataset/  -m   -l  log/test-set-pretrained-11-21 --dc config/labels/semantic-kitti.yaml

