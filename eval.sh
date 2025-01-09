#This is for eval DINO with the SH file

#$ -l tmem=12G
#$ -pe gpu 1
#$ -l gpu=true
#$ -l h_rt=24:00:00 
#$ -j y
#$ -N finetune_dino_APTOS2019
#$ -S /bin/bash
#$ -R y

# activate the virtual env

source activate dino
# Source file for CUDA11.0
# 24/02/23

source /share/apps/source_files/cuda/cuda-11.0.source

nvidia-smi

hostname
date

# enter the project path
cd /SAN/ioo/AlzeyeTempProjects/zijcheng/dino_cfp_transfer #the path should be modified


# running command
#################################################
#The data_path should be modified
for n_round in 0 1 2 3 4
do
torchrun --nproc_per_node=1 --master_port=48793 eval_finetune.py --data_path /SAN/ioo/AlzeyeTempProjects/zijcheng/dino_cfp_transfer/APTOS2019 \
--pretrained_weights ./ckp-300.pth \
--task dino_finetune_APTOS2019_${n_round}/ \
--num_labels 5 \
--arch vit_small \
--batch_size_per_gpu 16 \
--epochs 50 \
--seed ${n_round}
done

date