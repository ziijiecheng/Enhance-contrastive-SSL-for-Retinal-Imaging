1. Build Environment
conda env create -f dino.yml
2. evaluation
ckp-300 is pre-trained with strong augmentation; 
ckp-325 is pre-trained with weak augmentation; 
ckp-350 is pre-trained with weak+med augmentation

You could run the code through file eval.sh Or

torchrun --nproc_per_node=1 --master_port=48793 eval_finetune.py --data_path /root/autodl-tmp/DINO/APTOS2019 \
--pretrained_weights ./ckp-300.pth \
--task dino_finetune_APTOS2019/ \
--num_labels 5 \
--arch vit_small \
--batch_size_per_gpu 16 \
--epochs 50

Remember to modify the data path.
