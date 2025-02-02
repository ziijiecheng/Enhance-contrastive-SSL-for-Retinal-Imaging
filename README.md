Check point under strong augmentation: https://drive.google.com/file/d/1IFGy2Gh0bu-0trECtgq1RE0lAfv9pNQW/view?usp=sharing

Check point under weak augmentation: https://drive.google.com/file/d/1diaDwSeJuoFZU4PhB17YOrIuBYInFjvh/view?usp=sharing

Check point under weak+med augmenttaion: https://drive.google.com/file/d/1XNNsKi3C0iKRjyWlJ6G83Lw2Sfv-YNAo/view?usp=sharing


1. Build Environment:
conda create --name dino python=3.10
conda activate dino
pip install -r requirements.txt
3. evaluation
ckp-300 is pre-trained with strong augmentation; 
ckp-325 is pre-trained with weak augmentation; 
ckp-350 is pre-trained with weak+med augmentation

You could run the code through file eval.sh Or
torchrun --nproc_per_node=1 --master_port=48793 eval_finetune.py --data_path /root/autodl-tmp/DINO/APTOS2019 --pretrained_weights ./ckp-300.pth --task dino_finetune_APTOS2019/ --num_labels 5 --arch vit_small --batch_size_per_gpu 16 --epochs 50

Remember to modify the data path.

The APTOS2019 is a public dataset:
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
