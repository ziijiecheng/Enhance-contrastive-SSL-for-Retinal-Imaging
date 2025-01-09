1. Build Environment
conda create --name dino --file dino_requirements.txt
2. evaluation
ckp-300 is pre-trained with strong augmentation; 
ckp-325 is pre-trained with weak augmentation; 
ckp-350 is pre-trained with weak+med augmentation

You could run the code through file eval.sh 
Remember to modify the data path.