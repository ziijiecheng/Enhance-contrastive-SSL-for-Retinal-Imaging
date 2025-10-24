# Self-Supervised Learning for Medical Image Analysis (MIDL 2025)

## 🔥 News
Our paper, **[Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales]**, has been accepted to the ** Medical Imaging with Deep Learning (MIDL) 2025**! 
Read the paper here: [[https://openreview.net/forum?id=63igmyYaDc&referrer=%5Bthe%20profile%20of%20zijie%20cheng%5D(%2Fprofile%3Fid%3D~zijie_cheng2)]]

---

## 💾 Pre-trained Checkpoints

We provide pre-trained model weights under three different augmentation strategies. All checkpoints are stored as Google Drive links below:

| Checkpoint File | Pre-training Augmentation Strategy | Download Link |
| :--- | :--- | :--- |
| `ckp-300.pth` | **Strong** Augmentation | [Download Strong Aug. Checkpoint](https://drive.google.com/file/d/1IFGy2Gh0bu-0trECtgq1RE0lAfv9pNQW/view?usp=sharing) |
| `ckp-325.pth` | **Weak** Augmentation | [Download Weak Aug. Checkpoint](https://drive.google.com/file/d/1diaDwSeJuoFZU4PhB17YOrIuBYInFjvh/view?usp=sharing) |
| `ckp-350.pth` | **Weak + Medium** Augmentation | [Download Weak+Med Aug. Checkpoint](https://drive.google.com/file/d/1XNNsKi3C0iKRjyWlJ6G83Lw2Sfv-YNAo/view?usp=sharing) |

---

## 🛠️ Environment Setup

Follow these steps to set up the required Conda environment:

```bash
# 1. Create the environment (Name: dino, Python 3.10)
conda create --name dino python=3.10 

# 2. Activate the environment
conda activate dino 

# 3. Install dependencies from requirements.txt
pip install -r requirements.txt

▶️ Evaluation and Fine-tuning
You can run the evaluation using the provided eval.sh script or by directly executing the torchrun command.

The example below demonstrates a 50-epoch fine-tuning run using the strong augmentation checkpoint (ckp-300.pth):

torchrun --nproc_per_node=1 --master_port=48793 eval_finetune.py \
--data_path /root/autodl-tmp/DINO/APTOS2019 \
--pretrained_weights ./ckp-300.pth \
--task dino_finetune_APTOS2019/ \
--num_labels 5 \
--arch vit_small \
--batch_size_per_gpu 16 \
--epochs 50

⚠️ Important: You must modify the --data_path argument to correctly point to the root directory of the APTOS2019 dataset on your system.

📚 Dataset Reference
The APTOS2019 Blindness Detection dataset used for this work is publicly available on Kaggle:

Link: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
