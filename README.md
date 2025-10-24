Self-Supervised Learning for Medical Image Analysis (MIDL 2025)
This is the official repository for the MIDL 2025 paper: **Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales**.

This work introduces an adjusted augmentation strategy for contrastive learning in retinal imaging, improving downstream classification performance.

### 🔥 News
Our paper：Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales, has been accepted to the Medical Imaging with Deep Learning (MIDL) 2025!

Read the paper here: [https://openreview.net/forum?id=63igmyYaDc]

### 🛠️ Installation
Follow these steps to set up the required Conda environment:

1. Create the environment

conda create --name dino python=3.10

2. Activate the environment

conda activate dino

3. Install dependencies

pip install -r requirements.txt

### 💾 Pre-trained Checkpoints

The following pre-trained checkpoints are provided, each corresponding to a different data augmentation strategy used during the initial self-supervised training phase:

| Augmentation Strategy | Checkpoint File | Download Link |
| :--- | :--- | :--- |
| **Strong Augmentation** (ckp-300) | `ckp-300.pth` | <https://drive.google.com/file/d/1IFGy2Gh0bu-0trECtgq1RE0lAfv9pNQW/view?usp=sharing> |
| **Weak Augmentation** (ckp-325) | `ckp-325.pth` | <https://drive.google.com/file/d/1diaDwSeJuoFZU4PhB17YOrIuBYInFjvh/view?usp=sharing> |
| **Weak + Medium Augmentation** (ckp-350) | `ckp-350.pth` | <https://drive.google.com/file/d/1XNNsKi3C0iKRjyWlJ6G83Lw2Sfv-YNAo/view?usp=sharing> |

### ▶️ Usage: Fine-tuning & Evaluation

1. Dataset Preparation

This work uses the APTOS2019 Blindness Detection dataset.

Download the dataset from the official Kaggle competition page:

https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Unzip and organize the dataset into a directory.

You will need to provide the path to this directory in the --data_path argument.

2. Run Fine-tuning

The example below demonstrates a 50-epoch fine-tuning run on the APTOS2019 dataset using the strong augmentation checkpoint (ckp-300.pth).

torchrun --nproc_per_node=1 --master_port=48793 eval_finetune.py \
--data_path /path/to/your/APTOS2019 \
--pretrained_weights ./ckp-300.pth \
--task dino_finetune_APTOS2019/ \
--num_labels 5 \
--arch vit_small \
--batch_size_per_gpu 16 \
--epochs 50

⚠️ Important: You must modify the --data_path argument to point to the root directory of the APTOS2019 dataset on your system.

### 📚 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{Cheng2025-eo,
  title     = "Enhancing Contrastive Learning for Retinal Imaging via Adjusted Augmentation Scales",
  author    = "Cheng, Zijie and Li, Boxuan and Altmann, Andre and Keane, Pearse and Zhou, Yukun",
  booktitle = "Medical Imaging with Deep Learning",
  year      = 2025
}
