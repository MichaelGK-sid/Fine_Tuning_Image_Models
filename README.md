# Fine-Tuning Vision Models with PyTorch  

This repository provides tools for fine-tuning popular computer vision models (such as **ViT** and **VGG**) from `torchvision.models` on any image dataset.  

## Contents  

### 1. Jupyter Notebook (`.ipynb`)  
The notebook contains all the PyTorch code needed to:  
- Download pretrained models (ViT, VGG, etc.)  
- Modify classification heads for custom tasks  
- Unzip datasets (if provided as a `.zip` file)  
- Create datasets and dataloaders for training and testing  
- Train models on custom image datasets  
- Evaluate models and calculate test accuracy  
- Save trained model weights in a `.pth` file  
- Load pretrained model weights from a `.pth` file  
- Generate model architecture summaries using `torchinfo`  

### 2. Data Folder  
The `data/` folder contains a sample dataset of images (already split into training and testing sets).  
- This dataset can be used to quickly try out model training.  
- Since the images are organized in simple folder structures, the unzipping step is **not required**.  
