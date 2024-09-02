# MoleScan

This repository includes object detection, segmentation and classification models designed to track and analyze skin moles. The models for detection and segmentation are based on the YOLOv8n architecture, which has been pretrained on the COCO dataset and subsequently fine-tuned with a custom dataset of labeled mole images. The model for classification is based on the InceptionV3 architecture, initialized with ImageNet weights and fine-tuned to address the specific challenges of the Skin Cancer: HAM10000 dataset

## Models Included

1. **Mole counting model**: This model can detect and count the number of moles present in a given image.
2. **Mole size estimation model**: This model estimates the size of a mole in an image by using a reference circle.
3. **Dermoscopy image classification model**: This model classifies dermoscopic images between 7 classes of pigmented lesions.
   
## Streamlit Application

In addition to the models, this repository includes a Streamlit app that integrates the three models, providing a user-friendly interface for mole detection, size estimation, and dermoscopic image classification.

## Repository Structure

    MoleScan/
    ├── app/                    # Includes the Streamlit application code, which brings together the functionality of all three models in a single user interface.
    ├── data/                   # Contains the dataset files for the three models. Images have not been included in this repository, only configuration files.
    ├── notebooks/              # Jupyter notebooks used for data preprocessing and model training. 
    ├── runs/                   # Results of the experiments performed on each model.
    ├── requirements.txt
    ├── README.md
    └── LICENSE
