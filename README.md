# CIFAR-100 Classification with VGG-16 and GANs

This repository contains the implementation of a project aimed at enhancing image classification on the CIFAR-100 dataset using the VGG-16 convolutional neural network architecture. Synthetic images were generated and integrated into the dataset using Generative Adversarial Networks (GANs) to improve model performance.

## **Overview**
This project leverages GANs for synthetic data generation and combines it with deep learning models to improve classification performance on the CIFAR-100 dataset. The model achieved improved accuracy compared to baseline methods.

## **Dataset**
The CIFAR-100 dataset contains:
- **60,000 images** of size **32x32 pixels** in 100 classes.
- **50,000 training images** and **10,000 test images**.

## **Synthetic Data Generation**
GANs were used to generate synthetic images:
- Synthetic data mimics real CIFAR-100 images.
- These images were resized and normalized to match CIFAR-100 dimensions.
- Augmenting the dataset with synthetic images improved generalization and reduced overfitting.

## **Model Architecture**
### **VGG-16**
- A 16-layer convolutional neural network known for feature extraction and image classification.
- Used **Adam optimizer** and **categorical cross-entropy loss** for training.

### **GANs**
- **Generator**: Creates synthetic images based on the CIFAR-100 image distribution.
- **Discriminator**: Classifies images as real or generated, improving the generator's output through adversarial training.

## **Training**
- Batch size: **64**
- Early stopping was applied to monitor validation loss.
- Training and validation curves were plotted to analyze model performance.

## **Results**
- **Training Accuracy**: ~71.10%
- **Test Accuracy**: ~44.83%
- Loss curves and accuracy graphs highlight the performance trends.
- VGG-16 with GANs outperformed ResNet and ImageNet models for this task.

## **Implementation**
### Prerequisites
Install the following libraries:
- `tensorflow`
- `numpy`
- `matplotlib`
- `PIL`
- `seaborn`

### Steps
1. Preprocess the CIFAR-100 dataset by normalizing image pixel values.
2. Generate synthetic data using the GAN architecture.
3. Merge real and synthetic data for training.
4. Train the VGG-16 model and evaluate its performance.

### Execution
- The code is implemented in Python using Jupyter Notebook.
- Run the `VGG-16 with GAN` notebook for training and evaluation.

## **Repository Structure**
- `README.md`: This file, providing an overview of the project.
- `VGG-16 with GAN`: Jupyter Notebook containing the training and evaluation code.
- `synthetic_images/`: Folder with synthetic images generated using GANs.
- `RESNET`: File containing the code of the resnet architecture.
- `GAN to Generate CIFAR100 synthetic images`: The file contains the code which is used to generate synthetic images.

## **References**
For additional details on GANs and VGG-16, refer to:
- Ian Goodfellow et al., "Generative Adversarial Nets."
- Karen Simonyan and Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition."

---
