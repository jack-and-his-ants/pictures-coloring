# pictures-coloring
[![Python](https://shields.io)](https://python.org)
[![PyTorch](https://shields.io)](https://pytorch.org)
[![scikit-learn](https://shields.io)](https://scikit-learn.org)

## Overview

This project implements an automatic image colorization system using deep learning. The goal is to reconstruct color information from grayscale images by predicting chrominance values in the CIE LAB color space.

Instead of directly predicting continuous color channels, the problem is formulated as a pixel-wise classification task. Color values are quantized using K-Means clustering, allowing the neural network to predict one of a finite number of color classes for each pixel.

The project was developed in Python using PyTorch and trained on a dataset of natural color flowers images.

## Project Motivation

Automatic image colorization is a challenging computer vision problem because multiple plausible colors may correspond to the same grayscale intensity.

For example:

* Grass is usually green but may appear yellow or brown.
* Cars can have many different colors.
* The sky may be blue, orange, gray, or black depending on conditions.

As a result, image colorization is an inherently ambiguous problem that requires both local texture understanding and high-level semantic reasoning.

## Dataset Preparation
### Image Preprocessing

All images are resized to a fixed resolution:

* 128 × 128 pixels

Each RGB image is converted to the CIE LAB color space.

The LAB color space separates:

- L – luminance (brightness)
- a – green-red channel
- b – blue-yellow channel

Only the luminance channel is provided to the neural network during training and then inference.

### Color Quantization

Predicting continuous a and b values directly is difficult and often produces washed-out colors.

To address this problem, K-Means clustering is applied to the chrominance channels:

(a, b) → K color classes

The clustering algorithm is trained on a subset of pixels from the training dataset.

For this project:

Number of clusters: 32

Each pixel is assigned to the nearest cluster center.

The neural network therefore predicts a class label rather than raw color values.

## Neural Network Architecture
### Encoder

The model uses a pretrained ResNet18 backbone as a feature extractor.

Advantages:

- Transfer learning from ImageNet
- Strong semantic feature extraction
- Faster convergence
- Better generalization

The input grayscale image is replicated across three channels before being passed through the network.

The encoder consists of:

- Conv1
- Layer1
- Layer2
- Layer3
- Layer4

Only the deeper layers are fine-tuned during training while the earlier layers remain frozen.

### Decoder

The decoder follows a U-Net-inspired architecture.
Upsampling is performed using transposed convolutions:

- 4×4  → 8×8
- 8×8  → 16×16
- 16×16 → 32×32
- 32×32 → 64×64
- 64×64 → 128×128

Each decoder block contains:

- Convolution
- Batch Normalization
- ReLU activation
- Dropout
- Skip Connections

To preserve spatial details, skip connections are used between encoder and decoder stages.

These connections help reconstruct:

- object boundaries
- textures
- fine image structures

without losing information during downsampling.

### Loss Function

The network is trained using weighted cross-entropy loss.

Why weighted loss?

Some colors occur much more frequently than others.

For example gray, brown or dark green appear significantly more often than rare colors.

Without class weighting, the model tends to predict only the most common colors.

Class weights are computed from training set statistics and applied during optimization.

### Training
- Optimizer: AdamW
- Learning Rate: 1e-3
- Batch Size: 8
- Scheduler: ReduceLROnPlateau

The learning rate is automatically reduced when validation loss stops improving.

### Inference Pipeline

During inference:

- Load RGB image
- Resize to 128×128
- Convert RGB → LAB
- Extract L channel
- Predict color class for each pixel
- Replace class labels with K-Means centroids
- Reconstruct LAB image
- Convert LAB → RGB

The output is a fully colorized image.

## Results

The model is capable of generating colorizations for natural scenes.

Strengths:

- Produces realistic vegetation colors
- Preserves image structure
- Generates visually coherent outputs

Limitations:

- Color ambiguity may lead to unrealistic color choices
- The model is trained on flowers pictures

Examples include:

- interpreting trees as flowers
- changing colors of some flowers

These issues are common in image colorization systems due to the inherently multimodal nature of the problem.

## Future Improvements

Potential improvements include:

### Larger Color Vocabulary

Increase the number of K-Means clusters:
32 → 64 → 100

### Deeper Backbone

Replace ResNet18 with:

- ResNet34
- ResNet50
- EfficientNet
### Perceptual Loss

Combine cross-entropy with perceptual feature losses.

### Attention Mechanisms

Introduce self-attention or transformer-based blocks to improve global context understanding.

### Larger Training Dataset

More diverse training images would likely improve generalization and color realism.

## Technologies Used
- Python
- PyTorch
- NumPy
- scikit-image
- scikit-learn
- torchvision
- matplotlib
- joblib

## Bibliography
### Scientific Paper
1. Zhang, R., Isola, P., & Efros, A. A. (2016).
   *Colorful Image Colorization*.
   European Conference on Computer Vision (ECCV).
   https://doi.org/10.48550/arXiv.1603.08511

### Dataset

2. Flowers Dataset.
   Kaggle.
   https://www.kaggle.com/datasets/imsparsh/flowers-dataset