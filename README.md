# Cats vs. Dogs Image Classification

This project implements an image classification model to distinguish between cats and dogs using transfer learning with a pre-trained deep learning model. The goal is to leverage pre-trained weights to optimize performance and reduce training time for this binary classification task.

---

## Features
- Uses **TensorFlow** and **Keras** for model implementation.
- Employs **MobileNetV2** (a pre-trained CNN model) for transfer learning.
- Fine-tunes the model to classify images into "cat" or "dog."
- Includes visualization of training accuracy and loss.
- Allows real-time image prediction with confidence scores.

---

## Requirements
Ensure you have the following installed:

- Python 3.7+
- TensorFlow 2.0+
- Google Colab (optional, for cloud-based execution)

Install the required Python libraries:

```bash
pip install tensorflow matplotlib
```

---

## Dataset
The dataset used is the **Dogs vs. Cats Dataset** from Kaggle.

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
2. Extract the dataset into a folder named `dataset` with the following structure:

```
dataset/
    train/
        cat.0.jpg
        cat.1.jpg
        dog.0.jpg
        dog.1.jpg
    validation/
        cat.1000.jpg
        cat.1001.jpg
        dog.1000.jpg
        dog.1001.jpg
```

---

## Code Overview

### 1. Data Preparation
- Loads the dataset using TensorFlow's `image_dataset_from_directory`.
- Preprocesses images to resize them and normalize pixel values to the range [0, 1].

### 2. Model Implementation
- Utilizes **MobileNetV2** with ImageNet weights for transfer learning.
- Adds custom classification layers on top of the pre-trained base.
- Fine-tunes the model to learn features specific to the cats-vs-dogs task.

### 3. Model Training
- Compiles the model with the Adam optimizer, binary cross-entropy loss, and accuracy metrics.
- Trains the model with early stopping and model checkpointing.
- Plots training and validation accuracy/loss curves.

### 4. Image Prediction
- Allows prediction of new images with real-time inference.
- Displays the image along with the classification result and confidence score.

---
