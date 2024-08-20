# Face_Gender_-_Ethncity_Recognizer_Model
# 

This project is a deep learning-based model designed to recognize age, gender, and ethnicity from facial images. It uses Convolutional Neural Networks (CNNs) along with hyperparameter tuning to achieve optimal performance.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)

## Introduction

This project builds a model that predicts the gender, age, and ethnicity of individuals based on their facial images. The model is trained on the [Age Gender Dataset](https://www.kaggle.com/jangedoo/utkface-new) and employs hyperparameter tuning to find the optimal configuration for the neural network.

## Dataset

The dataset used in this project is a CSV file containing pixel values of facial images, along with their corresponding age, gender, and ethnicity labels.

- **Dataset Details:**
  - Images are grayscale, 48x48 pixels.
  - Labels include:
    - Age (continuous)
    - Gender (binary: Male/Female)
    - Ethnicity (categorical: 5 classes)

## Model Architecture

The model is a Convolutional Neural Network (CNN) with multiple convolutional layers followed by fully connected layers. It uses the following layers:
- Convolutional Layers with Leaky ReLU activation
- MaxPooling or AvgPooling Layers
- Global Average Pooling
- Dense Layers with Dropout for regularization
- Output Layers:
  - Gender: Binary classification with sigmoid activation
  - Ethnicity: Categorical classification with softmax activation
  - Age: Continuous output with mean squared error loss
