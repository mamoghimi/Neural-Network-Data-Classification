# Neural Network Visualization for Spiral and Circular Data Classification

This project demonstrates how to use a neural network to classify data points in two distinct patterns: spiral-shaped and concentric circular data. The goal is to visualize how a neural network can learn to separate classes with complex decision boundaries.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)

## Overview
This project illustrates how a simple neural network can be used to classify data with non-linear boundaries. The neural network is trained to classify two different datasets:
1. **Spiral Data:** Two classes of data points are arranged in a spiral pattern.
2. **Circular Data:** Two classes of data points are arranged in concentric circles.

The project leverages `PyTorch` for implementing the neural network, while `matplotlib` is used to visualize the decision boundaries during training. The interactive plots show the network's learning progress and provide a clear understanding of how it adapts to complex classification tasks.

## Features
- **Two different data patterns:** The project supports both spiral and circular datasets.
- **Interactive visualization:** The decision boundary is updated during training, providing real-time feedback on the model's performance.
- **Configurable neural network:** The architecture includes three hidden layers with ReLU activation, dropout regularization, and a fully connected output layer.
- **Training enhancements:** Uses an Adam optimizer with a learning rate scheduler and optional gradient clipping for stable training.

## Installation
To run this project, you need the following dependencies:
- `Python 3.x`
- `PyTorch`
- `matplotlib`
- `numpy`
- `scikit-learn`

You can install the required libraries using:
```bash
pip install torch matplotlib numpy scikit-learn
```
# Usage
To run the project, follow these steps:
- Clone the repository:
```bash
git clone https://github.com/your-username/neural-network-visualization.git
cd neural-network-visualization
```
- Run the script for spiral data classification:
```bash
python spiral_classification.py
```
- Run the script for circular data classification:
```bash
python circular_classification.py
```

# Examples
- Spiral Data Classification without Noise
https://github.com/user-attachments/assets/fc7dd3e4-51a0-4c76-b126-1d6b57230a48









