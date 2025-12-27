# My PyTorch for Deep Learning Journey

This repository contains my personal work and code as I follow along with the excellent ["PyTorch for Deep Learning"](https://www.learnpytorch.io/) course by Daniel Bourke. My goal is to replicate the course materials from scratch to build a strong foundation in PyTorch and fundamental deep learning concepts.

This project is for my personal learning and to showcase my progress in mastering PyTorch.

## Acknowledgements

A huge thank you to Daniel Bourke for creating such a comprehensive and accessible resource for the deep learning community.

*   **Original Course Website:** [learnpytorch.io](https://www.learnpytorch.io/)
*   **Original GitHub Repository:** [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)

## Course Progress

Here is a list of the course modules. I will be checking them off as I complete the notebooks and exercises for each section.

- [X] **00. PyTorch Fundamentals**
- [X] **01. PyTorch Workflow**
- [X] **02. PyTorch Neural Network Classification**
- [X] **03. PyTorch Computer Vision**
- [X] **04. PyTorch Custom Datasets**
- [X] **05. PyTorch Going Modular**
- [X] **06. PyTorch Transfer Learning**
- [X] **07. PyTorch Experiment Tracking**
- [X] **08. PyTorch Paper Replicating**
- [ ] **09. PyTorch Model Deployment**

## Key Concepts & Learnings

Throughout this course, I am focusing on understanding and implementing the following core concepts:

*   **Tensors:** The fundamental data structure in PyTorch.
*   **Computational Graphs:** How PyTorch builds and executes models.
*   **`torch.nn`:** Building neural network layers and models.
*   **`torch.utils.data.Dataset` & `DataLoader`:** Creating custom data pipelines for training.
*   **Loss Functions & Optimizers:** The mechanics of model training.
*   **Model Evaluation:** Assessing model performance with various metrics.
*   **GPU Acceleration:** Leveraging CUDA for faster training.
*   **Transfer Learning:** Using pre-trained models to solve new problems.
*   **Model Deployment:** Making a trained model available for inference.

## Structure & Usage

### Directory Structure

*   **`going_modular/`**: This directory contains modularized Python scripts for model building and training. This structure allows for cleaner notebooks and reusable code.
    *   `vit.py`: Contains the full implementation of the Vision Transformer (ViT) architecture, including `PatchEmbedding` and the Transformer Encoder.
    *   `engine.py`: The "engine" of the training process. It contains:
        *   `train_step`: Handles the training loop for a single epoch (forward pass, loss calculation, backpropagation, optimizer step).
        *   `test_step`: Handles the evaluation loop for a single epoch (forward pass, loss calculation, metric calculation).
        *   `train`: Combines `train_step` and `test_step` to train the model for multiple epochs and track results.
    *   `train.py`: An executable script to train the model from the command line. It orchestrates the entire process:
        *   Sets up command-line arguments (hyperparameters like epochs, batch size, learning rate).
        *   Prepares `DataLoaders` for training and testing.
        *   Initializes the model (e.g., ViT), loss function, and optimizer.
        *   Calls `engine.train()` to start training.
        *   Saves the trained model to a file.
    *   `helper_functions.py`: Provides utility functions, such as plotting loss curves to visualize model performance over epochs.

### Installation

To run the notebooks or scripts in this repository, ensure you have PyTorch and other dependencies installed:

```bash
# It is recommended to create a virtual environment first
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn
```