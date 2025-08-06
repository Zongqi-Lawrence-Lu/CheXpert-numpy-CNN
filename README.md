# CheXpert X-Ray Labeling Project

This project implements a Convolutional Neural Network (CNN) from scratch in NumPy to classify chest X-ray images from the CheXpert dataset (https://stanfordmlgroup.github.io/competitions/chexpert/) by the Stanford ML Group.

It is a self-contained prototype to study model construction and optimization without relying on libraries like PyTorch or TensorFlow.


# Usage

1. Preprocess the data:
   - Use "process_data.py" to convert downloaded CheXpert images and labels into NumPy arrays.
   - To manage memory, limit the dataset (~10,000) using a stopping condition in the loop.
   - (Optional) Replace the hardcoded conditions "male" and "frontal" with other filters (Here filters are used to simply training).

2. Train the CNN:
   - Run "network.py": to train the model.
   - If training is too long, use save/read functions to restart.


# Development Progress

Version | Features 
  v1.0  | Built the overall CNN architecture in NumPy
  v1.1  | Added momentum and batch processing
  v1.2  | Parallelized computation across channels in hidden layers


# Current Status

- This is a very early-stage prototype.
- Training is limited due to performance constraints (especially memory), but we observe decreasing cost and increasing success in the first few epoches.
- See the sample output file for training log (uses 30,000 images subset of the full data, test set is 108 fully labeled images).


# Limitations & Future Work

1. Convolutional layers currently lack customized dialation and padding options. 

2. Performance bottlenecks:
   - Training on 10,000 images takes ~1 hour, partly due to the inefficient nested for loops in convolutional layers.
   - Plans to improve:
     - Further parallelize convolution operations, or alternatively,
     - Implement convolutional layers with vectorized operations.
     - Switch from list-based channel representation to a full 4D tensor format: (channel_size, batch_size, height, width)
     - Possibly port to GPU with CuPy for speedups.

3. Architecture Improvements:
   - Add residual layers once performance bottlenecks are resolved, though deeper networks are not yet feasible due to runtime limitations.
   - Consider how to use multiple-view input (frontal/lateral) and tabular input processing (age/sex) to increase performance.


# Motivation

This project was built when reading Chapter 10 of Understanding Deep Learning by Simon Prince.
