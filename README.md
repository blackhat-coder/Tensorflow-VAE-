# Tensorflow VAE (Variational Autoencoder) ✌

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains an experimental implementation of a Variational Autoencoder (VAE) using TensorFlow. The project was developed for learning purposes, with a focus on exploring generative models through unsupervised learning.

## Overview 🗒️

A Variational Autoencoder (VAE) is a type of generative model that learns a probabilistic mapping from input data to a latent space, allowing the generation of new data samples similar to the training distribution. VAEs are commonly used for tasks such as image generation and dimensionality reduction.

This implementation provides a customizable framework for building and training a VAE model on various datasets using TensorFlow.

## Features 

- **Encoder-Decoder Architecture**: Implements the core VAE structure with an encoder to map inputs to latent variables and a decoder for reconstruction.
- **Latent Space Regularization**: Ensures the latent space follows a Gaussian distribution via the KL divergence loss component.
- **Customizable**: Modify the network architecture, including layers, units, and activation functions.
- **Training and Evaluation**: Scripts for training the model and visualizing results.

## Getting Started

### Prerequisites 

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualizations)

### Installation 💻

1. Clone this repository:
   ```bash
   git clone https://github.com/blackhat-coder/Tensorflow-VAE.git
   cd Tensorflow-VAE
   
2. ```bash
   pip install -r requirements.txt

### Usage 🔧

1. **Prepare your dataset** in a format compatible with the model (e.g., images in `.jpg` or `.png` format).

2. **Train the model** using the `train_vae.py` script:

   ```bash
   python train_vae.py --dataset <path_to_dataset> --epochs 100

Visualize Results using the visualize.py script:

   ```bash
   python visualize.py --model_path <path_to_saved_model> --latent_dim 2
   ```

## Contributing

We welcome contributions to improve this playground 😉!

## License

This project is licensed under the MIT License.

---

**MIT License**

This project is based on the concepts presented in the paper:

Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." arXiv preprint arXiv:1312.6114 (2013).

