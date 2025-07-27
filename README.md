# Diffusion Paths: 2D GMM Diffusion Model

A PyTorch implementation of a 2D Gaussian Mixture Model (GMM) diffusion model with support for multiple noise schedules.

## Overview

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for generating 2D data from a Gaussian Mixture Model. The model supports multiple noise schedules (linear, cosine, quadratic, exponential) and includes comprehensive evaluation metrics.

## Features

- **Multiple Noise Schedules**: Linear, Cosine, Quadratic, and Exponential beta schedules
- **Large Batch Training**: Optimized for large batch sizes (1280) with scaled learning rates
- **Comprehensive Evaluation**: MMD, Wasserstein distance, and test loss metrics
- **Checkpoint Management**: Automatic checkpoint saving and loading
- **Visualization**: Sample generation and comparison plots

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusion-paths
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model with a specific noise schedule:

```bash
# Linear schedule (default)
python train.py --run_name "linear_schedule" --schedule "linear" --epochs 1000 --save_every 10

# Cosine schedule
python train.py --run_name "cosine_schedule" --schedule "cosine" --epochs 1000 --save_every 10

# Quadratic schedule
python train.py --run_name "quadratic_schedule" --schedule "quadratic" --epochs 1000 --save_every 10

# Exponential schedule
python train.py --run_name "exponential_schedule" --schedule "exponential" --epochs 1000 --save_every 10
```

### Evaluation

Evaluate trained models:

```bash
# Evaluate all checkpoints
python test_model.py

# The script will automatically:
# - Load all checkpoint files
# - Generate samples from each model
# - Compute MMD and Wasserstein distances
# - Create comparison visualizations
```

## Project Structure

```
diffusion-paths/
├── model.py                 # Neural network architecture (Denoiser)
├── train.py                 # Training script with noise schedule support
├── test_model.py            # Evaluation script
├── data/
│   └── gmm_dataset.py      # GMM data generation
├── checkpoints/             # Saved model checkpoints
│   ├── linear_schedule/
│   ├── cosine_schedule/
│   ├── quadratic_schedule/
│   └── exponential_schedule/
├── test_results/            # Evaluation results and plots
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Noise Schedules

The model supports four different noise schedules:

1. **Linear**: `β_t = β_start + (β_end - β_start) * t/T`
2. **Cosine**: `β_t = β_start + (β_end - β_start) * (1 - cos(πt/2T))`
3. **Quadratic**: `β_t = β_start + (β_end - β_start) * (t/T)²`
4. **Exponential**: `β_t = exp(log(β_start) + (log(β_end) - log(β_start)) * t/T)`

## Model Architecture

- **Denoiser**: MLP with timestep embedding
- **Input**: 2D data points + timestep
- **Output**: Predicted noise
- **Loss**: MSE between predicted and true noise

## Evaluation Metrics

- **Test Loss**: MSE loss on noise prediction
- **MMD (RBF)**: Maximum Mean Discrepancy with RBF kernel
- **Wasserstein Distance**: Earth Mover's Distance
- **Sample Quality**: Visual comparison with original GMM data

## Default Configuration

- **Batch Size**: 1280 (large batch optimized)
- **Learning Rate**: 3.16e-5 (scaled for large batch)
- **Epochs**: 1000
- **Save Every**: 10 epochs
- **Noise Steps**: 1000
- **Data**: 8-component GMM with radius 5.0, std 0.2

## Results

The evaluation provides comprehensive comparison of different noise schedules:

- **Test Loss**: Measures noise prediction accuracy
- **Sample Quality**: MMD and Wasserstein metrics for sample distribution
- **Training Trajectories**: Loss curves over training epochs
- **Visual Comparisons**: Generated samples vs original data

## Excluded from Git

The following files/directories are excluded from version control:
- `checkpoints/` - Model checkpoints
- `test_results/` - Evaluation results
- `venv/` - Virtual environment
- `*.pt`, `*.pth` - PyTorch model files
- `*.png`, `*.jpg` - Generated images
- `*.npy`, `*.csv`, `*.json` - Data files
- `__pycache__/` - Python cache

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- SciPy

## License

This project is for educational and research purposes. 