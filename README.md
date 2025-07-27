# Diffusion Model Training and Evaluation

This repository contains a comprehensive implementation of diffusion model training and evaluation on 2D Gaussian Mixture Model (GMM) data, optimized for large batch training.

## Features

- **2D GMM Dataset**: Generate 2D data from a GMM where means are equally spaced on a circle
- **Large Batch Training**: Optimized for 1280 batch size with appropriate learning rate scaling
- **Complete DDPM Implementation**: Full diffusion probabilistic model with configurable parameters
- **Comprehensive Evaluation**: MMD (RBF/Linear), Wasserstein distance, test loss tracking
- **Visualization Suite**: Individual plots, comparison plots, learning curves
- **Model Analysis**: Metrics visualization and performance analysis tools

## Project Structure

```
.
├── model.py                          # Denoiser model implementation
├── train.py                          # Training script (large batch optimized)
├── test_model.py                     # Evaluation script
├── compare_models.py                 # Model comparison analysis
├── plot_metrics.py                   # Metrics visualization
├── data/
│   └── gmm_dataset.py               # GMM dataset implementation
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

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

The default configuration uses large batch training for optimal performance:

```bash
python train.py --epochs 1000 --batch_size 1280 --lr 3.16e-5 --save_every 10
```

**Default Parameters:**
- **Batch Size**: 1280 (optimized for large batch training)
- **Learning Rate**: 3.16e-5 (scaled by √10 for large batch stability)
- **Epochs**: 1000
- **Save Frequency**: Every 10 epochs

### Evaluation

```bash
python test_model.py
```

### Visualization

#### Metrics Analysis
```bash
python plot_metrics.py
```

## Model Architecture

The diffusion model uses a simple MLP-based denoiser:

- **Input**: 2D coordinates + timestep embedding
- **Architecture**: 3-layer MLP (128 → 128 → 2)
- **Timestep Embedding**: Sinusoidal positional encoding
- **Noise Schedule**: Linear beta schedule (T=1000)

## Dataset

The GMM dataset generates 2D data points distributed in clusters around a circle:

- **Components**: 8 Gaussian components
- **Radius**: 5.0 (circle radius)
- **Standard Deviation**: 0.2 (component spread)
- **Samples**: 10,000 training samples

## Evaluation Metrics

### Quantitative Metrics
- **MMD (RBF)**: Maximum Mean Discrepancy with RBF kernel
- **MMD (Linear)**: Maximum Mean Discrepancy with linear kernel
- **Wasserstein Distance**: Earth mover's distance
- **Test Loss**: MSE loss on noise prediction
- **Sample Statistics**: Range, standard deviation of generated samples

### Visualization
- **Individual Plots**: Per-epoch sample distributions
- **Comparison Plots**: Side-by-side model comparisons
- **Learning Curves**: Metric evolution over training
- **Distribution Analysis**: Sample quality assessment

## Training Configuration

### Large Batch Training (Default)
- **Batch Size**: 1280
- **Learning Rate**: 3.16e-5 (scaled by √10)
- **Epochs**: 1000
- **Save Frequency**: Every 10 epochs
- **Optimization**: Adam optimizer with large batch stability

## Results

The project includes comprehensive evaluation of the large batch training configuration:

### Performance Analysis
- **Convergence Speed**: Analysis of training dynamics
- **Final Quality**: Best-performing model identification
- **Stability**: Coefficient of variation analysis
- **Sample Quality**: Distribution matching metrics

### Key Benefits
- **Large Batch Efficiency**: Better gradient estimates with 10x batch size
- **Stable Training**: Proper learning rate scaling prevents divergence
- **Comprehensive Evaluation**: Multiple metrics for thorough analysis
- **Reproducible Results**: Deterministic training with proper seeding

## File Organization

### Excluded from Git
- `checkpoints/`: Model checkpoints (large files)
- `test_results/`: Evaluation results and plots
- `venv/`: Virtual environment
- `*.npy`, `*.png`, `*.csv`: Data and visualization files

### Included in Git
- All Python source code
- Configuration files
- Documentation
- Requirements specification

## Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization
- **Pandas**: Data analysis
- **Scikit-learn**: Machine learning utilities
- **SciPy**: Scientific computing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{diffusion_paths_2024,
  title={Diffusion Model Training and Evaluation on 2D GMM Data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/diffusion-paths}
}
``` 