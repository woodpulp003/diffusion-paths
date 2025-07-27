# Diffusion Model Training and Evaluation

This repository contains a comprehensive implementation of diffusion model training and evaluation on 2D Gaussian Mixture Model (GMM) data, with support for both standard and large batch training configurations.

## Features

- **2D GMM Dataset**: Generate 2D data from a GMM where means are equally spaced on a circle
- **Diffusion Model Training**: Complete DDPM implementation with configurable parameters
- **Large Batch Training**: Support for 10x batch size with appropriate learning rate scaling
- **Parallel Evaluation**: Multi-worker evaluation of model checkpoints with comprehensive metrics
- **Comprehensive Metrics**: MMD (RBF/Linear), Wasserstein distance, test loss tracking
- **Visualization Suite**: Individual plots, comparison plots, learning curves
- **Model Comparison**: Direct comparison between different training configurations

## Project Structure

```
.
├── model.py                          # Denoiser model implementation
├── train.py                          # Standard training script
├── train_large_batch.py              # Large batch training script
├── test_model.py                     # Standard evaluation script
├── test_model_parallel.py            # Parallel evaluation script
├── test_model_large_batch.py         # Large batch evaluation script
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

#### Standard Training (Batch Size 128)
```bash
python train.py --epochs 1000 --batch_size 128 --lr 1e-4 --save_every 10
```

#### Large Batch Training (Batch Size 1280)
```bash
python train_large_batch.py --epochs 1000 --batch_size 1280 --lr 3.16e-5 --save_every 10
```

### Evaluation

#### Standard Evaluation
```bash
python test_model_parallel.py
```

#### Large Batch Evaluation
```bash
python test_model_large_batch.py
```

#### Model Comparison
```bash
python compare_models.py
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

## Training Configurations

### Standard Training
- **Batch Size**: 128
- **Learning Rate**: 1e-4
- **Epochs**: 1000
- **Save Frequency**: Every 10 epochs

### Large Batch Training
- **Batch Size**: 1280 (10x standard)
- **Learning Rate**: 3.16e-5 (scaled by √10)
- **Epochs**: 1000
- **Save Frequency**: Every 10 epochs

## Results

The project includes comprehensive evaluation of both training configurations:

### Performance Comparison
- **Convergence Speed**: Analysis of training dynamics
- **Final Quality**: Comparison of best-performing models
- **Stability**: Coefficient of variation analysis
- **Sample Quality**: Distribution matching metrics

### Key Findings
- Large batch training shows different convergence patterns
- Learning rate scaling is crucial for large batch stability
- Both configurations achieve similar final performance with sufficient training
- Parallel evaluation enables efficient checkpoint analysis

## Parallel Processing

The evaluation system supports parallel processing with configurable workers:

- **Default Workers**: 10 parallel processes
- **GPU Support**: Automatic device assignment
- **Memory Efficient**: Streaming evaluation of large checkpoint sets
- **Comprehensive Logging**: Progress tracking and error handling

## File Organization

### Excluded from Git
- `checkpoints/`: Model checkpoints (large files)
- `checkpoints_large_batch/`: Large batch checkpoints
- `test_results/`: Evaluation results and plots
- `test_results_large_batch/`: Large batch evaluation results
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