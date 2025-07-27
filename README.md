# Diffusion Paths: Comprehensive Evaluation and Analysis

This repository contains a comprehensive evaluation framework for diffusion models with different noise schedules, focusing on both simple and complex datasets.

## 🏗️ Project Structure

```
diffusion-paths/
├── data/                          # Data generation and utilities
│   └── gmm_dataset.py            # Gaussian Mixture Model dataset generation
├── scripts/                       # Organized script directories
│   ├── training/                  # Training scripts
│   │   ├── train.py              # Simple dataset training
│   │   └── train_complex_dataset.py  # Complex dataset training
│   ├── evaluation/                # Evaluation scripts
│   │   ├── test_model.py         # Simple dataset evaluation
│   │   ├── test_complex_model.py # Complex dataset evaluation
│   │   └── evaluate_complex_dataset.py  # Complex dataset evaluation
│   ├── visualization/             # Visualization scripts
│   │   ├── create_complex_epoch_distributions.py
│   │   ├── create_complex_side_by_side_plots.py
│   │   └── complex_evaluation_summary.py
│   └── analysis/                  # Analysis scripts
│       ├── complex_geodesic_analysis.py
│       ├── geodesic_evaluation.py
│       ├── plot_geodesic_trajectories.py
│       └── quick_geodesic_test.py
├── test_results/                  # Simple dataset results
├── test_results_complex/          # Complex dataset results
│   ├── epoch_distributions/       # Epoch-by-epoch distribution plots
│   ├── complex_evaluation_metrics.json
│   ├── complex_evaluation_metrics.csv
│   └── [various visualization plots]
├── checkpoints/                   # Simple dataset model checkpoints
├── checkpoints_complex/           # Complex dataset model checkpoints
├── model.py                       # Denoiser model architecture
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## 🎯 Key Features

### 📊 Comprehensive Evaluation
- **Multiple Noise Schedules**: Linear, Cosine, Quadratic, Exponential
- **Parallel Processing**: Efficient evaluation with configurable workers
- **Multiple Metrics**: Test Loss, MMD (RBF & Linear), Wasserstein Distance
- **Epoch-by-Epoch Analysis**: Detailed progression tracking

### 🔬 Geodesic Analysis
- **Conditional Geodesic Evaluation**: Measures geodesic properties
- **Marginal Geodesic Evaluation**: Analyzes marginal distributions
- **Noise Schedule Comparison**: Comprehensive schedule analysis

### 📈 Visualization
- **Distribution Plots**: Side-by-side comparisons of learned vs original data
- **Progression Plots**: Epoch-by-epoch metric progression
- **Geodesic Plots**: Trajectory and deviation visualizations
- **Summary Plots**: Comprehensive comparison visualizations

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Models
```bash
# Train simple dataset models
python scripts/training/train.py

# Train complex dataset models
python scripts/training/train_complex_dataset.py
```

### 3. Evaluation
```bash
# Evaluate simple dataset
python scripts/evaluation/test_model.py

# Evaluate complex dataset
python scripts/evaluation/test_complex_model.py
```

### 4. Analysis
```bash
# Run geodesic analysis
python scripts/analysis/complex_geodesic_analysis.py

# Quick geodesic test
python scripts/analysis/quick_geodesic_test.py
```

### 5. Visualization
```bash
# Create epoch distribution plots
python scripts/visualization/create_complex_epoch_distributions.py

# Create side-by-side comparisons
python scripts/visualization/create_complex_side_by_side_plots.py
```

## 📋 Results Summary

### Complex Dataset Evaluation Results
- **Best Overall**: LINEAR schedule (Test Loss: 0.199254)
- **Best MMD RBF**: LINEAR schedule (0.042255)
- **Best MMD Linear**: LINEAR schedule (0.048721)
- **Best Wasserstein**: LINEAR schedule (0.353967)

### Geodesic Analysis Results
- **Best Overall Geodesic**: LINEAR (0.199254)
- **Best Conditional**: QUADRATIC (0.042255)
- **Best Marginal**: LINEAR (0.353967)

## 🔧 Configuration

### Noise Schedules
- **Linear**: `β_t = β_start + (β_end - β_start) * t/T`
- **Cosine**: `β_t = β_start + (β_end - β_start) * (1 - cos(πt/2T))`
- **Quadratic**: `β_t = β_start + (β_end - β_start) * (t/T)²`
- **Exponential**: `β_t = β_start + (β_end - β_start) * (exp(t/T) - 1)/(e - 1)`

### Evaluation Parameters
- **Timesteps**: 1000
- **Beta Range**: 1e-4 to 0.02
- **Evaluation Samples**: 1000 per epoch
- **Parallel Workers**: 5 (configurable)

## 📊 Generated Files

### Evaluation Results
- `complex_evaluation_metrics.json/csv`: Comprehensive evaluation metrics
- `complex_dataset_summary.csv`: Summary statistics
- `complex_geodesic_analysis_results.json`: Geodesic analysis results

### Visualizations
- `epoch_progression_metrics.png`: Metric progression plots
- `complex_side_by_side_distributions.png`: Distribution comparisons
- `complex_geodesic_analysis_plots.png`: Geodesic deviation plots
- `complex_geodesic_comparison.png`: Geodesic comparison
- `epoch_distributions/`: Individual epoch distribution plots

## 🛠️ Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **SciPy**: Scientific computing

## 📝 Documentation

- `GEODESIC_EVALUATION_README.md`: Detailed geodesic analysis documentation
- `GEODESIC_RESULTS_SUMMARY.md`: Geodesic analysis results summary

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 