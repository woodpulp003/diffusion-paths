# Geodesic Evaluation of Noise Schedules in Diffusion Models

This module provides comprehensive evaluation of how "geodesic-like" different noise schedules are in diffusion models, measuring their deviation from linear interpolation in distribution space.

## Theory

### What is a Geodesic?

In the context of diffusion models, a geodesic path represents the optimal (shortest) path between two distributions in distribution space. For diffusion models, we're interested in:

1. **Conditional Geodesic**: The path from a specific data point `x₀` to pure Gaussian noise via the chosen noise schedule
2. **Marginal Geodesic**: The path from the full data distribution `q₀(x)` to the noise distribution `qₜ(x) ~ N(0,I)`

### Why Evaluate Geodesic Properties?

- **Optimal Sampling**: More geodesic-like schedules may lead to better sampling quality
- **Training Stability**: Geodesic schedules might provide more stable training dynamics
- **Theoretical Understanding**: Helps understand the geometric properties of different noise schedules

## Evaluation Metrics

### 1. Conditional Geodesic Deviation

Measures how well the conditional path `q(xₜ | x₀)` follows a linear interpolation from `x₀` to pure noise.

**Method**: 
- Sample from actual noise schedule: `xₜ ~ q(xₜ | x₀)`
- Sample from linear interpolation: `xₜ_interp = (1-t)x₀ + t·noise`
- Compute MMD between these distributions

### 2. Marginal Geodesic Deviation

Measures how well the marginal distribution `qₜ(x)` follows a linear interpolation from `q₀(x)` to `qₜ(x)`.

**Method**:
- Sample from actual marginal: `xₜ ~ qₜ(x)`
- Sample from interpolated marginal: `xₜ_interp ~ (1-t)q₀(x) + t·N(0,I)`
- Compute Sliced Wasserstein Distance between these distributions

## Usage

### Quick Test

For a fast demonstration with reduced parameters:

```bash
python quick_geodesic_test.py
```

This will:
1. Visualize different noise schedules
2. Run a quick evaluation with 100 interpolation steps
3. Generate comparison plots and save results

### Full Evaluation

For comprehensive evaluation with full parameters:

```bash
python geodesic_evaluation.py
```

This will:
1. Evaluate all 4 noise schedules (linear, cosine, quadratic, exponential)
2. Use 1000 interpolation steps for high precision
3. Generate detailed results and visualizations

### Custom Evaluation

```python
from geodesic_evaluation import NoiseSchedule, evaluate_noise_schedule_geodesic
from data.gmm_dataset import generate_gmm_data

# Generate data
data_samples = generate_gmm_data(n_samples=10000, n_components=8, radius=5.0, std=0.2)
data_samples = torch.FloatTensor(data_samples).to(device)

# Create noise schedule
noise_schedule = NoiseSchedule(T=1000, schedule_type="cosine")

# Evaluate geodesic properties
results = evaluate_noise_schedule_geodesic(
    noise_schedule, 
    data_samples, 
    n_interpolation_steps=1000,
    n_samples_per_step=100
)

print(f"Overall geodesic score: {results['overall_geodesic_score']:.6f}")
```

## Noise Schedules Evaluated

1. **Linear**: `βₜ = β_start + (β_end - β_start) * t/T`
2. **Cosine**: `βₜ = β_start + (β_end - β_start) * (1 - cos(πt/2T))`
3. **Quadratic**: `βₜ = β_start + (β_end - β_start) * (t/T)²`
4. **Exponential**: `βₜ = exp(log(β_start) + (log(β_end) - log(β_start)) * t/T)`

## Output Files

### Results Files
- `geodesic_evaluation_results.json`: Full evaluation results
- `quick_geodesic_test_results.json`: Quick test results

### Visualization Files
- `geodesic_evaluation_plots.png`: Comprehensive comparison plots
- `quick_geodesic_test_plots.png`: Quick test plots
- `noise_schedule_comparison.png`: Noise schedule visualization

### Plot Descriptions

1. **Conditional Geodesic Deviations**: Shows MMD deviation over time for each schedule
2. **Marginal Geodesic Deviations**: Shows Sliced Wasserstein deviation over time
3. **Average Deviations**: Bar chart comparing average deviations across schedules
4. **Overall Scores**: Final ranking with overall geodesic scores

## Interpretation

### Lower Scores = Better Geodesic Properties

- **Overall Geodesic Score**: Combined measure of conditional and marginal geodesic properties
- **Conditional Deviation**: How well individual data points follow linear paths
- **Marginal Deviation**: How well the full distribution follows linear interpolation

### Expected Results

Based on theoretical considerations:
- **Linear schedule** should have the lowest geodesic deviation
- **Cosine schedule** may show moderate deviation
- **Quadratic/Exponential** schedules may show higher deviations

## Technical Details

### Distance Metrics

1. **MMD (Maximum Mean Discrepancy)**: Used for conditional geodesic evaluation
   - Kernel: RBF with γ=1.0
   - Computes distance between sample distributions

2. **Sliced Wasserstein Distance**: Used for marginal geodesic evaluation
   - Uses 100 random projections
   - More robust for distribution comparison

### Sampling Strategy

- **Conditional**: Sample from `q(xₜ | x₀)` vs linear interpolation
- **Marginal**: Sample from `qₜ(x)` vs interpolated distribution
- **Interpolation**: Linear interpolation between endpoints

### Parameters

- **T**: Number of diffusion timesteps (default: 1000)
- **n_interpolation_steps**: Number of evaluation points (default: 1000)
- **n_samples_per_step**: Samples per evaluation point (default: 100)

## Dependencies

Additional dependencies beyond the base project:
- `scipy>=1.7.0`: For statistical functions
- `scikit-learn>=1.0.0`: For MMD computation
- `tqdm>=4.62.0`: For progress bars

## Research Applications

This evaluation framework can be used for:

1. **Noise Schedule Design**: Compare new schedule proposals
2. **Model Analysis**: Understand geometric properties of trained models
3. **Theoretical Studies**: Validate theoretical predictions about geodesic properties
4. **Hyperparameter Tuning**: Optimize noise schedule parameters

## Future Extensions

Potential enhancements:
- **Higher Dimensions**: Extend to 3D+ data
- **Different Metrics**: Add other distribution distance measures
- **Adaptive Sampling**: Optimize sampling strategy for efficiency
- **Theoretical Bounds**: Compare with theoretical geodesic bounds 