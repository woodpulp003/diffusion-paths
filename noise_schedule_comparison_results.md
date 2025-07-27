# Noise Schedule Comparison Results

## Training Summary

All models were trained for 100 epochs with the following configurations:
- **Batch Size**: 1280 (large batch)
- **Learning Rate**: 3.16e-5 (scaled for large batch)
- **Epochs**: 100
- **Save Frequency**: Every 10 epochs

## Final Training Losses (Epoch 100)

| Schedule Type | Final Loss | Performance Rank |
|---------------|------------|------------------|
| **Linear**    | 0.643      | 🥇 **Best**      |
| **Cosine**    | 0.704      | 🥈 2nd           |
| **Exponential**| 0.923     | 🥉 3rd           |
| **Quadratic** | 0.989      | 4th              |

## Key Findings

### 1. **Linear Schedule Performs Best**
- **Final Loss**: 0.643
- **Convergence**: Stable and consistent
- **Reason**: Standard DDPM approach with gradual noise increase

### 2. **Cosine Schedule Shows Promise**
- **Final Loss**: 0.704
- **Convergence**: Smooth transitions
- **Advantage**: Often better for high-resolution data

### 3. **Exponential Schedule Struggles**
- **Final Loss**: 0.923
- **Issue**: Exponential decay may be too aggressive for this dataset

### 4. **Quadratic Schedule Performs Worst**
- **Final Loss**: 0.989
- **Issue**: Faster early noise increase may destabilize training

## Training Dynamics

### Loss Progression (Selected Epochs)
```
Linear:     0.995 → 0.991 → 0.989 → ... → 0.643
Cosine:     1.010 → 1.003 → 0.981 → ... → 0.704
Exponential: 1.029 → 1.003 → 0.981 → ... → 0.923
Quadratic:  1.010 → 1.003 → 0.981 → ... → 0.989
```

## Recommendations

1. **Use Linear Schedule**: Best performance for 2D GMM data
2. **Consider Cosine**: Good alternative for smoother transitions
3. **Avoid Quadratic/Exponential**: Too aggressive for this dataset size

## Model Checkpoints

All models are saved in organized directories:
- `checkpoints/linear_schedule/`
- `checkpoints/cosine_schedule/`
- `checkpoints/quadratic_schedule/`
- `checkpoints/exponential_schedule/`

Each directory contains:
- Checkpoint files: `model_epoch_*.pt`
- Loss history: `losses.csv`
- Final model: `model_final.pt`

## Next Steps

1. **Extended Training**: Train linear and cosine schedules for 1000 epochs
2. **Evaluation**: Run comprehensive evaluation on all schedules
3. **Visualization**: Compare sample quality across schedules
4. **Hyperparameter Tuning**: Optimize learning rates for each schedule

## Technical Details

### Schedule Implementations
- **Linear**: `β_t = β_start + (β_end - β_start) * t/T`
- **Cosine**: `β_t = β_end * sin²(πt/2T)`
- **Quadratic**: `β_t = (linear_betas)²`
- **Exponential**: `β_t = exp(log(β_start) + (log(β_end) - log(β_start)) * t/T)`

### Training Configuration
```python
T = 1000  # Timesteps
β_start = 1e-4
β_end = 0.02
Batch Size = 1280
Learning Rate = 3.16e-5
```

---

*Results generated on: July 26, 2024*
*Training completed successfully for all 4 noise schedules* 