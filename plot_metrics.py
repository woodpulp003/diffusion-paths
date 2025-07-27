import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Load the metrics data
df = pd.read_csv('test_results/evaluation_metrics.csv')

# Remove duplicate epochs (some checkpoints have the same epoch number)
df_unique = df.drop_duplicates(subset=['epoch']).sort_values('epoch')

# Create a comprehensive plot
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Diffusion Model Training Progress - All Metrics vs Epochs', fontsize=16, fontweight='bold')

# 1. Test Loss vs Training Loss
axes[0, 0].plot(df_unique['epoch'], df_unique['test_loss'], 'b-', label='Test Loss', linewidth=2)
axes[0, 0].plot(df_unique['epoch'], df_unique['training_loss'], 'r-', label='Training Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Evolution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. MMD (RBF) Evolution
axes[0, 1].plot(df_unique['epoch'], df_unique['mmd_rbf'], 'g-', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MMD (RBF)')
axes[0, 1].set_title('MMD (RBF) Evolution')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# 3. MMD (Linear) Evolution
axes[0, 2].plot(df_unique['epoch'], df_unique['mmd_linear'], 'm-', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('MMD (Linear)')
axes[0, 2].set_title('MMD (Linear) Evolution')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_yscale('log')

# 4. Wasserstein Distance Evolution
axes[1, 0].plot(df_unique['epoch'], df_unique['wasserstein_distance'], 'c-', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Wasserstein Distance')
axes[1, 0].set_title('Wasserstein Distance Evolution')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# 5. Sample Standard Deviation Evolution
axes[1, 1].plot(df_unique['epoch'], df_unique['large_samples_std'], 'orange', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Sample Standard Deviation')
axes[1, 1].set_title('Sample Distribution Spread')
axes[1, 1].grid(True, alpha=0.3)

# 6. Combined Metrics (normalized)
# Normalize all metrics to [0,1] for comparison
mmd_rbf_norm = (df_unique['mmd_rbf'] - df_unique['mmd_rbf'].min()) / (df_unique['mmd_rbf'].max() - df_unique['mmd_rbf'].min())
mmd_linear_norm = (df_unique['mmd_linear'] - df_unique['mmd_linear'].min()) / (df_unique['mmd_linear'].max() - df_unique['mmd_linear'].min())
wasserstein_norm = (df_unique['wasserstein_distance'] - df_unique['wasserstein_distance'].min()) / (df_unique['wasserstein_distance'].max() - df_unique['wasserstein_distance'].min())
test_loss_norm = (df_unique['test_loss'] - df_unique['test_loss'].min()) / (df_unique['test_loss'].max() - df_unique['test_loss'].min())

axes[1, 2].plot(df_unique['epoch'], mmd_rbf_norm, 'g-', label='MMD (RBF)', linewidth=2)
axes[1, 2].plot(df_unique['epoch'], mmd_linear_norm, 'm-', label='MMD (Linear)', linewidth=2)
axes[1, 2].plot(df_unique['epoch'], wasserstein_norm, 'c-', label='Wasserstein', linewidth=2)
axes[1, 2].plot(df_unique['epoch'], test_loss_norm, 'b-', label='Test Loss', linewidth=2)
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Normalized Metric Value')
axes[1, 2].set_title('Normalized Metrics Comparison')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_results/comprehensive_metrics_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Find best performing epochs
print("="*80)
print("BEST PERFORMING EPOCHS BY DIFFERENT METRICS")
print("="*80)

best_test_loss = df_unique.loc[df_unique['test_loss'].idxmin()]
best_mmd_rbf = df_unique.loc[df_unique['mmd_rbf'].idxmin()]
best_mmd_linear = df_unique.loc[df_unique['mmd_linear'].idxmin()]
best_wasserstein = df_unique.loc[df_unique['wasserstein_distance'].idxmin()]

print(f"Best Test Loss: Epoch {best_test_loss['epoch']:4d} = {best_test_loss['test_loss']:.6f}")
print(f"Best MMD (RBF): Epoch {best_mmd_rbf['epoch']:4d} = {best_mmd_rbf['mmd_rbf']:.6f}")
print(f"Best MMD (Linear): Epoch {best_mmd_linear['epoch']:4d} = {best_mmd_linear['mmd_linear']:.6f}")
print(f"Best Wasserstein: Epoch {best_wasserstein['epoch']:4d} = {best_wasserstein['wasserstein_distance']:.6f}")

# Show convergence analysis
print(f"\n" + "="*80)
print("CONVERGENCE ANALYSIS")
print("="*80)

# Check if metrics are still improving in the last 20% of epochs
last_20_percent = int(len(df_unique) * 0.2)
recent_epochs = df_unique.tail(last_20_percent)

print(f"Last {last_20_percent} epochs (epochs {recent_epochs['epoch'].min()}-{recent_epochs['epoch'].max()}):")

# Check if metrics are still decreasing
mmd_rbf_trend = recent_epochs['mmd_rbf'].iloc[-1] - recent_epochs['mmd_rbf'].iloc[0]
mmd_linear_trend = recent_epochs['mmd_linear'].iloc[-1] - recent_epochs['mmd_linear'].iloc[0]
wasserstein_trend = recent_epochs['wasserstein_distance'].iloc[-1] - recent_epochs['wasserstein_distance'].iloc[0]
test_loss_trend = recent_epochs['test_loss'].iloc[-1] - recent_epochs['test_loss'].iloc[0]

print(f"  MMD (RBF) trend: {mmd_rbf_trend:+.6f} ({'Improving' if mmd_rbf_trend < 0 else 'Worsening'})")
print(f"  MMD (Linear) trend: {mmd_linear_trend:+.6f} ({'Improving' if mmd_linear_trend < 0 else 'Worsening'})")
print(f"  Wasserstein trend: {wasserstein_trend:+.6f} ({'Improving' if wasserstein_trend < 0 else 'Worsening'})")
print(f"  Test Loss trend: {test_loss_trend:+.6f} ({'Improving' if test_loss_trend < 0 else 'Worsening'})")

# Show final values
print(f"\nFinal epoch ({df_unique['epoch'].max()}) values:")
final_metrics = df_unique.iloc[-1]
print(f"  Test Loss: {final_metrics['test_loss']:.6f}")
print(f"  MMD (RBF): {final_metrics['mmd_rbf']:.6f}")
print(f"  MMD (Linear): {final_metrics['mmd_linear']:.6f}")
print(f"  Wasserstein: {final_metrics['wasserstein_distance']:.6f}")

print(f"\nPlot saved to: test_results/comprehensive_metrics_analysis.png") 