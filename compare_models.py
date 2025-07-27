import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Load the metrics data for both models
df_original = pd.read_csv('test_results/evaluation_metrics.csv')
df_large_batch = pd.read_csv('test_results_large_batch/evaluation_metrics.csv')

# Remove duplicate epochs and sort
df_original_unique = df_original.drop_duplicates(subset=['epoch']).sort_values('epoch')
df_large_batch_unique = df_large_batch.drop_duplicates(subset=['epoch']).sort_values('epoch')

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Model Comparison: Original vs Large Batch (10x)', fontsize=16, fontweight='bold')

# 1. Test Loss Comparison
axes[0, 0].plot(df_original_unique['epoch'], df_original_unique['test_loss'], 'b-', label='Original Model', linewidth=2)
axes[0, 0].plot(df_large_batch_unique['epoch'], df_large_batch_unique['test_loss'], 'r-', label='Large Batch Model', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Test Loss')
axes[0, 0].set_title('Test Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. Training Loss Comparison
axes[0, 1].plot(df_original_unique['epoch'], df_original_unique['training_loss'], 'b-', label='Original Model', linewidth=2)
axes[0, 1].plot(df_large_batch_unique['epoch'], df_large_batch_unique['training_loss'], 'r-', label='Large Batch Model', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Training Loss')
axes[0, 1].set_title('Training Loss Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# 3. MMD (RBF) Comparison
axes[0, 2].plot(df_original_unique['epoch'], df_original_unique['mmd_rbf'], 'b-', label='Original Model', linewidth=2)
axes[0, 2].plot(df_large_batch_unique['epoch'], df_large_batch_unique['mmd_rbf'], 'r-', label='Large Batch Model', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('MMD (RBF)')
axes[0, 2].set_title('MMD (RBF) Comparison')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_yscale('log')

# 4. MMD (Linear) Comparison
axes[1, 0].plot(df_original_unique['epoch'], df_original_unique['mmd_linear'], 'b-', label='Original Model', linewidth=2)
axes[1, 0].plot(df_large_batch_unique['epoch'], df_large_batch_unique['mmd_linear'], 'r-', label='Large Batch Model', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MMD (Linear)')
axes[1, 0].set_title('MMD (Linear) Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# 5. Wasserstein Distance Comparison
axes[1, 1].plot(df_original_unique['epoch'], df_original_unique['wasserstein_distance'], 'b-', label='Original Model', linewidth=2)
axes[1, 1].plot(df_large_batch_unique['epoch'], df_large_batch_unique['wasserstein_distance'], 'r-', label='Large Batch Model', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Wasserstein Distance')
axes[1, 1].set_title('Wasserstein Distance Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yscale('log')

# 6. Sample Standard Deviation Comparison
axes[1, 2].plot(df_original_unique['epoch'], df_original_unique['large_samples_std'], 'b-', label='Original Model', linewidth=2)
axes[1, 2].plot(df_large_batch_unique['epoch'], df_large_batch_unique['large_samples_std'], 'r-', label='Large Batch Model', linewidth=2)
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Sample Standard Deviation')
axes[1, 2].set_title('Sample Distribution Spread Comparison')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print comprehensive comparison summary
print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Find best performing epochs for each model
print("\nBEST PERFORMING EPOCHS BY METRIC:")
print("-" * 50)

# Original model best epochs
best_orig_test_loss = df_original_unique.loc[df_original_unique['test_loss'].idxmin()]
best_orig_mmd_rbf = df_original_unique.loc[df_original_unique['mmd_rbf'].idxmin()]
best_orig_mmd_linear = df_original_unique.loc[df_original_unique['mmd_linear'].idxmin()]
best_orig_wasserstein = df_original_unique.loc[df_original_unique['wasserstein_distance'].idxmin()]

# Large batch model best epochs
best_lb_test_loss = df_large_batch_unique.loc[df_large_batch_unique['test_loss'].idxmin()]
best_lb_mmd_rbf = df_large_batch_unique.loc[df_large_batch_unique['mmd_rbf'].idxmin()]
best_lb_mmd_linear = df_large_batch_unique.loc[df_large_batch_unique['mmd_linear'].idxmin()]
best_lb_wasserstein = df_large_batch_unique.loc[df_large_batch_unique['wasserstein_distance'].idxmin()]

print(f"Test Loss:")
print(f"  Original: Epoch {best_orig_test_loss['epoch']:4d} = {best_orig_test_loss['test_loss']:.6f}")
print(f"  Large Batch: Epoch {best_lb_test_loss['epoch']:4d} = {best_lb_test_loss['test_loss']:.6f}")

print(f"\nMMD (RBF):")
print(f"  Original: Epoch {best_orig_mmd_rbf['epoch']:4d} = {best_orig_mmd_rbf['mmd_rbf']:.6f}")
print(f"  Large Batch: Epoch {best_lb_mmd_rbf['epoch']:4d} = {best_lb_mmd_rbf['mmd_rbf']:.6f}")

print(f"\nMMD (Linear):")
print(f"  Original: Epoch {best_orig_mmd_linear['epoch']:4d} = {best_orig_mmd_linear['mmd_linear']:.6f}")
print(f"  Large Batch: Epoch {best_lb_mmd_linear['epoch']:4d} = {best_lb_mmd_linear['mmd_linear']:.6f}")

print(f"\nWasserstein Distance:")
print(f"  Original: Epoch {best_orig_wasserstein['epoch']:4d} = {best_orig_wasserstein['wasserstein_distance']:.6f}")
print(f"  Large Batch: Epoch {best_lb_wasserstein['epoch']:4d} = {best_lb_wasserstein['wasserstein_distance']:.6f}")

# Final epoch comparison
print(f"\n" + "="*80)
print("FINAL EPOCH COMPARISON")
print("="*80)

final_orig = df_original_unique.iloc[-1]
final_lb = df_large_batch_unique.iloc[-1]

print(f"Original Model (Epoch {final_orig['epoch']}):")
print(f"  Test Loss: {final_orig['test_loss']:.6f}")
print(f"  Training Loss: {final_orig['training_loss']:.6f}")
print(f"  MMD (RBF): {final_orig['mmd_rbf']:.6f}")
print(f"  MMD (Linear): {final_orig['mmd_linear']:.6f}")
print(f"  Wasserstein: {final_orig['wasserstein_distance']:.6f}")

print(f"\nLarge Batch Model (Epoch {final_lb['epoch']}):")
print(f"  Test Loss: {final_lb['test_loss']:.6f}")
print(f"  Training Loss: {final_lb['training_loss']:.6f}")
print(f"  MMD (RBF): {final_lb['mmd_rbf']:.6f}")
print(f"  MMD (Linear): {final_lb['mmd_linear']:.6f}")
print(f"  Wasserstein: {final_lb['wasserstein_distance']:.6f}")

# Performance analysis
print(f"\n" + "="*80)
print("PERFORMANCE ANALYSIS")
print("="*80)

# Calculate improvements/worsening
test_loss_ratio = final_lb['test_loss'] / final_orig['test_loss']
mmd_rbf_ratio = final_lb['mmd_rbf'] / final_orig['mmd_rbf']
mmd_linear_ratio = final_lb['mmd_linear'] / final_orig['mmd_linear']
wasserstein_ratio = final_lb['wasserstein_distance'] / final_orig['wasserstein_distance']

print(f"Large Batch vs Original (Final Epoch):")
print(f"  Test Loss: {test_loss_ratio:.2f}x ({'Better' if test_loss_ratio < 1 else 'Worse'})")
print(f"  MMD (RBF): {mmd_rbf_ratio:.2f}x ({'Better' if mmd_rbf_ratio < 1 else 'Worse'})")
print(f"  MMD (Linear): {mmd_linear_ratio:.2f}x ({'Better' if mmd_linear_ratio < 1 else 'Worse'})")
print(f"  Wasserstein: {wasserstein_ratio:.2f}x ({'Better' if wasserstein_ratio < 1 else 'Worse'})")

# Training stability analysis
print(f"\n" + "="*80)
print("TRAINING STABILITY ANALYSIS")
print("="*80)

# Calculate coefficient of variation (std/mean) for loss
orig_loss_cv = df_original_unique['test_loss'].std() / df_original_unique['test_loss'].mean()
lb_loss_cv = df_large_batch_unique['test_loss'].std() / df_large_batch_unique['test_loss'].mean()

print(f"Test Loss Coefficient of Variation:")
print(f"  Original Model: {orig_loss_cv:.4f}")
print(f"  Large Batch Model: {lb_loss_cv:.4f}")
print(f"  {'Large batch more stable' if lb_loss_cv < orig_loss_cv else 'Original more stable'}")

# Convergence speed analysis
print(f"\nConvergence Analysis:")
print(f"  Original model epochs: {len(df_original_unique)}")
print(f"  Large batch model epochs: {len(df_large_batch_unique)}")

# Find when models reach similar performance
target_loss = 0.4  # Example threshold
orig_convergence = df_original_unique[df_original_unique['test_loss'] <= target_loss]
lb_convergence = df_large_batch_unique[df_large_batch_unique['test_loss'] <= target_loss]

if len(orig_convergence) > 0 and len(lb_convergence) > 0:
    orig_conv_epoch = orig_convergence.iloc[0]['epoch']
    lb_conv_epoch = lb_convergence.iloc[0]['epoch']
    print(f"  Original model reached {target_loss} loss at epoch {orig_conv_epoch}")
    print(f"  Large batch model reached {target_loss} loss at epoch {lb_conv_epoch}")
    print(f"  {'Large batch converged faster' if lb_conv_epoch < orig_conv_epoch else 'Original converged faster'}")

print(f"\nPlot saved to: model_comparison.png") 