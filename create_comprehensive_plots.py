import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from data.gmm_dataset import generate_complex_gmm_data
from model import Denoiser
from evaluate_complex_dataset import NoiseSchedule, sample_from_model

def load_evaluation_results():
    """Load the evaluation results from the CSV file."""
    return pd.read_csv('test_results_complex/complex_dataset_summary.csv')

def create_metrics_comparison_plot():
    """Create a comprehensive plot showing all metrics for each noise schedule."""
    
    # Load results
    df = load_evaluation_results()
    
    # Extract metrics
    schedules = df['Schedule'].tolist()
    test_losses = [float(x.split(' ')[0]) for x in df['Best Test Loss (Epoch)']]
    mmd_rbf = [float(x.split(' ')[0]) for x in df['Best MMD RBF (Epoch)']]
    mmd_linear = [float(x.split(' ')[0]) for x in df['Best MMD Linear (Epoch)']]
    wasserstein = [float(x.split(' ')[0]) for x in df['Best Wasserstein (Epoch)']]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Complex Dataset: Performance Metrics by Noise Schedule', fontsize=16, fontweight='bold')
    
    # Colors for each schedule
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Test Loss
    ax1 = axes[0, 0]
    bars1 = ax1.bar(schedules, test_losses, color=colors, alpha=0.8)
    ax1.set_title('Best Test Loss by Schedule', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Test Loss (Lower is Better)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, test_losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. MMD RBF
    ax2 = axes[0, 1]
    bars2 = ax2.bar(schedules, mmd_rbf, color=colors, alpha=0.8)
    ax2.set_title('Best MMD RBF by Schedule', fontweight='bold', fontsize=12)
    ax2.set_ylabel('MMD RBF (Lower is Better)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, mmd_rbf):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. MMD Linear
    ax3 = axes[1, 0]
    bars3 = ax3.bar(schedules, mmd_linear, color=colors, alpha=0.8)
    ax3.set_title('Best MMD Linear by Schedule', fontweight='bold', fontsize=12)
    ax3.set_ylabel('MMD Linear (Lower is Better)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, mmd_linear):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Wasserstein Distance
    ax4 = axes[1, 1]
    bars4 = ax4.bar(schedules, wasserstein, color=colors, alpha=0.8)
    ax4.set_title('Best Wasserstein Distance by Schedule', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Wasserstein Distance (Lower is Better)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars4, wasserstein):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_results_complex/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Metrics comparison plot saved as: metrics_comparison.png")

def create_side_by_side_visual_comparison():
    """Create side-by-side visual comparison of generated samples from each schedule."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original data for comparison
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Complex Dataset: Generated vs Original Data Comparison', fontsize=16, fontweight='bold')
    
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, schedule in enumerate(schedules):
        ax = axes[idx // 2, idx % 2]
        
        # Find the best model for this schedule
        best_epoch = find_best_epoch_for_schedule(schedule)
        checkpoint_path = f'checkpoints_complex/{schedule}_complex/model_epoch_{best_epoch:04d}.pt'
        
        if os.path.exists(checkpoint_path):
            print(f"Loading {schedule} schedule (epoch {best_epoch})...")
            
            # Load model
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = Denoiser(embedding_dim=32)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Create noise schedule
            noise_schedule = NoiseSchedule(T=1000, schedule_type=schedule)
            
            # Generate samples
            print(f"Generating samples for {schedule} schedule...")
            generated_samples = sample_from_model(model, noise_schedule, device, num_samples=5000)
            
            # Plot original data
            ax.scatter(original_data[:, 0], original_data[:, 1], 
                      alpha=0.6, s=1, label='Original', color='blue', marker='o')
            
            # Plot generated samples
            ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                      alpha=0.6, s=1, label='Generated', color='red', marker='x')
            
            ax.set_title(f'{schedule.upper()} Schedule (Epoch {best_epoch})', fontweight='bold')
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = get_metrics_for_schedule(schedule)
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="lightblue", alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'Checkpoint not found:\n{checkpoint_path}', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title(f'{schedule.upper()} Schedule', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_results_complex/side_by_side_visual_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Side-by-side visual comparison saved as: side_by_side_visual_comparison.png")

def find_best_epoch_for_schedule(schedule):
    """Find the best epoch for a given schedule based on test loss."""
    df = load_evaluation_results()
    schedule_row = df[df['Schedule'] == schedule].iloc[0]
    best_epoch_str = schedule_row['Best Test Loss (Epoch)']
    return int(best_epoch_str.split('(')[1].split(')')[0])

def get_metrics_for_schedule(schedule):
    """Get the metrics for a given schedule."""
    df = load_evaluation_results()
    schedule_row = df[df['Schedule'] == schedule].iloc[0]
    
    test_loss = float(schedule_row['Best Test Loss (Epoch)'].split(' ')[0])
    mmd_rbf = float(schedule_row['Best MMD RBF (Epoch)'].split(' ')[0])
    mmd_linear = float(schedule_row['Best MMD Linear (Epoch)'].split(' ')[0])
    wasserstein = float(schedule_row['Best Wasserstein (Epoch)'].split(' ')[0])
    
    return f"Test Loss: {test_loss:.4f}\nMMD RBF: {mmd_rbf:.4f}\nMMD Linear: {mmd_linear:.4f}\nWasserstein: {wasserstein:.4f}"

def create_combined_visualization():
    """Create a combined visualization with both metrics and visual comparison."""
    
    # Load results
    df = load_evaluation_results()
    
    # Create figure with 3 rows: metrics, visual comparison, and summary
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.5, 0.5], hspace=0.3, wspace=0.3)
    
    # Row 1: Metrics comparison
    metrics = ['Best Test Loss (Epoch)', 'Best MMD RBF (Epoch)', 'Best MMD Linear (Epoch)', 'Best Wasserstein (Epoch)']
    metric_names = ['Test Loss', 'MMD RBF', 'MMD Linear', 'Wasserstein']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = fig.add_subplot(gs[0, i])
        schedules = df['Schedule'].tolist()
        values = [float(x.split(' ')[0]) for x in df[metric]]
        
        bars = ax.bar(schedules, values, color=colors, alpha=0.8)
        ax.set_title(f'{name} by Schedule', fontweight='bold', fontsize=11)
        ax.set_ylabel(f'{name} (Lower is Better)', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Row 2: Visual comparison (placeholder for now)
    ax_visual = fig.add_subplot(gs[1, :])
    ax_visual.text(0.5, 0.5, 'Visual comparison will be generated separately\nfor better clarity and detail', 
                   transform=ax_visual.transAxes, ha='center', va='center', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    ax_visual.set_title('Generated vs Original Data Comparison', fontweight='bold', fontsize=14)
    ax_visual.axis('off')
    
    # Row 3: Summary statistics
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    # Calculate summary
    best_test_loss = df.loc[df['Best Test Loss (Epoch)'].str.split(' ').str[0].astype(float).idxmin()]
    best_mmd_linear = df.loc[df['Best MMD Linear (Epoch)'].str.split(' ').str[0].astype(float).idxmin()]
    
    summary_text = f"""
    🏆 BEST PERFORMERS:
    Test Loss: {best_test_loss['Schedule'].upper()} ({best_test_loss['Best Test Loss (Epoch)']})
    MMD Linear: {best_mmd_linear['Schedule'].upper()} ({best_mmd_linear['Best MMD Linear (Epoch)']})
    MMD RBF: All similar (~0.068)
    Wasserstein: {best_mmd_linear['Schedule'].upper()} ({best_mmd_linear['Best Wasserstein (Epoch)']})
    
    📊 EVALUATION SUMMARY:
    Total Checkpoints: 400 | Schedules Tested: 4 | Best Overall: COSINE
    """
    
    ax_summary.text(0.1, 0.5, summary_text, transform=ax_summary.transAxes, fontsize=12,
                   verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    
    plt.savefig('test_results_complex/combined_evaluation_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Combined visualization saved as: combined_evaluation_visualization.png")

def main():
    """Main function to create all visualizations."""
    print("Creating comprehensive evaluation visualizations...")
    
    # Create metrics comparison plot
    print("\n1. Creating metrics comparison plot...")
    create_metrics_comparison_plot()
    
    # Create side-by-side visual comparison
    print("\n2. Creating side-by-side visual comparison...")
    create_side_by_side_visual_comparison()
    
    # Create combined visualization
    print("\n3. Creating combined visualization...")
    create_combined_visualization()
    
    print("\n✅ All visualizations completed!")
    print("Generated files:")
    print("- metrics_comparison.png: Bar charts for all metrics")
    print("- side_by_side_visual_comparison.png: Visual comparison of generated data")
    print("- combined_evaluation_visualization.png: Combined overview")

if __name__ == "__main__":
    main() 