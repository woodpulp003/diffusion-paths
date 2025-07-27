import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import glob
from data.gmm_dataset import generate_complex_gmm_data
from model import Denoiser
from evaluate_complex_dataset import NoiseSchedule, sample_from_model, evaluate_checkpoint

def load_all_checkpoint_results():
    """Load all checkpoint evaluation results from the evaluation."""
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join('checkpoints_complex', "*", "model_epoch_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Evaluate all checkpoints
    all_results = []
    
    for checkpoint_path in checkpoint_files:
        try:
            result = evaluate_checkpoint(checkpoint_path, original_data, device, num_samples=2000)
            all_results.append(result)
            print(f"Evaluated: {os.path.basename(checkpoint_path)}")
        except Exception as e:
            print(f"Error evaluating {checkpoint_path}: {e}")
            continue
    
    return all_results

def create_epoch_progression_plots():
    """Create epoch-by-epoch progression plots for all metrics."""
    
    print("Loading evaluation results...")
    all_results = load_all_checkpoint_results()
    
    if not all_results:
        print("No results found. Please run the evaluation first.")
        return
    
    # Organize results by schedule
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    schedule_results = {schedule: [] for schedule in schedules}
    
    for result in all_results:
        schedule = result['schedule_type']
        if schedule in schedule_results:
            schedule_results[schedule].append(result)
    
    # Sort each schedule by epoch
    for schedule in schedules:
        schedule_results[schedule].sort(key=lambda x: x['epoch'])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Complex Dataset: Epoch-by-Epoch Metric Progression', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Test Loss over epochs
    ax1 = axes[0, 0]
    for i, schedule in enumerate(schedules):
        if schedule_results[schedule]:
            epochs = [r['epoch'] for r in schedule_results[schedule]]
            test_losses = [r['test_loss'] for r in schedule_results[schedule]]
            ax1.plot(epochs, test_losses, label=schedule.upper(), color=colors[i], 
                    marker='o', markersize=3, alpha=0.8, linewidth=1.5)
    
    ax1.set_title('Test Loss Progression', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Test Loss (Lower is Better)', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # 2. MMD RBF over epochs
    ax2 = axes[0, 1]
    for i, schedule in enumerate(schedules):
        if schedule_results[schedule]:
            epochs = [r['epoch'] for r in schedule_results[schedule]]
            mmd_rbf = [r['mmd_rbf'] for r in schedule_results[schedule]]
            ax2.plot(epochs, mmd_rbf, label=schedule.upper(), color=colors[i], 
                    marker='o', markersize=3, alpha=0.8, linewidth=1.5)
    
    ax2.set_title('MMD RBF Progression', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('MMD RBF (Lower is Better)', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MMD Linear over epochs
    ax3 = axes[1, 0]
    for i, schedule in enumerate(schedules):
        if schedule_results[schedule]:
            epochs = [r['epoch'] for r in schedule_results[schedule]]
            mmd_linear = [r['mmd_linear'] for r in schedule_results[schedule]]
            ax3.plot(epochs, mmd_linear, label=schedule.upper(), color=colors[i], 
                    marker='o', markersize=3, alpha=0.8, linewidth=1.5)
    
    ax3.set_title('MMD Linear Progression', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('MMD Linear (Lower is Better)', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Wasserstein Distance over epochs
    ax4 = axes[1, 1]
    for i, schedule in enumerate(schedules):
        if schedule_results[schedule]:
            epochs = [r['epoch'] for r in schedule_results[schedule]]
            wasserstein = [r['wass_dist'] for r in schedule_results[schedule]]
            ax4.plot(epochs, wasserstein, label=schedule.upper(), color=colors[i], 
                    marker='o', markersize=3, alpha=0.8, linewidth=1.5)
    
    ax4.set_title('Wasserstein Distance Progression', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('Wasserstein Distance (Lower is Better)', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results_complex/epoch_progression_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Epoch progression plots saved as: epoch_progression_metrics.png")
    
    # Create individual plots for each metric
    create_individual_metric_plots(schedule_results, colors)

def create_individual_metric_plots(schedule_results, colors):
    """Create individual detailed plots for each metric."""
    
    metrics = [
        ('test_loss', 'Test Loss', 'Test Loss (Lower is Better)'),
        ('mmd_rbf', 'MMD RBF', 'MMD RBF (Lower is Better)'),
        ('mmd_linear', 'MMD Linear', 'MMD Linear (Lower is Better)'),
        ('wass_dist', 'Wasserstein Distance', 'Wasserstein Distance (Lower is Better)')
    ]
    
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    
    for metric_key, metric_name, ylabel in metrics:
        plt.figure(figsize=(12, 8))
        
        for i, schedule in enumerate(schedules):
            if schedule_results[schedule]:
                epochs = [r['epoch'] for r in schedule_results[schedule]]
                values = [r[metric_key] for r in schedule_results[schedule]]
                
                plt.plot(epochs, values, label=schedule.upper(), color=colors[i], 
                        marker='o', markersize=4, alpha=0.8, linewidth=2)
        
        plt.title(f'Complex Dataset: {metric_name} Progression by Schedule', 
                 fontweight='bold', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if metric_key == 'test_loss':
            plt.yscale('log')  # Log scale for test loss
        
        plt.tight_layout()
        plt.savefig(f'test_results_complex/{metric_key}_progression.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ {metric_name} progression plot saved as: {metric_key}_progression.png")

def create_side_by_side_epoch_comparison():
    """Create side-by-side visual comparison at different epochs."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Select key epochs for comparison
    key_epochs = [100, 300, 500, 700, 900]
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    
    # Create figure
    fig, axes = plt.subplots(len(key_epochs), len(schedules), figsize=(20, 16))
    fig.suptitle('Complex Dataset: Generated Samples at Different Epochs', fontsize=16, fontweight='bold')
    
    for epoch_idx, epoch in enumerate(key_epochs):
        for schedule_idx, schedule in enumerate(schedules):
            ax = axes[epoch_idx, schedule_idx]
            
            checkpoint_path = f'checkpoints_complex/{schedule}_complex/model_epoch_{epoch:04d}.pt'
            
            if os.path.exists(checkpoint_path):
                try:
                    # Load model
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model = Denoiser(embedding_dim=32)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    
                    # Create noise schedule
                    noise_schedule = NoiseSchedule(T=1000, schedule_type=schedule)
                    
                    # Generate samples
                    generated_samples = sample_from_model(model, noise_schedule, device, num_samples=2000)
                    
                    # Plot
                    ax.scatter(original_data[:, 0], original_data[:, 1], 
                              alpha=0.4, s=1, label='Original', color='blue', marker='o')
                    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                              alpha=0.6, s=1, label='Generated', color='red', marker='x')
                    
                    ax.set_title(f'{schedule.upper()} Epoch {epoch}', fontweight='bold', fontsize=10)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error loading\n{os.path.basename(checkpoint_path)}', 
                           transform=ax.transAxes, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                    ax.set_title(f'{schedule.upper()} Epoch {epoch}', fontweight='bold', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'Not found:\n{os.path.basename(checkpoint_path)}', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                ax.set_title(f'{schedule.upper()} Epoch {epoch}', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('test_results_complex/side_by_side_epoch_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Side-by-side epoch comparison saved as: side_by_side_epoch_comparison.png")

def main():
    """Main function to create all epoch progression visualizations."""
    print("Creating epoch-by-epoch progression visualizations...")
    
    # Create epoch progression plots
    print("\n1. Creating epoch progression plots...")
    create_epoch_progression_plots()
    
    # Create side-by-side epoch comparison
    print("\n2. Creating side-by-side epoch comparison...")
    create_side_by_side_epoch_comparison()
    
    print("\n✅ All epoch progression visualizations completed!")
    print("Generated files:")
    print("- epoch_progression_metrics.png: All metrics progression")
    print("- test_loss_progression.png: Test loss over epochs")
    print("- mmd_rbf_progression.png: MMD RBF over epochs")
    print("- mmd_linear_progression.png: MMD Linear over epochs")
    print("- wass_dist_progression.png: Wasserstein distance over epochs")
    print("- side_by_side_epoch_comparison.png: Visual comparison at different epochs")

if __name__ == "__main__":
    main() 