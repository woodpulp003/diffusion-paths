import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import csv
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool, cpu_count
import pandas as pd

from model import Denoiser
from data.gmm_dataset import get_dataloader, generate_complex_gmm_data


def get_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for diffusion process."""
    return torch.linspace(beta_start, beta_end, T)


def get_cosine_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Cosine beta schedule for diffusion process."""
    steps = torch.linspace(0, 1, T)
    return beta_start + (beta_end - beta_start) * (1 - torch.cos(steps * torch.pi / 2))


def get_quadratic_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Quadratic beta schedule for diffusion process."""
    steps = torch.linspace(0, 1, T)
    return beta_start + (beta_end - beta_start) * steps ** 2


def get_exponential_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Exponential beta schedule for diffusion process."""
    steps = torch.linspace(0, 1, T)
    return beta_start + (beta_end - beta_start) * (torch.exp(steps) - 1) / (np.e - 1)


def get_beta_schedule(T: int, schedule_type: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Get beta schedule based on the specified type."""
    if schedule_type == "linear":
        return get_linear_beta_schedule(T, beta_start, beta_end)
    elif schedule_type == "cosine":
        return get_cosine_beta_schedule(T, beta_start, beta_end)
    elif schedule_type == "quadratic":
        return get_quadratic_beta_schedule(T, beta_start, beta_end)
    elif schedule_type == "exponential":
        return get_exponential_beta_schedule(T, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def compute_mmd(samples1, samples2, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of samples.
    
    Args:
        samples1: First set of samples (n_samples1, n_features)
        samples2: Second set of samples (n_samples2, n_features)
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel parameter
    
    Returns:
        MMD value
    """
    if kernel == 'rbf':
        # RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
        def rbf_kernel(x, y):
            dist_sq = np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=2)
            return np.exp(-gamma * dist_sq)
        
        # K(xx) - 2*K(xy) + K(yy)
        k_xx = rbf_kernel(samples1, samples1)
        k_yy = rbf_kernel(samples2, samples2)
        k_xy = rbf_kernel(samples1, samples2)
        
        mmd = (np.mean(k_xx) - 2 * np.mean(k_xy) + np.mean(k_yy))
        
    elif kernel == 'linear':
        # Linear kernel: K(x,y) = x^T y
        k_xx = np.dot(samples1, samples1.T)
        k_yy = np.dot(samples2, samples2.T)
        k_xy = np.dot(samples1, samples2.T)
        
        mmd = (np.mean(k_xx) - 2 * np.mean(k_xy) + np.mean(k_yy))
    
    return mmd


def compute_wasserstein_distance(samples1, samples2):
    """
    Compute Wasserstein distance between two sets of samples.
    Uses scipy's wasserstein_distance for 1D projections.
    
    Args:
        samples1: First set of samples (n_samples1, n_features)
        samples2: Second set of samples (n_samples2, n_features)
    
    Returns:
        Average Wasserstein distance across dimensions
    """
    distances = []
    for dim in range(samples1.shape[1]):
        dist = wasserstein_distance(samples1[:, dim], samples2[:, dim])
        distances.append(dist)
    
    return np.mean(distances)


def sample_from_model(model, noise_schedule, device, num_samples=1000, batch_size=128):
    """Sample from the trained diffusion model using DDPM sampling."""
    model.eval()
    
    # Get schedule parameters
    betas = noise_schedule['betas']
    alphas = noise_schedule['alphas']
    alpha_bars = noise_schedule['alpha_bars']
    
    samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Start from pure noise
            x_t = torch.randn(current_batch_size, 2, device=device)
            
            # Reverse process: T-1 to 0
            for t in range(len(betas) - 1, -1, -1):
                t_tensor = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise
                pred_noise = model(x_t, t_tensor)
                
                # Get schedule parameters
                alpha_bar_t = alpha_bars[t]
                alpha_t = alphas[t]
                beta_t = betas[t]
                
                # DDPM reverse step
                if t > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                
                # Reverse equation
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + torch.sqrt(beta_t) * noise
            
            samples.append(x_t.cpu().numpy())
    
    return np.vstack(samples)


def evaluate_single_checkpoint(args):
    """Evaluate a single checkpoint (for parallel processing)."""
    checkpoint_file, original_data, device, T, batch_size, large_n_samples = args
    
    try:
        # Extract epoch number and schedule type
        filename = os.path.basename(checkpoint_file)
        epoch = int(filename.split('_')[-1].split('.')[0])
        
        # Determine schedule type from path
        if 'cosine' in checkpoint_file:
            schedule_type = 'cosine'
        elif 'quadratic' in checkpoint_file:
            schedule_type = 'quadratic'
        elif 'exponential' in checkpoint_file:
            schedule_type = 'exponential'
        else:
            schedule_type = 'linear'
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = Denoiser(embedding_dim=32).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create noise schedule
        betas = get_beta_schedule(T, schedule_type)
        alpha = 1. - betas
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        # Move to device
        betas = betas.to(device)
        alpha_bar = alpha_bar.to(device)
        
        # Create noise schedule dict for sampling
        noise_schedule = {
            'betas': betas,
            'alphas': alpha,
            'alpha_bars': alpha_bar
        }
        
        # Test noise prediction
        total_loss = 0.0
        num_batches = 0
        loss_fn = nn.MSELoss()
        
        # Create test dataloader for complex data
        test_loader = get_dataloader(batch_size=batch_size, train=False, shuffle=False, complex_data=True)
        
        with torch.no_grad():
            for batch_idx, x0 in enumerate(test_loader):
                x0 = x0.to(device)
                
                # Sample random timesteps uniformly for each sample in batch
                t = torch.randint(0, T, (x0.shape[0],), device=device)
                
                # Get alpha_bar values for the sampled timesteps
                alpha_bar_t = alpha_bar[t].unsqueeze(1)
                
                # Sample Gaussian noise
                epsilon = torch.randn_like(x0, device=device)
                
                # Generate x_t using the forward noising equation
                x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
                
                # Predict noise using the model
                pred_epsilon = model(x_t, t)
                
                # Compute loss
                loss = loss_fn(pred_epsilon, epsilon)
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx >= 2:  # Test 3 batches
                    break
        
        avg_loss = total_loss / num_batches
        
        # Generate samples for metrics
        large_samples = sample_from_model(model, noise_schedule, device, num_samples=large_n_samples, batch_size=batch_size)
        
        # Compute quantitative metrics
        mmd_rbf = compute_mmd(large_samples, original_data, kernel='rbf', gamma=1.0)
        mmd_linear = compute_mmd(large_samples, original_data, kernel='linear')
        wasserstein_dist = compute_wasserstein_distance(large_samples, original_data)
        
        # Store metrics
        metrics_entry = {
            'epoch': epoch,
            'schedule_type': schedule_type,
            'test_loss': avg_loss,
            'training_loss': checkpoint.get('loss', 0.0),
            'mmd_rbf': float(mmd_rbf),
            'mmd_linear': float(mmd_linear),
            'wasserstein_distance': float(wasserstein_dist),
            'large_samples_range': [float(large_samples.min()), float(large_samples.max())],
            'large_samples_std': float(np.std(large_samples))
        }
        
        return metrics_entry
        
    except Exception as e:
        print(f"Error evaluating {checkpoint_file}: {e}")
        return None


def create_epoch_progression_plots(metrics_data, save_dir="test_results_complex"):
    """Create epoch-by-epoch progression plots for all metrics."""
    
    if not metrics_data:
        print("No metrics data available for plotting.")
        return
    
    # Organize results by schedule
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    schedule_results = {schedule: [] for schedule in schedules}
    
    for result in metrics_data:
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
            wasserstein = [r['wasserstein_distance'] for r in schedule_results[schedule]]
            ax4.plot(epochs, wasserstein, label=schedule.upper(), color=colors[i], 
                    marker='o', markersize=3, alpha=0.8, linewidth=1.5)
    
    ax4.set_title('Wasserstein Distance Progression', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('Wasserstein Distance (Lower is Better)', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_progression_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Epoch progression plots saved as: epoch_progression_metrics.png")
    
    # Create individual plots for each metric
    create_individual_metric_plots(schedule_results, colors, save_dir)


def create_individual_metric_plots(schedule_results, colors, save_dir):
    """Create individual detailed plots for each metric."""
    
    metrics = [
        ('test_loss', 'Test Loss', 'Test Loss (Lower is Better)'),
        ('mmd_rbf', 'MMD RBF', 'MMD RBF (Lower is Better)'),
        ('mmd_linear', 'MMD Linear', 'MMD Linear (Lower is Better)'),
        ('wasserstein_distance', 'Wasserstein Distance', 'Wasserstein Distance (Lower is Better)')
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
        plt.savefig(f'{save_dir}/{metric_key}_progression.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ {metric_name} progression plot saved as: {metric_key}_progression.png")


def main():
    # Set up device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Using device: {device}")
    
    # Test parameters
    T = 1000
    batch_size = 128
    large_n_samples = 2000  # Reduced for efficiency with 5 workers
    
    # Create output directory
    os.makedirs("test_results_complex", exist_ok=True)
    
    # Generate original complex data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Find all complex checkpoint files
    checkpoint_files = []
    complex_dirs = ["checkpoints_complex"] + [f"checkpoints_complex/{d}" for d in os.listdir("checkpoints_complex") if os.path.isdir(f"checkpoints_complex/{d}")]
    
    for complex_dir in complex_dirs:
        if os.path.exists(complex_dir):
            files = glob.glob(f"{complex_dir}/model_epoch_*.pt")
            checkpoint_files.extend(files)
    
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not checkpoint_files:
        print("No complex checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} complex checkpoint files:")
    for file in checkpoint_files[:5]:  # Show first 5
        print(f"  {file}")
    if len(checkpoint_files) > 5:
        print(f"  ... and {len(checkpoint_files) - 5} more")
    
    # Prepare arguments for parallel processing
    eval_args = [(checkpoint_file, original_data, device, T, batch_size, large_n_samples) 
                 for checkpoint_file in checkpoint_files]
    
    # Use parallel processing with 5 workers
    print(f"\nStarting evaluation with 5 workers...")
    with Pool(processes=5) as pool:
        results = pool.map(evaluate_single_checkpoint, eval_args)
    
    # Filter out None results (failed evaluations)
    metrics_data = [result for result in results if result is not None]
    
    print(f"Successfully evaluated {len(metrics_data)} checkpoints")
    
    # Print summary
    print(f"\n" + "="*80)
    print(f"COMPLEX DATASET EVALUATION SUMMARY")
    print(f"="*80)
    
    # Find best performing checkpoints by different metrics
    if metrics_data:
        best_loss_epoch = min(metrics_data, key=lambda x: x['test_loss'])
        best_mmd_rbf_epoch = min(metrics_data, key=lambda x: x['mmd_rbf'])
        best_mmd_linear_epoch = min(metrics_data, key=lambda x: x['mmd_linear'])
        best_wasserstein_epoch = min(metrics_data, key=lambda x: x['wasserstein_distance'])
        
        print(f"Best Test Loss: Epoch {best_loss_epoch['epoch']:4d} ({best_loss_epoch['schedule_type']}) = {best_loss_epoch['test_loss']:.6f}")
        print(f"Best MMD (RBF): Epoch {best_mmd_rbf_epoch['epoch']:4d} ({best_mmd_rbf_epoch['schedule_type']}) = {best_mmd_rbf_epoch['mmd_rbf']:.6f}")
        print(f"Best MMD (Linear): Epoch {best_mmd_linear_epoch['epoch']:4d} ({best_mmd_linear_epoch['schedule_type']}) = {best_mmd_linear_epoch['mmd_linear']:.6f}")
        print(f"Best Wasserstein: Epoch {best_wasserstein_epoch['epoch']:4d} ({best_wasserstein_epoch['schedule_type']}) = {best_wasserstein_epoch['wasserstein_distance']:.6f}")
    
    # Save metrics to JSON and CSV
    with open('test_results_complex/complex_evaluation_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    with open('test_results_complex/complex_evaluation_metrics.csv', 'w', newline='') as f:
        if metrics_data:
            writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
            writer.writeheader()
            writer.writerows(metrics_data)
    
    # Create epoch progression plots
    print("\nCreating epoch progression plots...")
    create_epoch_progression_plots(metrics_data)
    
    print(f"\nResults saved:")
    print(f"  - Metrics (JSON): test_results_complex/complex_evaluation_metrics.json")
    print(f"  - Metrics (CSV): test_results_complex/complex_evaluation_metrics.csv")
    print(f"  - Epoch progression plots: test_results_complex/*_progression.png")
    print(f"Complex dataset evaluation completed successfully!")


if __name__ == "__main__":
    main() 