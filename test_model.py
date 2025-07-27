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

from model import Denoiser
from data.gmm_dataset import get_dataloader


def get_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for diffusion process."""
    return torch.linspace(beta_start, beta_end, T)


def get_cosine_beta_schedule(T: int, beta_end: float = 0.02) -> torch.Tensor:
    """Cosine beta schedule for diffusion process."""
    steps = torch.linspace(0, torch.pi / 2, T)
    return beta_end * torch.sin(steps) ** 2


def get_quadratic_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Quadratic beta schedule for diffusion process."""
    linear_betas = torch.linspace(beta_start, beta_end, T)
    return linear_betas ** 2


def get_exponential_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Exponential beta schedule for diffusion process."""
    return torch.exp(torch.linspace(np.log(beta_start), np.log(beta_end), T))


def get_beta_schedule(T: int, schedule_type: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Get beta schedule based on the specified type."""
    if schedule_type == "linear":
        return get_linear_beta_schedule(T, beta_start, beta_end)
    elif schedule_type == "cosine":
        return get_cosine_beta_schedule(T, beta_end)
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


def visualize_checkpoint_comparison(original_data, checkpoint_samples, save_path="test_results/checkpoint_comparison.png"):
    """Visualize samples from different checkpoints."""
    num_checkpoints = len(checkpoint_samples)
    
    # For many checkpoints, create a larger grid
    if num_checkpoints <= 20:
        cols = 5
        rows = (num_checkpoints + 1 + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        axes = axes.flatten() if num_checkpoints > 1 else [axes]
    else:
        # For many checkpoints, show selected ones
        selected_epochs = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        selected_samples = {k: v for k, v in checkpoint_samples.items() if k in selected_epochs}
        num_checkpoints = len(selected_samples)
        cols = 4
        rows = (num_checkpoints + 1 + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        axes = axes.flatten()
        checkpoint_samples = selected_samples
    
    # Original data
    axes[0].scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, s=10, c='blue')
    axes[0].set_title('Original GMM Data')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Checkpoint samples
    for i, (epoch, samples) in enumerate(checkpoint_samples.items()):
        ax_idx = i + 1
        if ax_idx < len(axes):
            axes[ax_idx].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10, c='red')
            axes[ax_idx].set_title(f'Epoch {epoch}')
            axes[ax_idx].set_xlabel('X')
            axes[ax_idx].set_ylabel('Y')
            axes[ax_idx].axis('equal')
            axes[ax_idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(checkpoint_samples) + 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Checkpoint comparison saved to: {save_path}")


def main():
    # Set up device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Using device: {device}")
    
    # Test parameters (using large batch defaults)
    T = 1000
    batch_size = 1280  # Large batch default
    n_samples = 1000
    large_n_samples = 5000  # Larger sample set for robust evaluation
    
    # Find all checkpoint files (including subdirectories for different schedules)
    checkpoint_files = []
    checkpoint_dirs = ["checkpoints"] + [f"checkpoints/{d}" for d in os.listdir("checkpoints") if os.path.isdir(f"checkpoints/{d}")]
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            files = glob.glob(f"{checkpoint_dir}/model_epoch_*.pt")
            checkpoint_files.extend(files)
    
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for file in checkpoint_files[:5]:  # Show first 5
        print(f"  {file}")
    if len(checkpoint_files) > 5:
        print(f"  ... and {len(checkpoint_files) - 5} more")
    
    # Create test dataloader for original data
    test_loader = get_dataloader(
        batch_size=batch_size,
        train=True,
        n_samples=n_samples,
        shuffle=True
    )
    
    # Get original data for comparison
    all_original_data = []
    with torch.no_grad():
        for batch_idx, x0 in enumerate(test_loader):
            x0 = x0.to(device)
            all_original_data.append(x0.cpu().numpy())
            if batch_idx >= 0:  # Just get first batch
                break
    all_original_data = np.vstack(all_original_data)
    
    # Initialize metrics storage
    metrics_data = []
    
    # Test each checkpoint
    checkpoint_samples = {}
    checkpoint_losses = {}
    
    print(f"\nTesting all checkpoints with comprehensive evaluation...")
    print("-" * 80)
    
    for checkpoint_file in checkpoint_files:
        # Extract epoch number
        epoch = int(checkpoint_file.split('_')[-1].split('.')[0])
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = Denoiser().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Determine schedule type from checkpoint config
        schedule_type = checkpoint.get('config', {}).get('noise_schedule', {}).get('type', 'linear')
        print(f"Testing epoch {epoch:4d} (schedule: {schedule_type}, training loss: {checkpoint['loss']:.6f})...")
        
        # Create noise schedule based on checkpoint
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
                
                if batch_idx >= 0:  # Just test first batch
                    break
        
        avg_loss = total_loss / num_batches
        checkpoint_losses[epoch] = avg_loss
        
        # Generate small sample set for visualization
        print(f"  Generating small sample set (500 samples) for visualization...")
        small_samples = sample_from_model(model, noise_schedule, device, num_samples=500, batch_size=batch_size)
        checkpoint_samples[epoch] = small_samples
        
        # Generate large sample set for robust evaluation
        print(f"  Generating large sample set ({large_n_samples} samples) for metrics...")
        large_samples = sample_from_model(model, noise_schedule, device, num_samples=large_n_samples, batch_size=batch_size)
        
        # Compute quantitative metrics
        print(f"  Computing MMD and Wasserstein distances...")
        mmd_rbf = compute_mmd(large_samples, all_original_data, kernel='rbf', gamma=1.0)
        mmd_linear = compute_mmd(large_samples, all_original_data, kernel='linear')
        wasserstein_dist = compute_wasserstein_distance(large_samples, all_original_data)
        
        # Store metrics
        metrics_entry = {
            'epoch': epoch,
            'schedule_type': schedule_type,
            'test_loss': avg_loss,
            'training_loss': checkpoint['loss'],
            'mmd_rbf': float(mmd_rbf),
            'mmd_linear': float(mmd_linear),
            'wasserstein_distance': float(wasserstein_dist),
            'large_samples_range': [float(large_samples.min()), float(large_samples.max())],
            'large_samples_std': float(np.std(large_samples))
        }
        metrics_data.append(metrics_entry)
        
        print(f"  Epoch {epoch:4d}: Test loss = {avg_loss:.6f}")
        print(f"    MMD (RBF): {mmd_rbf:.6f}, MMD (Linear): {mmd_linear:.6f}")
        print(f"    Wasserstein: {wasserstein_dist:.6f}")
        print(f"    Large samples: {large_samples.shape[0]} samples, range [{large_samples.min():.3f}, {large_samples.max():.3f}]")
        print(f"    Small samples: {small_samples.shape[0]} samples, range [{small_samples.min():.3f}, {small_samples.max():.3f}]")
    
    # Print summary
    print(f"\n" + "="*80)
    print(f"COMPREHENSIVE EVALUATION SUMMARY")
    print(f"="*80)
    
    # Find best performing checkpoints by different metrics
    best_loss_epoch = min(metrics_data, key=lambda x: x['test_loss'])
    best_mmd_rbf_epoch = min(metrics_data, key=lambda x: x['mmd_rbf'])
    best_mmd_linear_epoch = min(metrics_data, key=lambda x: x['mmd_linear'])
    best_wasserstein_epoch = min(metrics_data, key=lambda x: x['wasserstein_distance'])
    
    print(f"Best Test Loss: Epoch {best_loss_epoch['epoch']:4d} ({best_loss_epoch['schedule_type']}) = {best_loss_epoch['test_loss']:.6f}")
    print(f"Best MMD (RBF): Epoch {best_mmd_rbf_epoch['epoch']:4d} ({best_mmd_rbf_epoch['schedule_type']}) = {best_mmd_rbf_epoch['mmd_rbf']:.6f}")
    print(f"Best MMD (Linear): Epoch {best_mmd_linear_epoch['epoch']:4d} ({best_mmd_linear_epoch['schedule_type']}) = {best_mmd_linear_epoch['mmd_linear']:.6f}")
    print(f"Best Wasserstein: Epoch {best_wasserstein_epoch['epoch']:4d} ({best_wasserstein_epoch['schedule_type']}) = {best_wasserstein_epoch['wasserstein_distance']:.6f}")
    
    # Create visualizations and save results
    os.makedirs("test_results", exist_ok=True)
    
    # Save small samples for visualization
    for epoch, samples in checkpoint_samples.items():
        np.save(f"test_results/samples_epoch_{epoch:04d}.npy", samples)
    
    # Save large samples for robust evaluation
    for metrics_entry in metrics_data:
        epoch = metrics_entry['epoch']
        schedule_type = metrics_entry['schedule_type']
        large_samples = sample_from_model(
            Denoiser().to(device).load_state_dict(
                torch.load(f"checkpoints/model_epoch_{epoch:04d}.pt", map_location=device)['model_state_dict']
            ),
            noise_schedule, device, num_samples=large_n_samples, batch_size=batch_size
        )
        np.save(f"test_results/samples_epoch_{epoch:04d}_large.npy", large_samples)
    
    # Save metrics to JSON and CSV
    with open('test_results/evaluation_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    with open('test_results/evaluation_metrics.csv', 'w', newline='') as f:
        if metrics_data:
            writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
            writer.writeheader()
            writer.writerows(metrics_data)
    
    # Create comparison plot
    visualize_checkpoint_comparison(all_original_data, checkpoint_samples)
    
    print(f"\nResults saved:")
    print(f"  - Small samples: test_results/samples_epoch_*.npy")
    print(f"  - Large samples: test_results/samples_epoch_*_large.npy")
    print(f"  - Metrics (JSON): test_results/evaluation_metrics.json")
    print(f"  - Metrics (CSV): test_results/evaluation_metrics.csv")
    print(f"  - Visualization: test_results/checkpoint_comparison.png")
    print(f"Test completed successfully!")


if __name__ == "__main__":
    main() 