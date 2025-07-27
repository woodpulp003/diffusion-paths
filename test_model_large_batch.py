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
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from model import Denoiser
from data.gmm_dataset import get_dataloader


def get_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for diffusion process."""
    return torch.linspace(beta_start, beta_end, T)


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


def visualize_checkpoint_comparison(original_data, checkpoint_samples, save_path="test_results_large_batch/checkpoint_comparison.png"):
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


def create_individual_plots(original_data, checkpoint_samples, save_dir="test_results_large_batch/individual_plots"):
    """Create individual plots for each checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch, samples in checkpoint_samples.items():
        plt.figure(figsize=(10, 8))
        
        # Plot original data in background
        plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.3, s=5, c='blue', label='Original Data')
        
        # Plot generated samples
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.7, s=10, c='red', label='Generated Samples')
        
        plt.title(f'Large Batch Diffusion Model Output - Epoch {epoch}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = os.path.join(save_dir, f'epoch_{epoch:04d}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Individual plots saved to: {save_dir}")


def create_metrics_plots(metrics_data, save_dir="test_results_large_batch"):
    """Create plots showing the evolution of metrics over epochs."""
    epochs = [m['epoch'] for m in metrics_data if 'error' not in m]
    test_losses = [m['test_loss'] for m in metrics_data if 'error' not in m]
    training_losses = [m['training_loss'] for m in metrics_data if 'error' not in m]
    mmd_rbf_values = [m['mmd_rbf'] for m in metrics_data if 'error' not in m]
    mmd_linear_values = [m['mmd_linear'] for m in metrics_data if 'error' not in m]
    wasserstein_values = [m['wasserstein_distance'] for m in metrics_data if 'error' not in m]
    
    # Learning progress plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, test_losses, 'b-', label='Test Loss')
    plt.plot(epochs, training_losses, 'r-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Large Batch Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, mmd_rbf_values, 'g-', label='MMD (RBF)')
    plt.xlabel('Epoch')
    plt.ylabel('MMD')
    plt.title('MMD (RBF) Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, mmd_linear_values, 'm-', label='MMD (Linear)')
    plt.xlabel('Epoch')
    plt.ylabel('MMD')
    plt.title('MMD (Linear) Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, wasserstein_values, 'c-', label='Wasserstein Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.title('Wasserstein Distance Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution comparison
    plt.subplot(2, 3, 5)
    plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, s=1, c='blue', label='Original')
    plt.title('Original Data Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Best model samples (lowest MMD RBF)
    best_idx = np.argmin(mmd_rbf_values)
    best_epoch = epochs[best_idx]
    best_samples = [m['small_samples'] for m in metrics_data if m.get('epoch') == best_epoch][0]
    
    plt.subplot(2, 3, 6)
    plt.scatter(best_samples[:, 0], best_samples[:, 1], alpha=0.6, s=1, c='red', label=f'Best Model (Epoch {best_epoch})')
    plt.title(f'Best Large Batch Model Distribution (Epoch {best_epoch})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning progress plot saved to: {os.path.join(save_dir, 'learning_progress.png')}")


def evaluate_single_checkpoint(checkpoint_file, device_id, original_data, noise_schedule, 
                             batch_size=128, n_samples=1000, large_n_samples=5000):
    """
    Evaluate a single checkpoint. This function will be run in parallel.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        device_id: GPU device ID to use (or 'cpu')
        original_data: Original training data for comparison
        noise_schedule: Noise schedule parameters
        batch_size: Batch size for sampling
        n_samples: Number of samples for small evaluation
        large_n_samples: Number of samples for robust evaluation
    
    Returns:
        Dictionary with evaluation results
    """
    # Set device
    if device_id == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_id}')
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Extract epoch number
    epoch = int(checkpoint_file.split('_')[-1].split('.')[0])
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model = Denoiser().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Worker {device_id}: Testing epoch {epoch:4d} (training loss: {checkpoint['loss']:.6f})...")
        
        # Test noise prediction
        total_loss = 0.0
        num_batches = 0
        loss_fn = nn.MSELoss()
        
        # Create test dataloader
        test_loader = get_dataloader(
            batch_size=batch_size,
            train=True,
            n_samples=n_samples,
            shuffle=True
        )
        
        T = len(noise_schedule['betas'])
        betas = noise_schedule['betas'].to(device)
        alpha_bar = noise_schedule['alpha_bars'].to(device)
        
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
        
        # Generate small sample set for visualization
        print(f"Worker {device_id}: Generating small sample set (500 samples) for visualization...")
        small_samples = sample_from_model(model, noise_schedule, device, num_samples=500, batch_size=batch_size)
        
        # Generate large sample set for robust evaluation
        print(f"Worker {device_id}: Generating large sample set ({large_n_samples} samples) for metrics...")
        large_samples = sample_from_model(model, noise_schedule, device, num_samples=large_n_samples, batch_size=batch_size)
        
        # Compute quantitative metrics
        print(f"Worker {device_id}: Computing MMD and Wasserstein distances...")
        mmd_rbf = compute_mmd(large_samples, original_data, kernel='rbf', gamma=1.0)
        mmd_linear = compute_mmd(large_samples, original_data, kernel='linear')
        wasserstein_dist = compute_wasserstein_distance(large_samples, original_data)
        
        # Store results
        results = {
            'epoch': epoch,
            'test_loss': avg_loss,
            'training_loss': checkpoint['loss'],
            'mmd_rbf': float(mmd_rbf),
            'mmd_linear': float(mmd_linear),
            'wasserstein_distance': float(wasserstein_dist),
            'large_samples_range': [float(large_samples.min()), float(large_samples.max())],
            'large_samples_std': float(np.std(large_samples)),
            'small_samples': small_samples,
            'large_samples': large_samples,
            'checkpoint_file': checkpoint_file
        }
        
        print(f"Worker {device_id}: Epoch {epoch:4d}: Test loss = {avg_loss:.6f}")
        print(f"Worker {device_id}:   MMD (RBF): {mmd_rbf:.6f}, MMD (Linear): {mmd_linear:.6f}")
        print(f"Worker {device_id}:   Wasserstein: {wasserstein_dist:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Worker {device_id}: Error processing {checkpoint_file}: {str(e)}")
        return {
            'epoch': epoch,
            'error': str(e),
            'checkpoint_file': checkpoint_file
        }


def process_worker_files(files, device_id, original_data, noise_schedule, 
                        batch_size, n_samples, large_n_samples):
    """Process a list of files for a single worker."""
    results = []
    for file in files:
        result = evaluate_single_checkpoint(
            file, device_id, original_data, noise_schedule,
            batch_size, n_samples, large_n_samples
        )
        results.append(result)
    return results


def main():
    # Set up device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Using device: {device}")
    
    # Test parameters
    T = 1000
    batch_size = 128
    n_samples = 1000
    large_n_samples = 5000  # Larger sample set for robust evaluation
    num_workers = 10  # Number of parallel workers
    
    # Create noise schedule (same as training)
    betas = get_linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02)
    alpha = 1. - betas
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # Create noise schedule dict for sampling
    noise_schedule = {
        'betas': betas,
        'alphas': alpha,
        'alpha_bars': alpha_bar
    }
    
    # Find all checkpoint files from large batch training
    checkpoint_files = glob.glob("checkpoints_large_batch/model_epoch_*.pt")
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not checkpoint_files:
        print("No large batch checkpoint files found!")
        return
    
    print(f"Found {len(checkpoint_files)} large batch checkpoint files:")
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
    
    print(f"Original data shape: {all_original_data.shape}")
    
    # Prepare for parallel processing
    print(f"\nStarting parallel evaluation with {num_workers} workers...")
    print("-" * 80)
    
    # Determine device assignment for workers
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        device_ids = [i % num_gpus for i in range(num_workers)]
    else:
        device_ids = ['cpu'] * num_workers
    
    # Split checkpoint files among workers
    files_per_worker = len(checkpoint_files) // num_workers
    remainder = len(checkpoint_files) % num_workers
    
    worker_files = []
    start_idx = 0
    for i in range(num_workers):
        end_idx = start_idx + files_per_worker + (1 if i < remainder else 0)
        worker_files.append(checkpoint_files[start_idx:end_idx])
        start_idx = end_idx
    
    print(f"Distributed {len(checkpoint_files)} files among {num_workers} workers:")
    for i, files in enumerate(worker_files):
        print(f"  Worker {i}: {len(files)} files (device: {device_ids[i]})")
    
    # Process checkpoints in parallel
    start_time = time.time()
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_worker = {}
        for worker_id, files in enumerate(worker_files):
            if files:  # Only submit if there are files to process
                future = executor.submit(
                    process_worker_files,
                    files,
                    device_ids[worker_id],
                    all_original_data,
                    noise_schedule,
                    batch_size,
                    n_samples,
                    large_n_samples
                )
                future_to_worker[future] = worker_id
        
        # Collect results
        for future in as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"Worker {worker_id} completed with {len(results)} results")
            except Exception as e:
                print(f"Worker {worker_id} failed: {str(e)}")
    
    end_time = time.time()
    print(f"\nParallel evaluation completed in {end_time - start_time:.2f} seconds")
    
    # Filter out error results and sort by epoch
    valid_results = [r for r in all_results if 'error' not in r]
    valid_results.sort(key=lambda x: x['epoch'])
    
    print(f"Successfully processed {len(valid_results)} checkpoints")
    
    # Create output directory
    os.makedirs("test_results_large_batch", exist_ok=True)
    
    # Save results
    for result in valid_results:
        epoch = result['epoch']
        
        # Save small samples for visualization
        np.save(f"test_results_large_batch/samples_epoch_{epoch:04d}.npy", result['small_samples'])
        
        # Save large samples for robust evaluation
        np.save(f"test_results_large_batch/samples_epoch_{epoch:04d}_large.npy", result['large_samples'])
        
        # Remove large arrays from result dict to save memory
        result.pop('small_samples', None)
        result.pop('large_samples', None)
    
    # Save metrics to JSON and CSV
    with open('test_results_large_batch/evaluation_metrics.json', 'w') as f:
        json.dump(valid_results, f, indent=2)
    
    with open('test_results_large_batch/evaluation_metrics.csv', 'w', newline='') as f:
        if valid_results:
            writer = csv.DictWriter(f, fieldnames=valid_results[0].keys())
            writer.writeheader()
            writer.writerows(valid_results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Prepare data for visualization
    checkpoint_samples = {}
    for result in valid_results:
        epoch = result['epoch']
        samples = np.load(f"test_results_large_batch/samples_epoch_{epoch:04d}.npy")
        checkpoint_samples[epoch] = samples
    
    # Create comparison plot
    visualize_checkpoint_comparison(all_original_data, checkpoint_samples)
    
    # Create individual plots
    create_individual_plots(all_original_data, checkpoint_samples)
    
    # Create metrics plots
    create_metrics_plots(valid_results)
    
    # Print summary
    print(f"\n" + "="*80)
    print(f"LARGE BATCH PARALLEL EVALUATION SUMMARY")
    print(f"="*80)
    
    # Find best performing checkpoints by different metrics
    best_loss_epoch = min(valid_results, key=lambda x: x['test_loss'])
    best_mmd_rbf_epoch = min(valid_results, key=lambda x: x['mmd_rbf'])
    best_mmd_linear_epoch = min(valid_results, key=lambda x: x['mmd_linear'])
    best_wasserstein_epoch = min(valid_results, key=lambda x: x['wasserstein_distance'])
    
    print(f"Best Test Loss: Epoch {best_loss_epoch['epoch']:4d} = {best_loss_epoch['test_loss']:.6f}")
    print(f"Best MMD (RBF): Epoch {best_mmd_rbf_epoch['epoch']:4d} = {best_mmd_rbf_epoch['mmd_rbf']:.6f}")
    print(f"Best MMD (Linear): Epoch {best_mmd_linear_epoch['epoch']:4d} = {best_mmd_linear_epoch['mmd_linear']:.6f}")
    print(f"Best Wasserstein: Epoch {best_wasserstein_epoch['epoch']:4d} = {best_wasserstein_epoch['wasserstein_distance']:.6f}")
    
    print(f"\nResults saved:")
    print(f"  - Small samples: test_results_large_batch/samples_epoch_*.npy")
    print(f"  - Large samples: test_results_large_batch/samples_epoch_*_large.npy")
    print(f"  - Metrics (JSON): test_results_large_batch/evaluation_metrics.json")
    print(f"  - Metrics (CSV): test_results_large_batch/evaluation_metrics.csv")
    print(f"  - Visualization: test_results_large_batch/checkpoint_comparison.png")
    print(f"  - Individual plots: test_results_large_batch/individual_plots/")
    print(f"  - Learning progress: test_results_large_batch/learning_progress.png")
    print(f"Large batch parallel test completed successfully!")


if __name__ == "__main__":
    main() 