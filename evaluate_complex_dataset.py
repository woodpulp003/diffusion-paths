import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
from data.gmm_dataset import get_dataloader, generate_complex_gmm_data
from model import Denoiser
import warnings
warnings.filterwarnings('ignore')

# Beta schedule functions
def get_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)

def get_cosine_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    steps = torch.linspace(0, 1, T)
    return beta_start + (beta_end - beta_start) * (1 - torch.cos(steps * torch.pi / 2))

def get_quadratic_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    steps = torch.linspace(0, 1, T)
    return beta_start + (beta_end - beta_start) * steps ** 2

def get_exponential_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    steps = torch.linspace(0, 1, T)
    return beta_start + (beta_end - beta_start) * (torch.exp(steps) - 1) / (np.e - 1)

def get_beta_schedule(T: int, schedule_type: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
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

class NoiseSchedule:
    def __init__(self, T: int, schedule_type: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.schedule_type = schedule_type.lower()
        self.betas = self._get_betas()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def _get_betas(self):
        return get_beta_schedule(self.T, self.schedule_type)

def compute_mmd(samples1, samples2, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy between two sets of samples.
    
    Args:
        samples1: numpy array of shape (n_samples, n_features)
        samples2: numpy array of shape (n_samples, n_features)
        kernel: 'rbf' or 'linear'
        gamma: parameter for RBF kernel
    
    Returns:
        MMD value
    """
    if kernel == 'rbf':
        def rbf_kernel(x, y):
            diff = x.reshape(-1, 1, x.shape[-1]) - y.reshape(1, -1, y.shape[-1])
            return np.exp(-gamma * np.sum(diff**2, axis=-1))
        
        # Compute kernel matrices
        K11 = rbf_kernel(samples1, samples1)
        K22 = rbf_kernel(samples2, samples2)
        K12 = rbf_kernel(samples1, samples2)
        
        # Compute MMD
        mmd = np.mean(K11) + np.mean(K22) - 2 * np.mean(K12)
        
    elif kernel == 'linear':
        # Linear kernel: k(x,y) = x^T y
        K11 = np.dot(samples1, samples1.T)
        K22 = np.dot(samples2, samples2.T)
        K12 = np.dot(samples1, samples2.T)
        
        mmd = np.mean(K11) + np.mean(K22) - 2 * np.mean(K12)
    
    return mmd

def compute_wasserstein_distance(samples1, samples2):
    """
    Compute Wasserstein distance (Earth Mover's Distance) between two sets of samples.
    Uses scipy's implementation.
    
    Args:
        samples1: numpy array of shape (n_samples, n_features)
        samples2: numpy array of shape (n_samples, n_features)
    
    Returns:
        Wasserstein distance
    """
    try:
        from scipy.stats import wasserstein_distance
        
        # Compute 1D Wasserstein distances for each dimension and sum them
        wass_dist = 0
        for dim in range(samples1.shape[1]):
            wass_dist += wasserstein_distance(samples1[:, dim], samples2[:, dim])
        
        return wass_dist
    except ImportError:
        print("Warning: scipy not available, using simplified Wasserstein distance")
        # Simplified version: compute mean absolute difference
        return np.mean(np.abs(samples1 - samples2))

def sample_from_model(model, noise_schedule, device, num_samples=5000, batch_size=128):
    """
    Sample from the trained diffusion model using DDPM reverse process.
    
    Args:
        model: trained Denoiser model
        noise_schedule: NoiseSchedule object
        device: torch device
        num_samples: number of samples to generate
        batch_size: batch size for generation
    
    Returns:
        numpy array of generated samples
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Start from pure noise
            x_t = torch.randn(current_batch_size, 2, device=device)
            
            # Reverse diffusion process
            for t in reversed(range(noise_schedule.T)):
                t_tensor = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = model(x_t, t_tensor)
                
                # Get schedule parameters
                alpha_t = noise_schedule.alphas[t]
                alpha_bar_t = noise_schedule.alpha_bars[t]
                beta_t = noise_schedule.betas[t]
                
                # Add small epsilon for numerical stability
                eps = 1e-8
                
                if t > 0:
                    # Add noise for stochastic sampling
                    noise = torch.randn_like(x_t)
                    x_t = (1 / torch.sqrt(alpha_t + eps)) * (
                        x_t - (beta_t / torch.sqrt(1 - alpha_bar_t + eps)) * predicted_noise
                    ) + torch.sqrt(beta_t) * noise
                else:
                    # Final step: no noise
                    x_t = (1 / torch.sqrt(alpha_t + eps)) * (
                        x_t - (beta_t / torch.sqrt(1 - alpha_bar_t + eps)) * predicted_noise
                    )
            
            samples.append(x_t.cpu().numpy())
    
    return np.vstack(samples)

def evaluate_checkpoint(checkpoint_path, original_data, device, num_samples=5000):
    """
    Evaluate a single checkpoint.
    
    Args:
        checkpoint_path: path to checkpoint file
        original_data: original training data
        device: torch device
        num_samples: number of samples to generate
    
    Returns:
        dictionary with evaluation metrics
    """
    print(f"Evaluating checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})
    
    # Determine schedule type from config or filename
    schedule_type = config.get('noise_schedule', {}).get('type', 'linear')
    if 'cosine' in checkpoint_path:
        schedule_type = 'cosine'
    elif 'quadratic' in checkpoint_path:
        schedule_type = 'quadratic'
    elif 'exponential' in checkpoint_path:
        schedule_type = 'exponential'
    
    # Create model and load state
    model = Denoiser(embedding_dim=32)
    model.load_state_dict(model_state)
    model.to(device)
    
    # Create noise schedule
    noise_schedule = NoiseSchedule(T=1000, schedule_type=schedule_type)
    
    # Generate samples
    generated_samples = sample_from_model(model, noise_schedule, device, num_samples)
    
    # Compute metrics
    mmd_rbf = compute_mmd(generated_samples, original_data, kernel='rbf')
    mmd_linear = compute_mmd(generated_samples, original_data, kernel='linear')
    wass_dist = compute_wasserstein_distance(generated_samples, original_data)
    
    # Compute test loss (noise prediction loss on test data)
    model.eval()
    test_loader = get_dataloader(batch_size=128, train=False, shuffle=False, complex_data=True)
    test_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 5:  # Limit to 5 batches for test loss
                break
            x_0 = batch.to(device)
            batch_size = x_0.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, noise_schedule.T, (batch_size,), device=device)
            
            # Sample noise
            noise = torch.randn_like(x_0)
            
            # Calculate alpha_bar_t for each t in the batch
            alpha_bar_t = noise_schedule.alpha_bars[t].unsqueeze(1)
            
            # Add noise to x_0
            x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
            
            # Predict noise
            pred_noise = model(x_t, t)
            
            # Calculate loss
            loss = nn.MSELoss()(pred_noise, noise)
            test_loss += loss.item()
            num_batches += 1
    
    avg_test_loss = test_loss / num_batches if num_batches > 0 else float('inf')
    
    # Extract epoch from filename
    filename = os.path.basename(checkpoint_path)
    if 'epoch' in filename:
        epoch = int(filename.split('epoch_')[1].split('.')[0])
    else:
        epoch = 1000  # Final model
    
    return {
        'checkpoint_path': checkpoint_path,
        'epoch': epoch,
        'schedule_type': schedule_type,
        'mmd_rbf': mmd_rbf,
        'mmd_linear': mmd_linear,
        'wass_dist': wass_dist,
        'test_loss': avg_test_loss,
        'generated_samples': generated_samples,
        'sample_stats': {
            'mean': np.mean(generated_samples, axis=0),
            'std': np.std(generated_samples, axis=0),
            'min': np.min(generated_samples, axis=0),
            'max': np.max(generated_samples, axis=0)
        }
    }

def create_comparison_plots(all_results, original_data, save_dir="test_results_complex"):
    """Create comparison plots for all schedules."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Complex Dataset: Comparison of Noise Schedules', fontsize=16)
    
    # Plot training loss trajectories
    ax1 = axes[0, 0]
    for result in all_results:
        schedule = result['schedule_type']
        epoch = result['epoch']
        test_loss = result['test_loss']
        ax1.scatter(epoch, test_loss, label=schedule, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('Test Loss vs Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MMD RBF
    ax2 = axes[0, 1]
    for result in all_results:
        schedule = result['schedule_type']
        epoch = result['epoch']
        mmd_rbf = result['mmd_rbf']
        ax2.scatter(epoch, mmd_rbf, label=schedule, alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MMD (RBF)')
    ax2.set_title('MMD (RBF) vs Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot MMD Linear
    ax3 = axes[1, 0]
    for result in all_results:
        schedule = result['schedule_type']
        epoch = result['epoch']
        mmd_linear = result['mmd_linear']
        ax3.scatter(epoch, mmd_linear, label=schedule, alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MMD (Linear)')
    ax3.set_title('MMD (Linear) vs Epoch')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot Wasserstein Distance
    ax4 = axes[1, 1]
    for result in all_results:
        schedule = result['schedule_type']
        epoch = result['epoch']
        wass_dist = result['wass_dist']
        ax4.scatter(epoch, wass_dist, label=schedule, alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Wasserstein Distance')
    ax4.set_title('Wasserstein Distance vs Epoch')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complex_dataset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create distribution comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Complex Dataset: Generated vs Original Distributions', fontsize=16)
    
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    for idx, schedule in enumerate(schedules):
        ax = axes[idx // 2, idx % 2]
        
        # Find the final model for this schedule
        schedule_results = [r for r in all_results if r['schedule_type'] == schedule]
        if schedule_results:
            final_result = max(schedule_results, key=lambda x: x['epoch'])
            generated_samples = final_result['generated_samples']
            
            # Plot original data
            ax.scatter(original_data[:, 0], original_data[:, 1], 
                      alpha=0.6, s=1, label='Original', color='blue')
            
            # Plot generated samples
            ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                      alpha=0.6, s=1, label='Generated', color='red')
            
            ax.set_title(f'{schedule.capitalize()} Schedule (Epoch {final_result["epoch"]})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'complex_dataset_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_results, save_dir="test_results_complex"):
    """Create a summary table of the best performing models."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Group results by schedule type
    schedule_results = {}
    for result in all_results:
        schedule = result['schedule_type']
        if schedule not in schedule_results:
            schedule_results[schedule] = []
        schedule_results[schedule].append(result)
    
    # Find best models for each metric
    best_models = {}
    for schedule, results in schedule_results.items():
        if not results:
            continue
            
        # Best by test loss
        best_test_loss = min(results, key=lambda x: x['test_loss'])
        
        # Best by MMD RBF
        best_mmd_rbf = min(results, key=lambda x: x['mmd_rbf'])
        
        # Best by MMD Linear
        best_mmd_linear = min(results, key=lambda x: x['mmd_linear'])
        
        # Best by Wasserstein
        best_wass = min(results, key=lambda x: x['wass_dist'])
        
        best_models[schedule] = {
            'best_test_loss': best_test_loss,
            'best_mmd_rbf': best_mmd_rbf,
            'best_mmd_linear': best_mmd_linear,
            'best_wass': best_wass
        }
    
    # Create summary table
    summary_data = []
    for schedule, bests in best_models.items():
        summary_data.append({
            'Schedule': schedule,
            'Best Test Loss (Epoch)': f"{bests['best_test_loss']['test_loss']:.6f} ({bests['best_test_loss']['epoch']})",
            'Best MMD RBF (Epoch)': f"{bests['best_mmd_rbf']['mmd_rbf']:.6f} ({bests['best_mmd_rbf']['epoch']})",
            'Best MMD Linear (Epoch)': f"{bests['best_mmd_linear']['mmd_linear']:.6f} ({bests['best_mmd_linear']['epoch']})",
            'Best Wasserstein (Epoch)': f"{bests['best_wass']['wass_dist']:.6f} ({bests['best_wass']['epoch']})"
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(save_dir, 'complex_dataset_summary.csv'), index=False)
    
    print("\n" + "="*80)
    print("COMPLEX DATASET EVALUATION SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate complex dataset models')
    parser.add_argument('--num_workers', type=int, default=20, help='Number of parallel workers')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to generate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_complex', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original complex data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    print(f"Original data shape: {original_data.shape}")
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(args.checkpoint_dir, "*", "model_epoch_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {args.checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Evaluate checkpoints
    print(f"Starting evaluation with {args.num_workers} workers...")
    
    # Prepare arguments for parallel processing
    eval_args = [(checkpoint_path, original_data, device, args.num_samples) 
                 for checkpoint_path in checkpoint_files]
    
    # Use parallel processing
    with Pool(processes=min(args.num_workers, cpu_count())) as pool:
        all_results = pool.starmap(evaluate_checkpoint, eval_args)
    
    print(f"Evaluation completed for {len(all_results)} checkpoints")
    
    # Create plots and summary
    create_comparison_plots(all_results, original_data)
    create_summary_table(all_results)
    
    print("Evaluation completed! Results saved to test_results_complex/")

if __name__ == "__main__":
    main() 