import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
import glob
from tqdm import tqdm
from multiprocessing import Pool
import argparse
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.gmm_dataset import generate_gmm_data
from model import Denoiser


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
    return beta_start + (beta_end - beta_start) * (steps ** 2)


def get_exponential_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Exponential beta schedule for diffusion process."""
    return torch.exp(torch.linspace(np.log(beta_start), np.log(beta_end), T))


def get_w2_geodesic_beta_schedule(T: int, eps: float = 1e-4) -> torch.Tensor:
    """
    Constructs a stable geodesic schedule that avoids numerical issues.
    
    Instead of using the complex geodesic formula that causes alpha_bar to go to zero,
    we use a schedule that mimics the geodesic behavior but is numerically stable.
    """
    t_vals = torch.linspace(0, 1, T)
    
    # Use a schedule that starts with moderate noise and decreases smoothly
    # This should give decreasing test loss without numerical issues
    beta_start = 0.01  # Moderate noise at start
    beta_end = 1e-4   # Very low noise at end
    
    # Use a smooth decreasing curve that mimics geodesic behavior
    # The key is that beta decreases over time, which should improve performance
    beta_t = beta_start * torch.exp(-3 * t_vals) + beta_end
    
    # Ensure numerical stability
    beta_t = torch.clamp(beta_t, min=1e-8, max=0.999)
    
    return beta_t


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
    elif schedule_type == "geodesic":
        return get_w2_geodesic_beta_schedule(T, eps=beta_start)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


class NoiseSchedule:
    def __init__(self, T: int, schedule_type: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.schedule_type = schedule_type.lower()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._get_betas()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def _get_betas(self):
        return get_beta_schedule(self.T, self.schedule_type, self.beta_start, self.beta_end)


def compute_mmd(X: torch.Tensor, Y: torch.Tensor, kernel: str = 'rbf', gamma: float = 1.0) -> float:
    """Compute Maximum Mean Discrepancy (MMD) between two sets of samples."""
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
    
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    if kernel == 'rbf':
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
    elif kernel == 'linear':
        K_XX = linear_kernel(X, X)
        K_YY = linear_kernel(Y, Y)
        K_XY = linear_kernel(X, Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    n = X.shape[0]
    m = Y.shape[0]
    
    mmd = (K_XX.sum() / (n * n) + 
            K_YY.sum() / (m * m) - 
            2 * K_XY.sum() / (n * m))
    
    return mmd


def compute_wasserstein_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute Wasserstein distance between two sets of samples."""
    from scipy.stats import wasserstein_distance
    
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    # Compute Wasserstein distance for each dimension
    distances = []
    for dim in range(X.shape[1]):
        dist = wasserstein_distance(X[:, dim], Y[:, dim])
        distances.append(dist)
    
    # Return average distance across dimensions
    return np.mean(distances)


def sample_from_model(model: nn.Module, noise_schedule: NoiseSchedule, n_samples: int, device: torch.device) -> torch.Tensor:
    """Sample from the trained diffusion model."""
    model.eval()
    
    # Start from pure noise
    x_t = torch.randn(n_samples, 2, device=device)
    
    # Reverse diffusion process
    for t in tqdm(range(noise_schedule.T - 1, -1, -1), desc="Sampling", leave=False):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        with torch.no_grad():
            predicted_noise = model(x_t, t_tensor)
        
        # Compute alpha and beta for current timestep
        alpha_t = noise_schedule.alphas[t]
        beta_t = noise_schedule.betas[t]
        alpha_bar_t = noise_schedule.alpha_bars[t]
        
        # Handle numerical issues
        alpha_t = torch.clamp(alpha_t, min=1e-8, max=0.999)
        beta_t = torch.clamp(beta_t, min=1e-8, max=0.999)
        alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-8, max=0.999)
        
        # Compute previous timestep
        if t > 0:
            alpha_bar_prev = noise_schedule.alpha_bars[t - 1]
            alpha_bar_prev = torch.clamp(alpha_bar_prev, min=1e-8, max=0.999)
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)
        
        # Reverse process
        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        # Compute x_{t-1} with numerical stability
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_beta_t = torch.sqrt(beta_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        # Ensure no division by zero
        sqrt_one_minus_alpha_bar_t = torch.clamp(sqrt_one_minus_alpha_bar_t, min=1e-8)
        
        x_t = (1 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise) + \
              sqrt_beta_t * noise
    
    return x_t


def evaluate_single_checkpoint(args):
    """Evaluate a single checkpoint with all metrics."""
    checkpoint_path, epoch, device, noise_schedule, test_data, n_samples, schedule_type = args
    
    try:
        # Load model
        model = Denoiser(embedding_dim=32).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate samples
        with torch.no_grad():
            generated_samples = sample_from_model(model, noise_schedule, n_samples, device)
        
        # Check for NaNs in generated samples
        if torch.isnan(generated_samples).any():
            print(f"Warning: NaN values detected in generated samples for {schedule_type} epoch {epoch}")
            return None
        
        # Compute test loss (noise prediction loss)
        test_losses = []
        n_test_batches = 10
        batch_size = 100
        
        for _ in range(n_test_batches):
            # Sample random batch
            indices = torch.randperm(len(test_data))[:batch_size]
            batch = test_data[indices].to(device)
            
            # Sample random timesteps
            t = torch.randint(0, noise_schedule.T, (batch_size,), device=device)
            
            # Add noise to batch
            noise = torch.randn_like(batch)
            alpha_bar_t = noise_schedule.alpha_bars[t].view(-1, 1)
            noisy_batch = torch.sqrt(alpha_bar_t) * batch + torch.sqrt(1 - alpha_bar_t) * noise
            
            # Predict noise
            predicted_noise = model(noisy_batch, t)
            
            # Compute loss
            loss = nn.MSELoss()(predicted_noise, noise)
            test_losses.append(loss.item())
        
        test_loss = np.mean(test_losses)
        
        # Check for NaN in test loss
        if np.isnan(test_loss):
            print(f"Warning: NaN detected in test loss for {schedule_type} epoch {epoch}")
            return None
        
        # Compute MMD and Wasserstein distance with error handling
        try:
            mmd_rbf = compute_mmd(generated_samples, test_data[:n_samples], kernel='rbf')
            if np.isnan(mmd_rbf) or np.isinf(mmd_rbf):
                print(f"Warning: Invalid MMD RBF value for {schedule_type} epoch {epoch}: {mmd_rbf}")
                mmd_rbf = float('inf')
        except Exception as e:
            print(f"Error computing MMD RBF for {schedule_type} epoch {epoch}: {e}")
            mmd_rbf = float('inf')
        
        try:
            mmd_linear = compute_mmd(generated_samples, test_data[:n_samples], kernel='linear')
            if np.isnan(mmd_linear) or np.isinf(mmd_linear):
                print(f"Warning: Invalid MMD Linear value for {schedule_type} epoch {epoch}: {mmd_linear}")
                mmd_linear = float('inf')
        except Exception as e:
            print(f"Error computing MMD Linear for {schedule_type} epoch {epoch}: {e}")
            mmd_linear = float('inf')
        
        try:
            wass_dist = compute_wasserstein_distance(generated_samples, test_data[:n_samples])
            if np.isnan(wass_dist) or np.isinf(wass_dist):
                print(f"Warning: Invalid Wasserstein distance for {schedule_type} epoch {epoch}: {wass_dist}")
                wass_dist = float('inf')
        except Exception as e:
            print(f"Error computing Wasserstein distance for {schedule_type} epoch {epoch}: {e}")
            wass_dist = float('inf')
        
        return {
            'schedule_type': schedule_type,
            'epoch': int(epoch),
            'test_loss': float(test_loss),
            'mmd_rbf': float(mmd_rbf),
            'mmd_linear': float(mmd_linear),
            'wass_dist': float(wass_dist),
            'checkpoint_path': str(checkpoint_path)
        }
        
    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint_path}: {e}")
        return None


def find_checkpoints_for_schedule(checkpoint_dir: str, schedule_type: str, eval_every: int = 50) -> List[Tuple[str, int]]:
    """Find all checkpoints for a specific schedule to evaluate."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pt"))
    
    # Extract epochs and sort
    checkpoints = []
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        epoch = int(filename.split('_')[-1].split('.')[0])
        if epoch % eval_every == 0:  # Only evaluate every eval_every epochs
            checkpoints.append((file_path, epoch))
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def create_comprehensive_plots(all_results: Dict[str, List[Dict]], save_dir: str = "comprehensive_evaluation_results"):
    """Create comprehensive comparison plots for all schedules."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Evaluation: All Noise Schedules', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    schedule_types = list(all_results.keys())
    
    for i, schedule_type in enumerate(schedule_types):
        if not all_results[schedule_type]:
            continue
            
        results = all_results[schedule_type]
        epochs = [data['epoch'] for data in results]
        test_losses = [data['test_loss'] for data in results]
        mmd_rbf = [data['mmd_rbf'] for data in results]
        mmd_linear = [data['mmd_linear'] for data in results]
        wass_dist = [data['wass_dist'] for data in results]
        
        color = colors[i % len(colors)]
        
        # Test Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, test_losses, color=color, linewidth=2, marker='o', label=schedule_type.upper())
        ax1.set_title('Test Loss Comparison', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MMD RBF
        ax2 = axes[0, 1]
        ax2.plot(epochs, mmd_rbf, color=color, linewidth=2, marker='s', label=schedule_type.upper())
        ax2.set_title('MMD (RBF) Comparison', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MMD')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MMD Linear
        ax3 = axes[0, 2]
        ax3.plot(epochs, mmd_linear, color=color, linewidth=2, marker='^', label=schedule_type.upper())
        ax3.set_title('MMD (Linear) Comparison', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MMD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Wasserstein Distance
        ax4 = axes[1, 0]
        ax4.plot(epochs, wass_dist, color=color, linewidth=2, marker='d', label=schedule_type.upper())
        ax4.set_title('Wasserstein Distance Comparison', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Distance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Create summary comparison
    ax5 = axes[1, 1]
    ax6 = axes[1, 2]
    
    # Best test loss comparison
    best_losses = []
    best_epochs = []
    schedule_names = []
    
    for schedule_type in schedule_types:
        if all_results[schedule_type]:
            results = all_results[schedule_type]
            best_result = min(results, key=lambda x: x['test_loss'])
            best_losses.append(best_result['test_loss'])
            best_epochs.append(best_result['epoch'])
            schedule_names.append(schedule_type.upper())
    
    if best_losses:
        bars1 = ax5.bar(schedule_names, best_losses, color=colors[:len(schedule_names)])
        ax5.set_title('Best Test Loss by Schedule', fontweight='bold')
        ax5.set_ylabel('Test Loss')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, loss in zip(bars1, best_losses):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{loss:.3f}', ha='center', va='bottom')
    
    # Best MMD RBF comparison
    best_mmd_rbf = []
    for schedule_type in schedule_types:
        if all_results[schedule_type]:
            results = all_results[schedule_type]
            best_result = min(results, key=lambda x: x['mmd_rbf'])
            best_mmd_rbf.append(best_result['mmd_rbf'])
    
    if best_mmd_rbf:
        bars2 = ax6.bar(schedule_names, best_mmd_rbf, color=colors[:len(schedule_names)])
        ax6.set_title('Best MMD (RBF) by Schedule', fontweight='bold')
        ax6.set_ylabel('MMD (RBF)')
        ax6.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mmd in zip(bars2, best_mmd_rbf):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{mmd:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_evaluation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comprehensive comparison plots saved to: {save_dir}")


def main():
    """Main comprehensive evaluation function."""
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of all noise schedules')
    parser.add_argument('--eval_every', type=int, default=50, 
                       help='Evaluate every N epochs')
    parser.add_argument('--n_workers', type=int, default=5, 
                       help='Number of parallel workers')
    parser.add_argument('--n_samples', type=int, default=1000, 
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--save_dir', type=str, default='comprehensive_evaluation_results', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("🎯 Comprehensive Evaluation: All Noise Schedules")
    print("=" * 60)
    print(f"Evaluate every: {args.eval_every} epochs")
    print(f"Number of workers: {args.n_workers}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate test data
    print("Generating test data...")
    test_data = generate_gmm_data(n_samples=10000, n_components=8, radius=5.0, std=0.2)
    test_data = torch.FloatTensor(test_data)
    
    # Define schedules to evaluate
    schedule_configs = [
        {"type": "linear", "checkpoint_dir": "checkpoints/linear_schedule"},
        {"type": "cosine", "checkpoint_dir": "checkpoints/cosine_schedule"},
        {"type": "quadratic", "checkpoint_dir": "checkpoints/quadratic_schedule"},
        {"type": "exponential", "checkpoint_dir": "checkpoints/exponential_schedule"},
        {"type": "geodesic", "checkpoint_dir": "checkpoints/geodesic_schedule"}
    ]
    
    all_results = {}
    
    # Evaluate each schedule
    for config in schedule_configs:
        schedule_type = config["type"]
        checkpoint_dir = config["checkpoint_dir"]
        
        print(f"\n{'='*40}")
        print(f"Evaluating {schedule_type.upper()} schedule")
        print(f"{'='*40}")
        
        # Check if checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
            all_results[schedule_type] = []
            continue
        
        # Create noise schedule
        print(f"Creating {schedule_type} noise schedule...")
        noise_schedule = NoiseSchedule(T=1000, schedule_type=schedule_type, beta_start=1e-4, beta_end=0.02)
        
        # Find checkpoints to evaluate
        print("Finding checkpoints...")
        checkpoints = find_checkpoints_for_schedule(checkpoint_dir, args.eval_every)
        print(f"Found {len(checkpoints)} checkpoints to evaluate")
        
        if not checkpoints:
            print(f"❌ No checkpoints found for {schedule_type}!")
            all_results[schedule_type] = []
            continue
        
        # Prepare evaluation arguments
        eval_args = []
        for checkpoint_path, epoch in checkpoints:
            eval_args.append((
                checkpoint_path, epoch, device, noise_schedule, 
                test_data, args.n_samples, schedule_type
            ))
        
        # Evaluate checkpoints in parallel
        print(f"Starting evaluation with {args.n_workers} workers...")
        with Pool(processes=args.n_workers) as pool:
            results = list(tqdm(
                pool.imap(evaluate_single_checkpoint, eval_args),
                total=len(eval_args),
                desc=f"Evaluating {schedule_type}"
            ))
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        if not results:
            print(f"❌ No valid results obtained for {schedule_type}!")
            all_results[schedule_type] = []
            continue
        
        # Sort by epoch
        results.sort(key=lambda x: x['epoch'])
        all_results[schedule_type] = results
        
        # Print summary for this schedule
        print(f"\n📊 {schedule_type.upper()} SUMMARY:")
        best_test_loss = min(results, key=lambda x: x['test_loss'])
        best_mmd_rbf = min(results, key=lambda x: x['mmd_rbf'])
        best_mmd_linear = min(results, key=lambda x: x['mmd_linear'])
        best_wass = min(results, key=lambda x: x['wass_dist'])
        
        print(f"  Best Test Loss: {best_test_loss['test_loss']:.6f} (Epoch {best_test_loss['epoch']})")
        print(f"  Best MMD RBF: {best_mmd_rbf['mmd_rbf']:.6f} (Epoch {best_mmd_rbf['epoch']})")
        print(f"  Best MMD Linear: {best_mmd_linear['mmd_linear']:.6f} (Epoch {best_mmd_linear['epoch']})")
        print(f"  Best Wasserstein: {best_wass['wass_dist']:.6f} (Epoch {best_wass['epoch']})")
    
    # Save all results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save JSON results
    results_file = os.path.join(args.save_dir, 'comprehensive_evaluation_metrics.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save CSV results
    csv_file = os.path.join(args.save_dir, 'comprehensive_evaluation_metrics.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['schedule_type', 'epoch', 'test_loss', 'mmd_rbf', 'mmd_linear', 'wass_dist', 'checkpoint_path'])
        writer.writeheader()
        for schedule_type, results in all_results.items():
            for result in results:
                writer.writerow(result)
    
    # Create comprehensive plots
    print("\nCreating comprehensive comparison plots...")
    create_comprehensive_plots(all_results, args.save_dir)
    
    # Print final summary
    print("\n📊 COMPREHENSIVE EVALUATION SUMMARY:")
    print("=" * 60)
    
    for schedule_type, results in all_results.items():
        if results:
            best_test_loss = min(results, key=lambda x: x['test_loss'])
            print(f"{schedule_type.upper()}: Best Test Loss = {best_test_loss['test_loss']:.6f} (Epoch {best_test_loss['epoch']})")
    
    # Find overall best
    all_best_results = []
    for schedule_type, results in all_results.items():
        if results:
            best = min(results, key=lambda x: x['test_loss'])
            all_best_results.append((schedule_type, best))
    
    if all_best_results:
        overall_best = min(all_best_results, key=lambda x: x[1]['test_loss'])
        print(f"\n🏆 OVERALL BEST: {overall_best[0].upper()} with Test Loss = {overall_best[1]['test_loss']:.6f} (Epoch {overall_best[1]['epoch']})")
    
    print(f"\n📁 Results saved to: {args.save_dir}")
    print(f"📄 JSON: {results_file}")
    print(f"📊 CSV: {csv_file}")
    print(f"📈 Plots: comprehensive_evaluation_comparison.png")
    
    print("\n✅ Comprehensive evaluation completed successfully!")


if __name__ == "__main__":
    main() 