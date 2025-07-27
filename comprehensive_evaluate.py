import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
from data.gmm_dataset import generate_gmm_data, get_dataloader
from model import Denoiser, get_timestep_embedding
import multiprocessing as mp
from functools import partial
import time

# Beta schedule functions
def get_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)

def get_cosine_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    steps = torch.linspace(0, 1, T)
    return beta_start + (beta_end - beta_start) * (1 - torch.cos(steps * torch.pi / 2))

def get_quadratic_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    steps = torch.linspace(0, 1, T)
    return beta_start + (beta_end - beta_start) * (steps ** 2)

def get_exponential_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.exp(torch.linspace(torch.log(torch.tensor(beta_start)), torch.log(torch.tensor(beta_end)), T))

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

def compute_mmd(samples1, samples2, kernel='rbf', gamma=1.0):
    """Compute Maximum Mean Discrepancy between two sets of samples."""
    if kernel == 'rbf':
        def rbf_kernel(x, y):
            x_norm = np.sum(x**2, axis=1, keepdims=True)
            y_norm = np.sum(y**2, axis=1, keepdims=True)
            xy = np.dot(x, y.T)
            return np.exp(-gamma * (x_norm + y_norm.T - 2*xy))
        
        K_xx = rbf_kernel(samples1, samples1)
        K_yy = rbf_kernel(samples2, samples2)
        K_xy = rbf_kernel(samples1, samples2)
        
        mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
        return mmd
    
    elif kernel == 'linear':
        def linear_kernel(x, y):
            return np.dot(x, y.T)
        
        K_xx = linear_kernel(samples1, samples1)
        K_yy = linear_kernel(samples2, samples2)
        K_xy = linear_kernel(samples1, samples2)
        
        mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
        return mmd

def compute_wasserstein_distance(samples1, samples2):
    """Compute Wasserstein distance between two sets of samples."""
    # Compute 1D Wasserstein distances for each dimension and sum them
    distances = []
    for dim in range(samples1.shape[1]):
        dist = wasserstein_distance(samples1[:, dim], samples2[:, dim])
        distances.append(dist)
    return sum(distances)

def sample_from_model(model, noise_schedule, device, num_samples=5000, batch_size=128):
    """Sample from the trained model using DDPM reverse process."""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Start from pure noise
            x_t = torch.randn(current_batch_size, 2, device=device)
            
            # Reverse diffusion process
            for t in range(len(noise_schedule) - 1, -1, -1):
                alpha_t = 1 - noise_schedule[t]
                alpha_bar_t = torch.prod(1 - noise_schedule[:t+1]) if t > 0 else 1 - noise_schedule[0]
                beta_t = noise_schedule[t]
                
                # Predict noise
                t_tensor = torch.full((current_batch_size,), t, dtype=torch.long, device=device)
                pred_noise = model(x_t, t_tensor)
                
                # Reverse equation
                if t > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise) + torch.sqrt(beta_t) * noise
            
            samples.append(x_t.cpu().numpy())
    
    return np.vstack(samples)

def evaluate_checkpoint(checkpoint_path, schedule_type, original_data, device):
    """Evaluate a single checkpoint and return metrics."""
    print(f"Evaluating {schedule_type} checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})
    
    # Create model and load state
    model = Denoiser(embedding_dim=32)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # Get noise schedule
    T = config.get('noise_schedule', {}).get('T', 1000)
    noise_schedule = get_beta_schedule(T, schedule_type)
    
    # Compute test loss
    test_loss = 0.0
    num_batches = 0
    
    # Create test data loader
    test_loader = get_dataloader(batch_size=128, train=False, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_0 = batch.to(device)
            batch_size = x_0.shape[0]
            
            # Random timestep for each sample in batch
            t = torch.randint(0, T, (batch_size,), device=device)
            
            # Add noise for each sample individually
            noise = torch.randn_like(x_0)
            x_t = torch.zeros_like(x_0)
            
            for i in range(batch_size):
                t_i = t[i]
                alpha_bar_t = torch.prod(1 - noise_schedule[:t_i+1]) if t_i > 0 else 1 - noise_schedule[0]
                x_t[i] = torch.sqrt(alpha_bar_t) * x_0[i] + torch.sqrt(1 - alpha_bar_t) * noise[i]
            
            # Predict noise
            pred_noise = model(x_t, t)
            
            # Compute loss
            loss = nn.MSELoss()(pred_noise, noise)
            test_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:  # Limit to 10 batches for efficiency
                break
    
    test_loss /= num_batches
    
    # Generate samples
    generated_samples = sample_from_model(model, noise_schedule, device, num_samples=5000)
    
    # Compute metrics
    mmd_rbf = compute_mmd(generated_samples, original_data, kernel='rbf')
    mmd_linear = compute_mmd(generated_samples, original_data, kernel='linear')
    wasserstein_dist = compute_wasserstein_distance(generated_samples, original_data)
    
    # Load training loss for this epoch
    checkpoint_dir = os.path.dirname(checkpoint_path)
    losses_file = os.path.join(checkpoint_dir, 'losses.csv')
    train_loss = None
    if os.path.exists(losses_file):
        try:
            losses_df = pd.read_csv(losses_file)
            # Extract epoch number from checkpoint filename
            epoch_str = os.path.basename(checkpoint_path).split('_')[-1].split('.')[0]
            if epoch_str.isdigit():
                epoch = int(epoch_str)
                if epoch <= len(losses_df):
                    train_loss = losses_df.iloc[epoch-1]['loss']
        except:
            pass
    
    return {
        'schedule_type': schedule_type,
        'checkpoint': os.path.basename(checkpoint_path),
        'test_loss': test_loss,
        'train_loss': train_loss,
        'mmd_rbf': mmd_rbf,
        'mmd_linear': mmd_linear,
        'wasserstein_distance': wasserstein_dist,
        'generated_samples': generated_samples
    }

def evaluate_schedule_parallel(args):
    """Evaluate all checkpoints for a single schedule using parallel processing."""
    schedule_name, checkpoint_dir, original_data, device_str = args
    
    print(f"Starting evaluation of {schedule_name} schedule...")
    
    # Set device
    device = torch.device(device_str)
    
    # Get all checkpoint files for this schedule
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pt")))
    
    results = []
    for checkpoint_path in checkpoint_files:
        try:
            result = evaluate_checkpoint(checkpoint_path, schedule_name, original_data, device)
            results.append(result)
            print(f"Completed {schedule_name}: {os.path.basename(checkpoint_path)} - Test Loss: {result['test_loss']:.4f}")
        except Exception as e:
            print(f"Error evaluating {checkpoint_path}: {e}")
            continue
    
    return results

def create_comparison_plots(all_results, original_data, save_path="comprehensive_evaluation_results.png"):
    """Create comprehensive comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Original data
    axes[0, 0].scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, s=1)
    axes[0, 0].set_title('Original GMM Data')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    # Plot 2: Generated samples comparison (final models)
    final_samples = {}
    for schedule_results in all_results:
        if schedule_results:
            final_result = schedule_results[-1]  # Last checkpoint
            final_samples[final_result['schedule_type']] = final_result['generated_samples']
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (schedule, samples) in enumerate(final_samples.items()):
        axes[0, 1].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=1, 
                           color=colors[i], label=schedule)
    axes[0, 1].set_title('Generated Samples (Final Models)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].legend()
    
    # Plot 3: Test Loss trajectories
    for schedule_results in all_results:
        if schedule_results:
            epochs = [int(r['checkpoint'].split('_')[-1].split('.')[0]) for r in schedule_results]
            test_losses = [r['test_loss'] for r in schedule_results]
            axes[0, 2].plot(epochs, test_losses, marker='o', label=schedule_results[0]['schedule_type'])
    axes[0, 2].set_title('Test Loss Trajectories')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Test Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: MMD RBF trajectories
    for schedule_results in all_results:
        if schedule_results:
            epochs = [int(r['checkpoint'].split('_')[-1].split('.')[0]) for r in schedule_results]
            mmd_values = [r['mmd_rbf'] for r in schedule_results]
            axes[1, 0].plot(epochs, mmd_values, marker='o', label=schedule_results[0]['schedule_type'])
    axes[1, 0].set_title('MMD RBF Trajectories')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MMD RBF')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Wasserstein distance trajectories
    for schedule_results in all_results:
        if schedule_results:
            epochs = [int(r['checkpoint'].split('_')[-1].split('.')[0]) for r in schedule_results]
            wass_values = [r['wasserstein_distance'] for r in schedule_results]
            axes[1, 1].plot(epochs, wass_values, marker='o', label=schedule_results[0]['schedule_type'])
    axes[1, 1].set_title('Wasserstein Distance Trajectories')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Wasserstein Distance')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Final metrics comparison
    final_metrics = []
    schedule_names = []
    for schedule_results in all_results:
        if schedule_results:
            final_result = schedule_results[-1]
            final_metrics.append([
                final_result['test_loss'],
                final_result['mmd_rbf'],
                final_result['wasserstein_distance']
            ])
            schedule_names.append(final_result['schedule_type'])
    
    if final_metrics:
        final_metrics = np.array(final_metrics)
        x = np.arange(len(schedule_names))
        width = 0.25
        
        axes[1, 2].bar(x - width, final_metrics[:, 0], width, label='Test Loss')
        axes[1, 2].bar(x, final_metrics[:, 1], width, label='MMD RBF')
        axes[1, 2].bar(x + width, final_metrics[:, 2], width, label='Wasserstein')
        
        axes[1, 2].set_title('Final Metrics Comparison')
        axes[1, 2].set_xlabel('Schedule Type')
        axes[1, 2].set_ylabel('Metric Value')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(schedule_names)
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plots saved to: {save_path}")

def create_summary_table(all_results, save_path="comprehensive_evaluation_summary.csv"):
    """Create a summary table of all results."""
    all_data = []
    
    for schedule_results in all_results:
        for result in schedule_results:
            all_data.append({
                'schedule_type': result['schedule_type'],
                'checkpoint': result['checkpoint'],
                'test_loss': result['test_loss'],
                'train_loss': result['train_loss'],
                'mmd_rbf': result['mmd_rbf'],
                'mmd_linear': result['mmd_linear'],
                'wasserstein_distance': result['wasserstein_distance']
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(save_path, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    
    for schedule_type in df['schedule_type'].unique():
        schedule_df = df[df['schedule_type'] == schedule_type]
        print(f"\n{schedule_type.upper()} SCHEDULE:")
        print("-" * 40)
        
        # Best test loss
        best_test = schedule_df.loc[schedule_df['test_loss'].idxmin()]
        print(f"Best Test Loss: {best_test['test_loss']:.4f} (Epoch {best_test['checkpoint']})")
        
        # Best MMD
        best_mmd = schedule_df.loc[schedule_df['mmd_rbf'].idxmin()]
        print(f"Best MMD RBF: {best_mmd['mmd_rbf']:.4f} (Epoch {best_mmd['checkpoint']})")
        
        # Best Wasserstein
        best_wass = schedule_df.loc[schedule_df['wasserstein_distance'].idxmin()]
        print(f"Best Wasserstein: {best_wass['wasserstein_distance']:.4f} (Epoch {best_wass['checkpoint']})")
    
    print(f"\nDetailed results saved to: {save_path}")
    return df

def main():
    # Set up device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original data
    print("Generating original GMM data...")
    original_data = generate_gmm_data(n_samples=10000, n_components=8, radius=5.0, std=0.2)
    
    # Define schedules to evaluate
    schedules = [
        ("linear", "checkpoints/linear_schedule"),
        ("cosine", "checkpoints/cosine_schedule"),
        ("quadratic", "checkpoints/quadratic_schedule"),
        ("exponential", "checkpoints/exponential_schedule")
    ]
    
    # Prepare arguments for parallel processing
    args_list = [(schedule_name, checkpoint_dir, original_data, str(device)) 
                 for schedule_name, checkpoint_dir in schedules]
    
    # Use parallel processing
    num_workers = 10
    print(f"Starting parallel evaluation with {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        all_results = pool.map(evaluate_schedule_parallel, args_list)
    
    # Create plots and summary
    print("Creating comparison plots...")
    create_comparison_plots(all_results, original_data)
    
    print("Creating summary table...")
    create_summary_table(all_results)
    
    print("Comprehensive evaluation completed!")

if __name__ == "__main__":
    main() 