import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from data.gmm_dataset import generate_complex_gmm_data
from model import Denoiser

def get_beta_schedule(T: int, schedule_type: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02):
    """Get beta schedule based on the specified type."""
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, T)
    elif schedule_type == "cosine":
        steps = torch.linspace(0, 1, T)
        return beta_start + (beta_end - beta_start) * (1 - torch.cos(steps * torch.pi / 2))
    elif schedule_type == "quadratic":
        steps = torch.linspace(0, 1, T)
        return beta_start + (beta_end - beta_start) * steps ** 2
    elif schedule_type == "exponential":
        steps = torch.linspace(0, 1, T)
        return beta_start + (beta_end - beta_start) * (torch.exp(steps) - 1) / (np.e - 1)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

def sample_from_model(model, noise_schedule, device, num_samples=5000, batch_size=128):
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

def load_best_checkpoint_for_schedule(schedule_type):
    """Load the best checkpoint for a given schedule based on test loss."""
    # Load evaluation results to find best checkpoint
    with open('test_results_complex/complex_evaluation_metrics.json', 'r') as f:
        data = json.load(f)
    
    # Filter for the specific schedule
    schedule_data = [entry for entry in data if entry['schedule_type'] == schedule_type]
    
    if not schedule_data:
        raise ValueError(f"No data found for schedule: {schedule_type}")
    
    # Find best by test loss
    best_checkpoint = min(schedule_data, key=lambda x: x['test_loss'])
    epoch = best_checkpoint['epoch']
    
    # Construct checkpoint path
    checkpoint_path = f'checkpoints_complex/{schedule_type}_complex/model_epoch_{epoch:04d}.pt'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path, epoch

def create_complex_side_by_side_plots():
    """Create side-by-side plots of learned distributions for all four schedules."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original complex data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Define schedules
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    schedule_names = ['Linear', 'Cosine', 'Quadratic', 'Exponential']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Complex Dataset: Learned Distributions by Noise Schedule', fontsize=16, fontweight='bold')
    
    for idx, (schedule, schedule_name) in enumerate(zip(schedules, schedule_names)):
        ax = axes[idx // 2, idx % 2]
        
        try:
            print(f"Processing {schedule} schedule...")
            
            # Load best checkpoint for this schedule
            checkpoint_path, epoch = load_best_checkpoint_for_schedule(schedule)
            print(f"  Loading checkpoint: {os.path.basename(checkpoint_path)} (epoch {epoch})")
            
            # Load model
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = Denoiser(embedding_dim=32)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Create noise schedule
            T = 1000
            betas = get_beta_schedule(T, schedule)
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
            
            # Generate samples
            print(f"  Generating samples...")
            generated_samples = sample_from_model(model, noise_schedule, device, num_samples=5000, batch_size=128)
            
            # Plot original data
            ax.scatter(original_data[:, 0], original_data[:, 1], 
                      alpha=0.6, s=1, label='Original', color='blue', marker='o')
            
            # Plot generated samples
            ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                      alpha=0.6, s=1, label='Generated', color='red', marker='x')
            
            # Get metrics for this schedule
            with open('test_results_complex/complex_evaluation_metrics.json', 'r') as f:
                data = json.load(f)
            
            schedule_data = [entry for entry in data if entry['schedule_type'] == schedule and entry['epoch'] == epoch]
            if schedule_data:
                metrics = schedule_data[0]
                metrics_text = f"Test Loss: {metrics['test_loss']:.4f}\nMMD RBF: {metrics['mmd_rbf']:.4f}\nMMD Linear: {metrics['mmd_linear']:.4f}\nWasserstein: {metrics['wasserstein_distance']:.4f}"
            else:
                metrics_text = "Metrics not available"
            
            ax.set_title(f'{schedule_name} Schedule (Epoch {epoch})', fontweight='bold', fontsize=12)
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add metrics text
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="lightblue", alpha=0.8))
            
            print(f"  ✅ {schedule_name} schedule completed")
            
        except Exception as e:
            print(f"  ❌ Error processing {schedule} schedule: {e}")
            ax.text(0.5, 0.5, f'Error loading\n{schedule} schedule', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title(f'{schedule_name} Schedule', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('test_results_complex/complex_side_by_side_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Side-by-side complex distributions plot saved as: complex_side_by_side_distributions.png")

def create_individual_schedule_plots():
    """Create individual detailed plots for each schedule."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original complex data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Define schedules
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    schedule_names = ['Linear', 'Cosine', 'Quadratic', 'Exponential']
    
    for schedule, schedule_name in zip(schedules, schedule_names):
        try:
            print(f"\nCreating individual plot for {schedule} schedule...")
            
            # Load best checkpoint for this schedule
            checkpoint_path, epoch = load_best_checkpoint_for_schedule(schedule)
            print(f"  Loading checkpoint: {os.path.basename(checkpoint_path)} (epoch {epoch})")
            
            # Load model
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = Denoiser(embedding_dim=32)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # Create noise schedule
            T = 1000
            betas = get_beta_schedule(T, schedule)
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
            
            # Generate samples
            print(f"  Generating samples...")
            generated_samples = sample_from_model(model, noise_schedule, device, num_samples=5000, batch_size=128)
            
            # Create individual plot
            plt.figure(figsize=(12, 8))
            
            # Plot original data
            plt.scatter(original_data[:, 0], original_data[:, 1], 
                       alpha=0.6, s=1, label='Original', color='blue', marker='o')
            
            # Plot generated samples
            plt.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                       alpha=0.6, s=1, label='Generated', color='red', marker='x')
            
            # Get metrics for this schedule
            with open('test_results_complex/complex_evaluation_metrics.json', 'r') as f:
                data = json.load(f)
            
            schedule_data = [entry for entry in data if entry['schedule_type'] == schedule and entry['epoch'] == epoch]
            if schedule_data:
                metrics = schedule_data[0]
                metrics_text = f"Test Loss: {metrics['test_loss']:.4f}\nMMD RBF: {metrics['mmd_rbf']:.4f}\nMMD Linear: {metrics['mmd_linear']:.4f}\nWasserstein: {metrics['wasserstein_distance']:.4f}"
            else:
                metrics_text = "Metrics not available"
            
            plt.title(f'Complex Dataset: {schedule_name} Schedule (Epoch {epoch})', fontweight='bold', fontsize=14)
            plt.xlabel('X', fontsize=12)
            plt.ylabel('Y', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Add metrics text
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'test_results_complex/complex_{schedule}_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"  ✅ {schedule_name} individual plot saved as: complex_{schedule}_distribution.png")
            
        except Exception as e:
            print(f"  ❌ Error processing {schedule} schedule: {e}")

def main():
    """Main function to create side-by-side plots."""
    print("Creating side-by-side complex distribution plots...")
    
    # Create combined side-by-side plot
    print("\n1. Creating combined side-by-side plot...")
    create_complex_side_by_side_plots()
    
    # Create individual plots
    print("\n2. Creating individual schedule plots...")
    create_individual_schedule_plots()
    
    print("\n✅ All complex distribution plots completed!")
    print("Generated files:")
    print("- complex_side_by_side_distributions.png: Combined comparison")
    print("- complex_linear_distribution.png: Linear schedule")
    print("- complex_cosine_distribution.png: Cosine schedule")
    print("- complex_quadratic_distribution.png: Quadratic schedule")
    print("- complex_exponential_distribution.png: Exponential schedule")

if __name__ == "__main__":
    main() 