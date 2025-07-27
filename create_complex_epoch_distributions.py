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

def create_epoch_distribution_plots():
    """Create learned distribution plots for every 100th epoch."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original complex data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Define schedules and epochs
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    schedule_names = ['Linear', 'Cosine', 'Quadratic', 'Exponential']
    epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Create output directory
    os.makedirs('test_results_complex/epoch_distributions', exist_ok=True)
    
    for schedule, schedule_name in zip(schedules, schedule_names):
        print(f"\nProcessing {schedule_name} schedule...")
        
        for epoch in epochs:
            try:
                print(f"  Processing epoch {epoch}...")
                
                # Construct checkpoint path
                checkpoint_path = f'checkpoints_complex/{schedule}_complex/model_epoch_{epoch:04d}.pt'
                
                if not os.path.exists(checkpoint_path):
                    print(f"    ⚠️  Checkpoint not found: {checkpoint_path}")
                    continue
                
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
                print(f"    Generating samples...")
                generated_samples = sample_from_model(model, noise_schedule, device, num_samples=3000, batch_size=128)
                
                # Create plot
                plt.figure(figsize=(12, 8))
                
                # Plot original data
                plt.scatter(original_data[:, 0], original_data[:, 1], 
                           alpha=0.6, s=1, label='Original', color='blue', marker='o')
                
                # Plot generated samples
                plt.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                           alpha=0.6, s=1, label='Generated', color='red', marker='x')
                
                plt.title(f'Complex Dataset: {schedule_name} Schedule (Epoch {epoch})', 
                         fontweight='bold', fontsize=14)
                plt.xlabel('X', fontsize=12)
                plt.ylabel('Y', fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                
                # Save plot
                save_path = f'test_results_complex/epoch_distributions/complex_{schedule}_epoch_{epoch:04d}.png'
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"    ✅ Saved: {os.path.basename(save_path)}")
                
            except Exception as e:
                print(f"    ❌ Error processing epoch {epoch}: {e}")
                continue
    
    print(f"\n✅ All epoch distribution plots completed!")
    print(f"Saved to: test_results_complex/epoch_distributions/")

def create_side_by_side_epoch_comparison():
    """Create side-by-side comparison plots for specific epochs."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original complex data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Define schedules and key epochs
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    schedule_names = ['Linear', 'Cosine', 'Quadratic', 'Exponential']
    key_epochs = [100, 300, 500, 700, 900]
    
    # Create figure
    fig, axes = plt.subplots(len(key_epochs), len(schedules), figsize=(20, 16))
    fig.suptitle('Complex Dataset: Learned Distributions at Different Epochs', fontsize=16, fontweight='bold')
    
    for epoch_idx, epoch in enumerate(key_epochs):
        for schedule_idx, (schedule, schedule_name) in enumerate(zip(schedules, schedule_names)):
            ax = axes[epoch_idx, schedule_idx]
            
            try:
                # Construct checkpoint path
                checkpoint_path = f'checkpoints_complex/{schedule}_complex/model_epoch_{epoch:04d}.pt'
                
                if not os.path.exists(checkpoint_path):
                    ax.text(0.5, 0.5, f'Not found:\n{os.path.basename(checkpoint_path)}', 
                           transform=ax.transAxes, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                    ax.set_title(f'{schedule_name} Epoch {epoch}', fontweight='bold', fontsize=10)
                    continue
                
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
                generated_samples = sample_from_model(model, noise_schedule, device, num_samples=2000, batch_size=128)
                
                # Plot
                ax.scatter(original_data[:, 0], original_data[:, 1], 
                          alpha=0.4, s=1, label='Original', color='blue', marker='o')
                ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                          alpha=0.6, s=1, label='Generated', color='red', marker='x')
                
                ax.set_title(f'{schedule_name} Epoch {epoch}', fontweight='bold', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{schedule_name} Epoch {epoch}', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                ax.set_title(f'{schedule_name} Epoch {epoch}', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('test_results_complex/complex_epoch_comparison_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Side-by-side epoch comparison grid saved as: complex_epoch_comparison_grid.png")

def create_individual_schedule_progression():
    """Create progression plots for each schedule showing evolution over epochs."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original complex data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    
    # Define schedules
    schedules = ['linear', 'cosine', 'quadratic', 'exponential']
    schedule_names = ['Linear', 'Cosine', 'Quadratic', 'Exponential']
    epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for schedule, schedule_name in zip(schedules, schedule_names):
        print(f"\nCreating progression plot for {schedule_name} schedule...")
        
        # Create figure with subplots for each epoch
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Complex Dataset: {schedule_name} Schedule Progression', fontsize=16, fontweight='bold')
        
        for idx, epoch in enumerate(epochs):
            ax = axes[idx // 5, idx % 5]
            
            try:
                # Construct checkpoint path
                checkpoint_path = f'checkpoints_complex/{schedule}_complex/model_epoch_{epoch:04d}.pt'
                
                if not os.path.exists(checkpoint_path):
                    ax.text(0.5, 0.5, f'Not found:\nEpoch {epoch}', 
                           transform=ax.transAxes, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                    ax.set_title(f'Epoch {epoch}', fontweight='bold', fontsize=10)
                    continue
                
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
                generated_samples = sample_from_model(model, noise_schedule, device, num_samples=2000, batch_size=128)
                
                # Plot
                ax.scatter(original_data[:, 0], original_data[:, 1], 
                          alpha=0.4, s=1, label='Original', color='blue', marker='o')
                ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                          alpha=0.6, s=1, label='Generated', color='red', marker='x')
                
                ax.set_title(f'Epoch {epoch}', fontweight='bold', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\nEpoch {epoch}', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                ax.set_title(f'Epoch {epoch}', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'test_results_complex/complex_{schedule}_progression.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✅ {schedule_name} progression plot saved as: complex_{schedule}_progression.png")

def main():
    """Main function to create epoch distribution plots."""
    print("Creating complex epoch distribution plots...")
    
    # Create individual epoch plots
    print("\n1. Creating individual epoch distribution plots...")
    create_epoch_distribution_plots()
    
    # Create side-by-side comparison grid
    print("\n2. Creating side-by-side epoch comparison grid...")
    create_side_by_side_epoch_comparison()
    
    # Create individual schedule progression plots
    print("\n3. Creating individual schedule progression plots...")
    create_individual_schedule_progression()
    
    print("\n✅ All complex epoch distribution plots completed!")
    print("Generated files:")
    print("- test_results_complex/epoch_distributions/complex_*_epoch_*.png: Individual epoch plots")
    print("- test_results_complex/complex_epoch_comparison_grid.png: Side-by-side comparison grid")
    print("- test_results_complex/complex_*_progression.png: Schedule progression plots")

if __name__ == "__main__":
    main() 