import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from data.gmm_dataset import generate_gmm_data, get_dataloader
from model import Denoiser, get_timestep_embedding

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

def sample_from_model(model, noise_schedule, device, num_samples=5000, batch_size=128):
    """Sample from the trained model using DDPM reverse process"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_size_actual = min(batch_size, num_samples - i)
            
            # Start from pure noise
            x_t = torch.randn(batch_size_actual, 2, device=device)
            
            # Reverse diffusion process
            for t in reversed(range(noise_schedule.T)):
                t_tensor = torch.full((batch_size_actual,), t, device=device, dtype=torch.long)
                
                # Predict noise
                pred_noise = model(x_t, t_tensor)
                
                # Get alpha and beta values
                alpha_t = 1 - noise_schedule.betas[t]
                alpha_bar_t = noise_schedule.alpha_bars[t]
                
                # Add small epsilon to prevent division by zero
                eps = 1e-8
                beta_t = noise_schedule.betas[t]
                
                # Reverse step
                if t > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                
                # DDPM reverse equation
                x_t = (1 / torch.sqrt(alpha_t + eps)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t + eps)) * pred_noise) + torch.sqrt(beta_t) * noise
            
            samples.append(x_t.cpu().numpy())
    
    return np.vstack(samples)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original data for comparison
    original_data = generate_gmm_data(n_samples=10000, n_components=8, radius=5.0, std=0.2)
    
    # Define schedules and their checkpoint directories
    schedules = [
        ("linear", "checkpoints/linear_schedule"),
        ("cosine", "checkpoints/cosine_schedule"),
        ("quadratic", "checkpoints/quadratic_schedule"),
        ("exponential", "checkpoints/exponential_schedule")
    ]
    
    # Find checkpoints for every 100th epoch (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
    target_epochs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(target_epochs), len(schedules), figsize=(20, 25))
    fig.suptitle('Side-by-Side Comparison: Generated Samples vs Original Distribution\n(5000 samples per plot)', fontsize=16)
    
    # Set up noise schedule
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    for epoch_idx, target_epoch in enumerate(target_epochs):
        print(f"\nProcessing epoch {target_epoch}...")
        
        for schedule_idx, (schedule_type, checkpoint_dir) in enumerate(schedules):
            print(f"  Evaluating {schedule_type} schedule...")
            
            # Find the checkpoint for this epoch
            if target_epoch == 0:
                checkpoint_pattern = f"{checkpoint_dir}/model_epoch_0010.pt"  # Use epoch 10 for "0"
            elif target_epoch == 100:
                checkpoint_pattern = f"{checkpoint_dir}/model_epoch_0100.pt"
            else:
                checkpoint_pattern = f"{checkpoint_dir}/model_epoch_{target_epoch:04d}.pt"
            
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            if not checkpoint_files:
                print(f"    No checkpoint found for {schedule_type} at epoch {target_epoch}")
                # Create empty subplot
                axes[epoch_idx, schedule_idx].text(0.5, 0.5, f'No checkpoint\n{schedule_type}\nEpoch {target_epoch}', 
                                                 ha='center', va='center', transform=axes[epoch_idx, schedule_idx].transAxes)
                axes[epoch_idx, schedule_idx].set_xlim(-6, 6)
                axes[epoch_idx, schedule_idx].set_ylim(-6, 6)
                continue
            
            checkpoint_path = checkpoint_files[0]
            print(f"    Found checkpoint: {checkpoint_path}")
            
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model_state = checkpoint['model_state_dict']
                
                # Create model and load state
                model = Denoiser(embedding_dim=32)
                model.load_state_dict(model_state)
                model.to(device)
                model.eval()
                
                # Create noise schedule
                betas = get_beta_schedule(T, schedule_type, beta_start, beta_end)
                alphas = 1 - betas
                alpha_bars = torch.cumprod(alphas, dim=0)
                
                class NoiseSchedule:
                    def __init__(self, betas, alpha_bars):
                        self.betas = betas
                        self.alpha_bars = alpha_bars
                        self.T = len(betas)
                
                noise_schedule = NoiseSchedule(betas, alpha_bars)
                
                # Generate samples
                print(f"    Generating 5000 samples...")
                generated_samples = sample_from_model(model, noise_schedule, device, num_samples=5000)
                
                # Plot
                ax = axes[epoch_idx, schedule_idx]
                
                # Plot original data as background
                ax.scatter(original_data[:, 0], original_data[:, 1], alpha=0.3, s=1, c='lightgray', label='Original')
                
                # Plot generated samples
                ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.7, s=2, c='red', label='Generated')
                
                ax.set_xlim(-6, 6)
                ax.set_ylim(-6, 6)
                ax.set_title(f'{schedule_type.capitalize()} - Epoch {target_epoch}')
                ax.set_aspect('equal')
                
                if epoch_idx == 0:  # Only add legend to top row
                    ax.legend()
                
                print(f"    Completed {schedule_type} at epoch {target_epoch}")
                
            except Exception as e:
                print(f"    Error processing {schedule_type} at epoch {target_epoch}: {e}")
                axes[epoch_idx, schedule_idx].text(0.5, 0.5, f'Error\n{schedule_type}\nEpoch {target_epoch}', 
                                                 ha='center', va='center', transform=axes[epoch_idx, schedule_idx].transAxes)
                axes[epoch_idx, schedule_idx].set_xlim(-6, 6)
                axes[epoch_idx, schedule_idx].set_ylim(-6, 6)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = "side_by_side_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSide-by-side comparison saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main() 