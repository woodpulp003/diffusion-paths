import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from data.gmm_dataset import generate_complex_gmm_data
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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate original complex data
    print("Generating original complex dataset...")
    original_data = generate_complex_gmm_data(10000, n_components=12)
    print(f"Original data shape: {original_data.shape}")
    
    # Define schedules and target epochs
    schedules = [
        ('linear', 'checkpoints_complex/linear_complex'),
        ('cosine', 'checkpoints_complex/cosine_complex'),
        ('quadratic', 'checkpoints_complex/quadratic_complex'),
        ('exponential', 'checkpoints_complex/exponential_complex')
    ]
    
    target_epochs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Create output directory
    os.makedirs('test_results_complex', exist_ok=True)
    
    # Generate side-by-side plots for each target epoch
    for target_epoch in target_epochs:
        print(f"\nGenerating side-by-side plot for epoch {target_epoch}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Complex Dataset: Side-by-Side Comparison at Epoch {target_epoch}', fontsize=16)
        
        for idx, (schedule_name, checkpoint_dir) in enumerate(schedules):
            ax = axes[idx // 2, idx % 2]
            
            # Find the closest checkpoint to target epoch
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pt"))
            if not checkpoint_files:
                print(f"No checkpoints found for {schedule_name}")
                continue
            
            # Extract epochs and find closest
            epochs = []
            for file in checkpoint_files:
                try:
                    epoch = int(file.split('epoch_')[1].split('.')[0])
                    epochs.append((epoch, file))
                except:
                    continue
            
            if not epochs:
                print(f"No valid checkpoints found for {schedule_name}")
                continue
            
            # Find closest epoch
            closest_epoch, closest_file = min(epochs, key=lambda x: abs(x[0] - target_epoch))
            
            print(f"  {schedule_name}: using epoch {closest_epoch} (target was {target_epoch})")
            
            try:
                # Load checkpoint
                checkpoint = torch.load(closest_file, map_location=device)
                model_state = checkpoint['model_state_dict']
                
                # Create model and load state
                model = Denoiser(embedding_dim=32)
                model.load_state_dict(model_state)
                model.to(device)
                
                # Create noise schedule
                noise_schedule = NoiseSchedule(T=1000, schedule_type=schedule_name)
                
                # Generate samples
                generated_samples = sample_from_model(model, noise_schedule, device, num_samples=5000)
                
                # Plot original data
                ax.scatter(original_data[:, 0], original_data[:, 1], 
                          alpha=0.6, s=1, label='Original', color='blue')
                
                # Plot generated samples
                ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                          alpha=0.6, s=1, label='Generated', color='red')
                
                ax.set_title(f'{schedule_name.capitalize()} Schedule (Epoch {closest_epoch})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error processing {schedule_name}: {e}")
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{schedule_name.capitalize()} Schedule')
        
        plt.tight_layout()
        plt.savefig(f'test_results_complex/side_by_side_epoch_{target_epoch:04d}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved side-by-side plot for epoch {target_epoch}")
    
    print("\nAll side-by-side plots generated!")

if __name__ == "__main__":
    main() 