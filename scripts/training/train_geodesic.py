import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import json

import sys
import os
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
    Constructs a beta schedule such that the marginal std follows the Wasserstein-2 geodesic 
    from N(0, eps^2 I) to N(0, I).
    
    The marginal std at time t is: sigma_t = sqrt((1 - t)^2 * eps^2 + t^2)
    Then convert to alphas and betas.
    """
    t_vals = torch.linspace(0, 1, T)
    sigma_t = torch.sqrt((1 - t_vals)**2 * eps**2 + t_vals**2)
    alpha_t = 1 / (1 + sigma_t**2)
    beta_t = 1 - alpha_t
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
        self.betas = self._get_betas()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def _get_betas(self):
        return get_beta_schedule(self.T, self.schedule_type)


def train_geodesic_model(epochs: int = 1000, batch_size: int = 1280, save_every: int = 10):
    """
    Train a diffusion model using the geodesic noise schedule.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_every: Save checkpoint every N epochs
    """
    print("🚀 Training Geodesic Diffusion Model")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating GMM dataset...")
    data_samples = generate_gmm_data(n_samples=100000, n_components=8, radius=5.0, std=0.2)
    data_samples = torch.FloatTensor(data_samples)
    
    # Create noise schedule
    print("Creating geodesic noise schedule...")
    noise_schedule = NoiseSchedule(T=1000, schedule_type="geodesic", beta_start=1e-4, beta_end=0.02)
    
    # Create model
    print("Initializing model...")
    model = Denoiser(embedding_dim=32).to(device)
    
    # Optimizer (same as other training scripts)
    optimizer = optim.AdamW(model.parameters(), lr=3.16e-5)  # Scaled for large batch
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints/geodesic_schedule"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Save every: {save_every} epochs")
    print("-" * 50)
    
    losses = []
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Sample random batch
        indices = torch.randperm(len(data_samples))[:batch_size]
        batch = data_samples[indices].to(device)
        
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1:04d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'schedule_type': 'geodesic',
                'noise_schedule': noise_schedule.schedule_type
            }, checkpoint_path)
            print(f"✅ Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epochs:04d}.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'schedule_type': 'geodesic',
        'noise_schedule': noise_schedule.schedule_type
    }, final_checkpoint_path)
    print(f"✅ Saved final checkpoint: {final_checkpoint_path}")
    
    # Save training losses
    losses_file = os.path.join(checkpoint_dir, "losses.csv")
    with open(losses_file, 'w') as f:
        f.write("epoch,loss\n")
        for i, loss_val in enumerate(losses):
            f.write(f"{i+1},{loss_val}\n")
    print(f"✅ Saved training losses: {losses_file}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss - Geodesic Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(checkpoint_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🎉 Training completed!")
    print(f"📁 Checkpoints saved in: {checkpoint_dir}")
    print(f"📊 Training losses saved in: {losses_file}")
    print(f"📈 Training curve saved as: training_loss.png")
    
    return model, losses


def main():
    """Main function to run geodesic training."""
    parser = argparse.ArgumentParser(description='Train diffusion model with geodesic noise schedule')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1280, help='Batch size for training')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    print("🎯 Geodesic Diffusion Model Training")
    print("=" * 50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save every: {args.save_every} epochs")
    print("=" * 50)
    
    # Train the model
    model, losses = train_geodesic_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_every=args.save_every
    )
    
    print("\n✅ Geodesic training completed successfully!")


if __name__ == "__main__":
    main() 