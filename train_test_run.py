import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np

from model import Denoiser
from data.gmm_dataset import get_dataloader


def get_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for diffusion process."""
    return torch.linspace(beta_start, beta_end, T)


def main():
    # Set up device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Using device: {device}")
    
    # Scaled up training parameters
    T = 1000
    batch_size = 64
    num_epochs = 1000  # Scaled up from 100
    learning_rate = 1e-3
    save_every = 10  # Save every 10 epochs
    
    # Create noise schedule
    betas = get_linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02)
    alpha = 1. - betas
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # Move to device
    betas = betas.to(device)
    alpha_bar = alpha_bar.to(device)
    
    # Create larger dataset for extended training
    # About 20 batches per epoch = 1280 samples
    dataset_size = batch_size * 20  # 1280 samples total
    loader = get_dataloader(
        batch_size=batch_size, 
        train=True, 
        n_samples=dataset_size,
        shuffle=True
    )
    
    # Create model and optimizer
    model = Denoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    print(f"Dataset size: {dataset_size} samples ({len(loader)} batches per epoch)")
    print(f"Total training steps: {num_epochs * len(loader)}")
    print(f"Checkpoints saved every {save_every} epochs")
    print(f"Total checkpoints to be saved: {num_epochs // save_every}")
    print("-" * 50)
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, x0 in enumerate(loader):
            x0 = x0.to(device)
            
            # Sample random timesteps uniformly for each sample in batch
            t = torch.randint(0, T, (x0.shape[0],), device=device)
            
            # Get alpha_bar values for the sampled timesteps
            alpha_bar_t = alpha_bar[t].unsqueeze(1)  # Shape: (batch_size, 1)
            
            # Sample Gaussian noise
            epsilon = torch.randn_like(x0, device=device)
            
            # Generate x_t using the forward noising equation
            x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
            
            # Predict noise using the model
            pred_epsilon = model(x_t, t)
            
            # Compute loss
            loss = loss_fn(pred_epsilon, epsilon)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Print epoch summary (less frequently for long training)
        if (epoch + 1) % 50 == 0 or epoch < 10:
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1:4d}/{num_epochs} | Avg Loss: {avg_epoch_loss:.6f}")
        
        # Save checkpoint every save_every epochs and at the final epoch
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            avg_epoch_loss = epoch_loss / num_batches
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1:04d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': {
                    'T': T,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'device': str(device),
                    'dataset_size': dataset_size,
                    'save_every': save_every
                }
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path} (Loss: {avg_epoch_loss:.6f})")
    
    print(f"\nTraining completed!")
    print(f"Final average loss: {avg_epoch_loss:.6f}")
    print(f"Total checkpoints saved: {num_epochs // save_every + 1}")


if __name__ == "__main__":
    main() 