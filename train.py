import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import csv
import numpy as np

from model import Denoiser
from data.gmm_dataset import get_dataloader


class NoiseSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def get_schedule(self):
        return {
            'betas': self.betas,
            'alphas': self.alphas,
            'alpha_bars': self.alpha_bars
        }


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_loop(model, dataloader, optimizer, noise_schedule, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, x0 in enumerate(dataloader):
        x0 = x0.to(device)
        batch_size = x0.shape[0]
        
        # Sample random timesteps uniformly
        t = torch.randint(0, noise_schedule.T, (batch_size,), device=device)
        
        # Get alpha_bar values for the sampled timesteps
        alpha_bar_t = noise_schedule.alpha_bars[t].unsqueeze(1)
        
        # Sample Gaussian noise
        epsilon = torch.randn_like(x0, device=device)
        
        # Generate x_t using the forward noising equation
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        
        # Predict noise using the model
        pred_epsilon = model(x_t, t)
        
        # Compute loss
        loss = nn.MSELoss()(pred_epsilon, epsilon)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}/{args.epochs}, Average Loss: {avg_loss:.6f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model on 2D GMM data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1280, help='Batch size (large batch default)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3.16e-5, help='Learning rate (scaled for large batch)')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples in dataset')
    parser.add_argument('--n_components', type=int, default=8, help='Number of GMM components')
    parser.add_argument('--radius', type=float, default=5.0, help='Radius of GMM circle')
    parser.add_argument('--std', type=float, default=0.2, help='Standard deviation of GMM components')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--losses_file', type=str, default='losses.csv', help='Losses CSV file')
    
    global args
    args = parser.parse_args()
    
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    
    print(f"Using device: {device}")
    print("Starting training with large batch configuration...")
    
    # Create noise schedule
    noise_schedule = NoiseSchedule()
    
    # Create dataloader
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        train=True,
        n_samples=args.n_samples,
        n_components=args.n_components,
        radius=args.radius,
        std=args.std
    )
    
    # Create model and optimizer
    model = Denoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training configuration
    config = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'n_samples': args.n_samples,
        'n_components': args.n_components,
        'radius': args.radius,
        'std': args.std,
        'save_every': args.save_every,
        'device': str(device),
        'noise_schedule': {
            'T': noise_schedule.T,
            'beta_start': noise_schedule.beta_start,
            'beta_end': noise_schedule.beta_end
        },
        'training_type': 'large_batch_default'
    }
    
    print(f"Config: {config}")
    print(f"Batch size: {args.batch_size} (large batch default)")
    print(f"Learning rate: {args.lr} (scaled for large batch)")
    
    # Training loop
    losses = []
    
    for epoch in range(args.epochs):
        avg_loss = train_loop(model, dataloader, optimizer, noise_schedule, device, epoch)
        losses.append(avg_loss)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1:04d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "model_final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }, final_checkpoint_path)
    print(f"Final model saved: {final_checkpoint_path}")
    
    # Save losses to CSV
    with open(args.losses_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for i, loss in enumerate(losses):
            writer.writerow([i+1, loss])
    
    print(f"Training completed! Losses saved to: {args.losses_file}")


if __name__ == "__main__":
    main() 