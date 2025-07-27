import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import pandas as pd
from data.gmm_dataset import get_dataloader, generate_complex_gmm_data
from model import Denoiser, get_timestep_embedding
import matplotlib.pyplot as plt

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

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_loop(model, optimizer, train_loader, noise_schedule, device, epoch, args):
    """Training loop for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
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
    parser = argparse.ArgumentParser(description='Train diffusion model on complex GMM dataset')
    parser.add_argument('--schedule', type=str, default='linear', 
                       choices=['linear', 'cosine', 'quadratic', 'exponential'],
                       help='Noise schedule type')
    parser.add_argument('--run_name', type=str, default='complex_dataset',
                       help='Name for this training run')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1280, help='Batch size')
    parser.add_argument('--lr', type=float, default=3.16e-5, help='Learning rate')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_components', type=int, default=12, help='Number of GMM components')
    parser.add_argument('--base_radius', type=float, default=3.0, help='Base radius for complex GMM')
    parser.add_argument('--max_radius', type=float, default=8.0, help='Max radius for complex GMM')
    parser.add_argument('--min_std', type=float, default=0.1, help='Min std for complex GMM')
    parser.add_argument('--max_std', type=float, default=0.4, help='Max std for complex GMM')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    print(f"Starting training with {args.schedule} noise schedule on complex dataset...")
    
    # Create data loaders
    train_loader = get_dataloader(
        batch_size=args.batch_size, 
        train=True, 
        n_samples=10000, 
        n_components=args.n_components,
        complex_data=True,
        base_radius=args.base_radius,
        max_radius=args.max_radius,
        min_std=args.min_std,
        max_std=args.max_std,
        shuffle=True,
        num_workers=4
    )
    
    # Create model
    model = Denoiser(embedding_dim=32)
    model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create noise schedule
    noise_schedule = NoiseSchedule(T=1000, schedule_type=args.schedule)
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints_complex/{args.run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Configuration
    config = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'n_samples': 10000,
        'n_components': args.n_components,
        'base_radius': args.base_radius,
        'max_radius': args.max_radius,
        'min_std': args.min_std,
        'max_std': args.max_std,
        'save_every': args.save_every,
        'device': str(device),
        'noise_schedule': {
            'type': args.schedule,
            'T': 1000,
            'beta_start': 0.0001,
            'beta_end': 0.02
        },
        'training_type': f'complex_dataset_{args.schedule}_schedule',
        'run_name': args.run_name
    }
    
    print(f"Config: {config}")
    print(f"Batch size: {args.batch_size} (large batch default)")
    print(f"Learning rate: {args.lr} (scaled for large batch)")
    print(f"Noise schedule: {args.schedule}")
    print(f"Run name: {args.run_name}")
    
    # Training loop
    losses = []
    
    for epoch in range(args.epochs):
        avg_loss = train_loop(model, optimizer, train_loader, noise_schedule, device, epoch, args)
        losses.append(avg_loss)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{(epoch+1):04d}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses[-1],
        'config': config
    }
    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    torch.save(final_checkpoint, final_path)
    print(f"Final model saved: {final_path}")
    
    # Save losses
    losses_df = pd.DataFrame({
        'epoch': range(1, len(losses) + 1),
        'loss': losses
    })
    losses_path = os.path.join(checkpoint_dir, "losses.csv")
    losses_df.to_csv(losses_path, index=False)
    print(f"Training completed! Losses saved to: {losses_path}")

if __name__ == "__main__":
    main() 