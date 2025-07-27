import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import csv
from typing import Dict, Any, Tuple
import argparse
from model import Denoiser
from data.gmm_dataset import GMM2DDataset, get_dataloader


class NoiseSchedule:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
    
    def get_params(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_idx = t - 1
        beta_t = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        alpha_bar_t = self.alpha_bars[t_idx]
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t_idx]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t_idx]
        
        return beta_t, alpha_t, alpha_bar_t, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_loop(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    noise_schedule: NoiseSchedule,
    device: torch.device,
    epoch: int,
    save_every: int,
    checkpoint_dir: str,
    losses_file: str,
    config: Dict[str, Any]
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, x_0 in enumerate(dataloader):
        x_0 = x_0.to(device)
        batch_size = x_0.shape[0]
        
        t = torch.randint(1, noise_schedule.T + 1, (batch_size,), device=device)
        
        beta_t, alpha_t, alpha_bar_t, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = noise_schedule.get_params(t)
        
        epsilon = torch.randn_like(x_0, device=device)
        
        x_t = sqrt_alpha_bar_t.view(-1, 1) * x_0 + sqrt_one_minus_alpha_bar_t.view(-1, 1) * epsilon
        
        epsilon_pred = model(x_t, t)
        
        loss = nn.MSELoss()(epsilon_pred, epsilon)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    
    with open(losses_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_loss])
    
    if (epoch + 1) % save_every == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, checkpoint_path)
        
        config_path = os.path.join(checkpoint_dir, f"config_epoch_{epoch+1}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model on 2D GMM data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples in dataset')
    parser.add_argument('--n_components', type=int, default=8, help='Number of GMM components')
    parser.add_argument('--radius', type=float, default=5.0, help='Radius of GMM circle')
    parser.add_argument('--std', type=float, default=0.2, help='Standard deviation of GMM components')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--losses_file', type=str, default='losses.csv', help='Losses CSV file')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
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
            'T': 1000,
            'beta_start': 1e-4,
            'beta_end': 0.02
        }
    }
    
    noise_schedule = NoiseSchedule(
        T=config['noise_schedule']['T'],
        beta_start=config['noise_schedule']['beta_start'],
        beta_end=config['noise_schedule']['beta_end']
    )
    
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        train=True,
        n_samples=args.n_samples,
        n_components=args.n_components,
        radius=args.radius,
        std=args.std
    )
    
    model = Denoiser().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    with open(args.losses_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
    
    print("Starting training...")
    print(f"Config: {config}")
    
    for epoch in range(args.epochs):
        avg_loss = train_loop(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            noise_schedule=noise_schedule,
            device=device,
            epoch=epoch,
            save_every=args.save_every,
            checkpoint_dir=args.checkpoint_dir,
            losses_file=args.losses_file,
            config=config
        )
        
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main() 