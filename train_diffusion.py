import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import numpy as np

from model import Denoiser
from data.gmm_dataset import get_dataloader


def get_linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule for diffusion process."""
    return torch.linspace(beta_start, beta_end, T)


def get_cosine_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Cosine beta schedule for diffusion process."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# Config
config = {
    "T": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "save_every": 10,
    "total_epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "verbose": True,
    "clamp_noise": True,  # Clamp predicted noise for safety during debugging
    "n_samples": 100000
}

# Set seeds
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

device = torch.device(config["device"])
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Diffusion hyperparameters
T = config["T"]
betas = get_linear_beta_schedule(T, config["beta_start"], config["beta_end"])
alpha = 1. - betas
alpha_bar = torch.cumprod(alpha, dim=0)

# Convert to torch tensors and move to device
betas = betas.to(device)
alpha_bar = alpha_bar.to(device)

# Data
loader = get_dataloader(batch_size=config["batch_size"], train=True, n_samples=config["n_samples"])

# Model
model = Denoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
loss_fn = nn.MSELoss()

# Save config and beta schedule
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

torch.save(betas, os.path.join(save_dir, "betas.pt"))

print(f"Training on device: {device}")
print(f"Config: {config}")

# Training loop
for epoch in range(config["total_epochs"]):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, x0 in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{config['total_epochs']}")):
        x0 = x0.to(device)

        # Sample t uniformly for each sample in batch
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        alpha_bar_t = alpha_bar[t].unsqueeze(1)  # shape: (B, 1)

        # Sample noise
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        # Predict noise
        pred_noise = model(xt, t)
        
        # Clamp predicted noise for safety during debugging
        if config["clamp_noise"]:
            pred_noise = torch.clamp(pred_noise, -1.0, 1.0)

        # Loss and backward
        loss = loss_fn(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        
        # Verbose logging
        if config["verbose"] and batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item():.4f}")

    avg_loss = running_loss / num_batches
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    # Save model
    if (epoch + 1) % config["save_every"] == 0 or (epoch + 1) == config["total_epochs"]:
        checkpoint_path = os.path.join(save_dir, f"model_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}") 