import torch
import torch.nn as nn
import math
from typing import Tuple


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class Denoiser(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(2 + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embedding = get_timestep_embedding(t, self.embedding_dim)
        x_t_flat = x_t.view(x_t.shape[0], -1)
        combined_input = torch.cat([x_t_flat, t_embedding], dim=1)
        eps_hat = self.mlp(combined_input)
        return eps_hat 