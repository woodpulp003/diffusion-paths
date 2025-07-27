import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import os
import json
from tqdm import tqdm

from data.gmm_dataset import generate_gmm_data, generate_complex_gmm_data
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
    """Noise schedule for diffusion process with geodesic evaluation capabilities."""
    
    def __init__(self, T: int, schedule_type: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.schedule_type = schedule_type.lower()
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Get betas based on schedule type
        self.betas = self._get_betas()
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def _get_betas(self):
        """Generate beta schedule based on the specified type."""
        return get_beta_schedule(self.T, self.schedule_type, self.beta_start, self.beta_end)
    
    def sample_q_t(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from q(x_t | x_0) for given timesteps t."""
        # Get alpha_bar for each timestep
        alpha_bar_t = self.alpha_bars[t]
        
        # Reshape for broadcasting
        alpha_bar_t = alpha_bar_t.view(-1, 1)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Compute x_t
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t
    
    def get_marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        """Get the standard deviation of the marginal q_t(x) at timestep t."""
        alpha_bar_t = self.alpha_bars[t]
        return torch.sqrt(1 - alpha_bar_t)


def compute_mmd(X: torch.Tensor, Y: torch.Tensor, kernel: str = 'rbf', gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of samples.
    
    Args:
        X: First set of samples (n_samples, n_features)
        Y: Second set of samples (n_samples, n_features)
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel parameter
    
    Returns:
        MMD value
    """
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    if kernel == 'rbf':
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
    elif kernel == 'linear':
        K_XX = linear_kernel(X, X)
        K_YY = linear_kernel(Y, Y)
        K_XY = linear_kernel(X, Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    n = X.shape[0]
    m = Y.shape[0]
    
    mmd = (np.sum(K_XX) / (n * n) + 
           np.sum(K_YY) / (m * m) - 
           2 * np.sum(K_XY) / (n * m))
    
    return mmd


def compute_sliced_wasserstein(X: torch.Tensor, Y: torch.Tensor, n_projections: int = 100) -> float:
    """
    Compute Sliced Wasserstein Distance between two sets of samples.
    
    Args:
        X: First set of samples (n_samples, n_features)
        Y: Second set of samples (n_samples, n_features)
        n_projections: Number of random projections to use
    
    Returns:
        Sliced Wasserstein distance
    """
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    
    # Generate random projection directions
    directions = np.random.randn(n_projections, X.shape[1])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    total_distance = 0.0
    
    for direction in directions:
        # Project samples onto direction
        proj_X = X @ direction
        proj_Y = Y @ direction
        
        # Sort projections
        proj_X_sorted = np.sort(proj_X)
        proj_Y_sorted = np.sort(proj_Y)
        
        # Compute Wasserstein distance for this projection
        distance = np.mean(np.abs(proj_X_sorted - proj_Y_sorted))
        total_distance += distance
    
    return total_distance / n_projections


def sample_from_interpolation(x_0: torch.Tensor, x_T: torch.Tensor, t: float) -> torch.Tensor:
    """
    Sample from linear interpolation between x_0 and x_T at time t.
    
    Args:
        x_0: Initial data point
        x_T: Final data point (noise)
        t: Interpolation time (0 to 1)
    
    Returns:
        Interpolated sample
    """
    return (1 - t) * x_0 + t * x_T


def evaluate_conditional_geodesic(noise_schedule: NoiseSchedule, 
                                data_samples: torch.Tensor,
                                n_interpolation_steps: int = 1000,
                                n_samples_per_step: int = 100) -> Dict[str, float]:
    """
    Evaluate how geodesic-like a noise schedule is relative to conditional interpolation paths.
    
    Args:
        noise_schedule: Noise schedule to evaluate
        data_samples: Original data samples
        n_interpolation_steps: Number of interpolation steps to evaluate
        n_samples_per_step: Number of samples to generate per step
    
    Returns:
        Dictionary with geodesic deviation metrics
    """
    device = data_samples.device
    total_deviation = 0.0
    deviations = []
    
    # Sample random data points
    n_data_samples = min(n_samples_per_step, len(data_samples))
    x_0_samples = data_samples[:n_data_samples]
    
    # Generate pure noise samples (x_T)
    x_T_samples = torch.randn_like(x_0_samples)
    
    # Evaluate at each interpolation step
    for step in tqdm(range(n_interpolation_steps), desc="Evaluating conditional geodesic"):
        t_ratio = step / (n_interpolation_steps - 1)
        t_timestep = int(t_ratio * (noise_schedule.T - 1))
        
        # Sample from actual noise schedule
        t_tensor = torch.full((n_data_samples,), t_timestep, device=device)
        actual_samples = noise_schedule.sample_q_t(x_0_samples, t_tensor)
        
        # Sample from linear interpolation
        interpolated_samples = sample_from_interpolation(x_0_samples, x_T_samples, t_ratio)
        
        # Compute deviation using MMD
        deviation = compute_mmd(actual_samples, interpolated_samples, kernel='rbf')
        deviations.append(deviation)
        total_deviation += deviation
    
    avg_deviation = total_deviation / n_interpolation_steps
    
    return {
        'conditional_geodesic_deviation': avg_deviation,
        'conditional_deviations': deviations,
        'max_conditional_deviation': max(deviations),
        'min_conditional_deviation': min(deviations)
    }


def evaluate_marginal_geodesic(noise_schedule: NoiseSchedule,
                              data_samples: torch.Tensor,
                              n_interpolation_steps: int = 1000,
                              n_samples_per_step: int = 100) -> Dict[str, float]:
    """
    Evaluate how geodesic-like a noise schedule is relative to the full data distribution.
    
    Args:
        noise_schedule: Noise schedule to evaluate
        data_samples: Original data samples
        n_interpolation_steps: Number of interpolation steps to evaluate
        n_samples_per_step: Number of samples to generate per step
    
    Returns:
        Dictionary with geodesic deviation metrics
    """
    device = data_samples.device
    total_deviation = 0.0
    deviations = []
    
    # Generate pure noise distribution (q_T)
    q_T_samples = torch.randn(n_samples_per_step, data_samples.shape[1], device=device)
    
    # Evaluate at each interpolation step
    for step in tqdm(range(n_interpolation_steps), desc="Evaluating marginal geodesic"):
        t_ratio = step / (n_interpolation_steps - 1)
        t_timestep = int(t_ratio * (noise_schedule.T - 1))
        
        # Sample from actual noise schedule marginal q_t
        t_tensor = torch.full((n_samples_per_step,), t_timestep, device=device)
        actual_samples = noise_schedule.sample_q_t(data_samples[:n_samples_per_step], t_tensor)
        
        # Sample from linear interpolation between q_0 and q_T
        interpolated_samples = sample_from_interpolation(
            data_samples[:n_samples_per_step], 
            q_T_samples, 
            t_ratio
        )
        
        # Compute deviation using Sliced Wasserstein Distance
        deviation = compute_sliced_wasserstein(actual_samples, interpolated_samples)
        deviations.append(deviation)
        total_deviation += deviation
    
    avg_deviation = total_deviation / n_interpolation_steps
    
    return {
        'marginal_geodesic_deviation': avg_deviation,
        'marginal_deviations': deviations,
        'max_marginal_deviation': max(deviations),
        'min_marginal_deviation': min(deviations)
    }


def evaluate_noise_schedule_geodesic(noise_schedule: NoiseSchedule,
                                   data_samples: torch.Tensor,
                                   n_interpolation_steps: int = 1000,
                                   n_samples_per_step: int = 100) -> Dict[str, float]:
    """
    Comprehensive evaluation of how geodesic-like a noise schedule is.
    
    Args:
        noise_schedule: Noise schedule to evaluate
        data_samples: Original data samples
        n_interpolation_steps: Number of interpolation steps to evaluate
        n_samples_per_step: Number of samples to generate per step
    
    Returns:
        Dictionary with comprehensive geodesic evaluation metrics
    """
    print(f"Evaluating {noise_schedule.schedule_type} noise schedule...")
    
    # Evaluate conditional geodesic
    conditional_results = evaluate_conditional_geodesic(
        noise_schedule, data_samples, n_interpolation_steps, n_samples_per_step
    )
    
    # Evaluate marginal geodesic
    marginal_results = evaluate_marginal_geodesic(
        noise_schedule, data_samples, n_interpolation_steps, n_samples_per_step
    )
    
    # Combine results
    results = {
        'schedule_type': noise_schedule.schedule_type,
        'T': noise_schedule.T,
        'beta_start': noise_schedule.beta_start,
        'beta_end': noise_schedule.beta_end,
        **conditional_results,
        **marginal_results
    }
    
    # Compute overall geodesic score (lower is better)
    overall_score = (conditional_results['conditional_geodesic_deviation'] + 
                    marginal_results['marginal_geodesic_deviation']) / 2
    
    results['overall_geodesic_score'] = overall_score
    
    return results


def plot_geodesic_deviations(results: Dict[str, Dict[str, float]], 
                            save_path: Optional[str] = None) -> None:
    """
    Plot geodesic deviations for all noise schedules.
    
    Args:
        results: Dictionary of results for each schedule
        save_path: Optional path to save the plot
    """
    schedules = list(results.keys())
    n_schedules = len(schedules)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Geodesic Deviation Analysis for Different Noise Schedules', fontsize=16)
    
    # Plot 1: Conditional geodesic deviations
    ax1 = axes[0, 0]
    for schedule in schedules:
        deviations = results[schedule]['conditional_deviations']
        timesteps = np.linspace(0, 1, len(deviations))
        ax1.plot(timesteps, deviations, label=schedule, alpha=0.8)
    ax1.set_title('Conditional Geodesic Deviations')
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('MMD Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Marginal geodesic deviations
    ax2 = axes[0, 1]
    for schedule in schedules:
        deviations = results[schedule]['marginal_deviations']
        timesteps = np.linspace(0, 1, len(deviations))
        ax2.plot(timesteps, deviations, label=schedule, alpha=0.8)
    ax2.set_title('Marginal Geodesic Deviations')
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Sliced Wasserstein Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average deviations comparison
    ax3 = axes[1, 0]
    schedule_names = list(results.keys())
    conditional_deviations = [results[s]['conditional_geodesic_deviation'] for s in schedule_names]
    marginal_deviations = [results[s]['marginal_geodesic_deviation'] for s in schedule_names]
    
    x = np.arange(len(schedule_names))
    width = 0.35
    
    ax3.bar(x - width/2, conditional_deviations, width, label='Conditional', alpha=0.8)
    ax3.bar(x + width/2, marginal_deviations, width, label='Marginal', alpha=0.8)
    ax3.set_title('Average Geodesic Deviations')
    ax3.set_xlabel('Noise Schedule')
    ax3.set_ylabel('Average Deviation')
    ax3.set_xticks(x)
    ax3.set_xticklabels(schedule_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Overall geodesic scores
    ax4 = axes[1, 1]
    overall_scores = [results[s]['overall_geodesic_score'] for s in schedule_names]
    colors = ['red' if s == min(overall_scores) else 'blue' for s in overall_scores]
    
    bars = ax4.bar(schedule_names, overall_scores, color=colors, alpha=0.8)
    ax4.set_title('Overall Geodesic Scores (Lower is Better)')
    ax4.set_xlabel('Noise Schedule')
    ax4.set_ylabel('Overall Score')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, overall_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Main function to evaluate geodesic properties of different noise schedules."""
    print("Starting geodesic evaluation of noise schedules...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating GMM dataset...")
    data_samples = generate_gmm_data(n_samples=10000, n_components=8, radius=5.0, std=0.2)
    data_samples = torch.FloatTensor(data_samples).to(device)
    
    # Define noise schedules to evaluate
    schedule_types = ["linear", "cosine", "quadratic", "exponential"]
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    # Evaluation parameters
    n_interpolation_steps = 1000
    n_samples_per_step = 100
    
    # Evaluate each schedule
    results = {}
    
    for schedule_type in schedule_types:
        print(f"\n{'='*50}")
        print(f"Evaluating {schedule_type} noise schedule")
        print(f"{'='*50}")
        
        # Create noise schedule
        noise_schedule = NoiseSchedule(T=T, schedule_type=schedule_type, 
                                     beta_start=beta_start, beta_end=beta_end)
        
        # Evaluate geodesic properties
        schedule_results = evaluate_noise_schedule_geodesic(
            noise_schedule, data_samples, n_interpolation_steps, n_samples_per_step
        )
        
        results[schedule_type] = schedule_results
        
        # Print summary
        print(f"\n{schedule_type.upper()} Schedule Results:")
        print(f"  Conditional Geodesic Deviation: {schedule_results['conditional_geodesic_deviation']:.6f}")
        print(f"  Marginal Geodesic Deviation: {schedule_results['marginal_geodesic_deviation']:.6f}")
        print(f"  Overall Geodesic Score: {schedule_results['overall_geodesic_score']:.6f}")
    
    # Create results summary
    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    
    # Sort by overall score (lower is better)
    sorted_schedules = sorted(results.keys(), 
                            key=lambda x: results[x]['overall_geodesic_score'])
    
    print("\nRanking by Overall Geodesic Score (Lower is Better):")
    for i, schedule in enumerate(sorted_schedules, 1):
        score = results[schedule]['overall_geodesic_score']
        print(f"{i}. {schedule.upper()}: {score:.6f}")
    
    # Save results
    results_file = "geodesic_evaluation_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for schedule, result in results.items():
            json_results[schedule] = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_results[schedule][key] = [float(x) for x in value.tolist()]
                elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                    json_results[schedule][key] = float(value)
                elif isinstance(value, list):
                    # Handle lists that might contain numpy types
                    json_results[schedule][key] = [float(x) if hasattr(x, 'dtype') else x for x in value]
                else:
                    json_results[schedule][key] = value
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create plots
    plot_geodesic_deviations(results, save_path="geodesic_evaluation_plots.png")
    
    print("\nGeodesic evaluation completed!")


if __name__ == "__main__":
    main() 