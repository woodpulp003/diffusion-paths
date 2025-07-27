import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from geodesic_evaluation import get_w2_geodesic_beta_schedule, NoiseSchedule


def test_w2_implementation():
    """Test that the W₂ geodesic implementation produces the correct theoretical path."""
    
    print("🧪 TESTING W₂ GEODESIC IMPLEMENTATION")
    print("=" * 50)
    
    T = 1000
    eps = 1e-4
    
    # Get the betas from our implementation
    betas = get_w2_geodesic_beta_schedule(T, eps)
    
    # Create the noise schedule
    schedule = NoiseSchedule(T=T, schedule_type="geodesic", beta_start=eps, beta_end=0.02)
    
    # Check if our betas match the schedule's betas
    print(f"📊 BETA COMPARISON:")
    print(f"   • Our betas[0]: {betas[0]:.10f}")
    print(f"   • Schedule betas[0]: {schedule.betas[0]:.10f}")
    print(f"   • Match: {'✅' if abs(betas[0] - schedule.betas[0]) < 1e-6 else '❌'}")
    
    # Check alpha_bars
    t_vals = torch.linspace(0, 1, T)
    sigma_t = (1 - t_vals) * eps + t_vals
    theoretical_alpha_bars = 1 - sigma_t**2
    
    print(f"\n📊 ALPHA_BAR COMPARISON:")
    print(f"   • Theoretical alpha_bar[0]: {theoretical_alpha_bars[0]:.10f}")
    print(f"   • Schedule alpha_bar[0]: {schedule.alpha_bars[0]:.10f}")
    print(f"   • Match: {'✅' if abs(theoretical_alpha_bars[0] - schedule.alpha_bars[0]) < 1e-6 else '❌'}")
    
    # Debug the computation
    print(f"\n🔍 DEBUGGING COMPUTATION:")
    print(f"   • eps: {eps}")
    print(f"   • t_vals[0]: {t_vals[0]:.10f}")
    print(f"   • sigma_t[0] = (1 - {t_vals[0]:.10f}) * {eps:.10f} + {t_vals[0]:.10f} = {sigma_t[0]:.10f}")
    print(f"   • sigma_t[0]^2 = {sigma_t[0]**2:.10f}")
    print(f"   • 1 - sigma_t[0]^2 = {1 - sigma_t[0]**2:.10f}")
    print(f"   • Theoretical alpha_bar[0] = {1 - sigma_t[0]**2:.10f}")
    
    # Check what our implementation is actually computing
    our_betas = get_w2_geodesic_beta_schedule(T, eps)
    our_alphas = 1 - our_betas
    our_alpha_bars = torch.cumprod(our_alphas, dim=0)
    
    print(f"\n🔍 OUR IMPLEMENTATION VALUES:")
    print(f"   • Our alphas[0]: {our_alphas[0]:.10f}")
    print(f"   • Our alpha_bars[0]: {our_alpha_bars[0]:.10f}")
    print(f"   • Our betas[0]: {our_betas[0]:.10f}")
    
    # Check marginal std
    theoretical_std = sigma_t
    implemented_std = schedule.get_marginal_std(torch.arange(T, device=schedule.betas.device))
    
    print(f"\n📊 MARGINAL STD COMPARISON:")
    print(f"   • Theoretical std[0]: {theoretical_std[0]:.10f}")
    print(f"   • Implemented std[0]: {implemented_std[0]:.10f}")
    print(f"   • Match: {'✅' if abs(theoretical_std[0] - implemented_std[0]) < 1e-6 else '❌'}")
    
    # Calculate overall error
    error = torch.abs(theoretical_std - implemented_std)
    max_error = torch.max(error).item()
    mean_error = torch.mean(error).item()
    
    print(f"\n📊 ERROR ANALYSIS:")
    print(f"   • Max error: {max_error:.10f}")
    print(f"   • Mean error: {mean_error:.10f}")
    print(f"   • Implementation correct: {'✅' if max_error < 1e-6 else '❌'}")
    
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'theoretical_std': theoretical_std,
        'implemented_std': implemented_std
    }


if __name__ == "__main__":
    results = test_w2_implementation() 