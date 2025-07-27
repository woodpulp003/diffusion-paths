import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from geodesic_evaluation import NoiseSchedule


def debug_w2_geodesic_implementation():
    """Debug the W₂ geodesic implementation to understand the mismatch."""
    
    print("🔍 DEBUGGING W₂ GEODESIC IMPLEMENTATION")
    print("=" * 50)
    
    T = 1000
    eps = 1e-4
    
    # Theoretical path
    t_vals = torch.linspace(0, 1, T)
    theoretical_std = (1 - t_vals) * eps + t_vals
    
    # Create geodesic schedule
    geodesic_schedule = NoiseSchedule(T=T, schedule_type="geodesic", beta_start=eps, beta_end=0.02)
    
    # Get implemented values
    implemented_std = geodesic_schedule.get_marginal_std(torch.arange(T, device=geodesic_schedule.betas.device)).cpu().numpy()
    theoretical_std_np = theoretical_std.cpu().numpy()
    
    # Check alpha_bar values
    alpha_bars = geodesic_schedule.alpha_bars.cpu().numpy()
    
    # Print key values
    print(f"\n📊 KEY VALUES:")
    print(f"   • eps: {eps}")
    print(f"   • T: {T}")
    print(f"   • Theoretical std at t=0: {theoretical_std_np[0]:.6f}")
    print(f"   • Implemented std at t=0: {implemented_std[0]:.6f}")
    print(f"   • Theoretical std at t=T-1: {theoretical_std_np[-1]:.6f}")
    print(f"   • Implemented std at t=T-1: {implemented_std[-1]:.6f}")
    
    # Check the actual computed values in the geodesic schedule
    print(f"\n🔍 DETAILED GEODESIC VALUES:")
    print(f"   • sigma_t[0] = {(1 - 0) * eps + 0:.10f}")
    print(f"   • sigma_t[0]^2 = {((1 - 0) * eps + 0)**2:.10f}")
    print(f"   • 1 - sigma_t[0]^2 = {1 - ((1 - 0) * eps + 0)**2:.10f}")
    print(f"   • alpha_bar_t[0] should be: {1 - ((1 - 0) * eps + 0)**2:.10f}")
    print(f"   • alpha_bar_t[0] actually is: {alpha_bars[0]:.10f}")
    
    print(f"\n📊 ALPHA_BAR VALUES:")
    print(f"   • alpha_bar[0]: {alpha_bars[0]:.6f}")
    print(f"   • alpha_bar[-1]: {alpha_bars[-1]:.6f}")
    print(f"   • 1 - alpha_bar[0]: {1 - alpha_bars[0]:.6f}")
    print(f"   • 1 - alpha_bar[-1]: {1 - alpha_bars[-1]:.6f}")
    print(f"   • sqrt(1 - alpha_bar[0]): {np.sqrt(1 - alpha_bars[0]):.6f}")
    print(f"   • sqrt(1 - alpha_bar[-1]): {np.sqrt(1 - alpha_bars[-1]):.6f}")
    
    # Check beta values
    betas = geodesic_schedule.betas.cpu().numpy()
    print(f"\n📊 BETA VALUES:")
    print(f"   • beta[0]: {betas[0]:.6f}")
    print(f"   • beta[-1]: {betas[-1]:.6f}")
    print(f"   • beta variance: {np.var(betas):.6f}")
    
    # Calculate errors
    error = np.abs(theoretical_std_np - implemented_std)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    print(f"\n📊 ERROR ANALYSIS:")
    print(f"   • Max error: {max_error:.6f}")
    print(f"   • Mean error: {mean_error:.6f}")
    print(f"   • Error at t=0: {error[0]:.6f}")
    print(f"   • Error at t=T-1: {error[-1]:.6f}")
    
    # Create debug plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Theoretical vs Implemented
    ax1 = axes[0, 0]
    ax1.plot(t_vals, theoretical_std_np, label='Theoretical', color='red', linewidth=2)
    ax1.plot(t_vals, implemented_std, label='Implemented', color='blue', linewidth=2)
    ax1.set_title('Theoretical vs Implemented W₂ Geodesic')
    ax1.set_xlabel('Normalized Time')
    ax1.set_ylabel('Standard Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error over time
    ax2 = axes[0, 1]
    ax2.plot(t_vals, error, color='orange', linewidth=2)
    ax2.set_title('Implementation Error Over Time')
    ax2.set_xlabel('Normalized Time')
    ax2.set_ylabel('Absolute Error')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=mean_error, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_error:.6f}')
    ax2.legend()
    
    # Plot 3: Alpha bar values
    ax3 = axes[1, 0]
    ax3.plot(t_vals, alpha_bars, color='green', linewidth=2)
    ax3.set_title('Alpha Bar Values')
    ax3.set_xlabel('Normalized Time')
    ax3.set_ylabel('Alpha Bar')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Beta values
    ax4 = axes[1, 1]
    ax4.plot(t_vals, betas, color='purple', linewidth=2)
    ax4.set_title('Beta Values')
    ax4.set_xlabel('Normalized Time')
    ax4.set_ylabel('Beta')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("w2_geodesic_debug.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Debug visualization saved to: w2_geodesic_debug.png")
    
    return {
        'theoretical_std': theoretical_std_np,
        'implemented_std': implemented_std,
        'alpha_bars': alpha_bars,
        'betas': betas,
        'error': error,
        'max_error': max_error,
        'mean_error': mean_error
    }


if __name__ == "__main__":
    debug_results = debug_w2_geodesic_implementation() 