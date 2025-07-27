import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from geodesic_evaluation import NoiseSchedule


def visualize_w2_geodesic_properties():
    """Visualize the W₂ geodesic path properties in detail."""
    
    print("🌐 W₂ Geodesic Path Visualization")
    print("=" * 50)
    
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    eps = beta_start
    
    # Create schedules
    schedules = {
        "linear": NoiseSchedule(T=T, schedule_type="linear", beta_start=beta_start, beta_end=beta_end),
        "cosine": NoiseSchedule(T=T, schedule_type="cosine", beta_start=beta_start, beta_end=beta_end),
        "quadratic": NoiseSchedule(T=T, schedule_type="quadratic", beta_start=beta_start, beta_end=beta_end),
        "exponential": NoiseSchedule(T=T, schedule_type="exponential", beta_start=beta_start, beta_end=beta_end),
        "geodesic": NoiseSchedule(T=T, schedule_type="geodesic", beta_start=beta_start, beta_end=beta_end)
    }
    
    timesteps = np.arange(T)
    t_normalized = timesteps / T
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('W₂ Geodesic Path Analysis', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Beta schedules
    ax1 = axes[0, 0]
    for i, (name, schedule) in enumerate(schedules.items()):
        betas = schedule.betas.cpu().numpy()
        ax1.plot(timesteps, betas, label=name.upper(), color=colors[i], alpha=0.8, linewidth=2)
    ax1.set_title('Beta Schedules', fontweight='bold')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Beta')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Alpha bar schedules
    ax2 = axes[0, 1]
    for i, (name, schedule) in enumerate(schedules.items()):
        alpha_bars = schedule.alpha_bars.cpu().numpy()
        ax2.plot(timesteps, alpha_bars, label=name.upper(), color=colors[i], alpha=0.8, linewidth=2)
    ax2.set_title('Alpha Bar Schedules', fontweight='bold')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Alpha Bar')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Noise variance schedules
    ax3 = axes[0, 2]
    for i, (name, schedule) in enumerate(schedules.items()):
        alpha_bars = schedule.alpha_bars.cpu().numpy()
        noise_variance = 1 - alpha_bars
        ax3.plot(timesteps, noise_variance, label=name.upper(), color=colors[i], alpha=0.8, linewidth=2)
    ax3.set_title('Noise Variance Schedules', fontweight='bold')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('1 - Alpha Bar')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Marginal standard deviation comparison
    ax4 = axes[1, 0]
    for i, (name, schedule) in enumerate(schedules.items()):
        marginal_std = schedule.get_marginal_std(torch.arange(T, device=schedule.betas.device)).cpu().numpy()
        ax4.plot(timesteps, marginal_std, label=name.upper(), color=colors[i], alpha=0.8, linewidth=2)
    
    # Add theoretical W₂ geodesic path
    theoretical_std = (1 - t_normalized) * eps + t_normalized
    ax4.plot(timesteps, theoretical_std, label='Theoretical W₂', color='black', linestyle='--', linewidth=3)
    
    ax4.set_title('Marginal Standard Deviation', fontweight='bold')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Marginal Std')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: W₂ geodesic path verification
    ax5 = axes[1, 1]
    geodesic_schedule = schedules["geodesic"]
    implemented_std = geodesic_schedule.get_marginal_std(torch.arange(T, device=geodesic_schedule.betas.device)).cpu().numpy()
    
    ax5.plot(timesteps, theoretical_std, label='Theoretical W₂', color='red', linewidth=3, linestyle='--')
    ax5.plot(timesteps, implemented_std, label='Implemented W₂', color='purple', linewidth=3)
    
    # Calculate and display error
    error = np.abs(theoretical_std - implemented_std)
    max_error = np.max(error)
    mean_error = np.mean(error)
    
    ax5.set_title(f'W₂ Geodesic Path Verification\nMax Error: {max_error:.6f}, Mean Error: {mean_error:.6f}', fontweight='bold')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Standard Deviation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error analysis
    ax6 = axes[1, 2]
    ax6.plot(timesteps, error, color='orange', linewidth=2)
    ax6.set_title('Implementation Error', fontweight='bold')
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Absolute Error')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=mean_error, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_error:.6f}')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig("w2_geodesic_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ W₂ geodesic analysis visualization saved to: w2_geodesic_analysis.png")
    
    # Print analysis summary
    print(f"\n📊 W₂ GEODESIC ANALYSIS SUMMARY:")
    print(f"   • Theoretical path: σ_t = (1-t)ε + t")
    print(f"   • Implementation error: Max={max_error:.6f}, Mean={mean_error:.6f}")
    print(f"   • Path verification: {'✅ PASS' if max_error < 1e-3 else '❌ FAIL'}")
    
    return {
        'theoretical_std': theoretical_std,
        'implemented_std': implemented_std,
        'max_error': max_error,
        'mean_error': mean_error
    }


def compare_geodesic_properties():
    """Compare geodesic properties across different schedules."""
    
    print("\n🌐 Geodesic Properties Comparison")
    print("=" * 50)
    
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    schedules = {
        "linear": NoiseSchedule(T=T, schedule_type="linear", beta_start=beta_start, beta_end=beta_end),
        "cosine": NoiseSchedule(T=T, schedule_type="cosine", beta_start=beta_start, beta_end=beta_end),
        "quadratic": NoiseSchedule(T=T, schedule_type="quadratic", beta_start=beta_start, beta_end=beta_end),
        "exponential": NoiseSchedule(T=T, schedule_type="exponential", beta_start=beta_start, beta_end=beta_end),
        "geodesic": NoiseSchedule(T=T, schedule_type="geodesic", beta_start=beta_start, beta_end=beta_end)
    }
    
    # Analyze properties
    properties = {}
    
    for name, schedule in schedules.items():
        betas = schedule.betas.cpu().numpy()
        alpha_bars = schedule.alpha_bars.cpu().numpy()
        marginal_std = schedule.get_marginal_std(torch.arange(T, device=schedule.betas.device)).cpu().numpy()
        
        # Calculate properties
        beta_variance = np.var(betas)
        alpha_bar_final = alpha_bars[-1]
        std_start = marginal_std[0]
        std_end = marginal_std[-1]
        std_range = std_end - std_start
        
        properties[name] = {
            'beta_variance': beta_variance,
            'alpha_bar_final': alpha_bar_final,
            'std_start': std_start,
            'std_end': std_end,
            'std_range': std_range
        }
        
        print(f"\n📊 {name.upper()} Properties:")
        print(f"   • Beta variance: {beta_variance:.6f}")
        print(f"   • Final alpha_bar: {alpha_bar_final:.6f}")
        print(f"   • Std range: {std_start:.6f} → {std_end:.6f} (Δ={std_range:.6f})")
    
    return properties


if __name__ == "__main__":
    # Run W₂ geodesic visualization
    analysis_results = visualize_w2_geodesic_properties()
    
    # Compare geodesic properties
    properties = compare_geodesic_properties()
    
    print("\n✅ W₂ geodesic analysis completed!")
    print("📊 Check the generated visualizations for detailed analysis") 