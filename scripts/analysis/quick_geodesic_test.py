import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from geodesic_evaluation import (
    NoiseSchedule, 
    evaluate_conditional_geodesic, 
    evaluate_marginal_geodesic,
    plot_geodesic_deviations
)
from data.gmm_dataset import generate_gmm_data


def quick_geodesic_test(n_interpolation_steps: int = 100, n_samples_per_step: int = 50):
    """
    Quick test of geodesic evaluation with reduced parameters for faster execution.
    
    Args:
        n_interpolation_steps: Number of interpolation steps (reduced for speed)
        n_samples_per_step: Number of samples per step (reduced for speed)
    """
    print("Running quick geodesic evaluation test...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate smaller dataset for quick test
    print("Generating GMM dataset...")
    data_samples = generate_gmm_data(n_samples=1000, n_components=8, radius=5.0, std=0.2)
    data_samples = torch.FloatTensor(data_samples).to(device)
    
    # Define noise schedules to evaluate
    schedule_types = ["linear", "cosine", "quadratic", "exponential", "geodesic"]
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    # Evaluate each schedule
    results = {}
    
    for schedule_type in schedule_types:
        print(f"\n{'='*40}")
        print(f"Evaluating {schedule_type} noise schedule")
        print(f"{'='*40}")
        
        # Create noise schedule
        noise_schedule = NoiseSchedule(T=T, schedule_type=schedule_type, 
                                     beta_start=beta_start, beta_end=beta_end)
        
        # Evaluate conditional geodesic
        print("Evaluating conditional geodesic...")
        conditional_results = evaluate_conditional_geodesic(
            noise_schedule, data_samples, n_interpolation_steps, n_samples_per_step
        )
        
        # Evaluate marginal geodesic
        print("Evaluating marginal geodesic...")
        marginal_results = evaluate_marginal_geodesic(
            noise_schedule, data_samples, n_interpolation_steps, n_samples_per_step
        )
        
        # Combine results
        schedule_results = {
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
        
        schedule_results['overall_geodesic_score'] = overall_score
        
        results[schedule_type] = schedule_results
        
        # Print summary
        print(f"\n{schedule_type.upper()} Schedule Results:")
        print(f"  Conditional Geodesic Deviation: {conditional_results['conditional_geodesic_deviation']:.6f}")
        print(f"  Marginal Geodesic Deviation: {marginal_results['marginal_geodesic_deviation']:.6f}")
        print(f"  Overall Geodesic Score: {overall_score:.6f}")
    
    # Create results summary
    print(f"\n{'='*40}")
    print("QUICK TEST RESULTS SUMMARY")
    print(f"{'='*40}")
    
    # Sort by overall score (lower is better)
    sorted_schedules = sorted(results.keys(), 
                            key=lambda x: results[x]['overall_geodesic_score'])
    
    print("\nRanking by Overall Geodesic Score (Lower is Better):")
    for i, schedule in enumerate(sorted_schedules, 1):
        score = results[schedule]['overall_geodesic_score']
        print(f"{i}. {schedule.upper()}: {score:.6f}")
    
    # Save results
    results_file = "quick_geodesic_test_results.json"
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
    plot_geodesic_deviations(results, save_path="quick_geodesic_test_plots.png")
    
    print("\nQuick geodesic test completed!")
    
    return results


def visualize_noise_schedule_comparison():
    """Visualize the different noise schedules for comparison."""
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    schedule_types = ["linear", "cosine", "quadratic", "exponential", "geodesic"]
    
    plt.figure(figsize=(15, 10))
    
    for schedule_type in schedule_types:
        noise_schedule = NoiseSchedule(T=T, schedule_type=schedule_type, 
                                     beta_start=beta_start, beta_end=beta_end)
        
        timesteps = np.arange(T)
        betas = noise_schedule.betas.cpu().numpy()
        alpha_bars = noise_schedule.alpha_bars.cpu().numpy()
        
        plt.subplot(2, 3, 1)
        plt.plot(timesteps, betas, label=schedule_type, alpha=0.8)
        plt.title('Beta Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('Beta')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(timesteps, alpha_bars, label=schedule_type, alpha=0.8)
        plt.title('Alpha Bar Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('Alpha Bar')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(timesteps, 1 - alpha_bars, label=schedule_type, alpha=0.8)
        plt.title('Noise Variance Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('1 - Alpha Bar')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.plot(timesteps, np.sqrt(1 - alpha_bars), label=schedule_type, alpha=0.8)
        plt.title('Noise Standard Deviation Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('sqrt(1 - Alpha Bar)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add marginal std visualization for geodesic comparison
        if schedule_type == "geodesic":
            plt.subplot(2, 3, 5)
            marginal_std = noise_schedule.get_marginal_std(torch.arange(T, device=noise_schedule.betas.device)).cpu().numpy()
            plt.plot(timesteps, marginal_std, label='W₂ Geodesic', color='purple', linewidth=2)
            plt.title('Marginal Standard Deviation (W₂ Geodesic)')
            plt.xlabel('Timestep')
            plt.ylabel('Marginal Std')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Show the theoretical W₂ geodesic path
            plt.subplot(2, 3, 6)
            eps = beta_start
            theoretical_std = (1 - timesteps/T) * eps + (timesteps/T)
            plt.plot(timesteps, theoretical_std, label='Theoretical W₂', color='red', linestyle='--', linewidth=2)
            plt.plot(timesteps, marginal_std, label='Implemented W₂', color='purple', linewidth=2)
            plt.title('W₂ Geodesic Path Comparison')
            plt.xlabel('Timestep')
            plt.ylabel('Standard Deviation')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("noise_schedule_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Noise schedule comparison visualization saved to: noise_schedule_comparison.png")


if __name__ == "__main__":
    print("Geodesic Evaluation Quick Test")
    print("=" * 40)
    
    # First, visualize the noise schedules
    print("1. Visualizing noise schedules...")
    visualize_noise_schedule_comparison()
    
    # Then run the quick test
    print("\n2. Running quick geodesic evaluation...")
    results = quick_geodesic_test(n_interpolation_steps=100, n_samples_per_step=50)
    
    print("\nQuick test completed! Check the generated files:")
    print("- quick_geodesic_test_results.json: Detailed results")
    print("- quick_geodesic_test_plots.png: Visualization plots")
    print("- noise_schedule_comparison.png: Noise schedule comparison") 