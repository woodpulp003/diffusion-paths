import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import os

from geodesic_evaluation import (
    NoiseSchedule, 
    evaluate_conditional_geodesic, 
    evaluate_marginal_geodesic
)
from data.gmm_dataset import generate_gmm_data


def plot_geodesic_trajectories(n_interpolation_steps: int = 100, n_samples_per_step: int = 50):
    """
    Plot conditional and marginal geodesic deviations along with step number.
    
    Args:
        n_interpolation_steps: Number of interpolation steps to evaluate
        n_samples_per_step: Number of samples per step
    """
    print("Generating geodesic trajectory plots...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating GMM dataset...")
    data_samples = generate_gmm_data(n_samples=1000, n_components=8, radius=5.0, std=0.2)
    data_samples = torch.FloatTensor(data_samples).to(device)
    
    # Define noise schedules to evaluate
    schedule_types = ["linear", "cosine", "quadratic", "exponential"]
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    # Store results for plotting
    results = {}
    
    # Evaluate each schedule
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
        
        # Store results
        results[schedule_type] = {
            'conditional_deviations': conditional_results['conditional_deviations'],
            'marginal_deviations': marginal_results['marginal_deviations'],
            'conditional_avg': conditional_results['conditional_geodesic_deviation'],
            'marginal_avg': marginal_results['marginal_geodesic_deviation']
        }
    
    # Create plots
    create_geodesic_trajectory_plots(results, n_interpolation_steps)
    
    return results


def create_geodesic_trajectory_plots(results: Dict, n_interpolation_steps: int):
    """
    Create comprehensive plots of geodesic trajectories.
    
    Args:
        results: Dictionary of results for each schedule
        n_interpolation_steps: Number of interpolation steps
    """
    schedules = list(results.keys())
    timesteps = np.linspace(0, 1, n_interpolation_steps)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Geodesic Trajectories: Conditional and Marginal Deviations vs Step Number', fontsize=16)
    
    # Plot 1: Conditional geodesic deviations over time
    ax1 = axes[0, 0]
    for schedule in schedules:
        deviations = results[schedule]['conditional_deviations']
        ax1.plot(timesteps, deviations, label=schedule, alpha=0.8, linewidth=2)
    
    ax1.set_title('Conditional Geodesic Deviations vs Normalized Time')
    ax1.set_xlabel('Normalized Time (t/T)')
    ax1.set_ylabel('MMD Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to better see differences
    
    # Plot 2: Marginal geodesic deviations over time
    ax2 = axes[0, 1]
    for schedule in schedules:
        deviations = results[schedule]['marginal_deviations']
        ax2.plot(timesteps, deviations, label=schedule, alpha=0.8, linewidth=2)
    
    ax2.set_title('Marginal Geodesic Deviations vs Normalized Time')
    ax2.set_xlabel('Normalized Time (t/T)')
    ax2.set_ylabel('Sliced Wasserstein Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to better see differences
    
    # Plot 3: Combined view (both conditional and marginal for each schedule)
    ax3 = axes[1, 0]
    colors = ['blue', 'red', 'green', 'orange']
    for i, schedule in enumerate(schedules):
        conditional = results[schedule]['conditional_deviations']
        marginal = results[schedule]['marginal_deviations']
        
        ax3.plot(timesteps, conditional, label=f'{schedule} (conditional)', 
                color=colors[i], alpha=0.8, linewidth=2)
        ax3.plot(timesteps, marginal, label=f'{schedule} (marginal)', 
                color=colors[i], alpha=0.4, linewidth=2, linestyle='--')
    
    ax3.set_title('Combined View: Conditional vs Marginal Deviations')
    ax3.set_xlabel('Normalized Time (t/T)')
    ax3.set_ylabel('Deviation')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Average deviations comparison
    ax4 = axes[1, 1]
    schedule_names = list(results.keys())
    conditional_avgs = [results[s]['conditional_avg'] for s in schedule_names]
    marginal_avgs = [results[s]['marginal_avg'] for s in schedule_names]
    
    x = np.arange(len(schedule_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, conditional_avgs, width, label='Conditional', alpha=0.8)
    bars2 = ax4.bar(x + width/2, marginal_avgs, width, label='Marginal', alpha=0.8)
    
    ax4.set_title('Average Deviations Comparison')
    ax4.set_xlabel('Noise Schedule')
    ax4.set_ylabel('Average Deviation')
    ax4.set_xticks(x)
    ax4.set_xticklabels(schedule_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("geodesic_trajectories.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed plots
    create_detailed_trajectory_plots(results, timesteps)


def create_detailed_trajectory_plots(results: Dict, timesteps: np.ndarray):
    """
    Create detailed individual plots for each schedule.
    
    Args:
        results: Dictionary of results for each schedule
        timesteps: Array of timesteps
    """
    schedules = list(results.keys())
    
    # Create individual plots for each schedule
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Geodesic Trajectories by Schedule', fontsize=16)
    
    for i, schedule in enumerate(schedules):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        conditional = results[schedule]['conditional_deviations']
        marginal = results[schedule]['marginal_deviations']
        
        # Plot both deviations
        ax.plot(timesteps, conditional, label='Conditional', color='blue', linewidth=2)
        ax.plot(timesteps, marginal, label='Marginal', color='red', linewidth=2)
        
        # Add average lines
        conditional_avg = results[schedule]['conditional_avg']
        marginal_avg = results[schedule]['marginal_avg']
        ax.axhline(y=conditional_avg, color='blue', alpha=0.5, linestyle='--', 
                  label=f'Conditional Avg: {conditional_avg:.4f}')
        ax.axhline(y=marginal_avg, color='red', alpha=0.5, linestyle='--', 
                  label=f'Marginal Avg: {marginal_avg:.4f}')
        
        ax.set_title(f'{schedule.upper()} Schedule')
        ax.set_xlabel('Normalized Time (t/T)')
        ax.set_ylabel('Deviation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("detailed_geodesic_trajectories.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create comparison plot with early, middle, and late time points
    create_time_point_comparison(results, timesteps)


def create_time_point_comparison(results: Dict, timesteps: np.ndarray):
    """
    Create a comparison plot showing deviations at specific time points.
    
    Args:
        results: Dictionary of results for each schedule
        timesteps: Array of timesteps
    """
    schedules = list(results.keys())
    
    # Define time points of interest
    time_points = {
        'Early (t/T=0.1)': 0.1,
        'Middle (t/T=0.5)': 0.5,
        'Late (t/T=0.9)': 0.9
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Geodesic Deviations at Specific Time Points', fontsize=16)
    
    for i, (time_name, time_ratio) in enumerate(time_points.items()):
        ax = axes[i]
        
        # Find closest timestep index
        time_idx = np.argmin(np.abs(timesteps - time_ratio))
        
        conditional_values = [results[s]['conditional_deviations'][time_idx] for s in schedules]
        marginal_values = [results[s]['marginal_deviations'][time_idx] for s in schedules]
        
        x = np.arange(len(schedules))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, conditional_values, width, label='Conditional', alpha=0.8)
        bars2 = ax.bar(x + width/2, marginal_values, width, label='Marginal', alpha=0.8)
        
        ax.set_title(f'{time_name}')
        ax.set_xlabel('Noise Schedule')
        ax.set_ylabel('Deviation')
        ax.set_xticks(x)
        ax.set_xticklabels(schedules)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("time_point_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def load_and_plot_existing_results(results_file: str = "quick_geodesic_test_results.json"):
    """
    Load existing results and create trajectory plots.
    
    Args:
        results_file: Path to the JSON results file
    """
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Running new evaluation...")
        return plot_geodesic_trajectories()
    
    print(f"Loading existing results from {results_file}...")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract deviations for plotting
    results = {}
    for schedule, schedule_data in data.items():
        results[schedule] = {
            'conditional_deviations': schedule_data['conditional_deviations'],
            'marginal_deviations': schedule_data['marginal_deviations'],
            'conditional_avg': schedule_data['conditional_geodesic_deviation'],
            'marginal_avg': schedule_data['marginal_geodesic_deviation']
        }
    
    n_interpolation_steps = len(results[list(results.keys())[0]]['conditional_deviations'])
    timesteps = np.linspace(0, 1, n_interpolation_steps)
    
    # Create plots
    create_geodesic_trajectory_plots(results, n_interpolation_steps)
    
    return results


def main():
    """Main function to generate geodesic trajectory plots."""
    print("Geodesic Trajectory Plotting")
    print("=" * 40)
    
    # Try to load existing results first
    try:
        results = load_and_plot_existing_results()
        print("\nPlots generated from existing results!")
    except Exception as e:
        print(f"Could not load existing results: {e}")
        print("Running new evaluation...")
        results = plot_geodesic_trajectories()
    
    print("\nGenerated files:")
    print("- geodesic_trajectories.png: Main trajectory plots")
    print("- detailed_geodesic_trajectories.png: Individual schedule plots")
    print("- time_point_comparison.png: Time point comparison")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for schedule, data in results.items():
        print(f"\n{schedule.upper()}:")
        print(f"  Conditional Avg: {data['conditional_avg']:.6f}")
        print(f"  Marginal Avg: {data['marginal_avg']:.6f}")
        print(f"  Combined Avg: {(data['conditional_avg'] + data['marginal_avg'])/2:.6f}")


if __name__ == "__main__":
    main() 