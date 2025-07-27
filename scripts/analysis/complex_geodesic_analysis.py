import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from data.gmm_dataset import generate_complex_gmm_data
from model import Denoiser
from geodesic_evaluation import (
    NoiseSchedule, 
    evaluate_conditional_geodesic, 
    evaluate_marginal_geodesic,
    plot_geodesic_deviations
)

def get_beta_schedule(T: int, schedule_type: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02):
    """Get beta schedule based on the specified type."""
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, T)
    elif schedule_type == "cosine":
        steps = torch.linspace(0, 1, T)
        return beta_start + (beta_end - beta_start) * (1 - torch.cos(steps * torch.pi / 2))
    elif schedule_type == "quadratic":
        steps = torch.linspace(0, 1, T)
        return beta_start + (beta_end - beta_start) * steps ** 2
    elif schedule_type == "exponential":
        steps = torch.linspace(0, 1, T)
        return beta_start + (beta_end - beta_start) * (torch.exp(steps) - 1) / (np.e - 1)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

def load_best_checkpoint_for_schedule(schedule_type):
    """Load the best checkpoint for a given schedule based on test loss."""
    # Load evaluation results to find best checkpoint
    with open('test_results_complex/complex_evaluation_metrics.json', 'r') as f:
        data = json.load(f)
    
    # Filter for the specific schedule
    schedule_data = [entry for entry in data if entry['schedule_type'] == schedule_type]
    
    if not schedule_data:
        raise ValueError(f"No data found for schedule: {schedule_type}")
    
    # Find best by test loss
    best_checkpoint = min(schedule_data, key=lambda x: x['test_loss'])
    epoch = best_checkpoint['epoch']
    
    # Construct checkpoint path
    checkpoint_path = f'checkpoints_complex/{schedule_type}_complex/model_epoch_{epoch:04d}.pt'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path, epoch

def create_complex_geodesic_analysis():
    """Perform geodesic analysis for complex dataset with all noise schedules."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate complex dataset
    print("Generating complex dataset...")
    data_samples = generate_complex_gmm_data(10000, n_components=12)
    data_samples = torch.FloatTensor(data_samples).to(device)
    
    # Define noise schedules to evaluate
    schedule_types = ["linear", "cosine", "quadratic", "exponential"]
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    # Parameters for geodesic evaluation
    n_interpolation_steps = 100
    n_samples_per_step = 50
    
    # Evaluate each schedule
    results = {}
    
    for schedule_type in schedule_types:
        print(f"\n{'='*40}")
        print(f"Evaluating {schedule_type} noise schedule")
        print(f"{'='*40}")
        
        try:
            # Load best checkpoint for this schedule
            checkpoint_path, epoch = load_best_checkpoint_for_schedule(schedule_type)
            print(f"Using checkpoint: {os.path.basename(checkpoint_path)} (epoch {epoch})")
            
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
                'checkpoint_epoch': epoch,
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
            
        except Exception as e:
            print(f"Error evaluating {schedule_type} schedule: {e}")
            continue
    
    # Create results summary
    print(f"\n{'='*40}")
    print("COMPLEX DATASET GEODESIC ANALYSIS RESULTS")
    print(f"{'='*40}")
    
    # Sort by overall score (lower is better)
    sorted_schedules = sorted(results.keys(), 
                            key=lambda x: results[x]['overall_geodesic_score'])
    
    print("\nRanking by Overall Geodesic Score (Lower is Better):")
    for i, schedule in enumerate(sorted_schedules, 1):
        score = results[schedule]['overall_geodesic_score']
        conditional = results[schedule]['conditional_geodesic_deviation']
        marginal = results[schedule]['marginal_geodesic_deviation']
        epoch = results[schedule]['checkpoint_epoch']
        print(f"{i}. {schedule.upper()}: {score:.6f} (Epoch {epoch})")
        print(f"    Conditional: {conditional:.6f}, Marginal: {marginal:.6f}")
    
    # Save results
    results_file = "test_results_complex/complex_geodesic_analysis_results.json"
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
    plot_geodesic_deviations(results, save_path="test_results_complex/complex_geodesic_analysis_plots.png")
    
    print("\nComplex geodesic analysis completed!")
    
    return results

def create_complex_geodesic_comparison_plots(results):
    """Create detailed comparison plots for complex geodesic analysis."""
    
    if not results:
        print("No results available for plotting.")
        return
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Complex Dataset: Geodesic Analysis by Noise Schedule', fontsize=16, fontweight='bold')
    
    schedules = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Overall Geodesic Score
    ax1 = axes[0, 0]
    scores = [results[schedule]['overall_geodesic_score'] for schedule in schedules]
    bars = ax1.bar(schedules, scores, color=colors, alpha=0.8)
    ax1.set_title('Overall Geodesic Score (Lower is Better)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Geodesic Score', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Conditional Geodesic Deviation
    ax2 = axes[0, 1]
    conditional_scores = [results[schedule]['conditional_geodesic_deviation'] for schedule in schedules]
    bars = ax2.bar(schedules, conditional_scores, color=colors, alpha=0.8)
    ax2.set_title('Conditional Geodesic Deviation', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Deviation', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, conditional_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(conditional_scores)*0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Marginal Geodesic Deviation
    ax3 = axes[1, 0]
    marginal_scores = [results[schedule]['marginal_geodesic_deviation'] for schedule in schedules]
    bars = ax3.bar(schedules, marginal_scores, color=colors, alpha=0.8)
    ax3.set_title('Marginal Geodesic Deviation', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Deviation', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, marginal_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(marginal_scores)*0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    best_overall = min(results.keys(), key=lambda x: results[x]['overall_geodesic_score'])
    best_conditional = min(results.keys(), key=lambda x: results[x]['conditional_geodesic_deviation'])
    best_marginal = min(results.keys(), key=lambda x: results[x]['marginal_geodesic_deviation'])
    
    summary_text = f"""
    🏆 BEST PERFORMERS:
    
    Overall Geodesic: {best_overall.upper()}
    ({results[best_overall]['overall_geodesic_score']:.4f})
    
    Conditional: {best_conditional.upper()}
    ({results[best_conditional]['conditional_geodesic_deviation']:.4f})
    
    Marginal: {best_marginal.upper()}
    ({results[best_marginal]['marginal_geodesic_deviation']:.4f})
    
    📊 ANALYSIS SUMMARY:
    Dataset: Complex GMM (12 components)
    Checkpoints: Best test loss per schedule
    Interpolation Steps: 100
    Samples per Step: 50
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.savefig('test_results_complex/complex_geodesic_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Complex geodesic comparison plot saved as: complex_geodesic_comparison.png")

def create_noise_schedule_visualization():
    """Create visualization of noise schedules for complex case."""
    T = 1000
    beta_start = 1e-4
    beta_end = 0.02
    
    schedule_types = ["linear", "cosine", "quadratic", "exponential"]
    
    plt.figure(figsize=(12, 8))
    
    for schedule_type in schedule_types:
        noise_schedule = NoiseSchedule(T=T, schedule_type=schedule_type, 
                                     beta_start=beta_start, beta_end=beta_end)
        
        timesteps = np.arange(T)
        betas = noise_schedule.betas.cpu().numpy()
        alpha_bars = noise_schedule.alpha_bars.cpu().numpy()
        
        plt.subplot(2, 2, 1)
        plt.plot(timesteps, betas, label=schedule_type, alpha=0.8)
        plt.title('Beta Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('Beta')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(timesteps, alpha_bars, label=schedule_type, alpha=0.8)
        plt.title('Alpha Bar Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('Alpha Bar')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(timesteps, 1 - alpha_bars, label=schedule_type, alpha=0.8)
        plt.title('Noise Variance Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('1 - Alpha Bar')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(timesteps, np.sqrt(1 - alpha_bars), label=schedule_type, alpha=0.8)
        plt.title('Noise Standard Deviation Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('sqrt(1 - Alpha Bar)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("test_results_complex/complex_noise_schedule_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Complex noise schedule comparison visualization saved to: complex_noise_schedule_comparison.png")

def main():
    """Main function to perform complex geodesic analysis."""
    print("Complex Dataset: Geodesic Analysis")
    print("=" * 40)
    
    # First, visualize the noise schedules
    print("1. Visualizing noise schedules...")
    create_noise_schedule_visualization()
    
    # Then run the geodesic analysis
    print("\n2. Running geodesic analysis...")
    results = create_complex_geodesic_analysis()
    
    # Create detailed comparison plots
    print("\n3. Creating detailed comparison plots...")
    create_complex_geodesic_comparison_plots(results)
    
    print("\nComplex geodesic analysis completed! Check the generated files:")
    print("- complex_geodesic_analysis_results.json: Detailed results")
    print("- complex_geodesic_analysis_plots.png: Geodesic deviation plots")
    print("- complex_geodesic_comparison.png: Comparison visualization")
    print("- complex_noise_schedule_comparison.png: Noise schedule comparison")

if __name__ == "__main__":
    main() 