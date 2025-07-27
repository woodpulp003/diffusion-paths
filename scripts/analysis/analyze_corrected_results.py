import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def analyze_corrected_results():
    """Analyze the comprehensive evaluation results with corrected beta schedules."""
    
    print("🔬 Analysis of Corrected Beta Schedule Results")
    print("=" * 60)
    
    # Load the comprehensive evaluation results
    csv_file = "comprehensive_evaluation_results/comprehensive_evaluation_metrics.csv"
    json_file = "comprehensive_evaluation_results/comprehensive_evaluation_metrics.json"
    
    if not os.path.exists(csv_file):
        print(f"❌ Results file not found: {csv_file}")
        return
    
    # Load data
    df = pd.read_csv(csv_file)
    
    print(f"📊 Loaded {len(df)} evaluation results")
    print(f"📈 Schedules evaluated: {df['schedule_type'].unique()}")
    
    # Analyze each schedule
    schedules = df['schedule_type'].unique()
    
    print("\n" + "="*60)
    print("📊 DETAILED SCHEDULE ANALYSIS")
    print("="*60)
    
    best_results = {}
    
    for schedule in schedules:
        schedule_data = df[df['schedule_type'] == schedule]
        
        print(f"\n🎯 {schedule.upper()} SCHEDULE:")
        print("-" * 40)
        
        # Best performance for each metric
        best_test_loss = schedule_data.loc[schedule_data['test_loss'].idxmin()]
        best_mmd_rbf = schedule_data.loc[schedule_data['mmd_rbf'].idxmin()]
        best_mmd_linear = schedule_data.loc[schedule_data['mmd_linear'].idxmin()]
        best_wass = schedule_data.loc[schedule_data['wass_dist'].idxmin()]
        
        print(f"  Best Test Loss: {best_test_loss['test_loss']:.6f} (Epoch {best_test_loss['epoch']})")
        print(f"  Best MMD RBF: {best_mmd_rbf['mmd_rbf']:.6f} (Epoch {best_mmd_rbf['epoch']})")
        print(f"  Best MMD Linear: {best_mmd_linear['mmd_linear']:.6f} (Epoch {best_mmd_linear['epoch']})")
        print(f"  Best Wasserstein: {best_wass['wass_dist']:.6f} (Epoch {best_wass['epoch']})")
        
        # Overall trend analysis
        initial_loss = schedule_data.iloc[0]['test_loss']
        final_loss = schedule_data.iloc[-1]['test_loss']
        improvement = initial_loss - final_loss
        
        print(f"  Loss Improvement: {improvement:.6f} ({initial_loss:.6f} → {final_loss:.6f})")
        
        # Store best results for comparison
        best_results[schedule] = {
            'best_test_loss': best_test_loss['test_loss'],
            'best_epoch': best_test_loss['epoch'],
            'final_loss': final_loss,
            'improvement': improvement
        }
    
    # Overall ranking
    print("\n" + "="*60)
    print("🏆 OVERALL RANKING BY BEST TEST LOSS")
    print("="*60)
    
    # Sort by best test loss (lower is better)
    sorted_schedules = sorted(best_results.items(), key=lambda x: x[1]['best_test_loss'])
    
    for i, (schedule, results) in enumerate(sorted_schedules, 1):
        print(f"{i}. {schedule.upper()}: {results['best_test_loss']:.6f} (Epoch {results['best_epoch']})")
    
    # Improvement ranking
    print("\n" + "="*60)
    print("📈 RANKING BY LOSS IMPROVEMENT")
    print("="*60)
    
    # Sort by improvement (higher is better)
    sorted_improvement = sorted(best_results.items(), key=lambda x: x[1]['improvement'], reverse=True)
    
    for i, (schedule, results) in enumerate(sorted_improvement, 1):
        print(f"{i}. {schedule.upper()}: {results['improvement']:.6f} improvement")
    
    # Create comparison plots
    create_comparison_plots(df)
    
    print("\n" + "="*60)
    print("✅ Analysis completed!")
    print("📊 Check the generated plots for visual comparisons")
    print("="*60)


def create_comparison_plots(df):
    """Create detailed comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Corrected Beta Schedule Comparison', fontsize=16, fontweight='bold')
    
    schedules = df['schedule_type'].unique()
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, schedule in enumerate(schedules):
        schedule_data = df[df['schedule_type'] == schedule]
        color = colors[i % len(colors)]
        
        epochs = schedule_data['epoch']
        test_losses = schedule_data['test_loss']
        mmd_rbf = schedule_data['mmd_rbf']
        mmd_linear = schedule_data['mmd_linear']
        wass_dist = schedule_data['wass_dist']
        
        # Test Loss
        axes[0, 0].plot(epochs, test_losses, color=color, linewidth=2, marker='o', 
                        label=schedule.upper(), alpha=0.8)
        axes[0, 0].set_title('Test Loss Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MMD RBF
        axes[0, 1].plot(epochs, mmd_rbf, color=color, linewidth=2, marker='s', 
                        label=schedule.upper(), alpha=0.8)
        axes[0, 1].set_title('MMD (RBF) Comparison', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MMD')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MMD Linear
        axes[1, 0].plot(epochs, mmd_linear, color=color, linewidth=2, marker='^', 
                        label=schedule.upper(), alpha=0.8)
        axes[1, 0].set_title('MMD (Linear) Comparison', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MMD')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Wasserstein Distance
        axes[1, 1].plot(epochs, wass_dist, color=color, linewidth=2, marker='d', 
                        label=schedule.upper(), alpha=0.8)
        axes[1, 1].set_title('Wasserstein Distance Comparison', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('corrected_beta_schedule_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Comparison plots saved as: corrected_beta_schedule_comparison.png")


def compare_with_previous_results():
    """Compare current results with previous results if available."""
    
    current_file = "comprehensive_evaluation_results/comprehensive_evaluation_metrics.csv"
    previous_file = "comprehensive_evaluation_results/comprehensive_evaluation_metrics.csv.bak"
    
    if not os.path.exists(previous_file):
        print("📝 No previous results found for comparison")
        return
    
    print("\n" + "="*60)
    print("🔄 COMPARISON WITH PREVIOUS RESULTS")
    print("="*60)
    
    current_df = pd.read_csv(current_file)
    previous_df = pd.read_csv(previous_file)
    
    schedules = current_df['schedule_type'].unique()
    
    for schedule in schedules:
        current_data = current_df[current_df['schedule_type'] == schedule]
        previous_data = previous_df[previous_df['schedule_type'] == schedule]
        
        if len(previous_data) == 0:
            continue
            
        current_best = current_data['test_loss'].min()
        previous_best = previous_data['test_loss'].min()
        
        improvement = previous_best - current_best
        
        print(f"\n{schedule.upper()}:")
        print(f"  Previous Best: {previous_best:.6f}")
        print(f"  Current Best:  {current_best:.6f}")
        print(f"  Improvement:   {improvement:.6f}")


if __name__ == "__main__":
    analyze_corrected_results()
    compare_with_previous_results() 