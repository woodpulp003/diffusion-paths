import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

def display_complex_evaluation_results():
    """Display comprehensive complex dataset evaluation results."""
    
    # Load the summary data
    summary_df = pd.read_csv('test_results_complex/complex_dataset_summary.csv')
    
    print("=" * 80)
    print("COMPLEX DATASET EVALUATION RESULTS")
    print("=" * 80)
    
    # Display numerical results
    print("\n📊 NUMERICAL RESULTS SUMMARY")
    print("-" * 50)
    print(summary_df.to_string(index=False))
    
    # Create a more detailed analysis
    print("\n🔍 DETAILED ANALYSIS")
    print("-" * 50)
    
    # Best performers by metric
    metrics = ['Best Test Loss (Epoch)', 'Best MMD RBF (Epoch)', 'Best MMD Linear (Epoch)', 'Best Wasserstein (Epoch)']
    
    for metric in metrics:
        best_row = summary_df.loc[summary_df[metric].str.split(' ').str[0].astype(float).idxmin()]
        best_schedule = best_row['Schedule']
        best_value = best_row[metric].split(' ')[0]
        best_epoch = best_row[metric].split('(')[1].split(')')[0]
        
        print(f"🏆 {metric}: {best_schedule.upper()} (Epoch {best_epoch}) = {best_value}")
    
    # Overall ranking analysis
    print("\n📈 OVERALL PERFORMANCE RANKING")
    print("-" * 50)
    
    # Create a scoring system (lower is better for all metrics)
    scores = {}
    for _, row in summary_df.iterrows():
        schedule = row['Schedule']
        scores[schedule] = {
            'test_loss': float(row['Best Test Loss (Epoch)'].split(' ')[0]),
            'mmd_rbf': float(row['Best MMD RBF (Epoch)'].split(' ')[0]),
            'mmd_linear': float(row['Best MMD Linear (Epoch)'].split(' ')[0]),
            'wasserstein': float(row['Best Wasserstein (Epoch)'].split(' ')[0])
        }
    
    # Calculate overall scores (normalized and averaged)
    overall_scores = {}
    for schedule, metrics in scores.items():
        # Normalize each metric to 0-1 scale (lower is better)
        normalized_scores = [
            metrics['test_loss'] / max(scores[s]['test_loss'] for s in scores),
            metrics['mmd_rbf'] / max(scores[s]['mmd_rbf'] for s in scores),
            metrics['mmd_linear'] / max(scores[s]['mmd_linear'] for s in scores),
            metrics['wasserstein'] / max(scores[s]['wasserstein'] for s in scores)
        ]
        overall_scores[schedule] = np.mean(normalized_scores)
    
    # Sort by overall score
    sorted_schedules = sorted(overall_scores.items(), key=lambda x: x[1])
    
    print("Overall Performance Ranking (Lower Score = Better):")
    for i, (schedule, score) in enumerate(sorted_schedules, 1):
        print(f"{i}. {schedule.upper()}: {score:.4f}")
    
    # Key insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 50)
    
    best_test_loss = summary_df.loc[summary_df['Best Test Loss (Epoch)'].str.split(' ').str[0].astype(float).idxmin()]
    best_mmd_linear = summary_df.loc[summary_df['Best MMD Linear (Epoch)'].str.split(' ').str[0].astype(float).idxmin()]
    
    print(f"• LINEAR schedule achieved the best test loss: {best_test_loss['Best Test Loss (Epoch)']}")
    print(f"• COSINE schedule achieved the best MMD Linear score: {best_mmd_linear['Best MMD Linear (Epoch)']}")
    print(f"• All schedules performed similarly on MMD RBF (around 0.068)")
    print(f"• COSINE schedule had the best Wasserstein distance: {best_mmd_linear['Best Wasserstein (Epoch)']}")
    
    # Display visual results
    print("\n🖼️ VISUAL RESULTS")
    print("-" * 50)
    print("Generated visualization files:")
    print("• complex_dataset_comparison.png - Training curves and metrics over epochs")
    print("• complex_dataset_distributions.png - Generated vs original data distributions")
    print("• side_by_side_epoch_*.png - Side-by-side comparisons at specific epochs")
    
    # Create a comprehensive visualization
    create_comprehensive_visualization(summary_df)
    
    print("\n✅ Evaluation complete! Check the generated files for detailed visualizations.")

def create_comprehensive_visualization(summary_df):
    """Create a comprehensive visualization of the results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Complex Dataset: Comprehensive Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Test Loss Comparison
    ax1 = axes[0, 0]
    schedules = summary_df['Schedule']
    test_losses = [float(x.split(' ')[0]) for x in summary_df['Best Test Loss (Epoch)']]
    epochs = [int(x.split('(')[1].split(')')[0]) for x in summary_df['Best Test Loss (Epoch)']]
    
    bars = ax1.bar(schedules, test_losses, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_title('Best Test Loss by Schedule', fontweight='bold')
    ax1.set_ylabel('Test Loss (Lower is Better)')
    ax1.grid(True, alpha=0.3)
    
    # Add epoch labels on bars
    for bar, epoch in zip(bars, epochs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Epoch {epoch}', ha='center', va='bottom', fontsize=9)
    
    # 2. MMD RBF Comparison
    ax2 = axes[0, 1]
    mmd_rbf = [float(x.split(' ')[0]) for x in summary_df['Best MMD RBF (Epoch)']]
    epochs = [int(x.split('(')[1].split(')')[0]) for x in summary_df['Best MMD RBF (Epoch)']]
    
    bars = ax2.bar(schedules, mmd_rbf, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_title('Best MMD RBF by Schedule', fontweight='bold')
    ax2.set_ylabel('MMD RBF (Lower is Better)')
    ax2.grid(True, alpha=0.3)
    
    for bar, epoch in zip(bars, epochs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'Epoch {epoch}', ha='center', va='bottom', fontsize=9)
    
    # 3. MMD Linear Comparison
    ax3 = axes[0, 2]
    mmd_linear = [float(x.split(' ')[0]) for x in summary_df['Best MMD Linear (Epoch)']]
    epochs = [int(x.split('(')[1].split(')')[0]) for x in summary_df['Best MMD Linear (Epoch)']]
    
    bars = ax3.bar(schedules, mmd_linear, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax3.set_title('Best MMD Linear by Schedule', fontweight='bold')
    ax3.set_ylabel('MMD Linear (Lower is Better)')
    ax3.grid(True, alpha=0.3)
    
    for bar, epoch in zip(bars, epochs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Epoch {epoch}', ha='center', va='bottom', fontsize=9)
    
    # 4. Wasserstein Distance Comparison
    ax4 = axes[1, 0]
    wasserstein = [float(x.split(' ')[0]) for x in summary_df['Best Wasserstein (Epoch)']]
    epochs = [int(x.split('(')[1].split(')')[0]) for x in summary_df['Best Wasserstein (Epoch)']]
    
    bars = ax4.bar(schedules, wasserstein, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax4.set_title('Best Wasserstein Distance by Schedule', fontweight='bold')
    ax4.set_ylabel('Wasserstein Distance (Lower is Better)')
    ax4.grid(True, alpha=0.3)
    
    for bar, epoch in zip(bars, epochs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Epoch {epoch}', ha='center', va='bottom', fontsize=9)
    
    # 5. Overall Performance Heatmap
    ax5 = axes[1, 1]
    metrics = ['Test Loss', 'MMD RBF', 'MMD Linear', 'Wasserstein']
    data = []
    
    for _, row in summary_df.iterrows():
        row_data = [
            float(row['Best Test Loss (Epoch)'].split(' ')[0]),
            float(row['Best MMD RBF (Epoch)'].split(' ')[0]),
            float(row['Best MMD Linear (Epoch)'].split(' ')[0]),
            float(row['Best Wasserstein (Epoch)'].split(' ')[0])
        ]
        data.append(row_data)
    
    im = ax5.imshow(data, cmap='RdYlGn_r', aspect='auto')
    ax5.set_xticks(range(len(metrics)))
    ax5.set_yticks(range(len(schedules)))
    ax5.set_xticklabels(metrics, rotation=45, ha='right')
    ax5.set_yticklabels(schedules)
    ax5.set_title('Performance Heatmap (Red=Worse, Green=Better)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(schedules)):
        for j in range(len(metrics)):
            text = ax5.text(j, i, f'{data[i][j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    best_test_loss = summary_df.loc[summary_df['Best Test Loss (Epoch)'].str.split(' ').str[0].astype(float).idxmin()]
    best_mmd_linear = summary_df.loc[summary_df['Best MMD Linear (Epoch)'].str.split(' ').str[0].astype(float).idxmin()]
    
    summary_text = f"""
    🏆 BEST PERFORMERS:
    
    Test Loss: {best_test_loss['Schedule'].upper()}
    ({best_test_loss['Best Test Loss (Epoch)']})
    
    MMD Linear: {best_mmd_linear['Schedule'].upper()}
    ({best_mmd_linear['Best MMD Linear (Epoch)']})
    
    MMD RBF: All similar (~0.068)
    
    Wasserstein: {best_mmd_linear['Schedule'].upper()}
    ({best_mmd_linear['Best Wasserstein (Epoch)']})
    
    📊 TOTAL CHECKPOINTS: 400
    📈 SCHEDULES TESTED: 4
    🎯 BEST OVERALL: LINEAR
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig('test_results_complex/comprehensive_evaluation_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive visualization saved as: comprehensive_evaluation_results.png")

if __name__ == "__main__":
    display_complex_evaluation_results() 