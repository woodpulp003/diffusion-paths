import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def create_detailed_visualization():
    """Create detailed visualizations of the learning progress."""
    
    # Load all the saved samples
    sample_files = glob.glob("test_results/samples_epoch_*.npy")
    sample_files.sort()
    
    if not sample_files:
        print("No sample files found!")
        return
    
    print(f"Found {len(sample_files)} sample files")
    
    # Load samples
    epoch_samples = {}
    for file in sample_files:
        epoch = int(file.split('_')[-1].split('.')[0])
        samples = np.load(file)
        epoch_samples[epoch] = samples
        print(f"Epoch {epoch:4d}: {samples.shape[0]} samples, range [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Create individual plots for selected epochs
    os.makedirs("test_results/individual_plots", exist_ok=True)
    
    # Select key epochs for visualization
    if len(epoch_samples) <= 20:
        selected_epochs = sorted(epoch_samples.keys())
    else:
        # For many epochs, select key milestones
        selected_epochs = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        selected_epochs = [e for e in selected_epochs if e in epoch_samples.keys()]
    
    # Create grid plot
    cols = 4
    rows = (len(selected_epochs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if len(selected_epochs) > 1 else [axes]
    
    for i, epoch in enumerate(selected_epochs):
        if epoch in epoch_samples:
            samples = epoch_samples[epoch]
            
            # Create individual plot
            plt.figure(figsize=(8, 6))
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=15, c='red')
            plt.title(f'Generated Samples - Epoch {epoch}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.xlim(-8, 8)
            plt.ylim(-8, 8)
            plt.savefig(f'test_results/individual_plots/epoch_{epoch:04d}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add to subplot
            if i < len(axes):
                axes[i].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10, c='red')
                axes[i].set_title(f'Epoch {epoch}')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('Y')
                axes[i].axis('equal')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim(-8, 8)
                axes[i].set_ylim(-8, 8)
    
    # Hide unused subplots
    for i in range(len(selected_epochs), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('test_results/learning_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a comparison with original data
    if os.path.exists("test_results/generated_samples.npy"):
        original_samples = np.load("test_results/generated_samples.npy")
        
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        axes = axes.flatten()
        
        # Original data
        axes[0].scatter(original_samples[:, 0], original_samples[:, 1], alpha=0.6, s=10, c='blue')
        axes[0].set_title('Original GMM Data')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].axis('equal')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(-8, 8)
        axes[0].set_ylim(-8, 8)
        
        # Generated samples from selected epochs
        for i, epoch in enumerate(selected_epochs):
            if epoch in epoch_samples and i + 1 < len(axes):
                samples = epoch_samples[epoch]
                ax_idx = i + 1
                
                axes[ax_idx].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=10, c='red')
                axes[ax_idx].set_title(f'Epoch {epoch}')
                axes[ax_idx].set_xlabel('X')
                axes[ax_idx].set_ylabel('Y')
                axes[ax_idx].axis('equal')
                axes[ax_idx].grid(True, alpha=0.3)
                axes[ax_idx].set_xlim(-8, 8)
                axes[ax_idx].set_ylim(-8, 8)
        
        # Hide unused subplots
        for i in range(len(selected_epochs) + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('test_results/full_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create loss progression plot
    epochs = sorted(epoch_samples.keys())
    sample_ranges = [epoch_samples[e].max() - epoch_samples[e].min() for e in epochs]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, sample_ranges, 'b-o')
    plt.title('Sample Range Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Sample Range (max - min)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sample_std = [np.std(epoch_samples[e]) for e in epochs]
    plt.plot(epochs, sample_std, 'r-o')
    plt.title('Sample Standard Deviation Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Sample Std Dev')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created visualizations:")
    print(f"  - Individual plots: test_results/individual_plots/")
    print(f"  - Learning progress: test_results/learning_progress.png")
    print(f"  - Full comparison: test_results/full_comparison.png")
    print(f"  - Training metrics: test_results/training_metrics.png")

if __name__ == "__main__":
    create_detailed_visualization() 