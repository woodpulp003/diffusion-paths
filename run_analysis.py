#!/usr/bin/env python3
"""
Diffusion Paths Analysis Runner

This script provides a simple interface to run different analyses
in the cleaned-up project structure.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_banner():
    """Print the project banner."""
    print("=" * 60)
    print("🎯 DIFFUSION PATHS: COMPREHENSIVE ANALYSIS RUNNER")
    print("=" * 60)
    print()

def print_menu():
    """Print the main menu."""
    print("📋 Available Analyses:")
    print()
    print("🏋️  TRAINING:")
    print("  1. Train simple dataset models")
    print("  2. Train complex dataset models")
    print()
    print("📊 EVALUATION:")
    print("  3. Evaluate simple dataset")
    print("  4. Evaluate complex dataset")
    print()
    print("🔬 ANALYSIS:")
    print("  5. Run geodesic analysis")
    print("  6. Quick geodesic test")
    print()
    print("📈 VISUALIZATION:")
    print("  7. Create epoch distribution plots")
    print("  8. Create side-by-side comparisons")
    print("  9. Generate evaluation summary")
    print()
    print("📚 DOCUMENTATION:")
    print("  10. Show project structure")
    print("  11. Show results summary")
    print()
    print("0. Exit")
    print()

def run_command(command, description):
    """Run a command with proper error handling."""
    print(f"\n🚀 Running: {description}")
    print("-" * 40)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def show_project_structure():
    """Show the cleaned-up project structure."""
    print("\n📁 PROJECT STRUCTURE:")
    print("=" * 40)
    
    structure = """
diffusion-paths/
├── data/                          # Data generation
│   └── gmm_dataset.py
├── scripts/                       # Organized scripts
│   ├── training/                  # Training scripts
│   │   ├── train.py
│   │   └── train_complex_dataset.py
│   ├── evaluation/                # Evaluation scripts
│   │   ├── test_model.py
│   │   ├── test_complex_model.py
│   │   └── evaluate_complex_dataset.py
│   ├── visualization/             # Visualization scripts
│   │   ├── create_complex_epoch_distributions.py
│   │   ├── create_complex_side_by_side_plots.py
│   │   └── complex_evaluation_summary.py
│   └── analysis/                  # Analysis scripts
│       ├── complex_geodesic_analysis.py
│       ├── geodesic_evaluation.py
│       ├── plot_geodesic_trajectories.py
│       └── quick_geodesic_test.py
├── test_results/                  # Simple dataset results
├── test_results_complex/          # Complex dataset results
├── checkpoints/                   # Model checkpoints
├── checkpoints_complex/           # Complex model checkpoints
├── model.py                       # Model architecture
├── requirements.txt               # Dependencies
└── README.md                     # Documentation
"""
    print(structure)

def show_results_summary():
    """Show a summary of key results."""
    print("\n📊 KEY RESULTS SUMMARY:")
    print("=" * 40)
    
    print("🏆 COMPLEX DATASET EVALUATION:")
    print("  • Best Overall: LINEAR schedule")
    print("  • Best MMD RBF: LINEAR schedule")
    print("  • Best MMD Linear: LINEAR schedule")
    print("  • Best Wasserstein: LINEAR schedule")
    print()
    
    print("🔬 GEODESIC ANALYSIS:")
    print("  • Best Overall Geodesic: LINEAR (0.199254)")
    print("  • Best Conditional: QUADRATIC (0.042255)")
    print("  • Best Marginal: LINEAR (0.353967)")
    print()
    
    print("📈 GENERATED FILES:")
    print("  • complex_evaluation_metrics.json/csv")
    print("  • complex_geodesic_analysis_results.json")
    print("  • Various visualization plots in test_results_complex/")

def main():
    """Main function to run the analysis runner."""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (0-11): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice == "1":
                run_command("python scripts/training/train.py", 
                          "Training simple dataset models")
            elif choice == "2":
                run_command("python scripts/training/train_complex_dataset.py", 
                          "Training complex dataset models")
            elif choice == "3":
                run_command("python scripts/evaluation/test_model.py", 
                          "Evaluating simple dataset")
            elif choice == "4":
                run_command("python scripts/evaluation/test_complex_model.py", 
                          "Evaluating complex dataset")
            elif choice == "5":
                run_command("python scripts/analysis/complex_geodesic_analysis.py", 
                          "Running geodesic analysis")
            elif choice == "6":
                run_command("python scripts/analysis/quick_geodesic_test.py", 
                          "Running quick geodesic test")
            elif choice == "7":
                run_command("python scripts/visualization/create_complex_epoch_distributions.py", 
                          "Creating epoch distribution plots")
            elif choice == "8":
                run_command("python scripts/visualization/create_complex_side_by_side_plots.py", 
                          "Creating side-by-side comparisons")
            elif choice == "9":
                run_command("python scripts/visualization/complex_evaluation_summary.py", 
                          "Generating evaluation summary")
            elif choice == "10":
                show_project_structure()
            elif choice == "11":
                show_results_summary()
            else:
                print("❌ Invalid choice. Please enter a number between 0-11.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main() 