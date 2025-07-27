import json

def display_complex_geodesic_results():
    """Display the complex geodesic analysis results."""
    
    with open('test_results_complex/complex_geodesic_analysis_results.json', 'r') as f:
        data = json.load(f)
    
    print("🏆 COMPLEX GEODESIC ANALYSIS RESULTS")
    print("=" * 50)
    
    # Sort by overall score (lower is better)
    sorted_schedules = sorted(data.keys(), 
                            key=lambda x: data[x]['overall_geodesic_score'])
    
    print("\nRanking by Overall Geodesic Score (Lower is Better):")
    for i, schedule in enumerate(sorted_schedules, 1):
        results = data[schedule]
        print(f"{i}. {schedule.upper()}:")
        print(f"   Overall Score: {results['overall_geodesic_score']:.6f}")
        print(f"   Conditional: {results['conditional_geodesic_deviation']:.6f}")
        print(f"   Marginal: {results['marginal_geodesic_deviation']:.6f}")
        print(f"   Epoch: {results['checkpoint_epoch']}")
        print()
    
    # Find best performers
    best_overall = min(data.keys(), key=lambda x: data[x]['overall_geodesic_score'])
    best_conditional = min(data.keys(), key=lambda x: data[x]['conditional_geodesic_deviation'])
    best_marginal = min(data.keys(), key=lambda x: data[x]['marginal_geodesic_deviation'])
    
    print("🏆 BEST PERFORMERS:")
    print(f"  Overall Geodesic: {best_overall.upper()} ({data[best_overall]['overall_geodesic_score']:.6f})")
    print(f"  Conditional: {best_conditional.upper()} ({data[best_conditional]['conditional_geodesic_deviation']:.6f})")
    print(f"  Marginal: {best_marginal.upper()} ({data[best_marginal]['marginal_geodesic_deviation']:.6f})")
    
    print("\n📊 ANALYSIS SUMMARY:")
    print("  Dataset: Complex GMM (12 components)")
    print("  Checkpoints: Best test loss per schedule")
    print("  Interpolation Steps: 100")
    print("  Samples per Step: 50")
    print("  Generated files:")
    print("    - complex_geodesic_analysis_results.json")
    print("    - complex_geodesic_analysis_plots.png")
    print("    - complex_noise_schedule_comparison.png")

if __name__ == "__main__":
    display_complex_geodesic_results() 