import os
import json
import time
import glob
from datetime import datetime

def monitor_evaluation_progress():
    """Monitor the evaluation progress and show current status."""
    
    print("=" * 60)
    print("COMPLEX MODEL EVALUATION MONITOR")
    print("=" * 60)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if results directory exists
    results_dir = "test_results_complex"
    if not os.path.exists(results_dir):
        print("❌ Results directory not found yet.")
        return
    
    # Check for metrics files
    json_file = os.path.join(results_dir, "complex_evaluation_metrics.json")
    csv_file = os.path.join(results_dir, "complex_evaluation_metrics.csv")
    
    if os.path.exists(json_file):
        print("✅ Metrics JSON file found!")
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"📊 Evaluated checkpoints: {len(data)}")
        
        if data:
            # Show some statistics
            schedules = {}
            for entry in data:
                schedule = entry['schedule_type']
                if schedule not in schedules:
                    schedules[schedule] = []
                schedules[schedule].append(entry)
            
            print(f"\n📈 Progress by schedule:")
            for schedule, entries in schedules.items():
                epochs = [e['epoch'] for e in entries]
                print(f"  {schedule.upper()}: {len(entries)} checkpoints (epochs {min(epochs)}-{max(epochs)})")
            
            # Show best performers so far
            if data:
                best_loss = min(data, key=lambda x: x['test_loss'])
                best_mmd_rbf = min(data, key=lambda x: x['mmd_rbf'])
                best_mmd_linear = min(data, key=lambda x: x['mmd_linear'])
                best_wasserstein = min(data, key=lambda x: x['wasserstein_distance'])
                
                print(f"\n🏆 Best performers so far:")
                print(f"  Test Loss: Epoch {best_loss['epoch']} ({best_loss['schedule_type']}) = {best_loss['test_loss']:.6f}")
                print(f"  MMD RBF: Epoch {best_mmd_rbf['epoch']} ({best_mmd_rbf['schedule_type']}) = {best_mmd_rbf['mmd_rbf']:.6f}")
                print(f"  MMD Linear: Epoch {best_mmd_linear['epoch']} ({best_mmd_linear['schedule_type']}) = {best_mmd_linear['mmd_linear']:.6f}")
                print(f"  Wasserstein: Epoch {best_wasserstein['epoch']} ({best_wasserstein['schedule_type']}) = {best_wasserstein['wasserstein_distance']:.6f}")
    else:
        print("⏳ Metrics file not found yet - evaluation still in progress...")
    
    # Check for plot files
    plot_files = glob.glob(os.path.join(results_dir, "*.png"))
    if plot_files:
        print(f"\n📊 Generated plots ({len(plot_files)}):")
        for plot_file in sorted(plot_files):
            file_size = os.path.getsize(plot_file) / 1024  # KB
            print(f"  {os.path.basename(plot_file)} ({file_size:.1f} KB)")
    else:
        print("\n⏳ No plot files generated yet...")
    
    # Check total expected checkpoints
    checkpoint_files = []
    complex_dirs = ["checkpoints_complex"] + [f"checkpoints_complex/{d}" for d in os.listdir("checkpoints_complex") if os.path.isdir(f"checkpoints_complex/{d}")]
    
    for complex_dir in complex_dirs:
        if os.path.exists(complex_dir):
            files = glob.glob(f"{complex_dir}/model_epoch_*.pt")
            checkpoint_files.extend(files)
    
    total_checkpoints = len(checkpoint_files)
    print(f"\n📋 Total checkpoints to evaluate: {total_checkpoints}")
    
    if os.path.exists(json_file) and data:
        progress = len(data) / total_checkpoints * 100
        print(f"📈 Progress: {len(data)}/{total_checkpoints} ({progress:.1f}%)")
    
    print("\n" + "=" * 60)

def check_process_status():
    """Check if the evaluation process is still running."""
    import subprocess
    
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'test_complex_model.py' in result.stdout:
            print("✅ Evaluation process is still running...")
            return True
        else:
            print("❌ Evaluation process not found - may have completed or crashed.")
            return False
    except:
        print("⚠️  Could not check process status.")
        return None

if __name__ == "__main__":
    monitor_evaluation_progress()
    check_process_status() 