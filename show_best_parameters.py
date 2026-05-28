import os
import glob
import pickle

def main():
    results_dir = './results/multivariate/'
    files = glob.glob(os.path.join(results_dir, "*", "NHITS", "hyperopt_*.p"))
    
    if not files:
        print("No trials files found.")
        return

    for file_path in sorted(files):
        print("\n" + 80*"=")
        print(f"File: {file_path}")
        print(80*"=")
        
        try:
            with open(file_path, 'rb') as f:
                trials = pickle.load(f)
        except Exception as e:
            print(f"Error loading file: {e}")
            continue
            
        completed_trials = [t for t in trials.trials if 'result' in t and t['result'].get('status') == 'ok']
        if not completed_trials:
            print("No completed trials found in this file.")
            continue
            
        best_trial = sorted(completed_trials, key=lambda x: x['result']['loss'])[0]
        best_mc = best_trial['result']['mc']
        
        print("Best Hyperparameters Selected:")
        for k in sorted(best_mc.keys()):
            # Print key architecture and optimization hyperparameters
            if k in ['n_layers', 'n_hidden', 'n_pool_kernel_size', 'n_freq_downsample', 'learning_rate', 'n_time_in', 'n_time_out', 'pooling_mode', 'interpolation_mode', 'loss_train', 'early_stop_patience']:
                print(f"  {k:22}: {best_mc[k]}")
        
        test_losses = best_trial['result'].get('test_losses', {})
        if test_losses:
            print("\nAssociated Test Metrics:")
            print(f"  MSE: {test_losses.get('mse')}")
            print(f"  MAE: {test_losses.get('mae')}")

if __name__ == '__main__':
    main()
