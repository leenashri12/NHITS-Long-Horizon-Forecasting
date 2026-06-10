import os
import pickle

def load_best_config(file_path):
    if not os.path.isfile(file_path):
        v2_path = file_path.replace('.p', '_v2.p')
        if os.path.isfile(v2_path):
            file_path = v2_path
        else:
            return None
    try:
        with open(file_path, 'rb') as f:
            trials = pickle.load(f)
        completed = [t for t in trials.trials if 'result' in t and t['result'].get('status') == 'ok']
        if not completed:
            return None
        best_trial = sorted(completed, key=lambda x: x['result']['loss'])[0]
        return best_trial['result']['mc']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def print_comparison(dataset, baseline_path, best_path):
    baseline_mc = load_best_config(baseline_path)
    best_mc = load_best_config(best_path)
    
    if not baseline_mc and not best_mc:
        print(f"No files found for {dataset}")
        return
        
    print(f"\n=================== Hyperparameter Comparison for {dataset.upper()} ===================")
    print(f"| Hyperparameter | Baseline (Paper Run - Worse) | Deep Search (Best Run - Optimized) |")
    print(f"|---|---|---|")
    
    keys = sorted(set(list(baseline_mc.keys() if baseline_mc else []) + list(best_mc.keys() if best_mc else [])))
    
    # We are interested in these key design and optimization parameters
    interesting_keys = [
        'n_layers', 'n_hidden', 'n_pool_kernel_size', 'n_freq_downsample',
        'learning_rate', 'n_time_in', 'n_time_out', 'pooling_mode',
        'interpolation_mode', 'loss_train', 'early_stop_patience', 'max_steps'
    ]
    
    for k in interesting_keys:
        val_base = baseline_mc.get(k, 'N/A') if baseline_mc else 'N/A'
        val_best = best_mc.get(k, 'N/A') if best_mc else 'N/A'
        print(f"| `{k}` | {val_base} | {val_best} |")

def main():
    # ECL 96 comparison
    print_comparison(
        dataset='ECL',
        baseline_path='./results/multivariate/ECL_96/NHITS/hyperopt_gpu_paper_run.p',
        best_path='./results/multivariate/ECL_96/NHITS/hyperopt_gpu_deep_search.p'
    )
    
    # Weather 96 comparison
    print_comparison(
        dataset='Weather',
        baseline_path='./results/multivariate/weather_96/NHITS/hyperopt_gpu_paper_run.p',
        best_path='./results/multivariate/weather_96/NHITS/hyperopt_gpu_deep_search.p'
    )

if __name__ == '__main__':
    main()
