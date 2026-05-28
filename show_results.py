import os
import glob
import pickle
import numpy as np
import pandas as pd

def get_score_min_val(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

    min_val_loss = float('inf')
    mae_best = None
    mse_best = None

    for trial in result.trials:
        # Some trials might have failed or not completed
        if 'result' not in trial or trial['result'].get('status') != 'ok':
            continue
        
        val_loss = trial['result'].get('loss')
        if val_loss is not None and val_loss < min_val_loss:
            min_val_loss = val_loss
            test_losses = trial['result'].get('test_losses', {})
            mae_best = test_losses.get('mae')
            mse_best = test_losses.get('mse')

    return mae_best, mse_best

def main():
    results_dir = './results/multivariate/'
    datasets = ['ETTm2', 'Exchange', 'weather', 'ili', 'ECL', 'traffic']
    horizons = [96, 192, 336, 720]
    ili_horizons = [24, 36, 48, 60]

    data = []

    for dataset in datasets:
        dataset_horizons = ili_horizons if dataset == 'ili' else horizons
        for horizon in dataset_horizons:
            pattern = os.path.join(results_dir, f"{dataset}_{horizon}", "NHITS", "hyperopt_*.p")
            files = glob.glob(pattern)
            
            if not files:
                data.append({
                    'Dataset': dataset,
                    'Horizon': horizon,
                    'MSE': 'N/A',
                    'MAE': 'N/A',
                    'Trials File': 'Not Found'
                })
                continue
                
            for file_path in files:
                file_name = os.path.basename(file_path)
                experiment = file_name.replace('hyperopt_', '').replace('.p', '')
                mae, mse = get_score_min_val(file_path)
                
                mae_str = f"{mae:.4f}" if mae is not None else "N/A"
                mse_str = f"{mse:.4f}" if mse is not None else "N/A"
                
                data.append({
                    'Dataset': dataset,
                    'Horizon': horizon,
                    'MSE': mse_str,
                    'MAE': mae_str,
                    'Trials File': f"{file_name} ({experiment})"
                })

    df = pd.DataFrame(data)
    print("\n--- N-HiTS Training Results Summary ---")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
