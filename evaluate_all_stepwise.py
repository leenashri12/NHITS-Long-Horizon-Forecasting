import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import gc

from src.losses.numpy import mae, mse
from src.experiments.utils import model_fit_predict

def get_split_lengths(dataset):
    if dataset == 'ETTm2':
        return 11520, 11520
    elif dataset == 'Exchange':
        return 760, 1517
    elif dataset == 'ECL':
        return 2632, 5260
    elif dataset == 'traffic':
        return 1756, 3508
    elif dataset == 'weather':
        return 5270, 10539
    elif dataset == 'ili':
        return 97, 193
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def main():
    results_dir = './results/multivariate/'
    pattern = os.path.join(results_dir, "*", "NHITS", "hyperopt_gpu_deep_search.p")
    files = glob.glob(pattern)
    
    if not files:
        print("No hyperopt_gpu_deep_search.p files found. Check your directories.")
        return

    # Sort files to process in a consistent order
    files = sorted(files)
    print(f"Found {len(files)} deep search trial files to evaluate:")
    for f in files:
        print(f"  - {f}")

    # We will write the report to stepwise_report.md
    report_lines = []
    report_lines.append("# N-HiTS Stepwise Evaluation Report")
    report_lines.append(f"Generated automatically from completed deep search GPU runs.")
    report_lines.append("")

    # Cache dataset loads to save time
    dataset_cache = {}

    for file_path in files:
        norm_path = os.path.normpath(file_path).replace("\\", "/")
        path_parts = norm_path.split('/')
        
        # Find folder with dataset and horizon
        dataset_horizon_str = None
        for part in path_parts:
            if '_' in part and any(d in part for d in ['ECL', 'weather', 'Exchange', 'ETTm2', 'ili', 'traffic']):
                dataset_horizon_str = part
                break
                
        if dataset_horizon_str is None:
            print(f"Could not parse dataset/horizon from: {file_path}")
            continue
            
        dataset, horizon_str = dataset_horizon_str.split('_')
        horizon = int(horizon_str)
        
        print("\n" + 80*"=")
        print(f"Processing Dataset: {dataset} | Horizon: {horizon}")
        print(80*"=")
        
        try:
            with open(file_path, 'rb') as f:
                trials = pickle.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
            
        completed_trials = [t for t in trials.trials if 'result' in t and t['result'].get('status') == 'ok']
        if not completed_trials:
            print(f"No completed trials in {file_path}")
            continue
            
        best_trial = sorted(completed_trials, key=lambda x: x['result']['loss'])[0]
        best_mc = best_trial['result']['mc'].copy()
        
        # Load dataset
        data_path = f'./data/{dataset}/M/df_y.csv'
        if dataset not in dataset_cache:
            print(f"Loading data from {data_path}...")
            dataset_cache[dataset] = pd.read_csv(data_path)
            
        Y_df = dataset_cache[dataset]
        X_df = None
        S_df = None
        
        len_val, len_test = get_split_lengths(dataset)
        
        # Run training to extract predictions
        print(f"Fitting final model to retrieve predictions...")
        try:
            results = model_fit_predict(mc=best_mc,
                                        S_df=S_df,
                                        Y_df=Y_df,
                                        X_df=X_df,
                                        f_cols=[],
                                        evaluate_train=False,
                                        ds_in_val=len_val,
                                        ds_in_test=len_test,
                                        return_arrays=True,
                                        loss_fns_val={'val_loss': mae},
                                        loss_fns_test={'mae': mae, 'mse': mse})
        except Exception as e:
            print(f"Error during model fit for {dataset}_{horizon}: {e}")
            continue
            
        y_true = results['test_y_true']
        y_hat = results['test_y_hat']
        mask = results['test_mask']
        
        print(f"Predictions extracted. y_true shape: {y_true.shape}")
        
        # Define evaluation marks (in physical hours)
        hours_to_evaluate = [1, 4, 6, 12, 24, 48]
        
        # Map physical hours to step index
        if dataset == 'weather':
            # Weather is 10-minute intervals (6 steps per hour)
            hour_to_step_factor = 6
            desc = "Weather (10-min interval, 6 steps/hr)"
        else:
            # ECL and others are hourly (1 step per hour)
            hour_to_step_factor = 1
            desc = f"{dataset} (hourly, 1 step/hr)"
            
        report_lines.append(f"## Dataset: {dataset.upper()} | Horizon: {horizon}")
        report_lines.append(f"**Data Profile:** {desc}")
        report_lines.append("")
        report_lines.append("| Time Ahead | Horizon Step Index | MSE | MAE | Status |")
        report_lines.append("|---|---|---|---|---|")
        
        for hr in hours_to_evaluate:
            step_number = hr * hour_to_step_factor
            idx = step_number - 1
            
            if idx >= horizon:
                report_lines.append(f"| T+{hr} Hour(s) | {idx} | N/A | N/A | Exceeds Horizon |")
                continue
                
            y_t_step = y_true[..., idx]
            y_h_step = y_hat[..., idx]
            mask_step = mask[..., idx]
            
            valid_mask = mask_step > 0
            y_t_valid = y_t_step[valid_mask]
            y_h_valid = y_h_step[valid_mask]
            
            step_mse = np.mean((y_t_valid - y_h_valid) ** 2)
            step_mae = np.mean(np.abs(y_t_valid - y_h_valid))
            
            report_lines.append(f"| T+{hr} Hour(s) | {idx} | {step_mse:.6f} | {step_mae:.6f} | OK |")
            
        report_lines.append("")
        
        # Free memory
        del y_true, y_hat, mask, results
        gc.collect()
        
    # Write to report file
    report_content = "\n".join(report_lines)
    with open("stepwise_report.md", "w") as rf:
        rf.write(report_content)
        
    print("\n" + 80*"=")
    print("Stepwise report compiled and saved to stepwise_report.md!")
    print(80*"=")

if __name__ == '__main__':
    main()
