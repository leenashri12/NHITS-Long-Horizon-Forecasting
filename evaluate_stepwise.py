import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import gc

from src.losses.numpy import mae, mse
from src.experiments.utils import model_fit_predict

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate step-wise forecasting performance")
    parser.add_argument('--file_path', required=True, type=str, 
                        help='Path to the hyperopt trials pickle file (e.g. ./results/multivariate/weather_96/NHITS/hyperopt_gpu_deep_search.p)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.isfile(args.file_path):
        print(f"Error: File {args.file_path} not found.")
        return

    # Parse dataset and horizon from the file path
    # Path format: ./results/multivariate/DATASET_HORIZON/NHITS/hyperopt_EXPERIMENT.p
    norm_path = os.path.normpath(args.file_path).replace("\\", "/")
    path_parts = norm_path.split('/')
    
    dataset_horizon_str = None
    for part in path_parts:
        if '_' in part and ('ECL' in part or 'weather' in part or 'Exchange' in part or 'ETTm2' in part):
            dataset_horizon_str = part
            break
            
    if dataset_horizon_str is None:
        print("Error: Could not infer dataset and horizon from path. Make sure path matches results directory structure.")
        return
        
    dataset, horizon_str = dataset_horizon_str.split('_')
    horizon = int(horizon_str)
    
    print(f"Inferred Dataset: {dataset} | Inferred Horizon: {horizon}")
    
    # Establish train/val/test split indices
    if dataset == 'ETTm2':
        len_val = 11520
        len_test = 11520
    elif dataset == 'Exchange':
        len_val = 760
        len_test = 1517
    elif dataset == 'ECL':
        len_val = 2632
        len_test = 5260
    elif dataset == 'traffic':
        len_val = 1756
        len_test = 3508
    elif dataset == 'weather':
        len_val = 5270
        len_test = 10539
    elif dataset == 'ili':
        len_val = 97
        len_test = 193
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load the best trials
    with open(args.file_path, 'rb') as f:
        trials = pickle.load(f)
        
    completed_trials = [t for t in trials.trials if 'result' in t and t['result'].get('status') == 'ok']
    if not completed_trials:
        print("Error: No completed trials found in this file.")
        return
        
    best_trial = sorted(completed_trials, key=lambda x: x['result']['loss'])[0]
    best_mc = best_trial['result']['mc'].copy()
    
    # Load dataset
    data_path = f'./data/{dataset}/M/df_y.csv'
    print(f"Loading data from {data_path}...")
    Y_df = pd.read_csv(data_path)
    X_df = None
    S_df = None

    # Run single training run to get full arrays
    print(f"Training final model on {dataset} to extract predictions...")
    results = model_fit_predict(mc=best_mc,
                                S_df=S_df,
                                Y_df=Y_df,
                                X_df=X_df,
                                f_cols=[],
                                evaluate_train=False,
                                ds_in_val=len_val,
                                ds_in_test=len_test,
                                return_arrays=True, # Critical: we want raw outputs
                                loss_fns_val={'val_loss': mae},
                                loss_fns_test={'mae': mae, 'mse': mse})
                                
    y_true = results['test_y_true']
    y_hat = results['test_y_hat']
    mask = results['test_mask']
    
    # Shapes are: (n_windows, n_series, horizon)
    print(f"Predictions extracted successfully. Shape: {y_true.shape}")
    
    # Evaluate at specific steps: T+1, T+4, T+6, T+12, T+24, T+48
    steps_to_evaluate = [1, 4, 6, 12, 24, 48]
    
    print("\n" + 60*"=")
    print(f"   Step-Wise Performance Summary for {dataset.upper()} (Horizon: {horizon})")
    print(60*"=")
    print(f"  {'Step':<10} | {'Horizon Index':<15} | {'MSE':<12} | {'MAE':<12}")
    print(60*"-")
    
    for step in steps_to_evaluate:
        idx = step - 1
        if idx >= horizon:
            print(f"  T+{step:<8} | Index {idx:<14} | {'N/A':<12} | {'N/A (Exceeds Horizon)':<12}")
            continue
            
        y_t_step = y_true[..., idx]
        y_h_step = y_hat[..., idx]
        mask_step = mask[..., idx]
        
        # Calculate masked errors
        valid_mask = mask_step > 0
        y_t_valid = y_t_step[valid_mask]
        y_h_valid = y_h_step[valid_mask]
        
        step_mse = np.mean((y_t_valid - y_h_valid) ** 2)
        step_mae = np.mean(np.abs(y_t_valid - y_h_valid))
        
        print(f"  T+{step:<8} | Index {idx:<14} | {step_mse:<12.6f} | {step_mae:<12.6f}")
        
    print(60*"=")
    
    # Explicitly garbage collect
    del Y_df, y_true, y_hat, mask, results
    gc.collect()

if __name__ == '__main__':
    main()
