import os
import pickle
import time
import argparse
import pandas as pd
import gc
import torch

# Limit PyTorch CPU threads to keep the laptop fully responsive if CUDA is not available
if not torch.cuda.is_available():
    torch.set_num_threads(2)

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from src.losses.numpy import mae, mse
from src.experiments.utils import hyperopt_tunning, model_fit_predict

def get_experiment_space(args):
    space = {
        # Architecture parameters
        'model': 'nhits',
        'mode': 'simple',
        'n_time_in': hp.choice('n_time_in', [5 * args.horizon]),
        'n_time_out': hp.choice('n_time_out', [args.horizon]),
        'n_x_hidden': hp.choice('n_x_hidden', [0]),
        'n_s_hidden': hp.choice('n_s_hidden', [0]),
        'shared_weights': hp.choice('shared_weights', [False]),
        'activation': hp.choice('activation', ['ReLU']),
        'initialization': hp.choice('initialization', ['lecun_normal']),
        'stack_types': hp.choice('stack_types', [3 * ['identity']]),
        'n_blocks': hp.choice('n_blocks', [3 * [1]]),
        'n_layers': hp.choice('n_layers', [3 * [2], 9 * [2]]),  # CPU-friendly lighter stack option
        'n_hidden': hp.choice('n_hidden', [256, 512]),          # CPU-friendly smaller width option
        'n_pool_kernel_size': hp.choice('n_pool_kernel_size', [3 * [1], 3 * [2], 3 * [4], 3 * [8], [8, 4, 1], [16, 8, 1]]),
        'n_freq_downsample': hp.choice('n_freq_downsample', [
            [168, 24, 1], [24, 12, 1],
            [180, 60, 1], [60, 8, 1],
            [40, 20, 1]
        ]),
        'pooling_mode': hp.choice('pooling_mode', ['max']),
        'interpolation_mode': hp.choice('interpolation_mode', ['linear']),
        
        # Regularization and optimization parameters
        'batch_normalization': hp.choice('batch_normalization', [False]),
        'dropout_prob_theta': hp.choice('dropout_prob_theta', [0]),
        'dropout_prob_exogenous': hp.choice('dropout_prob_exogenous', [0]),
        'learning_rate': hp.choice('learning_rate', [0.001, 0.0005]),
        'lr_decay': hp.choice('lr_decay', [0.5]),
        'n_lr_decays': hp.choice('n_lr_decays', [3]),
        'weight_decay': hp.choice('weight_decay', [0]),
        'max_epochs': hp.choice('max_epochs', [None]),
        'max_steps': hp.choice('max_steps', [args.max_steps]),
        'early_stop_patience': hp.choice('early_stop_patience', [10]),
        'eval_freq': hp.choice('eval_freq', [50]),
        'loss_train': hp.choice('loss', ['MAE']),
        'loss_hypar': hp.choice('loss_hypar', [0.5]),
        'loss_valid': hp.choice('loss_valid', ['MAE']),
        'l1_theta': hp.choice('l1_theta', [0]),
        
        # Data parameters
        'normalizer_y': hp.choice('normalizer_y', [None]),
        'normalizer_x': hp.choice('normalizer_x', [None]),
        'complete_windows': hp.choice('complete_windows', [True]),
        'frequency': hp.choice('frequency', ['H']),
        'seasonality': hp.choice('seasonality', [24]),
        'idx_to_sample_freq': hp.choice('idx_to_sample_freq', [1]),
        'val_idx_to_sample_freq': hp.choice('val_idx_to_sample_freq', [1]),
        'batch_size': hp.choice('batch_size', [1]),
        'n_windows': hp.choice('n_windows', [args.n_windows]),
        'random_seed': hp.quniform('random_seed', 1, 10, 1)
    }
    return space

def main(args):
    #----------------------------------------------- Load Data -----------------------------------------------#
    data_path = f'./data/{args.dataset}/M/df_y.csv'
    print(f"Loading data from {data_path}...")
    Y_df = pd.read_csv(data_path)

    X_df = None
    S_df = None

    print('Y_df head:\n', Y_df.head())
    
    # Establish train/val/test split indices
    if args.dataset == 'ETTm2':
        len_val = 11520
        len_test = 11520
    elif args.dataset == 'Exchange':
        len_val = 760
        len_test = 1517
    elif args.dataset == 'ECL':
        len_val = 2632
        len_test = 5260
    elif args.dataset == 'traffic':
        len_val = 1756
        len_test = 3508
    elif args.dataset == 'weather':
        len_val = 5270
        len_test = 10539
    elif args.dataset == 'ili':
        len_val = 97
        len_test = 193
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    space = get_experiment_space(args)

    #---------------------------------------------- Directories ----------------------------------------------#
    output_dir = f'./results/multivariate/{args.dataset}_{args.horizon}/NHITS/'
    os.makedirs(output_dir, exist_ok=True)

    hyperopt_file = output_dir + f'hyperopt_{args.experiment_id}.p'

    if not os.path.isfile(hyperopt_file):
        print(f"\n=================== Stage 1: Tuning on Weather ===================")
        # If GPU is available, we evaluate test loss on all trials during search just like the paper
        loss_fns_test_tuning = {'mae': mae, 'mse': mse} if torch.cuda.is_available() else {}

        # Weather has 21 series in total, so we tune on the whole dataset
        trials = hyperopt_tunning(space=space,
                                  hyperopt_max_evals=args.hyperopt_max_evals,
                                  loss_function_val=mae,
                                  loss_functions_test=loss_fns_test_tuning,
                                  Y_df=Y_df, X_df=X_df, S_df=S_df, f_cols=[],
                                  evaluate_train=False,  # CRITICAL memory optimization (set to False to avoid OOM crash)
                                  ds_in_val=len_val,
                                  ds_in_test=len_test,
                                  return_forecasts=False,
                                  results_file=hyperopt_file,
                                  save_progress=True,
                                  loss_kwargs={})

        print(f"\n=================== Stage 2: Final Model Evaluation ===================")
        # Extract the best trials based on validation loss
        best_trial = sorted(trials.trials, key=lambda x: x['result']['loss'])[0]
        
        # If test loss was already evaluated in the trial (which happens when CUDA is available), we reuse it
        if 'test_losses' in best_trial['result'] and best_trial['result']['test_losses']:
            print(f"Final Test Evaluation (reused from best trial): {best_trial['result']['test_losses']}")
        else:
            best_mc = best_trial['result']['mc'].copy()
            # Override steps for the final training run to ensure convergence
            best_mc['max_steps'] = 1000 if args.max_steps > 50 else args.max_steps
            
            print(f"Training best model with max_steps={best_mc['max_steps']} on ALL {Y_df['unique_id'].nunique()} series...")
            final_results = model_fit_predict(mc=best_mc,
                                               S_df=S_df,
                                               Y_df=Y_df,  # Run on FULL dataset
                                               X_df=X_df,
                                               f_cols=[],
                                               evaluate_train=False,
                                               ds_in_val=len_val,
                                               ds_in_test=len_test,
                                               return_arrays=False,
                                               loss_fns_val={'val_loss': mae},
                                               loss_fns_test={'mae': mae, 'mse': mse})
                                               
            # Update trials record with final test evaluation metrics
            best_trial['result']['test_losses'] = final_results['test_losses']
            print(f"Final Test Evaluation: {final_results['test_losses']}")

        with open(hyperopt_file, "wb") as f:
            pickle.dump(trials, f)
        print(f"Tuning and final training complete. Saved trials to {hyperopt_file}")
    else:
        print(f"Hyperparameter optimization already done for {hyperopt_file}!")

    # Explicitly garbage collect
    del Y_df
    gc.collect()

def parse_args():
    desc = "Example of hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--hyperopt_max_evals', type=int, default=10, help='Maximum hyperopt evaluations')
    parser.add_argument('--experiment_id', default='eval_train', type=str, help='String identifier for the experiment')
    parser.add_argument('--horizon', default=None, type=int, help='Forecasting horizon (leave None to run all paper horizons)')
    parser.add_argument('--max_steps', default=300, type=int, help='Maximum optimization steps per trial (CPU-friendly default: 300)')
    parser.add_argument('--n_windows', default=128, type=int, help='Number of windows to sample per batch (CPU-friendly default: 128)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()

    dataset = 'weather'
    
    if args.horizon is not None:
        horizons = [args.horizon]
    else:
        horizons = [96, 192, 336, 720]

    for horizon in horizons:
        print(50*'-', f"Dataset: {dataset} | Horizon: {horizon}", 50*'-')
        start = time.time()

        import copy
        temp_args = copy.deepcopy(args)
        temp_args.dataset = dataset
        temp_args.horizon = horizon

        main(temp_args)

        print(f"Horizon {horizon} Time: {time.time() - start:.2f} seconds")
        gc.collect()
