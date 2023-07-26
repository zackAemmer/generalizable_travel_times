#!/usr/bin python3


import json
import os
import random
import shutil
import time
import sys

import lightning.pytorch as pl
import numpy as np
import torch
from tabulate import tabulate
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn import metrics
from sklearn.model_selection import KFold
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler

from models import grids
from models import ff, conv, rnn, transformer, avg_speed, schedule, persistent
from utils import data_loader, data_utils, model_utils


if __name__=="__main__":

    torch.set_default_dtype(torch.float)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42, workers=True)

    model_type = sys.argv[1]
    run_folder = sys.argv[2]
    network_folder = sys.argv[3]
    skip_gtfs = sys.argv[4]
    is_param_search = sys.argv[5]

    grid_s_size=500
    n_folds=5

    if skip_gtfs=="True":
        skip_gtfs=True
    else:
        skip_gtfs=False
    if is_param_search=="True":
        is_param_search=True
    else:
        is_param_search=False
    if network_folder=="kcm/":
        holdout_routes=[100252,100139,102581,100341,102720]
    elif network_folder=="atb/":
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    else:
        holdout_routes=None

    NUM_WORKERS=4
    PIN_MEMORY=False

    print("="*30)
    print(f"RUN: '{run_folder}'")
    print(f"MODEL: {model_type}")
    print(f"NETWORK: '{network_folder}'")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"PIN_MEMORY: {PIN_MEMORY}")

    # Create folder structure; delete older results
    base_folder = f"{run_folder}{network_folder}"
    model_folder = f"{run_folder}{network_folder}models/{model_type}/"
    try:
        shutil.rmtree(model_folder)
        os.mkdir(model_folder)
    except:
        print("Model folder not found to remove")
        os.mkdir(model_folder)

    # Define embedded variables for network models
    embed_dict = {
        'timeID': {
            'vocab_size': 1440,
            'embed_dims': 8
        },
        'weekID': {
            'vocab_size': 7,
            'embed_dims': 3
        }
    }
    # Sample parameter values for random search
    if is_param_search:
        hyperparameter_sample_dict = {
            'n_param_samples': 1,
            'batch_size': [512],
            'hidden_size': [16, 32, 64, 128],
            'num_layers': [2, 3, 4, 5],
            'dropout_rate': [.05, .1, .2, .4]
        }
        hyperparameter_dict = model_utils.random_param_search(hyperparameter_sample_dict, ["FF","CONV","GRU","TRSF"])
        data_utils.write_pkl(hyperparameter_sample_dict, f"{model_folder}param_search_dict.pkl")
        data_utils.write_pkl(hyperparameter_dict, f"{model_folder}param_search_dict_sample.pkl")
    # Manually specified run without testing hyperparameters
    else:
        hyperparameter_dict = {
            'FF': {
                'batch_size': 1024,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_rate': .2
            },
            'CONV': {
                'batch_size': 1024,
                'hidden_size': 64,
                'num_layers': 3,
                'dropout_rate': .1
            },
            'GRU': {
                'batch_size': 1024,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout_rate': .05
            },
            'TRSF': {
                'batch_size': 1024,
                'hidden_size': 64,
                'num_layers': 3,
                'dropout_rate': .1
            },
            'DEEPTTE': {
                'batch_size': 1024
            }
        }

    # Data loading and fold setup
    with open(f"{base_folder}deeptte_formatted/train_summary_config.json", "r") as f:
        config = json.load(f)

    print(f"Building grid on fold training data")
    train_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/train", config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    train_ngrid = grids.NGridBetter(config['grid_bounds'][0], grid_s_size)
    train_ngrid.add_grid_content(train_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    train_ngrid.build_cell_lookup()
    train_dataset.grid = train_ngrid

    print(f"Building grid on fold testing data")
    test_dataset = data_loader.LoadSliceDataset(f"{base_folder}deeptte_formatted/test", config, holdout_routes=holdout_routes, skip_gtfs=skip_gtfs)
    test_ngrid = grids.NGridBetter(config['grid_bounds'][0], grid_s_size)
    test_ngrid.add_grid_content(test_dataset.get_all_samples(keep_cols=['shingle_id','locationtime','x','y','speed_m_s','bearing']), trace_format=True)
    test_ngrid.build_cell_lookup()
    test_dataset.grid = test_ngrid

    k_fold = KFold(n_folds, shuffle=True, random_state=42)
    run_results = []

    for fold_num, (train_idx,val_idx) in enumerate(k_fold.split(np.arange(len(train_dataset)))):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Declare models
        base_model_list, nn_model = model_utils.make_one_model(model_type, hyperparameter_dict=hyperparameter_dict, embed_dict=embed_dict, config=config, skip_gtfs=skip_gtfs)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SequentialSampler(val_idx)
        model_names = [m.model_name for m in base_model_list]
        model_names.append(nn_model.model_name)
        print(f"Model name: {model_names}")
        print(f"NN model total parameters: {sum(p.numel() for p in nn_model.parameters())}")

        # Keep track of all model performances
        model_fold_results = {}
        for x in model_names:
            model_fold_results[x] = {
                "Labels":[],
                "Preds":[]
            }

        # Total run samples
        print(f"Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples")

        # Train/Test baseline models
        for b_model in base_model_list:
            train_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=b_model.collate_fn, sampler=train_sampler, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, multiprocessing_context="fork")
            val_loader = DataLoader(train_dataset, batch_size=1024, collate_fn=b_model.collate_fn, sampler=val_sampler, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, multiprocessing_context="fork")
            test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=b_model.collate_fn, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, multiprocessing_context="fork")
            print(f"Network {network_folder} Fold {fold_num} Model {b_model.model_name}")
            # Train base model on all fold data
            b_model.train_time = 0.0
            b_model.train(train_loader, config)
            print(f"Fold final evaluation for: {b_model.model_name}")
            labels, preds = b_model.evaluate(test_loader, config)
            model_fold_results[b_model.model_name]["Labels"].extend(list(labels))
            model_fold_results[b_model.model_name]["Preds"].extend(list(preds))
            data_utils.write_pkl(b_model, f"{base_folder}models/{b_model.model_name}_{fold_num}.pkl")

        # Train/Test nn model
        train_dataset.add_grid_features = nn_model.requires_grid
        test_dataset.add_grid_features = nn_model.requires_grid
        train_loader = DataLoader(train_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, sampler=train_sampler, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, multiprocessing_context="fork")
        val_loader = DataLoader(train_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, sampler=val_sampler, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, multiprocessing_context="fork")
        test_loader = DataLoader(test_dataset, batch_size=nn_model.batch_size, collate_fn=nn_model.collate_fn, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, multiprocessing_context="fork")
        t0=time.time()
        trainer = pl.Trainer(
            check_val_every_n_epoch=1,
            max_epochs=30,
            min_epochs=5,
            # accelerator="cpu",
            logger=CSVLogger(save_dir=f"{model_folder}logs/", name=nn_model.model_name),
            callbacks=[EarlyStopping(monitor=f"{nn_model.model_name}_valid_loss", min_delta=.001, patience=3)],
        )
        trainer.fit(model=nn_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        nn_model.train_time = time.time() - t0
        trainer.test(model=nn_model, dataloaders=test_loader)
        preds_and_labels = trainer.predict(model=nn_model, dataloaders=test_loader)
        preds = np.concatenate([p['out_agg'] for p in preds_and_labels])
        labels = np.concatenate([l['y_agg'] for l in preds_and_labels])
        model_fold_results[nn_model.model_name]["Labels"].extend(list(labels))
        model_fold_results[nn_model.model_name]["Preds"].extend(list(preds))

        # After all models have trained for this fold, calculate various losses
        train_times = [x.train_time for x in base_model_list]
        train_times.append(nn_model.train_time)
        fold_results = {
            "Model_Names": model_names,
            "Fold": fold_num,
            "All_Losses": [],
            "Train_Times": train_times
        }
        for mname in fold_results["Model_Names"]:
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]), 2))
            fold_results['All_Losses'].append(_)

        # Print results of this fold
        print(tabulate(fold_results['All_Losses'], headers=["Model", "MAPE", "RMSE", "MAE"]))
        run_results.append(fold_results)

    # Save full run results
    data_utils.write_pkl(run_results, f"{model_folder}model_results.pkl")
    print(f"MODEL RUN COMPLETED {model_folder}")