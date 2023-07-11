#!/usr/bin python3


import json
import os
import random
import shutil
import time
import pathlib

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn import metrics
from sklearn.model_selection import KFold
from tabulate import tabulate

from models import grids
from utils import data_loader, data_utils, model_utils

from torch.profiler import profile, record_function, ProfilerActivity


def run_models(run_folder, network_folder, **kwargs):
    """
    Train each of the specified models on bus data found in the data folder.
    The data folder is generated using prepare_run.py.
    The data in the folder will have many attributes, not all necessariliy used.
    Use k-fold cross validation to split the data n times into train/test.
    The test set is used as validation during the training process.
    Model accuracy is measured at the end of each folds.
    Save the resulting models, and the data generated during training to the results folder.
    These are then analyzed in a jupyter notebook.
    """

    # with profile(activities=[ProfilerActivity.CPU]) as prof:

    print("="*30)
    print(f"RUN MODELS: '{run_folder}'")
    print(f"NETWORK: '{network_folder}'")

    # Create folder structure; delete older results
    base_folder = f"{run_folder}{network_folder}"
    if "model_results_temp.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}model_results_temp.pkl")
    if "model_results.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}model_results.pkl")
    if "model_generalization_results.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}model_generalization_results.pkl")
    if "param_search_dict.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}param_search_dict.pkl")
    if "param_search_dict_sample.pkl" in os.listdir(f"{base_folder}"):
        os.remove(f"{base_folder}param_search_dict_sample.pkl")
    shutil.rmtree(f"{base_folder}models")
    os.mkdir(f"{base_folder}models")

    # Select device to train on, and number workers if GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        NUM_WORKERS = 8
        PIN_MEMORY = True
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
        NUM_WORKERS = 0
        PIN_MEMORY = False
    print(f"DEVICE: {device}")
    print(f"WORKERS: {NUM_WORKERS}")

    # Define embedded variables for network models
    embed_dict = {
        'timeID': {
            'vocab_size': 1440,
            'embed_dims': 27
        },
        'weekID': {
            'vocab_size': 7,
            'embed_dims': 4
        }
    }
    # Sample parameter values for random search
    if kwargs['is_param_search']:
        hyperparameter_sample_dict = {
            'N_PARAM_SAMPLES': 10,
            'EPOCHS': [20, 40, 80],
            'BATCH_SIZE': [16, 32, 128, 512],
            'LEARN_RATE': [.1, .01, .001],
            'HIDDEN_SIZE': [32, 64, 128],
            'NUM_LAYERS': [2, 3, 4],
            'DROPOUT_RATE': [.1, .2, .3]
        }
        # hyperparameter_sample_dict = {
        #     'N_PARAM_SAMPLES': 1,
        #     'EPOCHS': [2],
        #     'BATCH_SIZE': [512],
        #     'LEARN_RATE': [.001],
        #     'HIDDEN_SIZE': [32],
        #     'NUM_LAYERS': [2],
        #     'DROPOUT_RATE': [.1]
        # }
        hyperparameter_dict = model_utils.random_param_search(hyperparameter_sample_dict, ["FF","CONV","GRU","TRSF"])
        data_utils.write_pkl(hyperparameter_sample_dict, f"{base_folder}param_search_dict.pkl")
        data_utils.write_pkl(hyperparameter_dict, f"{base_folder}param_search_dict_sample.pkl")
    # Manually specified run without testing hyperparameters
    else:
        hyperparameter_dict = {
            'FF': {
                'EPOCHS': 2,
                'BATCH_SIZE': 512,
                'LEARN_RATE': .001,
                'HIDDEN_SIZE': 32,
                'NUM_LAYERS': 2,
                'DROPOUT_RATE': .1
            },
            'CONV': {
                'EPOCHS': 2,
                'BATCH_SIZE': 512,
                'LEARN_RATE': .001,
                'HIDDEN_SIZE': 32,
                'NUM_LAYERS': 2,
                'DROPOUT_RATE': .1
            },
            'GRU': {
                'EPOCHS': 2,
                'BATCH_SIZE': 512,
                'LEARN_RATE': .001,
                'HIDDEN_SIZE': 32,
                'NUM_LAYERS': 2,
                'DROPOUT_RATE': .1
            },
            'TRSF': {
                'EPOCHS': 2,
                'BATCH_SIZE': 512,
                'LEARN_RATE': .001,
                'HIDDEN_SIZE': 32,
                'NUM_LAYERS': 2,
                'DROPOUT_RATE': .1
            },
            'DEEPTTE': {
                'EPOCHS': 2,
                'BATCH_SIZE': 512,
                'LEARN_RATE': .001
            }
        }

    # Data loading and fold setup
    with open(f"{run_folder}{network_folder}deeptte_formatted/train_config.json", "r") as f:
        config = json.load(f)
    dataset = data_loader.GenericDataset(f"{run_folder}{network_folder}deeptte_formatted/train", config, holdout_routes=kwargs['holdout_routes'], skip_gtfs=kwargs['skip_gtfs'])
    splits = KFold(kwargs['n_folds'], shuffle=True, random_state=0)
    run_results = []

    # Run full training process for each model during each validation fold
    for fold_num, (train_idx,test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Random samplers for indices from this fold
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # Declare models
        if kwargs['is_param_search']:
            model_list = model_utils.make_param_search_models(hyperparameter_dict, embed_dict, device, config)
        elif not kwargs['skip_gtfs']:
            model_list = model_utils.make_all_models(hyperparameter_dict, embed_dict, device, config)
        else:
            model_list = model_utils.make_all_models_nosch(hyperparameter_dict, embed_dict, device, config)
        model_names = [m.model_name for m in model_list]
        print(f"Model names: {model_names}")
        print(f"Model total parameters: {[sum(p.numel() for p in m.parameters()) for m in model_list if m.is_nn]}")

        # Keep track of all model performances
        model_fold_results = {}
        for x in model_names:
            model_fold_results[x] = {
                "Labels":[],
                "Preds":[]
            }
        # Keep track of train/test curves during training for network models
        model_fold_curves = {}
        for x in model_list:
            if x.is_nn:
                model_fold_curves[x.model_name] = {
                    "Train":[],
                    "Test":[]
                }

        # Total run samples
        print(f"{len(dataset.pq_dataset.to_table().to_pandas())} points in dataset, {len(dataset.content)} samples")

        # Build grid using only data from this fold
        print(f"Building grid on fold training data")
        train_ngrid = grids.NGridBetter(config['grid_bounds'][0], kwargs['grid_s_size'])
        train_ngrid.add_grid_content(dataset.pq_dataset.to_table(filter=ds.field("shingle_id").isin(train_idx)).to_pandas(), trace_format=True)
        train_ngrid.build_cell_lookup()
        print(f"Building grid on fold testing data")
        test_ngrid = grids.NGridBetter(config['grid_bounds'][0], kwargs['grid_s_size'])
        test_ngrid.add_grid_content(dataset.pq_dataset.to_table(filter=ds.field("shingle_id").isin(test_idx)).to_pandas(), trace_format=True)
        test_ngrid.build_cell_lookup()

        # Train all models
        for model in model_list:
            print(f"Fold training for: {model.model_name}")
            dataset.add_grid_features = model.requires_grid
            if not model.is_nn:
                # Train base model on all fold data
                dataset.grid = train_ngrid
                loader = DataLoader(dataset, sampler=train_sampler, collate_fn=model.collate_fn, batch_size=int(model.hyperparameter_dict['BATCH_SIZE']), pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
                model.train(loader, config)
                print(f"Fold final evaluation for: {model.model_name}")
                dataset.grid = test_ngrid
                loader = DataLoader(dataset, sampler=test_sampler, collate_fn=model.collate_fn, batch_size=int(model.hyperparameter_dict['BATCH_SIZE']), pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Preds"].extend(list(preds))
                data_utils.write_pkl(model, f"{base_folder}models/{model.model_name}_{fold_num}.pkl")
            else:
                # Train NN model on all fold data for n epochs
                optimizer = torch.optim.Adam(model.parameters(), lr=model.hyperparameter_dict['LEARN_RATE'])
                for epoch in range(model.hyperparameter_dict['EPOCHS']):
                    print(f"NETWORK: {base_folder}, FOLD: {fold_num}/{kwargs['n_folds']-1}, MODEL: {model.model_name}, EPOCH: {epoch}/{model.hyperparameter_dict['EPOCHS']-1}")
                    dataset.grid = train_ngrid
                    loader = DataLoader(dataset, sampler=train_sampler, collate_fn=model.collate_fn, batch_size=int(model.hyperparameter_dict['BATCH_SIZE']), pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
                    t0 = time.time()
                    avg_batch_loss = model_utils.train(model, loader, optimizer)
                    model.train_time += (time.time() - t0)
                    # Evaluate NN curves at regular epochs
                    if epoch % kwargs['EPOCH_EVAL_FREQ'] == 0 and model.is_nn:
                        print(f"Curve evaluation for: {model.model_name} train data")
                        train_losses = 0.0
                        dataset.grid = train_ngrid
                        loader = DataLoader(dataset, sampler=train_sampler, collate_fn=model.collate_fn, batch_size=int(model.hyperparameter_dict['BATCH_SIZE']), pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
                        labels, preds = model.evaluate(loader, config)
                        train_losses += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                        model_fold_curves[model.model_name]['Train'].append(train_losses)
                        print(f"Curve evaluation for: {model.model_name} test data")
                        test_losses = 0.0
                        dataset.grid = test_ngrid
                        loader = DataLoader(dataset, sampler=test_sampler, collate_fn=model.collate_fn, batch_size=int(model.hyperparameter_dict['BATCH_SIZE']), pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
                        labels, preds = model.evaluate(loader, config)
                        test_losses += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                        model_fold_curves[model.model_name]['Test'].append(test_losses)
                # After model is trained, evaluate on test set for this fold
                print(f"Fold final evaluation for: {model.model_name}")
                dataset.grid = test_ngrid
                loader = DataLoader(dataset, sampler=test_sampler, collate_fn=model.collate_fn, batch_size=int(model.hyperparameter_dict['BATCH_SIZE']), pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Preds"].extend(list(preds))
                torch.save(model.state_dict(), f"{base_folder}models/{model.model_name}_{fold_num}.pt")

        # After all models have trained for this fold, calculate various losses
        train_times = [x.train_time for x in model_list]
        fold_results = {
            "Model_Names": model_names,
            "Fold": fold_num,
            "All_Losses": [],
            "Loss_Curves": [{model: curve_dict} for model, curve_dict in zip(model_fold_curves.keys(), list(model_fold_curves.values()))],
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

        # Save temp run results after each fold
        data_utils.write_pkl(run_results, f"{base_folder}model_results_temp.pkl")

    # Save full run results
    data_utils.write_pkl(run_results, f"{base_folder}model_results.pkl")
    print(f"MODEL RUN COMPLETED '{base_folder}'")

    # prof.export_chrome_trace("profile.json")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

if __name__=="__main__":
    torch.set_default_dtype(torch.float)

    # DEBUG
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/debug/",
        network_folder="kcm/",
        EPOCH_EVAL_FREQ=10,
        grid_s_size=500,
        n_folds=2,
        holdout_routes=[100252,100139,102581,100341,102720],
        skip_gtfs=False,
        is_param_search=False
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/debug/",
        network_folder="atb/",
        EPOCH_EVAL_FREQ=10,
        grid_s_size=500,
        n_folds=2,
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"],
        skip_gtfs=False,
        is_param_search=False
    )
    # DEBUG MIXED
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/debug_nosch/",
        network_folder="kcm_atb/",
        EPOCH_EVAL_FREQ=10,
        grid_s_size=500,
        n_folds=2,
        holdout_routes=None,
        skip_gtfs=True,
        is_param_search=False
    )

    # PARAM SEARCH
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/param_search/",
        network_folder="kcm/",
        EPOCH_EVAL_FREQ=10,
        grid_s_size=500,
        n_folds=5,
        holdout_routes=None,
        skip_gtfs=False,
        is_param_search=True
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/param_search/",
        network_folder="atb/",
        EPOCH_EVAL_FREQ=10,
        grid_s_size=500,
        n_folds=5,
        holdout_routes=None,
        skip_gtfs=False,
        is_param_search=True
    )

    # FULL RUN