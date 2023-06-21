#!/usr/bin python3


import gc
import itertools
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from sklearn import metrics
from sklearn.model_selection import KFold
from tabulate import tabulate

from models import avg_speed, conv, ff, grids, persistent, rnn, schedule, transformer
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
    #     with record_function("train_models"):

    print("="*30)
    print(f"RUN MODEL: '{run_folder}'")
    print(f"NETWORK: '{network_folder}'")

    # Select device to train on, and number workers if GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        NUM_WORKERS = 8
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
        NUM_WORKERS = 0
    print(f"DEVICE: {device}")
    print(f"WORKERS: {NUM_WORKERS}")

    # Set hyperparameters
    EPOCHS = kwargs['EPOCHS']
    BATCH_SIZE = kwargs['BATCH_SIZE']
    LEARN_RATE = kwargs['LEARN_RATE']
    HIDDEN_SIZE = kwargs['HIDDEN_SIZE']
    EPOCH_EVAL_FREQ = 5

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

    # Data loading and fold setup
    data_folder = f"{run_folder}{network_folder}deeptte_formatted/"
    print(f"DATA: '{data_folder}'")
    with open(f"{data_folder}train_config.json", "r") as f:
        config = json.load(f)
    dataset = data_loader.GenericDataset(f"{data_folder}train", config, holdout_routes=kwargs['holdout_routes'])
    splits = KFold(kwargs['n_folds'], shuffle=True, random_state=0)
    run_results = []

    # Run full training process for each model during each validation fold
    for fold_num, (train_idx,test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Random samplers for indices from this fold
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # Declare baseline models
        base_model_list = []
        base_model_list.append(avg_speed.AvgHourlySpeedModel("AVG"))
        base_model_list.append(schedule.TimeTableModel("SCH"))
        base_model_list.append(persistent.PersistentTimeSeqModel("PER_TIM"))
        # Declare neural network models
        nn_model_list = model_utils.make_all_models(HIDDEN_SIZE, BATCH_SIZE, embed_dict, device)
        nn_optimizer_list = [torch.optim.Adam(model.parameters(), lr=LEARN_RATE) for model in nn_model_list]
        print(f"Model names: {[m.model_name for m in nn_model_list]}")
        print(f"Model total parameters: {[sum(p.numel() for p in m.parameters()) for m in nn_model_list]}")
        # Summarize models in run
        model_names = [x.model_name for x in base_model_list]
        model_names.extend([x.model_name for x in nn_model_list])
        print(f"Model names: {model_names}")
        print(f"NN model total parameters: {[sum(p.numel() for p in m.parameters()) for m in nn_model_list]}")
        # Keep track of all model performances
        model_fold_results = {}
        for x in model_names:
            model_fold_results[x] = {
                "Labels":[],
                "Preds":[]
            }
        # Keep track of train/test curves during training for network models
        model_fold_curves = {}
        for x in nn_model_list:
            model_fold_curves[x.model_name] = {
                "Train":[],
                "Test":[]
            }

        # # Build grid using only data from this fold
        # train_data = [dataset.content[i] for i in train_idx]
        # train_data = data_utils.map_from_deeptte(train_data, ["locationtime","x","y","speed_m_s","bearing"])
        # train_ngrid = grids.NGridBetter(config['grid_bounds'], kwargs['grid_s_size'])
        # train_ngrid.add_grid_content(train_data)
        # train_ngrid.build_cell_lookup()
        # test_data = [dataset.content[i] for i in test_idx]
        # test_data = data_utils.map_from_deeptte(test_data, ["locationtime","x","y","speed_m_s","bearing"])
        # test_ngrid = grids.NGridBetter(config['grid_bounds'], kwargs['grid_s_size'])
        # test_ngrid.add_grid_content(test_data)
        # test_ngrid.build_cell_lookup()

        # xbins, ybins = train_ngrid.digitize_points(train_data[:,1], train_data[:,2])
        # feats = train_ngrid.get_grid_features(xbins[:10000], ybins[:10000], train_data[:10000,0])
        # return None
        # dataset.grid = train_ngrid

        # Train all models on each training file
        for epoch in range(EPOCHS):
            print(f"FOLD: {fold_num}, EPOCH: {epoch}")
            # Construct dataloaders
            base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(dataset, train_sampler, BATCH_SIZE, NUM_WORKERS)

            # Train all models
            for model, loader in zip(base_model_list, base_dataloaders):
                print(f"Training: {model.model_name}")
                t0 = time.time()
                model.train(loader, config)
                model.train_time += (time.time() - t0)
            for model, loader, optimizer in zip(nn_model_list, nn_dataloaders, nn_optimizer_list):
                print(f"Training: {model.model_name}")
                t0 = time.time()
                avg_batch_loss = model_utils.train(model, loader, optimizer, LEARN_RATE)
                model.train_time += (time.time() - t0)

            if epoch % EPOCH_EVAL_FREQ == 0:
                # Save current model states
                print(f"Reached epoch {epoch} checkpoint, saving model states and curve values...")
                for model in base_model_list:
                    model.save_to(f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pkl")
                for model in nn_model_list:
                    torch.save(model.state_dict(), f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pt")

                # Record model curves on all train/test files for this fold]
                train_losses = [0.0 for x in nn_model_list]
                test_losses = [0.0 for x in nn_model_list]

                # Construct dataloaders for network models
                _, train_dataloaders = model_utils.make_all_dataloaders(dataset, train_sampler, BATCH_SIZE, NUM_WORKERS)
                _, test_dataloaders = model_utils.make_all_dataloaders(dataset, test_sampler, BATCH_SIZE, NUM_WORKERS)

                # Test all NN models on training and testing sets for this fold, across all files
                for i, (model, train_loader, test_loader) in enumerate(zip(nn_model_list, train_dataloaders, test_dataloaders)):
                    print(f"Evaluating: {model.model_name}")
                    labels, preds = model.evaluate(train_loader, config)
                    train_losses[i] += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                    labels, preds = model.evaluate(test_loader, config)
                    test_losses[i] += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                # Record average train/test losses of all models across all files
                for i, model in enumerate(nn_model_list):
                    model_fold_curves[model.model_name]['Train'].append(train_losses[i])
                    model_fold_curves[model.model_name]['Test'].append(test_losses[i])

        # Save final fold models
        print(f"Fold {fold_num} training complete, saving model states and metrics...")
        for model in base_model_list:
            model.save_to(f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pkl")
        for model in nn_model_list:
            torch.save(model.state_dict(), f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pt")

        # Calculate performance metrics for fold
        base_test_dataloaders, nn_test_dataloaders = model_utils.make_all_dataloaders(dataset, test_sampler, BATCH_SIZE, NUM_WORKERS)
        for model, loader in zip(base_model_list, base_test_dataloaders):
            print(f"Evaluating: {model.model_name}")
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Preds"].extend(list(preds))
        for model, loader in zip(nn_model_list, nn_test_dataloaders):
            print(f"Evaluating: {model.model_name}")
            labels, preds = model.evaluate(loader, config)
            model_fold_results[model.model_name]["Labels"].extend(list(labels))
            model_fold_results[model.model_name]["Preds"].extend(list(preds))
        # Calculate various losses:
        model_names = [x.model_name for x in base_model_list]
        model_names.extend([x.model_name for x in nn_model_list])
        train_times = [x.train_time for x in base_model_list]
        train_times.extend([x.train_time for x in nn_model_list])
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
        data_utils.write_pkl(run_results, f"{run_folder}{network_folder}model_results_temp.pkl")

        # Clean memory at end of each fold
        gc.collect()
        if device==torch.device("cuda"):
            torch.cuda.empty_cache()

    # Save run results
    data_utils.write_pkl(run_results, f"{run_folder}{network_folder}model_results.pkl")
    print(f"MODEL RUN COMPLETED '{run_folder}{network_folder}'")

    # prof.export_chrome_trace("trace.json")
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

if __name__=="__main__":
    torch.set_default_dtype(torch.float)

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_models(
    #     run_folder="./results/debug/",
    #     network_folder="kcm/",
    #     EPOCHS=30,
    #     BATCH_SIZE=512,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     grid_s_size=500,
    #     n_folds=5,
    #     holdout_routes=[100252,100139,102581,100341,102720]
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_models(
    #     run_folder="./results/debug/",
    #     network_folder="atb/",
    #     EPOCHS=30,
    #     BATCH_SIZE=512,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     grid_s_size=500,
    #     n_folds=5,
    #     holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    # )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/small/",
        network_folder="kcm/",
        EPOCHS=30,
        BATCH_SIZE=512,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        n_folds=3,
        holdout_routes=[100252,100139,102581,100341,102720]
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/small/",
        network_folder="atb/",
        EPOCHS=30,
        BATCH_SIZE=512,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        n_folds=3,
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    )