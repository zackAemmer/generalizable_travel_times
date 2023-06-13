#!/usr/bin python3


import gc
import itertools
import json
import os
import random
import time

import numpy as np
import torch
from sklearn import metrics
from tabulate import tabulate

from models import avg_speed, conv, ff, persistent, rnn, schedule, transformer
from utils import data_loader, data_utils, model_utils


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

    # Get list of available train/test files
    data_folder = f"{run_folder}{network_folder}deeptte_formatted/"
    print(f"DATA: '{data_folder}'")
    train_file_list = list(filter(lambda x: x[:5]=="train" and len(x)==6, os.listdir(data_folder)))
    train_file_list.sort()
    test_file_list = list(filter(lambda x: x[:4]=="test" and len(x)==5, os.listdir(data_folder)))
    test_file_list.sort()
    print(f"TRAIN FILES: {train_file_list}")
    print(f"VALID FILES: {test_file_list}")

    # Run full training process for each model during each validation fold
    run_results = []
    start_fold = kwargs['n_folds']
    if 'start_fold' in kwargs:
        print(f"STARTING ON FOLD: {kwargs['start_fold']}")
        run_results = data_utils.load_pkl(f"{run_folder}{network_folder}model_results_temp.pkl")
        start_fold = kwargs['start_fold']
    else:
        start_fold = 0
    for fold_num in range(start_fold, kwargs['n_folds']):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Declare baseline models
        base_model_list = []
        base_model_list.append(avg_speed.AvgHourlySpeedModel("AVG"))
        base_model_list.append(schedule.TimeTableModel("SCH"))
        base_model_list.append(persistent.PersistentTimeSeqModel("PER_TIM"))

        # Declare neural network models
        nn_model_list = []
        nn_model_list.append(ff.FF(
            "FF",
            n_features=12,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(ff.FF_GRID(
            "FF_NGRID_IND",
            n_features=12,
            n_grid_features=3*3*5*5,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(rnn.GRU(
            "GRU",
            n_features=10,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(rnn.GRU_GRID(
            "GRU_NGRID_IND",
            n_features=10,
            n_grid_features=3*3*5*5,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF(
            "TRSF",
            n_features=10,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF_GRID(
            "TRSF_NGRID_IND",
            n_features=10,
            n_grid_features=3*3*5*5,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF_GRID_ATTN(
            "TRSF_NGRID_CRS",
            n_features=10,
            n_grid_features=3*3*5*5,
            n_channels=3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))

        all_model_list = []
        all_model_list.extend(base_model_list)
        all_model_list.extend(nn_model_list)

        print(f"Model names: {[m.model_name for m in nn_model_list]}")
        print(f"Model total parameters: {[sum(p.numel() for p in m.parameters()) for m in nn_model_list]}")

        # Keep track of all model performances
        model_fold_results = {}
        for x in all_model_list:
            model_fold_results[x.model_name] = {"Labels":[], "Preds":[]}
        # Keep track of train/test curves during training for network models
        model_fold_curves = {}
        for x in nn_model_list:
            model_fold_curves[x.model_name] = {"Train":[], "Test":[]}

        for epoch in range(EPOCHS):
            print(f"FOLD: {fold_num}, EPOCH: {epoch}")
            # Train all models on each training file; split samples in each file by fold
            for train_file in list(train_file_list):
                # Load data and config for this training fold/file
                train_data, test_data, ngrid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
                ngrid_content = ngrid.get_fill_content()
                print(f"TRAIN FILE: {train_file}, {len(train_data)} train samples, {len(test_data)} test samples")
                with open(f"{data_folder}train_config.json", "r") as f:
                    config = json.load(f)

                # Construct dataloaders
                base_dataloaders, nn_dataloaders = model_utils.make_all_dataloaders(train_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, combine=False, holdout_routes=kwargs['holdout_routes'])

                # Train all models
                for model, loader in zip(base_model_list, base_dataloaders):
                    print(f"Training: {model.model_name}")
                    t0 = time.time()
                    model.train(loader, config)
                    model.train_time += (time.time() - t0)
                for model, loader in zip(nn_model_list, nn_dataloaders):
                    print(f"Training: {model.model_name}")
                    t0 = time.time()
                    avg_batch_loss = model_utils.train(model, loader, LEARN_RATE)
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
                for train_file in train_file_list:
                    # Load data and config for this training fold
                    train_data, test_data, ngrid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
                    ngrid_content = ngrid.get_fill_content()
                    print(f"TEST FILE: {train_file}, {len(train_data)} train samples, {len(test_data)} test samples")
                    with open(f"{data_folder}train_config.json", "r") as f:
                        config = json.load(f)

                    # Construct dataloaders for network models
                    _, train_dataloaders = model_utils.make_all_dataloaders(train_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, combine=False, holdout_routes=kwargs['holdout_routes'])
                    _, test_dataloaders = model_utils.make_all_dataloaders(test_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, combine=False, holdout_routes=kwargs['holdout_routes'])

                    # Test all NN models on training and testing sets for this fold, across all files
                    for i, (model, train_loader, test_loader) in enumerate(zip(nn_model_list, train_dataloaders, test_dataloaders)):
                        print(f"Evaluating: {model.model_name}")
                        labels, preds = model.evaluate(train_loader, config)
                        train_losses[i] += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                        labels, preds = model.evaluate(test_loader, config)
                        test_losses[i] += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                # Record average train/test losses of all models across all files
                for i, model in enumerate(nn_model_list):
                    model_fold_curves[model.model_name]['Train'].append(train_losses[i] / len(train_file_list))
                    model_fold_curves[model.model_name]['Test'].append(test_losses[i] / len(train_file_list))

        # Save final fold models
        print(f"Fold {fold_num} training complete, saving model states and metrics...")
        for model in base_model_list:
            model.save_to(f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pkl")
        for model in nn_model_list:
            torch.save(model.state_dict(), f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pt")

        # Calculate performance metrics for fold
        for train_file in train_file_list:
            train_data, test_data, ngrid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
            ngrid_content = ngrid.get_fill_content()
            print(f"TEST FILE: {train_file}, {len(test_data)} test samples")
            with open(f"{data_folder}train_config.json", "r") as f:
                config = json.load(f)

            dataloaders = model_utils.make_all_dataloaders(test_data, config, BATCH_SIZE, NUM_WORKERS, ngrid_content, combine=True, holdout_routes=kwargs['holdout_routes'])

            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                print(f"Evaluating: {model.model_name}")
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Preds"].extend(list(preds))

        # Calculate various losses:
        fold_results = {
            "Model_Names": [x.model_name for x in all_model_list],
            "Fold": fold_num,
            "All_Losses": [],
            "Loss_Curves": [{model: curve_dict} for model, curve_dict in zip(model_fold_curves.keys(), list(model_fold_curves.values()))],
            "Train_Times": [x.train_time for x in all_model_list]
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


if __name__=="__main__":
    torch.set_default_dtype(torch.float)

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_models(
    #     run_folder="./results/debug/",
    #     network_folder="kcm/",
    #     EPOCHS=10,
    #     BATCH_SIZE=512,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     n_folds=3,
    #     holdout_routes=[100252,100139,102581,100341,102720]
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_models(
    #     run_folder="./results/debug/",
    #     network_folder="atb/",
    #     EPOCHS=10,
    #     BATCH_SIZE=512,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     n_folds=3,
    #     holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    # )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/cross_attn/",
        network_folder="kcm/",
        EPOCHS=50,
        BATCH_SIZE=512,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        n_folds=5,
        holdout_routes=[100252,100139,102581,100341,102720]
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/cross_attn/",
        network_folder="atb/",
        EPOCHS=50,
        BATCH_SIZE=512,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        n_folds=5,
        holdout_routes=["ATB:Line:2_28","ATB:Line:2_3","ATB:Line:2_9","ATB:Line:2_340","ATB:Line:2_299"]
    )