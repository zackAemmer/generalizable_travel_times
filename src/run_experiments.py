#!/usr/bin python3


import gc
import json
import os
import random

import numpy as np
import torch
from sklearn import metrics

from models import avg_speed, conv, ff, persistent, rnn, schedule, transformer
from utils import data_utils, model_utils


def run_experiments(run_folder, train_network_folder, test_network_folder, **kwargs):
    print("="*30)
    print(f"RUN EXPERIMENTS: '{run_folder}'")
    print(f"TRAINED ON NETWORK: '{train_network_folder}'")
    print(f"TEST ON NETWORK: '{test_network_folder}'")

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
    BATCH_SIZE = kwargs['BATCH_SIZE']
    HIDDEN_SIZE = kwargs['HIDDEN_SIZE']

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

    # Get list of available test files
    train_data_folder = f"{run_folder}{train_network_folder}deeptte_formatted/"
    test_data_folder = f"{run_folder}{test_network_folder}deeptte_formatted/"
    print(f"DATA: Train '{train_data_folder}', Test '{test_data_folder}'")
    train_file_list = list(filter(lambda x: x[:4]=="test" and len(x)==5, os.listdir(train_data_folder)))
    train_file_list.sort()
    test_file_list = list(filter(lambda x: x[:4]=="test" and len(x)==5, os.listdir(test_data_folder)))
    test_file_list.sort()
    print(f"TRAIN FILES: {train_file_list}")
    print(f"TEST FILES: {test_file_list}")
    print("="*30)

    run_results = []
    for fold_num in range(kwargs['n_folds']):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Declare baseline models
        base_model_list = []
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/AVG_{fold_num}.pkl"))
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/SCH_{fold_num}.pkl"))
        base_model_list.append(data_utils.load_pkl(f"{run_folder}{train_network_folder}models/PER_TIM_{fold_num}.pkl"))

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
            "FF_GRID_IND",
            n_features=12,
            n_grid_features=8*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(ff.FF_GRID_ATTN(
            "FF_GRID_ATTN",
            n_features=12,
            n_grid_features=8*3*3,
            n_channels=8,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(ff.FF_GRID(
            "FF_NGRID_IND",
            n_features=12,
            n_grid_features=4*3*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(rnn.GRU(
            "GRU",
            n_features=9,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(rnn.GRU_GRID(
            "GRU_GRID_IND",
            n_features=9,
            n_grid_features=8*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(rnn.GRU_GRID_ATTN(
            "GRU_GRID_ATTN",
            n_features=9,
            n_grid_features=8*3*3,
            n_channels=8,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(rnn.GRU_GRID(
            "GRU_NGRID_IND",
            n_features=9,
            n_grid_features=4*3*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF(
            "TRSF",
            n_features=9,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF_GRID(
            "TRSF_IND",
            n_features=9,
            n_grid_features=8*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF_GRID_ATTN(
            "TRSF_GRID_ATTN",
            n_features=9,
            n_grid_features=8*3*3,
            n_channels=8,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device))
        nn_model_list.append(transformer.TRSF_GRID(
            "TRSF_NGRID_IND",
            n_features=9,
            n_grid_features=4*3*3*3,
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

        # Load all model weights
        for m in nn_model_list:
            m = m.load_state_dict(torch.load(f"{run_folder}{train_network_folder}models/{m.model_name}_{fold_num}.pt"))

        # Test models on different networks
        model_fold_results = {}
        for x in all_model_list:
            model_fold_results[x.model_name] = {"Train_Labels":[], "Train_Preds":[], "Test_Labels":[], "Test_Preds":[]}

        print(f"Evaluating {run_folder}{train_network_folder} on {train_data_folder}")
        for valid_file in train_file_list:
            print(f"VALIDATE FILE: {valid_file}")
            valid_data, grid, ngrid = data_utils.load_all_data(train_data_folder, valid_file)
            grid_content = grid.get_fill_content()
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            # Construct dataloaders for all models
            dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, grid_content, ngrid, data_subset=kwargs['data_subset'])
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                print(f"Evaluating: {model.model_name}")
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Train_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Train_Preds"].extend(list(preds))

        print(f"Evaluating {run_folder}{train_network_folder} on {test_data_folder}")
        for valid_file in test_file_list:
            print(f"VALIDATE FILE: {valid_file}")
            valid_data, grid, ngrid = data_utils.load_all_data(test_data_folder, valid_file)
            grid_content = grid.get_fill_content()
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            # Construct dataloaders for all models
            dataloaders = model_utils.make_all_dataloaders(valid_data, config, BATCH_SIZE, NUM_WORKERS, grid_content, ngrid, data_subset=kwargs['data_subset'])
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                print(f"Evaluating: {model.model_name}")
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Test_Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Test_Preds"].extend(list(preds))

        # Calculate various losses:
        fold_results = {
            "Model_Names": [x.model_name for x in all_model_list],
            "Fold": fold_num,
            "Train_Losses": [],
            "Test_Losses": []
        }
        for mname in fold_results["Model_Names"]:
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Train_Labels"], model_fold_results[mname]["Train_Preds"]), 2))
            fold_results['Train_Losses'].append(_)
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Test_Labels"], model_fold_results[mname]["Test_Preds"]), 2))
            fold_results['Test_Losses'].append(_)

        # Save fold
        run_results.append(fold_results)

        # Clean memory at end of each fold
        gc.collect()
        if device==torch.device("cuda"):
            torch.cuda.empty_cache()

    # Save run results
    data_utils.write_pkl(run_results, f"{run_folder}{train_network_folder}model_generalization_results.pkl")
    print(f"EXPERIMENTS COMPLETED '{run_folder}{train_network_folder}'")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_experiments(
        run_folder="./results/debug/",
        train_network_folder="kcm/",
        test_network_folder="atb/",
        EPOCHS=50,
        BATCH_SIZE=512,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        data_subset=.1,
        n_folds=2,
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_experiments(
        run_folder="./results/debug/",
        train_network_folder="atb/",
        test_network_folder="kcm/",
        EPOCHS=50,
        BATCH_SIZE=512,
        LEARN_RATE=1e-3,
        HIDDEN_SIZE=32,
        data_subset=.1,
        n_folds=2,
    )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_experiments(
    #     run_folder="./results/small/",
    #     train_network_folder="kcm/",
    #     test_network_folder="atb/",
    #     EPOCHS=50,
    #     BATCH_SIZE=512,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     n_folds=5,
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_experiments(
    #     run_folder="./results/small/",
    #     train_network_folder="atb/",
    #     test_network_folder="kcm/",
    #     EPOCHS=50,
    #     BATCH_SIZE=512,
    #     LEARN_RATE=1e-3,
    #     HIDDEN_SIZE=32,
    #     n_folds=5,
    # )