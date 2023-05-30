#!/usr/bin python3


import gc
import itertools
import json
import os
import random

import numpy as np
import torch
from sklearn import metrics
from tabulate import tabulate

from models import avg_speed, conv, ff, persistent, rnn, schedule, transformer
from utils import data_loader, data_utils, model_utils


def run_experiments(run_folder, train_network_folder, test_network_folder, hyperparameters, **kwargs):
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
    BATCH_SIZE = hyperparameters['BATCH_SIZE']
    HIDDEN_SIZE = hyperparameters['HIDDEN_SIZE']

    # Define embedded variables for network models
    embed_dict = {
        'timeID': {
            'vocab_size': 1440,
            'embed_dims': 24
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
    print(f"VALID FILES: {train_file_list}")
    print(f"TEST FILES: {test_file_list}")
    print("="*30)

    run_results = []
    for fold_num in range(kwargs['n_folds']):

        # Declare baseline models
        avg_model = data_utils.load_pkl(f"{run_folder}{train_network_folder}models/AVG_{fold_num}.pkl")
        sch_model = data_utils.load_pkl(f"{run_folder}{train_network_folder}models/SCH_{fold_num}.pkl")
        tim_model = data_utils.load_pkl(f"{run_folder}{train_network_folder}models/PER_TIM_{fold_num}.pkl")

        # Declare neural network models
        ff_model = ff.FF(
            "FF",
            n_features=11,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        ff_grid_model1 = ff.FF_GRID(
            "FF_GRID_IND",
            n_features=11,
            n_grid_features=8*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        ff_grid_model2 = ff.FF_GRID_ATTN(
            "FF_GRID_ATTN",
            n_features=11,
            n_grid_features=8*3*3,
            n_channels=8,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        ff_grid_model3 = ff.FF_GRID(
            "FF_NGRID_IND",
            n_features=11,
            n_grid_features=4*3*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        gru_model = rnn.GRU(
            "GRU",
            n_features=8,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        gru_grid_model1 = rnn.GRU_GRID(
            "GRU_GRID_IND",
            n_features=8,
            n_grid_features=8*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        gru_grid_model2 = rnn.GRU_GRID_ATTN(
            "GRU_GRID_ATTN",
            n_features=8,
            n_grid_features=8*3*3,
            n_channels=8,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        gru_grid_model3 = rnn.GRU_GRID(
            "GRU_NGRID_IND",
            n_features=8,
            n_grid_features=4*3*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        # gru_mto_model = rnn.GRU_RNN_MTO(
        #     "GRU_RNN_MTO",
        #     n_features=8,
        #     hidden_size=HIDDEN_SIZE,
        #     batch_size=BATCH_SIZE,
        #     embed_dict=embed_dict,
        #     device=device
        # ).to(device)
        # conv1d_model = conv.CONV(
        #     "CONV1D",
        #     n_features=8,
        #     hidden_size=HIDDEN_SIZE,
        #     batch_size=BATCH_SIZE,
        #     embed_dict=embed_dict,
        #     device=device
        # ).to(device)
        trs_model = transformer.TRSF(
            "TRSF",
            n_features=8,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        trs_grid_model1 = transformer.TRSF_GRID(
            "TRSF_IND",
            n_features=8,
            n_grid_features=8*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        trs_grid_model2 = transformer.TRSF_GRID_ATTN(
            "TRSF_GRID_ATTN",
            n_features=8,
            n_grid_features=8*3*3,
            n_channels=8,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        trs_grid_model3 = transformer.TRSF_GRID(
            "TRSF_NGRID_IND",
            n_features=8,
            n_grid_features=4*3*3*3,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=8,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)

        # Add all models to results list
        base_model_list = []
        base_model_list.append(avg_model)
        base_model_list.append(sch_model)
        base_model_list.append(tim_model)
        nn_model_list = []
        nn_model_list.append(ff_model)
        nn_model_list.append(ff_grid_model1)
        nn_model_list.append(ff_grid_model2)
        nn_model_list.append(ff_grid_model3)
        nn_model_list.append(gru_model)
        nn_model_list.append(gru_grid_model1)
        nn_model_list.append(gru_grid_model2)
        nn_model_list.append(gru_grid_model3)
        # nn_model_list.append(gru_mto_model)
        # nn_model_list.append(conv1d_model)
        nn_model_list.append(trs_model)
        nn_model_list.append(trs_grid_model1)
        nn_model_list.append(trs_grid_model2)
        nn_model_list.append(trs_grid_model3)
        all_model_list = []
        all_model_list.extend(base_model_list)
        all_model_list.extend(nn_model_list)

        # Load all model weights
        for m in nn_model_list:
            m = m.load_state_dict(torch.load(f"{run_folder}{train_network_folder}models/{m.model_name}_{fold_num}.pt"))
        print(f"Model names: {[m.model_name for m in nn_model_list]}")
        print(f"Model total parameters: {[sum(p.numel() for p in m.parameters()) for m in nn_model_list]}")

        # Test models on the network they trained on
        print(f"Evaluating {train_data_folder} on {train_data_folder}")
        train_results = []
        model_fold_results = {}
        for x in all_model_list:
            model_fold_results[x.model_name] = {"Labels":[], "Preds":[]}
        for valid_file in train_file_list:
            valid_data, grid, ngrid = data_utils.load_all_data(train_data_folder, valid_file)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            grid_content = grid.get_fill_content()
            print(f"VALIDATE FILE: {valid_file}")
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            # Construct dataloaders for all models
            dataloaders = []
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
            # dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                print(f"Evaluating: {model.model_name}")
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Preds"].extend(list(preds))
            train_results.append(model_fold_results)

        # Test models on network they did not train on
        print(f"Evaluating {train_data_folder} on {test_data_folder}")
        test_results = []
        model_fold_results = {}
        for x in all_model_list:
            model_fold_results[x.model_name] = {"Labels":[], "Preds":[]}
        for valid_file in train_file_list:
            valid_data, grid, ngrid = data_utils.load_all_data(test_data_folder, valid_file)
            print(f"Successfully loaded {len(valid_data)} testing samples.")
            grid_content = grid.get_fill_content()
            print(f"VALIDATE FILE: {valid_file}")
            with open(f"{train_data_folder}train_config.json", "r") as f:
                config = json.load(f)
            # Construct dataloaders for all models
            dataloaders = []
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
            # dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(valid_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                print(f"Evaluating: {model.model_name}")
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Preds"].extend(list(preds))
            test_results.append(model_fold_results)

            # Calculate various losses:
            fold_results = {
                "Model Names": [x.model_name for x in all_model_list],
                "Fold": fold_num,
                "All Losses": [],
            }
            for mname in fold_results["Model Names"]:
                _ = [mname]
                _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]), 2))
                _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"])), 2))
                _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]), 2))
                fold_results['All Losses'].append(_)

        # Save fold
        run_results.append(fold_results)

        # Clean memory at end of each fold
        gc.collect()
        if device==torch.device("cuda"):
            torch.cuda.empty_cache()

    # Save run results
    data_utils.write_pkl(run_results, f"{run_folder}{train_network_folder}model_generalization_results.pkl")
    print(f"MODEL RUN COMPLETED '{run_folder}{train_network_folder}'")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_experiments(
        run_folder="./results/debug/",
        train_network_folder="kcm/",
        test_network_folder="atb/",
        hyperparameters={
            "EPOCHS": 50,
            "BATCH_SIZE": 512,
            "LEARN_RATE": 1e-3,
            "HIDDEN_SIZE": 32
        },
        n_folds=5,
        fold_model=0
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_experiments(
        run_folder="./results/debug/",
        train_network_folder="atb/",
        test_network_folder="kcm/",
        hyperparameters={
            "EPOCHS": 50,
            "BATCH_SIZE": 512,
            "LEARN_RATE": 1e-3,
            "HIDDEN_SIZE": 32
        },
        n_folds=5,
        fold_model=0
    )