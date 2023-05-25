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


def run_models(run_folder, network_folder, hyperparameters, **kwargs):
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
    EPOCHS = hyperparameters['EPOCHS']
    BATCH_SIZE = hyperparameters['BATCH_SIZE']
    LEARN_RATE = hyperparameters['LEARN_RATE']
    HIDDEN_SIZE = hyperparameters['HIDDEN_SIZE']
    EPOCH_EVAL_FREQ = 5

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
    for fold_num in range(0, kwargs['n_folds']):
        print("="*30)
        print(f"BEGIN FOLD: {fold_num}")

        # Declare baseline models
        avg_model = avg_speed.AvgHourlySpeedModel("AVG")
        sch_model = schedule.TimeTableModel("SCH")
        tim_model = persistent.PersistentTimeSeqModel("PER_TIM")

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
            "TRSF_ENC",
            n_features=8,
            hidden_size=HIDDEN_SIZE,
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
        all_model_list = []
        all_model_list.extend(base_model_list)
        all_model_list.extend(nn_model_list)

        print(f"Model names: {[m.model_name for m in nn_model_list]}")
        print(f"Model total parameters: {[sum(p.numel() for p in m.parameters()) for m in nn_model_list]}")

        # Keep track of train/test curves during training for network models
        model_fold_curves = {}
        for x in nn_model_list:
            if hasattr(x, "hidden_size"):
                model_fold_curves[x.model_name] = {"Train":[], "Test":[]}

        # Keep track of all model performances
        model_fold_results = {}
        for x in all_model_list:
            model_fold_results[x.model_name] = {"Labels":[], "Preds":[]}

        for epoch in range(EPOCHS):
            print(f"FOLD: {fold_num}, EPOCH: {epoch}")
            # Train all models on each training file; split samples in each file by fold
            for train_file in list(train_file_list):
                # Load data and config for this training fold/file
                train_data, test_data, grid, ngrid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
                grid_content = grid.get_fill_content()
                print(f"TRAIN FILE: {train_file}, {len(train_data)} train samples, {len(test_data)} test samples")
                with open(f"{data_folder}train_config.json", "r") as f:
                    config = json.load(f)

                # Construct dataloaders
                base_dataloaders = []
                nn_dataloaders = []
                base_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
                base_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
                base_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
                # nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS))
                nn_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.transformer_collate, NUM_WORKERS))

                # Train all models
                for model, loader in zip(base_model_list, base_dataloaders):
                    print(f"Training: {model.model_name}")
                    model.train(loader, config)
                for model, loader in zip(nn_model_list, nn_dataloaders):
                    print(f"Training: {model.model_name}")
                    avg_batch_loss = model_utils.train(model, loader, LEARN_RATE)

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
                    train_data, test_data, grid, ngrid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
                    grid_content = grid.get_fill_content()
                    print(f"TEST FILE: {train_file}, {len(train_data)} train samples, {len(test_data)} test samples")
                    with open(f"{data_folder}train_config.json", "r") as f:
                        config = json.load(f)

                    # Construct dataloaders for network models
                    train_dataloaders = []
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
                    # train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS))
                    train_dataloaders.append(data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.transformer_collate, NUM_WORKERS))
                    test_dataloaders = []
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
                    # test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS))
                    test_dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.transformer_collate, NUM_WORKERS))

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
            train_data, test_data, grid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
            grid_content = grid.get_fill_content()
            with open(f"{data_folder}train_config.json", "r") as f:
                config = json.load(f)

            dataloaders = []
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=grid_content, buffer=1))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_grid_collate, NUM_WORKERS, grid=ngrid, is_ngrid=True, buffer=1))
            # dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS))
            dataloaders.append(data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.transformer_collate, NUM_WORKERS))

            # Test all models
            for model, loader in zip(all_model_list, dataloaders):
                print(f"Evaluating: {model.model_name}")
                labels, preds = model.evaluate(loader, config)
                model_fold_results[model.model_name]["Labels"].extend(list(labels))
                model_fold_results[model.model_name]["Preds"].extend(list(preds))

        # Calculate various losses:
        fold_results = {
            "Model Names": [x.model_name for x in all_model_list],
            "Fold": fold_num,
            "All Losses": [],
            "Loss Curves": [{model: curve_dict} for model, curve_dict in zip(model_fold_curves.keys(), list(model_fold_curves.values()))]
        }
        for mname in fold_results["Model Names"]:
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]), 2))
            fold_results['All Losses'].append(_)

        # Print results of this fold
        print(tabulate(fold_results['All Losses'], headers=["Model", "MAPE", "RMSE", "MAE"]))
        run_results.append(fold_results)

        # Clean memory at end of each fold
        gc.collect()
        if device==torch.device("cuda"):
            torch.cuda.empty_cache()

    # Save run results
    data_utils.write_pkl(run_results, f"{run_folder}{network_folder}model_results.pkl")
    print(f"MODEL RUN COMPLETED '{run_folder}{network_folder}'")


if __name__=="__main__":
    torch.set_default_dtype(torch.float)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/debug/",
        network_folder="kcm/",
        hyperparameters={
            "EPOCHS": 50,
            "BATCH_SIZE": 512,
            "LEARN_RATE": 1e-3,
            "HIDDEN_SIZE": 32
        },
        n_folds=5
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/debug/",
        network_folder="atb/",
        hyperparameters={
            "EPOCHS": 50,
            "BATCH_SIZE": 512,
            "LEARN_RATE": 1e-3,
            "HIDDEN_SIZE": 32
        },
        n_folds=5
    )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_models(
    #     run_folder="./results/small/",
    #     network_folder="kcm/",
    #     hyperparameters={
    #         "EPOCHS": 50,
    #         "BATCH_SIZE": 512,
    #         "LEARN_RATE": 1e-3,
    #         "HIDDEN_SIZE": 32
    #     },
    #     n_folds=5
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_models(
    #     run_folder="./results/small/",
    #     network_folder="atb/",
    #     hyperparameters={
    #         "EPOCHS": 50,
    #         "BATCH_SIZE": 512,
    #         "LEARN_RATE": 1e-3,
    #         "HIDDEN_SIZE": 32
    #     },
    #     n_folds=5
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_models(
    #     run_folder="./results/medium/",
    #     network_folder="kcm/",
    #     hyperparameters={
    #         "EPOCHS": 50,
    #         "BATCH_SIZE": 512,
    #         "LEARN_RATE": 1e-3,
    #         "HIDDEN_SIZE": 32
    #     },
    #     n_folds=5
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # run_models(
    #     run_folder="./results/medium/",
    #     network_folder="atb/",
    #     hyperparameters={
    #         "EPOCHS": 50,
    #         "BATCH_SIZE": 512,
    #         "LEARN_RATE": 1e-3,
    #         "HIDDEN_SIZE": 32
    #     },
    #     n_folds=5
    # )