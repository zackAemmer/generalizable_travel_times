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

        # Declare network models
        ff_model = ff.FF(
            "FF",
            n_features=11,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        # Individual grid cells
        ff_grid_model1 = ff.FF_GRID(
            "FF_GRID_IND",
            n_features=11,
            n_grid_features=72,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=4,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        # Attention on unrolled grid
        ff_grid_model2 = ff.FF_GRID_ATTN(
            "FF_GRID_ATTN",
            n_features=11,
            n_grid_features=72,
            n_channels=8,
            hidden_size=HIDDEN_SIZE,
            grid_compression_size=4,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)
        # gru_model = rnn.GRU_RNN(
        #     "GRU_RNN",
        #     8,
        #     1,
        #     HIDDEN_SIZE,
        #     BATCH_SIZE,
        #     embed_dict,
        #     device
        # ).to(device)
        # gru_mto_model = rnn.GRU_RNN_MTO(
        #     "GRU_RNN_MTO",
        #     8,
        #     1,
        #     HIDDEN_SIZE,
        #     BATCH_SIZE,
        #     embed_dict,
        #     device
        # ).to(device)
        # conv1d_model = conv.CONV(
        #     "CONV1D",
        #     8,
        #     1,
        #     HIDDEN_SIZE,
        #     BATCH_SIZE,
        #     embed_dict,
        #     device
        # ).to(device)
        trs_model = transformer.TRANSFORMER(
            "TRSF_ENC",
            n_features=8,
            hidden_size=HIDDEN_SIZE,
            batch_size=BATCH_SIZE,
            embed_dict=embed_dict,
            device=device
        ).to(device)

        # Add all models to results list
        model_list = []
        model_list.append(avg_model)
        model_list.append(sch_model)
        model_list.append(tim_model)
        model_list.append(ff_model)
        model_list.append(ff_grid_model1)
        model_list.append(ff_grid_model2)
        # model_list.append(gru_model)
        # model_list.append(gru_mto_model)
        # model_list.append(conv1d_model)
        model_list.append(trs_model)

        # Keep track of train/test curves during training for network models
        model_fold_curves = {}
        for x in model_list:
            if hasattr(x, "hidden_size"):
                model_fold_curves[x.model_name] = {"Train":[], "Test":[]}

        # Keep track of all model performances
        model_fold_results = {}
        for x in model_list:
            model_fold_results[x.model_name] = {"Labels":[], "Preds":[]}

        for epoch in range(EPOCHS):
            print(f"FOLD: {fold_num}, EPOCH: {epoch}")
            # Train all models on each training file; split samples in each file by fold
            for train_file in list(train_file_list):
                # Load data and config for this training fold
                train_data, test_data, grid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
                print(f"TRAIN FILE: {train_file}, {len(train_data)} train samples, {len(test_data)} test samples")
                with open(f"{data_folder}train_config.json", "r") as f:
                    config = json.load(f)

                # Construct dataloaders for network models
                train_dataloader_basic = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS)
                train_dataloader_basic_grid = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid, buffer=1)
                train_dataloader_seq = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS)
                # train_dataloader_seq_mto = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS)
                train_dataloader_trs = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.transformer_collate, NUM_WORKERS)

                # Train all models
                avg_model.fit(train_dataloader_basic, config)
                sch_model.fit(train_dataloader_basic, config)
                tim_model.fit(train_dataloader_seq, config)

                avg_batch_loss = model_utils.train(ff_model, train_dataloader_basic, LEARN_RATE)
                avg_batch_loss = model_utils.train(ff_grid_model1, train_dataloader_basic_grid, LEARN_RATE)
                avg_batch_loss = model_utils.train(ff_grid_model2, train_dataloader_basic_grid, LEARN_RATE)
                # avg_batch_loss = model_utils.train(gru_model, train_dataloader_seq, LEARN_RATE)
                # avg_batch_loss = model_utils.train(gru_mto_model, train_dataloader_seq_mto, LEARN_RATE)
                # avg_batch_loss = model_utils.train(conv1d_model, train_dataloader_seq, LEARN_RATE)
                avg_batch_loss = model_utils.train(trs_model, train_dataloader_trs, LEARN_RATE)

            if epoch % EPOCH_EVAL_FREQ == 0:
                # Save current model states
                print(f"Reached epoch {epoch} checkpoint, saving model states and curve values...")
                avg_model.save_to(f"{run_folder}{network_folder}models/{avg_model.model_name}_{fold_num}.pkl")
                sch_model.save_to(f"{run_folder}{network_folder}models/{sch_model.model_name}_{fold_num}.pkl")
                tim_model.save_to(f"{run_folder}{network_folder}models/{tim_model.model_name}_{fold_num}.pkl")
                torch.save(ff_model.state_dict(), f"{run_folder}{network_folder}models/{ff_model.model_name}_{fold_num}.pt")
                torch.save(ff_grid_model1.state_dict(), f"{run_folder}{network_folder}models/{ff_grid_model1.model_name}_{fold_num}.pt")
                torch.save(ff_grid_model2.state_dict(), f"{run_folder}{network_folder}models/{ff_grid_model2.model_name}_{fold_num}.pt")
                # torch.save(gru_model.state_dict(), f"{run_folder}{network_folder}models/{gru_model.model_name}_{fold_num}.pt")
                # torch.save(gru_mto_model.state_dict(), f"{run_folder}{network_folder}models/{gru_mto_model.model_name}_{fold_num}.pt")
                # torch.save(conv1d_model.state_dict(), f"{run_folder}{network_folder}models/{conv1d_model.model_name}_{fold_num}.pt")
                torch.save(trs_model.state_dict(), f"{run_folder}{network_folder}models/{trs_model.model_name}_{fold_num}.pt")

                # Record model curves on all train/test files for this fold
                ff_train = 0.0
                ff_grid1_train = 0.0
                ff_grid2_train = 0.0
                # gru_train = 0.0
                # gru_mto_train = 0.0
                # conv1d_train = 0.0
                trs_train = 0.0

                ff_test = 0.0
                ff_grid1_test = 0.0
                ff_grid2_test = 0.0
                # gru_test = 0.0
                # gru_mto_test = 0.0
                # conv1d_test = 0.0
                trs_test = 0.0

                for train_file in train_file_list:
                    # Load data and config for this training fold
                    train_data, test_data, grid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
                    print(f"TEST FILE: {train_file}, {len(train_data)} train samples, {len(test_data)} test samples")
                    with open(f"{data_folder}train_config.json", "r") as f:
                        config = json.load(f)

                    # Construct dataloaders for network models
                    train_dataloader_basic = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS)
                    train_dataloader_basic_grid = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid, buffer=1)
                    # train_dataloader_seq = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS)
                    # train_dataloader_seq_mto = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS)
                    train_dataloader_trs = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, data_loader.transformer_collate, NUM_WORKERS)

                    test_dataloader_basic = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS)
                    test_dataloader_basic_grid = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid, buffer=1)
                    # test_dataloader_seq = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS)
                    # test_dataloader_seq_mto = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS)
                    test_dataloader_trs = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.transformer_collate, NUM_WORKERS)

                    # Test all NN models on training and testing sets for this fold, across all files
                    labels, preds = ff_model.evaluate(train_dataloader_basic, config)
                    ff_train += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                    labels, preds = ff_model.evaluate(test_dataloader_basic, config)
                    ff_test += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                    labels, preds = ff_grid_model1.evaluate(train_dataloader_basic_grid, config)
                    ff_grid1_train += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                    labels, preds = ff_grid_model1.evaluate(test_dataloader_basic_grid, config)
                    ff_grid1_test += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                    labels, preds = ff_grid_model2.evaluate(train_dataloader_basic_grid, config)
                    ff_grid2_train += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                    labels, preds = ff_grid_model2.evaluate(test_dataloader_basic_grid, config)
                    ff_grid2_test += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                    # labels, preds = gru_model.evaluate(train_dataloader_seq, config)
                    # gru_train += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                    # labels, preds = gru_model.evaluate(test_dataloader_seq, config)
                    # gru_test += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                    # labels, preds = gru_mto_model.evaluate(train_dataloader_seq_mto, config)
                    # gru_mto_train += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                    # labels, preds = gru_mto_model.evaluate(test_dataloader_seq_mto, config)
                    # gru_mto_test += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                    # labels, preds = conv1d_model.evaluate(train_dataloader_seq, config)
                    # conv1d_train += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                    # labels, preds = conv1d_model.evaluate(test_dataloader_seq, config)
                    # conv1d_test += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                    labels, preds = trs_model.evaluate(train_dataloader_trs, config)
                    trs_train += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)
                    labels, preds = trs_model.evaluate(test_dataloader_trs, config)
                    trs_test += np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2)

                model_fold_curves[ff_model.model_name]['Train'].append(ff_train / len(train_file_list))
                model_fold_curves[ff_model.model_name]['Test'].append(ff_test / len(train_file_list))
                model_fold_curves[ff_grid_model1.model_name]['Train'].append(ff_grid1_train / len(train_file_list))
                model_fold_curves[ff_grid_model1.model_name]['Test'].append(ff_grid1_test / len(train_file_list))
                model_fold_curves[ff_grid_model2.model_name]['Train'].append(ff_grid2_train / len(train_file_list))
                model_fold_curves[ff_grid_model2.model_name]['Test'].append(ff_grid2_test / len(train_file_list))
                # model_fold_curves[gru_model.model_name]['Train'].append(gru_train / len(train_file_list))
                # model_fold_curves[gru_model.model_name]['Test'].append(gru_test / len(train_file_list))
                # model_fold_curves[gru_mto_model.model_name]['Train'].append(gru_mto_train / len(train_file_list))
                # model_fold_curves[gru_mto_model.model_name]['Test'].append(gru_mto_test / len(train_file_list))
                # model_fold_curves[conv1d_model.model_name]['Train'].append(conv1d_train / len(train_file_list))
                # model_fold_curves[conv1d_model.model_name]['Test'].append(conv1d_test / len(train_file_list))
                model_fold_curves[trs_model.model_name]['Train'].append(trs_train / len(train_file_list))
                model_fold_curves[trs_model.model_name]['Test'].append(trs_test / len(train_file_list))

        # Calculate performance metrics for fold
        print(f"Fold {fold_num} training complete, saving model metrics...")
        for train_file in train_file_list:
            train_data, test_data, grid = data_utils.load_fold_data(data_folder, train_file, fold_num, kwargs['n_folds'])
            with open(f"{data_folder}train_config.json", "r") as f:
                config = json.load(f)
            test_dataloader_basic = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_collate, NUM_WORKERS)
            test_dataloader_basic_grid = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.basic_grid_collate, NUM_WORKERS, grid=grid, buffer=1)
            test_dataloader_seq = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_collate, NUM_WORKERS)
            # test_dataloader_seq_mto = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.sequential_mto_collate, NUM_WORKERS)
            test_dataloader_trs = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, data_loader.transformer_collate, NUM_WORKERS)

            labels, preds = avg_model.predict(test_dataloader_basic, config)
            model_fold_results[avg_model.model_name]["Labels"].extend(list(labels))
            model_fold_results[avg_model.model_name]["Preds"].extend(list(preds))

            labels, preds = sch_model.predict(test_dataloader_basic, config)
            model_fold_results[sch_model.model_name]["Labels"].extend(list(labels))
            model_fold_results[sch_model.model_name]["Preds"].extend(list(preds))

            labels, preds = tim_model.predict(test_dataloader_seq, config)
            model_fold_results[tim_model.model_name]["Labels"].extend(list(labels))
            model_fold_results[tim_model.model_name]["Preds"].extend(list(preds))

            labels, preds = ff_model.evaluate(test_dataloader_basic, config)
            model_fold_results[ff_model.model_name]["Labels"].extend(list(labels))
            model_fold_results[ff_model.model_name]["Preds"].extend(list(preds))

            labels, preds = ff_grid_model1.evaluate(test_dataloader_basic_grid, config)
            model_fold_results[ff_grid_model1.model_name]["Labels"].extend(list(labels))
            model_fold_results[ff_grid_model1.model_name]["Preds"].extend(list(preds))

            labels, preds = ff_grid_model2.evaluate(test_dataloader_basic_grid, config)
            model_fold_results[ff_grid_model2.model_name]["Labels"].extend(list(labels))
            model_fold_results[ff_grid_model2.model_name]["Preds"].extend(list(preds))

            # gru_labels, gru_preds = gru_model.evaluate(test_dataloader_seq, config)
            # model_fold_results[gru_model.model_name]["Labels"].extend(list(gru_labels))
            # model_fold_results[gru_model.model_name]["Preds"].extend(list(gru_preds))

            # gru_mto_labels, gru_mto_preds = gru_mto_model.evaluate(test_dataloader_seq_mto, config)
            # model_fold_results[gru_mto_model.model_name]["Labels"].extend(list(gru_mto_labels))
            # model_fold_results[gru_mto_model.model_name]["Preds"].extend(list(gru_mto_preds))

            # conv1d_labels, conv1d_preds = conv1d_model.evaluate(test_dataloader_seq, config)
            # model_fold_results[conv1d_model.model_name]["Labels"].extend(list(conv1d_labels))
            # model_fold_results[conv1d_model.model_name]["Preds"].extend(list(conv1d_preds))

            trs_labels, trs_preds = trs_model.evaluate(test_dataloader_trs, config)
            model_fold_results[trs_model.model_name]["Labels"].extend(list(trs_labels))
            model_fold_results[trs_model.model_name]["Preds"].extend(list(trs_preds))

        fold_results = {
            "Model Names": [x.model_name for x in model_list],
            "Fold": fold_num,
            "All Losses": [],
            "Loss Curves": [{model: curve_dict} for model, curve_dict in zip(model_fold_curves.keys(), list(model_fold_curves.values()))]
        }
        # Calculate various losses:
        for mname in fold_results["Model Names"]:
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"])), 2))
            _.append(np.round(metrics.mean_absolute_error(model_fold_results[mname]["Labels"], model_fold_results[mname]["Preds"]), 2))
            fold_results['All Losses'].append(_)
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