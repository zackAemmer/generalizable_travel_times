#!/usr/bin python3
import itertools
import json
import random

import numpy as np
import torch
from sklearn import metrics
from tabulate import tabulate
from torch.utils.data import DataLoader

from models import avg_speed, ff, persistent_speed, rnn, time_table, conv
from utils import data_loader, data_utils, model_utils


def run_models(run_folder, network_folder, hyperparameters):
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
        NUM_WORKERS = 10
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
        NUM_WORKERS = 0
    print(f"Using device: {device}")
    print(f"Using num_workers: {NUM_WORKERS}")

    ### Set run and hyperparameters
    EPOCHS = hyperparameters['EPOCHS']
    BATCH_SIZE = hyperparameters['BATCH_SIZE']
    LEARN_RATE = hyperparameters['LEARN_RATE']
    HIDDEN_SIZE = hyperparameters['HIDDEN_SIZE']

    ### Load train/test data
    print("="*30)
    data_folder = run_folder + network_folder + "deeptte_formatted/"
    print(f"Loading data from '{data_folder}'...")
    # Load config
    with open(data_folder + "config.json", "r") as f:
        config = json.load(f)
    # Load GTFS-RT samples
    train_data_chunks, valid_data, train_grid, valid_grid = data_utils.load_train_test_data(data_folder, config['n_folds'])
    # Load GTFS data
    print(f"Loading and merging GTFS files from '{config['gtfs_folder']}'...")
    gtfs_data = data_utils.merge_gtfs_files(config['gtfs_folder'])

    ### Run full training process for each model during each validation fold
    run_results = []
    for fold_num in range(0, len(train_data_chunks)):
        print("="*30)
        print(f"FOLD: {fold_num} NETWORK: {network_folder}")

        # Set aside the train/test data according to the current fold number
        test_data = train_data_chunks[fold_num]
        train_data = [x for i,x in enumerate(train_data_chunks) if i!=fold_num]

        # Combine the training data to single object
        train_data = list(itertools.chain.from_iterable(train_data))

        # Construct dataloaders for Pytorch models
        train_dataloader_basic = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "basic", NUM_WORKERS)
        test_dataloader_basic = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "basic", NUM_WORKERS)

        train_dataloader_grid = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "basic_grid", NUM_WORKERS, grid=train_grid, n_prior=1)
        test_dataloader_grid = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "basic_grid", NUM_WORKERS, grid=train_grid, n_prior=1)

        train_dataloader_seq = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "sequential", NUM_WORKERS)
        test_dataloader_seq = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "sequential", NUM_WORKERS)
        _, _ = data_utils.get_seq_info(train_dataloader_seq)
        _, test_mask_seq = data_utils.get_seq_info(test_dataloader_seq)

        train_dataloader_seq_spd = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "sequential_spd", NUM_WORKERS)
        test_dataloader_seq_spd = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "sequential_spd", NUM_WORKERS)
        _, _ = data_utils.get_seq_info(train_dataloader_seq_spd)
        _, test_mask_spd = data_utils.get_seq_info(test_dataloader_seq_spd)

        train_dataloader_seq_mto = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "sequential_mto", NUM_WORKERS)
        test_dataloader_seq_mto = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "sequential_mto", NUM_WORKERS)

        print(f"Successfully loaded {len(train_data)} training samples and {len(test_data)} testing samples.")

        # Define embedded variables for nn models
        embed_dict = {
            'timeID': {
                'vocab_size': 1440,
                'embed_dims': 24
            },
            'weekID': {
                'vocab_size': 7,
                'embed_dims': 4
            },
            'driverID': {
                'vocab_size': config['n_unique_veh'],
                'embed_dims': 6
            },
            'tripID': {
                'vocab_size': config['n_unique_trip'],
                'embed_dims': 20
            }
        }

        # Keep track of all models trained during this run
        model_list = []
        model_labels = []
        model_preds = []
        curve_models = []
        curves = []

        #### TOTAL TRAVEL TIME TASK ####
        print("="*30)
        model = avg_speed.AvgHourlySpeedModel("AVG", config)
        print(f"Training {model.model_name} model...")
        model.fit(train_dataloader_basic)
        model.save_to(f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pkl")
        avg_labels, avg_preds = model.predict(test_dataloader_basic)
        model_list.append(model)
        model_labels.append(avg_labels)
        model_preds.append(avg_preds)

        # print("="*30)
        # model = time_table.TimeTableModel("SCH", config)
        # print(f"Training {model.model_name} model...")
        # model.save_to(f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pkl")
        # labels, preds = model.predict(test_dataloader_basic)
        # model_list.append(model)
        # model_labels.append(avg_labels)
        # model_preds.append(preds)

        # print("="*30)
        # model = ff.FF(
        #     "FF",
        #     8,
        #     HIDDEN_SIZE,
        #     embed_dict,
        #     device
        # ).to(device)
        # print(f"Training {model.model_name} model...")
        # train_losses, test_losses, labels, preds = model.fit_to_data(train_dataloader_basic, test_dataloader_basic, config, LEARN_RATE, EPOCHS)
        # torch.save(model.state_dict(), run_folder + network_folder + f"models/{model.model_name}_{fold_num}.pt")
        # model_list.append(model)
        # model_labels.append(avg_labels)
        # model_preds.append(preds)
        # curve_models.append(model.model_name)
        # curves.append({"Train":train_losses, "Test":test_losses})

        # print("="*30)
        # model = ff.FF_GRID(
        #     "FF_GRID",
        #     8,
        #     HIDDEN_SIZE,
        #     embed_dict,
        #     device
        # ).to(device)
        # print(f"Training {model.model_name} model...")
        # train_losses, test_losses, labels, preds = model.fit_to_data(train_dataloader_grid, test_dataloader_grid, config, LEARN_RATE, EPOCHS)
        # torch.save(model.state_dict(), run_folder + network_folder + f"models/{model.model_name}_{fold_num}.pt")
        # model_list.append(model)
        # model_labels.append(avg_labels)
        # model_preds.append(preds)
        # curve_models.append(model.model_name)
        # curves.append({"Train":train_losses, "Test":test_losses})

        #### FORECAST TASK ####
        # print("="*30)
        # model = persistent_speed.PersistentSpeedSeqModel("PER", config, 2.0)
        # print(f"Training {model.model_name} model...")
        # model.save_to(f"{run_folder}{network_folder}models/{model.model_name}_{fold_num}.pkl")
        # labels, preds = model.predict(test_dataloader_seq_spd)
        # preds = data_utils.convert_speeds_to_tts(preds, test_dataloader_seq_spd, test_mask_spd, config)
        # labels = data_utils.convert_speeds_to_tts(labels, test_dataloader_seq_spd, test_mask_spd, config)
        # model_list.append(model)
        # model_labels.append(avg_labels)
        # model_preds.append(preds)

        print("="*30)
        model = rnn.GRU_RNN(
            "GRU_RNN",
            8,
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict,
            device
        ).to(device)
        print(f"Training {model.model_name} model...")
        train_losses, test_losses, labels, preds = model.fit_to_data(train_dataloader_seq, test_dataloader_seq, test_mask_seq, config, LEARN_RATE, EPOCHS)
        torch.save(model.state_dict(), run_folder + network_folder + f"models/{model.model_name}_{fold_num}.pt")
        model_list.append(model)
        model_labels.append(avg_labels)
        model_preds.append(preds)
        curve_models.append(model.model_name)
        curves.append({"Train":train_losses, "Test":test_losses})

        print("="*30)
        model = rnn.GRU_RNN_MTO(
            "GRU_RNN_MTO",
            8,
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict,
            device
        ).to(device)
        print(f"Training {model.model_name} model...")
        train_losses, test_losses, labels, preds = model.fit_to_data(train_dataloader_seq_mto, test_dataloader_seq_mto, config, LEARN_RATE, EPOCHS)
        torch.save(model.state_dict(), run_folder + network_folder + f"models/{model.model_name}_{fold_num}.pt")
        model_list.append(model)
        model_labels.append(avg_labels)
        model_preds.append(preds)
        curve_models.append(model.model_name)
        curves.append({"Train":train_losses, "Test":test_losses})

        print("="*30)
        model = conv.CONV(
            "CONV1D_MTM",
            8,
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict,
            device
        ).to(device)
        print(f"Training {model.model_name} model...")
        train_losses, test_losses, labels, preds = model.fit_to_data(train_dataloader_seq, test_dataloader_seq, test_mask_seq, config, LEARN_RATE, EPOCHS)
        torch.save(model.state_dict(), run_folder + network_folder + f"models/{model.model_name}_{fold_num}.pt")
        model_list.append(model)
        model_labels.append(avg_labels)
        model_preds.append(preds)
        curve_models.append(model.model_name)
        curves.append({"Train":train_losses, "Test":test_losses})

        #### CALCULATE METRICS ####
        print("="*30)
        print(f"Saving model metrics from fold {fold_num}...")
        model_names = [x.model_name for x in model_list]
        fold_results = {
            "Model Names": model_names,
            "Fold": fold_num,
            "All Losses": [],
            "Loss Curves": [{model: curve_dict} for model, curve_dict in zip(curve_models, curves)]
        }
        # Add new losses here:
        for mname, mlabels, mpreds in zip(model_names, model_labels, model_preds):
            _ = [mname]
            _.append(np.round(metrics.mean_absolute_percentage_error(mlabels, mpreds), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(mlabels, mpreds)), 2))
            _.append(np.round(metrics.mean_absolute_error(mlabels, mpreds), 2))
            fold_results['All Losses'].append(_)
        print(tabulate(fold_results['All Losses'], headers=["Model", "MAPE", "RMSE", "MAE"]))
        run_results.append(fold_results)

    # Save results
    data_utils.write_pkl(run_results, run_folder + network_folder + f"model_results.pkl")
    print(f"MODEL RUN COMPLETED '{run_folder}{network_folder}'")
    return None


if __name__=="__main__":
    torch.set_default_dtype(torch.float)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/small/",
        network_folder="kcm/",
        hyperparameters={
            "EPOCHS": 30,
            "BATCH_SIZE": 512,
            "LEARN_RATE": 1e-3,
            "HIDDEN_SIZE": 32
        }
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/small/",
        network_folder="atb/",
        hyperparameters={
            "EPOCHS": 30,
            "BATCH_SIZE": 512,
            "LEARN_RATE": 1e-3,
            "HIDDEN_SIZE": 32
        }
    )