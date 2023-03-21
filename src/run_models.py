#!/usr/bin python3
import itertools
import json
import random

import numpy as np
import torch
from sklearn import metrics
from tabulate import tabulate
from torch.utils.data import DataLoader

from models import (avg_speed, basic_ff, gru_rnn, persistent_speed, time_table)
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
    train_data_chunks, valid_data = data_utils.load_train_test_data(data_folder, config['n_folds']) # Validation data no longer used
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
        train_dataloader = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "basic", NUM_WORKERS)
        test_dataloader = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "basic", NUM_WORKERS)

        train_dataloader_seq_spd = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "sequential_spd", NUM_WORKERS)
        test_dataloader_seq_spd = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "sequential_spd", NUM_WORKERS)
        _, _ = data_utils.get_seq_info(train_dataloader_seq_spd)
        _, test_mask_spd = data_utils.get_seq_info(test_dataloader_seq_spd)

        train_dataloader_seq_tt = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "sequential_tt", NUM_WORKERS)
        test_dataloader_seq_tt = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "sequential_tt", NUM_WORKERS)
        _, _ = data_utils.get_seq_info(train_dataloader_seq_tt)
        _, test_mask_tt = data_utils.get_seq_info(test_dataloader_seq_tt)
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
        avg_model = avg_speed.AvgHourlySpeedModel("AVG", config)
        print(f"Training {avg_model.model_name} model...")
        avg_model.fit(train_dataloader)
        avg_model.save_to(f"{run_folder}{network_folder}models/{avg_model.model_name}_{fold_num}.pkl")
        avg_labels, avg_preds = avg_model.predict(test_dataloader)
        model_list.append(avg_model)
        model_labels.append(avg_labels)
        model_preds.append(avg_preds)

        print("="*30)
        sch_model = time_table.TimeTableModel("SCH", config)
        print(f"Training {sch_model.model_name} model...")
        sch_model.save_to(f"{run_folder}{network_folder}models/{sch_model.model_name}_{fold_num}.pkl")
        labels, preds = sch_model.predict(test_dataloader)
        model_list.append(sch_model)
        model_labels.append(avg_labels)
        model_preds.append(preds)

        print("="*30)
        ff_model = basic_ff.BasicFeedForward(
            "FF",
            8,
            embed_dict,
            HIDDEN_SIZE
        ).to(device)
        print(f"Training {ff_model.model_name} model...")
        train_losses, test_losses = model_utils.fit_to_data(ff_model, train_dataloader, test_dataloader, LEARN_RATE, EPOCHS, config, device)
        torch.save(ff_model.state_dict(), run_folder + network_folder + f"models/{ff_model.model_name}_{fold_num}.pt")
        labels, preds, avg_loss = model_utils.predict(ff_model, test_dataloader, device)
        labels = data_utils.de_normalize(labels, config['time_mean'], config['time_std'])
        preds = data_utils.de_normalize(preds, config['time_mean'], config['time_std'])
        model_list.append(ff_model)
        model_labels.append(avg_labels)
        model_preds.append(preds)
        curve_models.append(ff_model.model_name)
        curves.append({"Train":train_losses, "Test":test_losses})

        ### FORECAST TASK ####
        print("="*30)
        per_model = persistent_speed.PersistentSpeedSeqModel("PER", config, 2.0)
        print(f"Training {per_model.model_name} model...")
        per_model.save_to(f"{run_folder}{network_folder}models/{per_model.model_name}_{fold_num}.pkl")
        labels, preds = per_model.predict(test_dataloader_seq_spd)
        preds = data_utils.convert_speeds_to_tts(preds, test_dataloader_seq_spd, test_mask_spd, config)
        labels = data_utils.convert_speeds_to_tts(labels, test_dataloader_seq_spd, test_mask_spd, config)
        model_list.append(per_model)
        model_labels.append(avg_labels)
        model_preds.append(preds)

        print("="*30)
        rnn_spd = gru_rnn.GRU_RNN_SPD(
            "RNN_SPD",
            8,
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict
        ).to(device)
        print(f"Training {rnn_spd.model_name} model...")
        train_losses, test_losses = model_utils.fit_to_data(rnn_spd, train_dataloader_seq_spd, test_dataloader_seq_spd, LEARN_RATE, EPOCHS, config, device, sequential_flag=True)
        torch.save(rnn_spd.state_dict(), run_folder + network_folder + f"models/{rnn_spd.model_name}_{fold_num}.pt")
        labels, preds, avg_loss = model_utils.predict(rnn_spd, test_dataloader_seq_spd, device, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['speed_m_s_mean'], config['speed_m_s_std'])
        preds = data_utils.de_normalize(preds, config['speed_m_s_mean'], config['speed_m_s_std'])
        preds = data_utils.convert_speeds_to_tts(preds, test_dataloader_seq_spd, test_mask_tt, config)
        labels = data_utils.convert_speeds_to_tts(labels, test_dataloader_seq_spd, test_mask_tt, config)
        model_list.append(rnn_spd)
        model_labels.append(avg_labels)
        model_preds.append(preds)
        curve_models.append(rnn_spd.model_name)
        curves.append({"Train":train_losses, "Test":test_losses})

        print("="*30)
        rnn_tt_nopack = gru_rnn.GRU_RNN_NOPACK(
            "GRU_RNN_NOPACK",
            8,
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict
        ).to(device)
        train_losses, test_losses = model_utils.fit_to_data(rnn_tt_nopack, train_dataloader_seq_tt, test_dataloader_seq_tt, LEARN_RATE, EPOCHS, config, device, sequential_flag=True)
        torch.save(rnn_tt_nopack.state_dict(), run_folder + network_folder + f"models/{rnn_tt_nopack.model_name}_{fold_num}.pt")
        labels, preds, avg_loss = model_utils.predict(rnn_tt_nopack, test_dataloader_seq_tt, device, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.aggregate_tts(preds, test_mask_tt)
        labels = data_utils.aggregate_tts(labels, test_mask_tt)
        model_list.append(rnn_tt_nopack)
        model_labels.append(avg_labels)
        model_preds.append(preds)
        curve_models.append(rnn_tt_nopack.model_name)
        curves.append({"Train":train_losses, "Test":test_losses})

        print("="*30)
        rnn_best = gru_rnn.GRU_RNN(
            "GRU_RNN",
            8,
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict
        ).to(device)
        train_losses, test_losses = model_utils.fit_to_data(rnn_best, train_dataloader_seq_tt, test_dataloader_seq_tt, LEARN_RATE, EPOCHS, config, device, sequential_flag=True)
        torch.save(rnn_best.state_dict(), run_folder + network_folder + f"models/{rnn_best.model_name}_{fold_num}.pt")
        labels, preds, avg_loss = model_utils.predict(rnn_best, test_dataloader_seq_tt, device, sequential_flag=True)
        labels = data_utils.de_normalize(labels, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.de_normalize(preds, config['time_calc_s_mean'], config['time_calc_s_std'])
        preds = data_utils.aggregate_tts(preds, test_mask_tt)
        labels = data_utils.aggregate_tts(labels, test_mask_tt)
        model_list.append(rnn_best)
        model_labels.append(avg_labels)
        model_preds.append(preds)
        curve_models.append(rnn_best.model_name)
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
        run_folder="./results/debug/",
        network_folder="kcm/",
        hyperparameters={
            "EPOCHS": 20,
            "BATCH_SIZE": 512,
            "LEARN_RATE": 1e-3,
            "HIDDEN_SIZE": 32
        }
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/debug/",
        network_folder="atb/",
        hyperparameters={
            "EPOCHS": 20,
            "BATCH_SIZE": 512,
            "LEARN_RATE": 1e-3,
            "HIDDEN_SIZE": 32
        }
    )