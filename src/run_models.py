#!/usr/bin python3
import itertools
import json
import random

import numpy as np
import torch
from sklearn import metrics
from tabulate import tabulate
from torch.utils.data import DataLoader

from models import (avg_speed, avg_speed_seq, basic_ff, basic_rnn, gru_rnn,
                    persistent_speed, time_table)
from utils import data_loader, data_utils, model_utils


def run_models(run_folder, network_folder):
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
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    ### Set run and hyperparameters
    EPOCHS = 3
    BATCH_SIZE = 256
    LEARN_RATE = 1e-3
    HIDDEN_SIZE = 32

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
        print(f"FOLD: {fold_num}")

        # Set aside the train/test data according to the current fold number
        test_data = train_data_chunks[fold_num]
        train_data = [x for i,x in enumerate(train_data_chunks) if i!=fold_num]

        # Combine the training data to single object
        train_data = list(itertools.chain.from_iterable(train_data))

        # Construct dataloaders for Pytorch models
        train_dataloader = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "basic")
        test_dataloader = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "basic")
        train_dataloader_seq = data_loader.make_generic_dataloader(train_data, config, BATCH_SIZE, "sequential")
        test_dataloader_seq = data_loader.make_generic_dataloader(test_data, config, BATCH_SIZE, "sequential")
        train_lens, train_mask = data_utils.get_seq_info(train_dataloader_seq)
        test_lens, test_mask = data_utils.get_seq_info(test_dataloader_seq)
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

        #### TRAVEL TIME TASK ####
        ### Train average time model
        print("="*30)
        print(f"Training average hourly speed model...")
        avg_model = avg_speed.AvgHourlySpeedModel(config)
        avg_model.fit(train_dataloader)
        avg_model.save_to(f"{run_folder}{network_folder}models/avg_model_{fold_num}.pkl")
        avg_labels, avg_preds = avg_model.predict(test_dataloader)

        ### Train time table model
        print("="*30)
        print(f"Training time table model...")
        sch_model = time_table.TimeTableModel(config)
        sch_model.save_to(f"{run_folder}{network_folder}models/sch_model_{fold_num}.pkl")
        sch_labels, sch_preds = sch_model.predict_simple_sch(test_dataloader)

        ### Train ff model
        print("="*30)
        print(f"Training basic ff model...")
        ff_model = basic_ff.BasicFeedForward(
            8,
            embed_dict,
            HIDDEN_SIZE
        ).to(device)
        ff_train_losses, ff_test_losses = model_utils.fit_to_data(ff_model, train_dataloader, test_dataloader, LEARN_RATE, EPOCHS, config, device)
        torch.save(ff_model.state_dict(), run_folder + network_folder + f"models/ff_model_{fold_num}.pt")
        ff_labels, ff_preds, ff_avg_loss = model_utils.predict(ff_model, test_dataloader, device)
        ff_labels = data_utils.de_normalize(ff_labels, config['time_mean'], config['time_std'])
        ff_preds = data_utils.de_normalize(ff_preds, config['time_mean'], config['time_std'])

        #### FORECAST SPEED TASK ####
        # TODO: check masking rnns

        ### Train persistent speed sequence model
        print("="*30)
        print(f"Training persistent speed model...")
        persistent_seq_model = persistent_speed.PersistentSpeedSeqModel(config, min_value=3.0)
        persistent_seq_model.save_to(f"{run_folder}{network_folder}models/persistent_seq_model_{fold_num}.pkl")
        persistent_seq_labels, persistent_seq_preds = persistent_seq_model.predict(test_dataloader_seq)

        ### Train Basic RNN model
        print("="*30)
        print(f"Training basic rnn model...")
        rnn_base_model = basic_rnn.BasicRNN(
            5,
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict
        ).to(device)
        rnn_base_train_losses, rnn_base_test_losses = model_utils.fit_to_data(rnn_base_model, train_dataloader_seq, test_dataloader_seq, LEARN_RATE, EPOCHS, config, device, sequential_flag=True)
        torch.save(rnn_base_model.state_dict(), run_folder + network_folder + f"models/rnn_base_model_{fold_num}.pt")
        rnn_base_labels, rnn_base_preds, rnn_base_avg_loss = model_utils.predict(rnn_base_model, test_dataloader_seq, device, sequential_flag=True)
        rnn_base_labels = data_utils.de_normalize(rnn_base_labels, config['speed_m_s_mean'], config['speed_m_s_std'])
        rnn_base_preds = data_utils.de_normalize(rnn_base_preds, config['speed_m_s_mean'], config['speed_m_s_std'])

        ### Train RNN model
        print("="*30)
        print(f"Training rnn model...")
        rnn_model = gru_rnn.GRU_RNN(
            5,
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict
        ).to(device)
        rnn_train_losses, rnn_test_losses = model_utils.fit_to_data(rnn_model, train_dataloader_seq, test_dataloader_seq, LEARN_RATE, EPOCHS, config, device, sequential_flag=True)
        torch.save(rnn_model.state_dict(), run_folder + network_folder + f"models/rnn_model_{fold_num}.pt")
        rnn_labels, rnn_preds, rnn_avg_loss = model_utils.predict(rnn_model, test_dataloader_seq, device, sequential_flag=True)
        rnn_labels = data_utils.de_normalize(rnn_labels, config['speed_m_s_mean'], config['speed_m_s_std'])
        rnn_preds = data_utils.de_normalize(rnn_preds, config['speed_m_s_mean'], config['speed_m_s_std'])

        #### CALCULATE METRICS ####
        # avg_seq_preds_tt = model_utils.convert_speeds_to_tts(avg_seq_preds, test_dataloader_seq, test_seq_mask, config)
        persistent_seq_preds_tt = data_utils.convert_speeds_to_tts(persistent_seq_preds, test_dataloader_seq, test_mask, config)
        rnn_base_preds_tt = data_utils.convert_speeds_to_tts(rnn_base_preds, test_dataloader_seq, test_mask, config)
        rnn_preds_tt = data_utils.convert_speeds_to_tts(rnn_preds, test_dataloader_seq, test_mask, config)

        print("="*30)
        print(f"Saving model metrics from fold {fold_num}...")
        # Add new models here:
        model_names = ["AVG","SCH","FF","PERSISTENT","RNN_BASE","RNN"]
        # Note that all model labels for same task should be identical, as a check, use only one set of labels here to evaluate
        model_labels = [avg_labels, avg_labels, avg_labels, avg_labels, avg_labels, avg_labels, avg_labels]
        model_preds = [avg_preds, sch_preds, ff_preds, persistent_seq_preds_tt, rnn_base_preds_tt, rnn_preds_tt]
        fold_results = {
            "Fold": fold_num,
            "All Losses": [],
            "FF Train Losses": ff_train_losses,
            "FF Valid Losses": ff_test_losses,
            "RNN_BASE Train Losses": rnn_base_train_losses,
            "RNN_BASE Valid Losses": rnn_base_test_losses,
            "RNN Train Losses": rnn_train_losses,
            "RNN Valid Losses": rnn_test_losses
        }
        # Add new losses here:
        for mname, mlabels, mpreds in zip(model_names, model_labels, model_preds):
            _ = [mname]
            # if mname not in ["AVG_SEQ","PERSISTENT","RNN_BASE","RNN"]:
            if mname not in ["NONE"]:
                # 0.0 speed is common in ytrue so don't use MAPE for these models
                _.append(np.round(metrics.mean_absolute_percentage_error(mlabels, mpreds), 2))
            else:
                _.append(0.0)
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
    torch.set_default_dtype(torch.float64)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/throwaway/",
        network_folder="kcm/"
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/throwaway/",
        network_folder="atb/"
    )