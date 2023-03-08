#!/usr/bin python3
import itertools
import json
import numpy as np
import random
from sklearn import metrics
from tabulate import tabulate
import torch
from torch.utils.data import DataLoader

from database import data_utils, data_loader, model_utils
from models import avg_speed, time_table, basic_ff, basic_rnn


def run_models(run_folder, network_folder):
    """
    Train each of the specified models on bus data found in the data folder.
    The data folder is generated using 03_map_features_to_deeptte.
    The data in the folder will have many attributes, not all necessariliy used.
    Use k-fold cross validation to split the data n times into train/test.
    The test set is used as validation during the training process.
    Model accuracy is measured across the n folds.
    Save the resulting models, and the data generated during training to the same data folder.
    These are then analyzed in the notebook 04_explore_results.
    """
    print("="*30)
    print(f"RUN MODEL: '{run_folder}'")
    print(f"NETWORK: '{network_folder}'")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ### Set run and hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 512
    LEARN_RATE = 1e-3
    HIDDEN_SIZE = 32
    SEQ_LEN = 2

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
        train_dataset = data_loader.make_dataset(train_data, config)
        test_dataset = data_loader.make_dataset(test_data, config)
        train_dataset_seq = data_loader.make_seq_dataset(train_data, SEQ_LEN)
        test_dataset_seq = data_loader.make_seq_dataset(test_data, SEQ_LEN)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
        train_dataloader_seq = DataLoader(train_dataset_seq, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
        test_dataloader_seq = DataLoader(test_dataset_seq, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
        print(f"Successfully loaded {len(train_data)} training samples and {len(test_data)} testing samples.")

        # Define embedded variables for nn models
        embed_dict = {
            'timeID': {
                'vocab_size': 1440,
                'embed_dims': 24,
                'col': 8
            },
            'weekID': {
                'vocab_size': 7,
                'embed_dims': 4,
                'col': 9
            },
            'driverID': {
                'vocab_size': config['n_unique_veh'],
                'embed_dims': 12,
                'col': 10
            }
        }

        ### Train average time model
        print("="*30)
        print(f"Training average hourly speed model...")
        avg_model = avg_speed.AvgHourlySpeedModel()
        avg_model.fit(train_data)
        avg_model.save_to(f"{run_folder}{network_folder}models/avg_model_{fold_num}.pkl")
        avg_preds = avg_model.predict(test_data)

        ### Train time table model
        print("="*30)
        print(f"Training time table model...")
        sch_model = time_table.TimeTableModel(config['gtfs_folder'])
        sch_model.save_to(f"{run_folder}{network_folder}models/sch_model_{fold_num}.pkl")
        sch_preds = sch_model.predict_simple_sch(test_data, gtfs_data)

        ### Train ff model
        print("="*30)
        print(f"Training basic ff model...")
        ff_model = basic_ff.BasicFeedForward(
            # X tensor, first element, n_features
            train_dataloader.dataset.tensors[0].shape[1],
            embed_dict,
            HIDDEN_SIZE
        ).to(device)
        ff_train_losses, ff_test_losses = model_utils.fit_to_data(ff_model, train_dataloader, test_dataloader, LEARN_RATE, EPOCHS, config, device)
        torch.save(ff_model.state_dict(), run_folder + network_folder + f"models/ff_model_{fold_num}.pt")
        ff_labels, ff_preds, ff_avg_loss = model_utils.predict(ff_model, test_dataloader, config, device)
        ff_labels = data_utils.de_normalize(ff_labels, config['time_mean'], config['time_std'])
        ff_preds = data_utils.de_normalize(ff_preds, config['time_mean'], config['time_std'])

        ### Train RNN model
        print("="*30)
        print(f"Training basic rnn model...")
        rnn_model = basic_rnn.BasicRNN(
            # X tensor, first element, traj component, n_features
            train_dataloader_seq.dataset.tensors[0][0][0].shape[1],
            1,
            HIDDEN_SIZE,
            BATCH_SIZE,
            embed_dict
        ).to(device)
        rnn_train_losses, rnn_test_losses = model_utils.fit_to_data(rnn_model, train_dataloader_seq, test_dataloader_seq, LEARN_RATE, EPOCHS, config, device, sequential_flag=True)
        torch.save(rnn_model.state_dict(), run_folder + network_folder + f"models/rnn_model_{fold_num}.pt")
        rnn_labels, rnn_preds, rnn_avg_loss = model_utils.predict(rnn_model, test_dataloader_seq, config, device, sequential_flag=True)
        rnn_labels = data_utils.de_normalize(rnn_labels, config['speed_m_s_mean'], config['speed_m_s_std'])
        rnn_preds = data_utils.de_normalize(rnn_preds, config['speed_m_s_mean'], config['speed_m_s_std'])

        ### Calculate metrics
        print("="*30)
        # Add new models here:
        models = ["AVG","SCH","FF","RNN"]
        preds = [avg_preds, sch_preds, ff_preds, rnn_preds]
        labels = np.array([x['time_gap'][-1] for x in test_data])
        fold_results = {
            "Fold": fold_num,
            "All Losses": [],
            "FF Train Losses": ff_train_losses,
            "FF Valid Losses": ff_test_losses,
            "RNN Train Losses": rnn_train_losses,
            "RNN Valid Losses": rnn_test_losses
        }
        # Add new losses here:
        for model_name, preds in zip(models, preds):
            _ = [model_name]
            _.append(np.round(metrics.mean_absolute_percentage_error(labels, preds), 2))
            _.append(np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2))
            _.append(np.round(metrics.mean_absolute_error(labels, preds), 2))
            fold_results['All Losses'].append(_)
        print(tabulate(fold_results['All Losses'], headers=["Model", "MAPE", "RMSE", "MAE"]))
        run_results.append(fold_results)

    # Save results
    data_utils.write_pkl(run_results, run_folder + network_folder + f"model_results.pkl")
    print(f"MODEL RUN COMPLETED '{run_folder}{network_folder}'")
    return None


if __name__=="__main__":
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