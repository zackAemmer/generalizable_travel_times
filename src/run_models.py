#!/usr/bin python3
import itertools
import json
import numpy as np
import random
from sklearn import metrics
from tabulate import tabulate
import torch
from torch.utils.data import DataLoader

from database import data_utils, data_loader
from models import avg_speed, time_table, basic_ff


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

    ### Set run and hyperparameters
    device = torch.device("cpu")
    EPOCHS = 50
    BATCH_SIZE = 16
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
    train_data_chunks, valid_data = data_utils.load_train_test_data(data_folder, config['n_folds'])
    # Load GTFS data
    print(f"Loading and merging GTFS files from '{config['gtfs_folder']}'...")
    gtfs_data = data_utils.merge_gtfs_files(config['gtfs_folder'])

    ### Run full training process for each validation fold
    run_results = []
    for fold_num in range(0, len(train_data_chunks)):
        print("="*30)
        print(f"FOLD: {fold_num}")
        test_data = train_data_chunks[fold_num]
        train_data = [x for i,x in enumerate(train_data_chunks) if i!=fold_num]
        train_data = list(itertools.chain.from_iterable(train_data))

        # Construct dataloaders for Pytorch models
        train_dataset = data_loader.make_dataset(train_data, config, device)
        test_dataset = data_loader.make_dataset(test_data, config, device)
        valid_dataset = data_loader.make_dataset(valid_data, config, device)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
        print(f"Successfully loaded {len(train_data)} training samples and {len(test_data)} testing samples.")

        ### Train average time model
        print("="*30)
        print(f"Training average hourly speed model...")
        avg_model = avg_speed.AvgHourlySpeedModel()
        avg_model.fit(train_data)
        avg_model.save_to(run_folder + network_folder + f"models/avg_model_{fold_num}.pkl")
        avg_preds = avg_model.predict(valid_data)

        ### Train time table model
        print("="*30)
        print(f"Training time table model...")
        sch_model = time_table.TimeTableModel(config['gtfs_folder'])
        sch_model.save_to(run_folder + network_folder + f"models/sch_model_{fold_num}.pkl")
        sch_preds = sch_model.predict_simple_sch(valid_data, gtfs_data)

        ### Train basic ff model
        print("="*30)
        print(f"Training basic ff network model...")
        # Variables must be changed here, and in data_loader.
        # Must change model file if using only continuous or only embedded variables
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
        ff_model = basic_ff.BasicFeedForward(
            train_dataloader.dataset[0][0].shape[0],
            embed_dict,
            HIDDEN_SIZE
        ).to(device)
        ff_train_losses, ff_test_losses = ff_model.fit_to_data(train_dataloader, test_dataloader, LEARN_RATE, EPOCHS)
        torch.save(ff_model.state_dict(), run_folder + network_folder + f"models/ff_model_{fold_num}.pt")
        ff_model.eval()
        ff_labels, ff_preds = ff_model.predict(valid_dataloader, config)

        ### Calculate metrics
        print("="*30)
        models = ["AVG","SCH","FF"]
        preds = [avg_preds, sch_preds, ff_preds]
        labels = np.array([x['time_gap'][-1] for x in valid_data])
        fold_results = {
            "Fold": fold_num,
            "All Losses": [],
            "FF Train Losses": ff_train_losses,
            "FF Valid Losses": ff_test_losses
        }
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
        run_folder="./results/3_month_test/",
        network_folder="kcm/"
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    run_models(
        run_folder="./results/3_month_test/",
        network_folder="atb/"
    )