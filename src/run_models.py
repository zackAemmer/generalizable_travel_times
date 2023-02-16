#!/usr/bin python3
import itertools
import json
import logging
import numpy as np
from sklearn import metrics
from tabulate import tabulate
import torch
from torch.utils.data import DataLoader

from database import data_utils, data_loader
from models import avg_speed, time_table, basic_ff


def load_train_test_data(data_folder):
    # Read in train data
    train_data = []
    for i in range(0,5):
        contents = open(data_folder + "train_0" + str(i), "r").read()
        train_data.append([json.loads(str(item)) for item in contents.strip().split('\n')])
    train_data = list(itertools.chain.from_iterable(train_data))
    # Read in test data
    contents = open(data_folder + "test", "r").read()
    test_data = [json.loads(str(item)) for item in contents.strip().split('\n')]
    return train_data, test_data

def run_models(run_folder, network_folder, gtfs_folder):
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
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARN_RATE = 1e-3
    HIDDEN_SIZE = 512

    ### Load train/test data
    print("="*30)
    data_folder = run_folder + network_folder + "deeptte_formatted/"
    print(f"Loading data from '{data_folder}'...")
    train_data, test_data = load_train_test_data(data_folder)

    # Load config with feature mean/std
    with open(data_folder + "config.json", "r") as f:
        config = json.load(f)

    # Load GTFS data
    print(f"Loading and merging GTFS files from '{gtfs_folder}'...")
    gtfs_data = data_utils.merge_gtfs_files(gtfs_folder)

    # Construct dataloaders for Pytorch models
    train_dataset = data_loader.make_dataset(train_data, config, device)
    test_dataset = data_loader.make_dataset(test_data, config, device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)
    print(f"Successfully loaded {len(train_data)} training samples and {len(test_data)} testing samples.")

    ### Train average time model
    print("="*30)
    print(f"Training average hourly speed model...")
    avg_model = avg_speed.AvgHourlySpeedModel()
    avg_model.fit(train_data)
    avg_model.save_to(run_folder + network_folder + "models/", "avg_model")
    avg_preds = avg_model.predict(test_data)

    ### Train time table model
    print("="*30)
    print(f"Training time table model...")
    sch_model = time_table.TimeTableModel(gtfs_data)
    sch_model.save_to(run_folder + network_folder + "models/", "sch_model")
    sch_preds = sch_model.predict_simple_sch(test_data)

    ### Train basic ff model
    print("="*30)
    print(f"Training basic ff network model...")
    # Variables must be changed here, and in data_loader.
    # Must change model file if using only continuous or only embedded variables
    embed_dict = {
        'timeID': {
            'vocab_size': 1440,
            'embed_dims': 24,
            'col': 2
        },
        'weekID': {
            'vocab_size': 7,
            'embed_dims': 4,
            'col': 3
        },
        'driverID': {
            'vocab_size': 1147, #646 nwy
            'embed_dims': 12,
            'col':4
        }
    }
    ff_model = basic_ff.BasicFeedForward(
        train_dataloader.dataset[0][0].shape[0],
        embed_dict,
        HIDDEN_SIZE
    ).to(device)
    ff_model.fit_to_data(train_dataloader, test_dataloader, LEARN_RATE, EPOCHS)
    ff_labels, ff_preds = ff_model.predict(test_dataloader, config)

    ### Calculate metrics
    print("="*30)
    models = ["AVG","SCH","FF"]
    preds = [avg_preds, sch_preds, ff_preds]
    labels = np.array([x['time_gap'][-1] for x in test_data])
    model_results = []
    for model_name, preds in zip(models, preds):
        _ = [model_name]
        _.append(np.round(metrics.mean_absolute_percentage_error(labels, preds), 2))
        _.append(np.round(np.sqrt(metrics.mean_squared_error(labels, preds)), 2))
        _.append(np.round(metrics.mean_absolute_error(labels, preds), 2))
        model_results.append(_)
    print(tabulate(model_results, headers=["Model", "MAPE", "RMSE", "MAE"]))

    # Save results
    data_utils.write_pkl(model_results, run_folder + network_folder + "model_results.pkl")

    return None


if __name__=="__main__":
    # run_models(
    #     run_folder="./results/end_to_end/",
    #     network_folder="kcm/",
    #     gtfs_folder="./data/kcm_gtfs/2022_09_19/"
    # )
    run_models(
        run_folder="./results/end_to_end/",
        network_folder="atb/",
        gtfs_folder="./data/nwy_gtfs/2022_12_01/"
    )