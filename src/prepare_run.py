#!/usr/bin python3
import itertools
import json
import logging
import os
import pickle
import random
import shutil
import torch

import numpy as np
import pandas as pd
from tabulate import tabulate

from database import data_utils, data_loader


def prepare_run(overwrite, run_name, network_name, gtfs_folder, raw_data_folder, timezone, given_names, train_dates, test_dates, n_folds):
    """
    Set up the folder and data structure for a set of k-fold validated model runs.
    All run data is copied from the original download directory to the run folder.
    Separate folders are made for the ATB and KCM networks.
    The formatted data is saved in "deeptte_formatted". Since we are benchmarking
    with that model, it is convenient to use their same format for all models.
    Create a separate folder for baseline and new model results.
    """
    print("="*30)
    print(f"PREPARE RUN: '{run_name}'")
    print(f"NETWORK: '{network_name}'")

    ### Create folder structure
    print("="*30)
    base_folder = "./results/" + run_name + "/" + network_name + "/"    
    if run_name not in os.listdir("./results/"):
        os.mkdir("./results/" + run_name)
    if network_name in os.listdir("results/" + run_name + "/") and overwrite:
        shutil.rmtree(base_folder)
    if network_name not in os.listdir("results/" + run_name + "/"):
        os.mkdir(base_folder)
        os.mkdir(base_folder + "deeptte_formatted/")
        os.mkdir(base_folder + "deeptte_results/")
        os.mkdir(base_folder + "models/")
        print(f"Created new results folder for '{run_name}'")
    else:
        print(f"Run '{run_name}/{network_name}' folder already exists in 'results/', delete the folder if new run desired.")
        return None

    ### Load data from raw bus data files
    print("="*30)
    print(f"Combining raw bus data files...")
    # Get traces from all data
    train_data, train_fail_dates = data_utils.combine_pkl_data(raw_data_folder, train_dates, given_names)
    test_data, test_fail_dates = data_utils.combine_pkl_data(raw_data_folder, test_dates, given_names)
    print(f"Lost dates train: {train_fail_dates}, {len(train_data)} samples kept.")
    print(f"Lost dates test: {test_fail_dates}, {len(test_data)} samples kept.")

    # Load the GTFS
    print(f"Loading and merging GTFS files from '{gtfs_folder}'...")
    gtfs_data = data_utils.merge_gtfs_files(gtfs_folder)
    
    # Shingle each trajectory into a subset of smaller chunks
    print(f"Shingling traces into smaller chunks...")
    train_traces = data_utils.shingle(train_data, 2, 5)
    test_traces = data_utils.shingle(test_data, 2, 5)
    print(f"Cumulative {np.round(len(train_traces) / len(train_data) * 100, 1)}% of train data retained. Saving {len(train_traces)} samples.")
    print(f"Cumulative {np.round(len(test_traces) / len(test_data) * 100, 1)}% of test data retained. Saving {len(test_traces)} samples.")

    # Calculate distance between points in each trajectory, do some filtering on speed, n_points
    print(f"Calculating trace trajectories, filtering on speed and number of trajectory points...")
    train_traces = data_utils.calculate_trace_df(train_traces, timezone)
    test_traces = data_utils.calculate_trace_df(test_traces, timezone)
    print(f"Cumulative {np.round(len(train_traces) / len(train_traces) * 100, 1)}% of train data retained.")
    print(f"Cumulative {np.round(len(test_traces) / len(test_traces) * 100, 1)}% of test data retained.")

    # Match trajectories to timetables and do filtering on stop distance, availability
    print(f"Matching traces to GTFS timetables, filtering on schedule availability...")
    train_traces = data_utils.parallelize_clean_trace_df_w_timetables(train_traces, gtfs_data)
    test_traces = data_utils.parallelize_clean_trace_df_w_timetables(test_traces, gtfs_data)
    # train_traces = data_utils.clean_trace_df_w_timetables(train_traces, gtfs_data)
    # test_traces = data_utils.clean_trace_df_w_timetables(test_traces, gtfs_data)
    print(f"Cumulative {np.round(len(train_traces) / len(train_data) * 100, 1)}% of train data retained. Saving {len(train_traces)} samples.")
    print(f"Cumulative {np.round(len(test_traces) / len(test_data) * 100, 1)}% of test data retained. Saving {len(test_traces)} samples.")

    # Get unique vehicle ids
    (train_traces, test_traces), n_unique_veh = data_utils.remap_vehicle_ids([train_traces, test_traces])
    print(f"Found {n_unique_veh} unique vehicle IDs in this data.")

    ### Save processed data from raw bus data files
    print("="*30)
    print(f"Mapping trace data to DeepTTE format...")
    deeptte_formatted_path = base_folder + "deeptte_formatted/"
    train_traces = data_utils.map_to_deeptte(train_traces, deeptte_formatted_path, n_folds)
    test_traces = data_utils.map_to_deeptte(test_traces, deeptte_formatted_path, n_folds, is_test=True)
    summary_config = data_utils.get_summary_config(train_traces, n_unique_veh, gtfs_folder, n_folds)

    # Write summary dict to config file
    with open(deeptte_formatted_path+"config.json", mode="a") as out_file:
        json.dump(summary_config, out_file)

    print(f"Saving processed bus data files...")
    # Save trace dataframes
    data_utils.write_pkl(train_traces, base_folder+"train_traces.pkl")
    data_utils.write_pkl(test_traces, base_folder+"test_traces.pkl")

    print(f"RUN PREPARATION COMPLETED '{run_name}/{network_name}'")


if __name__=="__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="3_month_test",
        network_name="kcm",
        gtfs_folder="./data/kcm_gtfs/2020_09_23/",
        raw_data_folder="./data/kcm_all/",
        timezone="America/Los_Angeles",
        given_names=['tripid','file','locationtime','lat','lon','vehicleid'],
        train_dates=data_utils.get_date_list("2020_10_24", 90),
        test_dates=data_utils.get_date_list("2021_03_01", 7),
        n_folds=5
    )
    # For now, we can use Norway dates that are post-2022_11_02
    # Need to get mapping of old IDs to new IDs in order to use schedule data from prior to that date
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="3_month_test",
        network_name="atb",
        gtfs_folder="./data/nwy_gtfs/2022_12_01/",
        raw_data_folder="./data/atb_all/",
        timezone="Europe/Oslo",
        given_names=['datedvehiclejourney','file','locationtime','lat','lon','vehicle'],
        train_dates=data_utils.get_date_list("2022_11_01", 28),
        test_dates=data_utils.get_date_list("2022_11_29", 12),
        n_folds=5
    )