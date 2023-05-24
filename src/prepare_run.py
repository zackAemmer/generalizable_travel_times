#!/usr/bin python3


import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from utils import data_utils


def process_data_parallel(data, **kwargs):
    # Clean and transform raw bus data records
    traces = data_utils.shingle(data, 2, 5)
    traces = data_utils.calculate_trace_df(traces, kwargs['timezone'], kwargs['epsg'], kwargs['grid_x_bounds'], kwargs['grid_y_bounds'], kwargs['data_dropout'])
    traces = data_utils.clean_trace_df_w_timetables(traces, kwargs['gtfs_folder'], kwargs['epsg'])
    traces = data_utils.calculate_cumulative_values(traces)
    return traces

def clean_data(dates, n_save_files, train_or_test, base_folder, **kwargs):
    # Handle cleaning of a set of dates across n files (allocated to training or validation)
    print(f"Processing {train_or_test} data from {dates}, saving across {n_save_files} files...")
    date_splits = np.array_split(dates, n_save_files)
    date_splits = [list(x) for x in date_splits]
    # For each file, parallel process the dates, join, and save them to a combined file
    file_num_mod = 0
    for file_num, date_list in enumerate(date_splits):
        print(f"Processing file number {file_num}/{len(date_splits)-1} containing dates {date_list}...")
        file_num = file_num - file_num_mod
        # Load all data for a set of dates, and break into sub-chunks to parallel process
        traces, fail_dates = data_utils.combine_pkl_data(kwargs['raw_data_folder'], date_list, kwargs['given_names'])
        if len(fail_dates) > 0:
            print(f"Failed to load data for dates: {fail_dates}")
        num_raw_points = len(traces)
        print(f"Found {num_raw_points} points, beginning parallel data processing for this chunk, using {kwargs['n_trace_splits']} chunks...")
        traces = np.array_split(traces, kwargs['n_trace_splits'])
        traces = Parallel(n_jobs=kwargs['n_jobs'])(delayed(process_data_parallel)(x, **kwargs) for x in traces)
        # Re-establish unique shingle IDs across parallel chunks
        max_shingle_id = 0
        unique_shingles = [pd.unique(x['shingle_id']) for x in traces]
        # For each shingle ID set, map the IDs to increasing numbers, then increase the max ID
        for chunk_num, shingle_set in enumerate(unique_shingles):
            if len(shingle_set)==0:
                # The process either received no data or filtered out all shingles, returning nothing
                continue
            shingle_remap = dict(zip(shingle_set, np.arange(max_shingle_id, max_shingle_id+len(shingle_set))))
            max_shingle_id = max(list(shingle_remap.values()))+1
            traces[chunk_num]['shingle_id'] = traces[chunk_num]['shingle_id'].replace(shingle_remap)
        # Join results for this set of dates, write to file
        traces = pd.concat(traces)
        # Sometimes, the chunk of dates has no data remaining. For example when AtB switches trip_id's
        # In this case, skip writing the chunk, and subtract 1 from the file number to keep them sequential
        if len(traces)==0:
            print("No traces retained...")
            file_num_mod += 1
            continue
        print(f"Saving {len(traces)} samples to run folder, retained {np.round(len(traces)/num_raw_points, 2)*100}% of original data points...")
        deeptte_formatted_path = f"{base_folder}deeptte_formatted/{train_or_test}{file_num}"
        traces, train_n_grid, train_fill_grid = data_utils.map_to_deeptte(traces, deeptte_formatted_path, grid_x_bounds=kwargs['grid_x_bounds'], grid_y_bounds=kwargs['grid_y_bounds'], grid_s_res=kwargs['grid_s_res'], grid_t_res=kwargs['grid_t_res'], grid_n_res=kwargs['grid_n_res'])
        summary_config = data_utils.get_summary_config(traces, kwargs['gtfs_folder'], n_save_files, kwargs['epsg'])
        # Save config, and traces to file for notebook analyses
        with open(f"{deeptte_formatted_path}_config.json", mode="a") as out_file:
            json.dump(summary_config, out_file)
        data_utils.write_pkl(traces, f"{base_folder}{train_or_test}{file_num}_traces.pkl")
        data_utils.write_pkl(train_n_grid, f"{base_folder}{train_or_test}{file_num}_grid.pkl")
        data_utils.write_pkl(train_fill_grid, f"{base_folder}{train_or_test}{file_num}_grid_ffill.pkl")
    # Combine normalization metrics and other values across all files
    data_utils.combine_config_files(f"{base_folder}deeptte_formatted/", n_save_files, train_or_test)

def prepare_run(overwrite, run_name, network_name, train_dates, test_dates, **kwargs):
    """
    Set up the folder and data structure for a set of k-fold validated model runs.
    All run data is copied from the original download directory to the run folder.
    Separate folders are made for the ATB and KCM networks.
    The formatted data is saved in "deeptte_formatted". Since we are benchmarking
    with that model, it is convenient to use the same data format for all models.
    """
    print("="*30)
    print(f"PREPARE RUN: '{run_name}'")
    print(f"NETWORK: '{network_name}'")
    # Create folder structure
    base_folder = f"./results/{run_name}/{network_name}/"
    if run_name not in os.listdir("./results/"):
        os.mkdir(f"./results/{run_name}")
    if network_name in os.listdir(f"results/{run_name}/") and overwrite:
        shutil.rmtree(base_folder)
    if network_name not in os.listdir(f"results/{run_name}/"):
        os.mkdir(base_folder)
        os.mkdir(f"{base_folder}deeptte_formatted/")
        os.mkdir(f"{base_folder}deeptte_results/")
        os.mkdir(f"{base_folder}models/")
        print(f"Created new results folder for '{run_name}'")
    else:
        print(f"Run '{run_name}/{network_name}' folder already exists in 'results/', delete the folder if new run desired.")
        return None
    # Split train/test dates into arbitrary number of chunks. More chunks = more training files = less ram pressure (chunk files are loaded 1 at a time)
    print(f"Processing training dates...")
    clean_data(train_dates, kwargs['num_train_files'], "train", base_folder, **kwargs)
    print(f"Processing testing dates...")
    clean_data(test_dates, kwargs['num_test_files'], "test", base_folder, **kwargs)
    print(f"RUN PREPARATION COMPLETED '{run_name}/{network_name}'")

if __name__=="__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="debug",
        network_name="kcm",
        train_dates=data_utils.get_date_list("2023_03_17", 3),
        test_dates=data_utils.get_date_list("2023_03_20", 3),
        num_train_files=2,
        num_test_files=2,
        n_jobs=5,
        n_trace_splits=5,
        data_dropout=0.2,
        gtfs_folder="./data/kcm_gtfs/",
        raw_data_folder="./data/kcm_all/",
        timezone="America/Los_Angeles",
        epsg="32148",
        grid_x_bounds=[368279,416973], # 48694 x 76563 = 3000sqkm
        grid_y_bounds=[15350, 91913],
        grid_s_res=500,
        grid_t_res=120,
        grid_n_res=3,
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id']
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="debug",
        network_name="atb",
        train_dates=data_utils.get_date_list("2023_03_17", 3),
        test_dates=data_utils.get_date_list("2023_03_20", 3),
        num_train_files=2,
        num_test_files=2,
        n_jobs=5,
        n_trace_splits=5,
        data_dropout=0.2,
        gtfs_folder="./data/atb_gtfs/",
        raw_data_folder="./data/atb_all_new/",
        timezone="Europe/Oslo",
        epsg="32632",
        grid_x_bounds=[533551, 677502], # 143951 x 164933 = 23000sqkm
        grid_y_bounds=[7011518, 7176451],
        grid_s_res=500,
        grid_t_res=120,
        grid_n_res=3,
        given_names=['trip_id','file','locationtime','lat','lon','vehicle_id']
    )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # prepare_run(
    #     overwrite=True,
    #     run_name="small",
    #     network_name="kcm",
    #     train_dates=data_utils.get_date_list("2023_02_15", 30),
    #     test_dates=data_utils.get_date_list("2023_04_20", 7),
    #     num_train_files=5,
    #     num_test_files=5,
    #     n_jobs=8,
    #     n_trace_splits=8,
    #     data_dropout=0.2,
    #     gtfs_folder="./data/kcm_gtfs/",
    #     raw_data_folder="./data/kcm_all/",
    #     timezone="America/Los_Angeles",
    #     epsg="32148",
    #     given_names=['trip_id','file','locationtime','lat','lon','vehicle_id']
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # prepare_run(
    #     overwrite=True,
    #     run_name="small",
    #     network_name="atb",
    #     train_dates=data_utils.get_date_list("2023_02_15", 30),
    #     test_dates=data_utils.get_date_list("2023_04_20", 7),
    #     num_train_files=5,
    #     num_test_files=5,
    #     n_jobs=8,
    #     n_trace_splits=8,
    #     data_dropout=0.2,
    #     gtfs_folder="./data/atb_gtfs/",
    #     raw_data_folder="./data/atb_all_new/",
    #     timezone="Europe/Oslo",
    #     epsg="32632",
    #     given_names=['trip_id','file','locationtime','lat','lon','vehicle_id']
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # prepare_run(
    #     overwrite=True,
    #     run_name="medium",
    #     network_name="kcm",
    #     train_dates=data_utils.get_date_list("2023_02_15", 60),
    #     test_dates=data_utils.get_date_list("2023_04_20", 7),
    #     num_train_files=5,
    #     num_test_files=5,
    #     n_jobs=8,
    #     n_trace_splits=8,
    #     data_dropout=0.2,
    #     gtfs_folder="./data/kcm_gtfs/",
    #     raw_data_folder="./data/kcm_all/",
    #     timezone="America/Los_Angeles",
    #     epsg="32148",
    #     given_names=['trip_id','file','locationtime','lat','lon','vehicle_id']
    # )
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # prepare_run(
    #     overwrite=True,
    #     run_name="medium",
    #     network_name="atb",
    #     train_dates=data_utils.get_date_list("2023_02_15", 60),
    #     test_dates=data_utils.get_date_list("2023_04_20", 7),
    #     num_train_files=5,
    #     num_test_files=5,
    #     n_jobs=8,
    #     n_trace_splits=8,
    #     data_dropout=0.2,
    #     gtfs_folder="./data/atb_gtfs/",
    #     raw_data_folder="./data/atb_all_new/",
    #     timezone="Europe/Oslo",
    #     epsg="32632",
    #     given_names=['trip_id','file','locationtime','lat','lon','vehicle_id']
    # )