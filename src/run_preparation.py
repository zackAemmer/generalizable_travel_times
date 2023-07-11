#!/usr/bin python3


import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from joblib import Parallel, delayed

from utils import data_utils


def process_data_parallel(date_list, i, **kwargs):
    # Load data from dates
    traces, fail_dates = data_utils.combine_pkl_data(kwargs['raw_data_folder'][i], date_list, kwargs['given_names'][i])
    if len(fail_dates) > 0:
        print(f"Failed to load data for dates: {fail_dates}")
    num_raw_points = len(traces)
    print(f"Chunk {date_list} found {num_raw_points} points...")
    # Clean and transform raw bus data records
    traces = data_utils.shingle(traces, 2, 5)
    traces = data_utils.calculate_trace_df(traces, kwargs['timezone'][i], kwargs['epsg'][i], kwargs['grid_bounds'][i], kwargs['coord_ref_center'][i], kwargs['data_dropout'])
    if not kwargs['skip_gtfs']:
        traces = data_utils.clean_trace_df_w_timetables(traces, kwargs['gtfs_folder'][i], kwargs['epsg'][i], kwargs['coord_ref_center'][i])
    traces = data_utils.calculate_cumulative_values(traces, kwargs['skip_gtfs'])
    # Reduce to absolute minimum variables
    if not kwargs['skip_gtfs']:
        traces = traces[[
            "shingle_id",
            "file",
            "trip_id",
            "weekID",
            "timeID",
            "timeID_s",
            "locationtime",
            "lon",
            "lat",
            "x",
            "y",
            "x_cent",
            "y_cent",
            "dist_calc_km",
            "time_calc_s",
            "dist_cumulative_km",
            "time_cumulative_s",
            "speed_m_s",
            "bearing",
            "route_id",
            "stop_x_cent",
            "stop_y_cent",
            "scheduled_time_s",
            "stop_dist_km",
            "passed_stops_n"
        ]].convert_dtypes()
    else:
        traces = traces[[
            "shingle_id",
            "file",
            "trip_id",
            "weekID",
            "timeID",
            "timeID_s",
            "locationtime",
            "lon",
            "lat",
            "x",
            "y",
            "x_cent",
            "y_cent",
            "dist_calc_km",
            "time_calc_s",
            "dist_cumulative_km",
            "time_cumulative_s",
            "speed_m_s",
            "bearing",
            # "route_id",
            # "stop_x_cent",
            # "stop_y_cent",
            # "scheduled_time_s",
            # "stop_dist_km",
            # "passed_stops_n"
        ]].convert_dtypes()
    print(f"Retained {np.round(len(traces)/num_raw_points, 2)*100}% of original data points...")
    return traces

def clean_data(dates, **kwargs):
    # Clean a set of dates (allocated to training or testing)
    print(f"Processing {kwargs['train_or_test']} data from {dates} across {kwargs['n_jobs']} jobs...")
    date_splits = np.array_split(dates, min(kwargs['n_jobs'],len(dates)-1))
    date_splits = [list(x) for x in date_splits]
    # Handle mixed network datasets
    j=0
    traces = []
    for i in range(len(kwargs['raw_data_folder'])):
        trace_chunks = Parallel(n_jobs=min(kwargs['n_workers'], len(date_splits)))(delayed(process_data_parallel)(x, i, **kwargs) for x in date_splits)
        for chunk in trace_chunks:
            chunk["chunk"] = j
            j+=1
        chunk = pd.concat(trace_chunks)
        traces.append(chunk)
    # Combine all parallel results, reset the shingle id to accommodate those initialized in different chunks
    traces = pd.concat(traces)
    traces.insert(loc=0, column='shingle_id_new', value=traces.set_index(['chunk','shingle_id']).index.factorize()[0])
    traces["shingle_id"] = traces["shingle_id_new"]
    traces = traces.drop(columns=["shingle_id_new", "chunk"])
    # Write the traces to parquet file
    print(f"Saving {len(pd.unique(traces['shingle_id']))} samples to {kwargs['train_or_test']} file...")
    table = pa.Table.from_pandas(traces)
    writer = pq.ParquetWriter(f"{kwargs['base_folder']}deeptte_formatted/{kwargs['train_or_test']}", table.schema)
    writer.write_table(table)
    writer.close()
    # Save configs
    print(f"Saving {kwargs['train_or_test']} config and shingle lookup...")
    summary_config = data_utils.get_summary_config(traces, kwargs['gtfs_folder'], kwargs['epsg'], kwargs['grid_bounds'], kwargs['coord_ref_center'], kwargs['skip_gtfs'])
    with open(f"{kwargs['base_folder']}deeptte_formatted/{kwargs['train_or_test']}_config.json", mode="a") as out_file:
        json.dump(summary_config, out_file)
    shingle_config = table.group_by("shingle_id").aggregate([("shingle_id","count")]).to_pandas()
    shingle_config = shingle_config.to_dict(orient="records")
    data_utils.write_pkl(shingle_config, f"{kwargs['base_folder']}deeptte_formatted/{kwargs['train_or_test']}_shingle_config.json")

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
    if len(network_name) > 1:
        kwargs['is_mixed'] = True
        network_name = "_".join(network_name)
    else:
        kwargs['is_mixed'] = False
        network_name = network_name[0]
    base_folder = f"./results/{run_name}/{network_name}/"
    if run_name not in os.listdir("./results/"):
        os.mkdir(f"./results/{run_name}")
    if network_name in os.listdir(f"results/{run_name}/") and overwrite:
        shutil.rmtree(base_folder)
    if network_name not in os.listdir(f"results/{run_name}/"):
        os.mkdir(base_folder)
        os.mkdir(f"{base_folder}deeptte_formatted/")
        os.mkdir(f"{base_folder}models/")
        print(f"Created new results folder for '{run_name}'")
    else:
        print(f"Run '{run_name}/{network_name}' folder already exists in 'results/', delete the folder if new run desired.")
        return None
    kwargs['base_folder'] = base_folder
    # Split train/test dates into arbitrary number of chunks. More chunks = more training files = less ram pressure (chunk files are loaded 1 at a time)
    print(f"Processing training dates...")
    kwargs['train_or_test'] = "train"
    clean_data(train_dates, **kwargs)
    print(f"Processing testing dates...")
    kwargs['train_or_test'] = "test"
    clean_data(test_dates, **kwargs)
    print(f"RUN PREPARATION COMPLETED '{run_name}/{network_name}'")

if __name__=="__main__":

    # DEBUG
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="debug",
        network_name=["kcm"],
        train_dates=data_utils.get_date_list("2023_03_15", 3),
        test_dates=data_utils.get_date_list("2023_05_16", 3),
        n_workers=2,
        n_jobs=2,
        data_dropout=0.2,
        gtfs_folder=["./data/kcm_gtfs/"],
        raw_data_folder=["./data/kcm_all_new/"],
        timezone=["America/Los_Angeles"],
        epsg=["32148"],
        grid_bounds=[[369903,37911,409618,87758]],
        coord_ref_center=[[386910,69022]],
        given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
        skip_gtfs=False
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="debug",
        network_name=["atb"],
        train_dates=data_utils.get_date_list("2023_03_15", 3),
        test_dates=data_utils.get_date_list("2023_05_16", 3),
        n_workers=2,
        n_jobs=2,
        data_dropout=0.2,
        gtfs_folder=["./data/atb_gtfs/"],
        raw_data_folder=["./data/atb_all_new/"],
        timezone=["Europe/Oslo"],
        epsg=["32632"],
        grid_bounds=[[550869,7012847,579944,7039521]],
        coord_ref_center=[[569472,7034350]],
        given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
        skip_gtfs=False
    )
    # DEBUG MIXED
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="debug_nosch",
        network_name=["kcm","atb"],
        train_dates=data_utils.get_date_list("2023_03_15", 3),
        test_dates=data_utils.get_date_list("2023_05_16", 3),
        n_workers=2,
        n_jobs=2,
        data_dropout=0.2,
        gtfs_folder=["./data/kcm_gtfs/","./data/atb_gtfs/"],
        raw_data_folder=["./data/kcm_all_new/","./data/atb_all_new/"],
        timezone=["America/Los_Angeles","Europe/Oslo"],
        epsg=["32148","32632"],
        grid_bounds=[[369903,37911,409618,87758],[550869,7012847,579944,7039521]],
        coord_ref_center=[[386910,69022],[569472,7034350]],
        given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id'],['trip_id','file','locationtime','lat','lon','vehicle_id']],
        skip_gtfs=True
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="debug_nosch",
        network_name=["rut"],
        train_dates=data_utils.get_date_list("2023_03_15", 3),
        test_dates=data_utils.get_date_list("2023_05_16", 3),
        n_workers=2,
        n_jobs=2,
        data_dropout=0.2,
        gtfs_folder=["./data/rut_gtfs/"],
        raw_data_folder=["./data/rut_all_new/"],
        timezone=["Europe/Oslo"],
        epsg=["32632"],
        grid_bounds=[[589080,6631314,604705,6648420]],
        coord_ref_center=[[597427,6642805]],
        given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
        skip_gtfs=True
    )

    # PARAM SEARCH
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="param_search",
        network_name=["kcm"],
        train_dates=data_utils.get_date_list("2023_03_15", 60),
        test_dates=data_utils.get_date_list("2023_05_16", 12),
        n_workers=8,
        n_jobs=12,
        data_dropout=0.2,
        gtfs_folder=["./data/kcm_gtfs/"],
        raw_data_folder=["./data/kcm_all_new/"],
        timezone=["America/Los_Angeles"],
        epsg=["32148"],
        grid_bounds=[[369903,37911,409618,87758]],
        coord_ref_center=[[386910,69022]],
        given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
        skip_gtfs=False
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    prepare_run(
        overwrite=True,
        run_name="param_search",
        network_name=["atb"],
        train_dates=data_utils.get_date_list("2023_03_15", 60),
        test_dates=data_utils.get_date_list("2023_05_16", 7),
        n_workers=8,
        n_jobs=12,
        data_dropout=0.2,
        gtfs_folder=["./data/atb_gtfs/"],
        raw_data_folder=["./data/atb_all_new/"],
        timezone=["Europe/Oslo"],
        epsg=["32632"],
        grid_bounds=[[550869,7012847,579944,7039521]],
        coord_ref_center=[[569472,7034350]],
        given_names=[['trip_id','file','locationtime','lat','lon','vehicle_id']],
        skip_gtfs=False
    )

    # # FULL RUN
    