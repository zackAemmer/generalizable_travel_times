"""
Functions for processing tracked bus and timetable data.
"""
import itertools
import json
import os
import pickle
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from random import sample

import numpy as np
import pandas as pd
import pyproj
from sklearn import metrics
import torch

from utils import shape_utils


# Set of unified feature names and dtypes for variables in the GTFS-RT data
FEATURE_NAMES = ['trip_id','file','locationtime','lat','lon','vehicle_id']
FEATURE_TYPES = ['object','object','int','float','float','object']
FEATURE_LOOKUP = dict(zip(FEATURE_NAMES, FEATURE_TYPES))

# Set of unified feature names and dtypes for variables in the GTFS data
GTFS_NAMES = ['trip_id','stop_id','stop_lat','stop_lon','arrival_time']
GTFS_TYPES = [str,str,float,float,str]
GTFS_LOOKUP = dict(zip(GTFS_NAMES, GTFS_TYPES))


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return None

def normalize(ary, mean, std):
    """
    Z = (x - u) / sigma
    """
    return (ary - mean) / std

def de_normalize(ary, mean, std):
    """
    x = (Z * sigma) + u
    """
    return (ary * std) + mean

def recode_nums(ary):
    """
    Get new numeric codes starting from 0 for an array with random numbers.
    ary: array to recode
    Returns: dictionary that maps old numbers to new numbers starting from 0
    """
    old_codes = np.sort(ary).astype(int)
    new_codes = np.arange(0,len(old_codes))
    return dict(zip(old_codes, new_codes))

def calculate_gps_dist(end_lons, end_lats, start_lons, start_lats):
    """
    Calculate the spherical distance between a series of points.
    lons1/lats1: arrays of coordinates for the end points
    lons2/lats2: arrays of coordinates for the start points
    Returns: array of distances in meters.
    """
    geodesic = pyproj.Geod(ellps='WGS84')
    f_azis, b_azis, dists = geodesic.inv(start_lons, start_lats, end_lons, end_lats)
    return dists, f_azis

def calculate_trip_speeds(data):
    """
    Calculate speeds between consecutive trip locations.
    data: pandas dataframe of bus data with unified columns
    Returns: array of speeds, dist_diff, time_diff between consecutive points.
    Nan for first point of a trip.
    """
    x = data[['shingle_id','lat','lon','locationtime']]
    y = data[['shingle_id','lat','lon','locationtime']].shift()
    y.columns = [colname+"_shift" for colname in y.columns]
    z = pd.concat([x,y], axis=1)
    z['dist_diff'], z['bearing'] = calculate_gps_dist(z['lon'].values, z['lat'].values, z['lon_shift'].values, z['lat_shift'].values)
    z['time_diff'] = z['locationtime'] - z['locationtime_shift']
    z['speed_m_s'] = z['dist_diff'] / z['time_diff']
    return z['speed_m_s'].values, z['dist_diff'].values, z['time_diff'].values, z['bearing'].values

def get_validation_dates(validation_path):
    """
    Get a list of date strings corresponding to all trace files stored in a folder.
    validation_path: string containing the path to the folder
    Returns: list of date strings
    """
    dates = []
    files = os.listdir(validation_path)
    for file in files:
        labels = file.split("-")
        dates.append(labels[2] + "-" + labels[3] + "-" + labels[4].split("_")[0])
    return dates

def get_date_list(start, n_days):
    """
    Get a list of date strings starting at a given day and continuing for n days.
    start: date string formatted as 'yyyy_mm_dd'
    n_days: int number of days forward to include from start day
    Returns: list of date strings
    """
    year, month, day = start.split("_")
    base = date(int(year), int(month), int(day))
    date_list = [base + timedelta(days=x) for x in range(n_days)]
    return [f"{date.strftime('%Y_%m_%d')}.pkl" for date in date_list]

def load_train_test_data(data_folder, n_folds):
    """
    Load files with training/testing samples that have been formatted into json.
    Files are train_00 - train_05, and test
    data_folder: location that contains the train/test files
    Returns: list of trainning json samples, and list of test samples.
    """
    train_data_chunks = []
    for i in range(0, n_folds):
        train_data = []
        contents = open(data_folder + "train_0" + str(i), "r").read()
        train_data.append([json.loads(str(item)) for item in contents.strip().split('\n')])
        train_data = list(itertools.chain.from_iterable(train_data))
        train_data_chunks.append(train_data)
    # Read in test data
    contents = open(data_folder + "test", "r").read()
    test_data = [json.loads(str(item)) for item in contents.strip().split('\n')]
    return train_data_chunks, test_data

def load_run_input_data(run_folder, network_folder):
    train_traces = load_pkl(f"{run_folder}{network_folder}train_traces.pkl")
    test_traces = load_pkl(f"{run_folder}{network_folder}test_traces.pkl")
    with open(f"{run_folder}{network_folder}/deeptte_formatted/config.json") as f:
        config = json.load(f)
    gtfs_data = merge_gtfs_files(f".{config['gtfs_folder']}")
    tte_train_chunks, tte_test = load_train_test_data(f"{run_folder}{network_folder}/deeptte_formatted/", config['n_folds'])
    return {
        "train_traces": train_traces,
        "test_traces": test_traces,
        "config": config,
        "gtfs_data": gtfs_data,
        "tte_train_chunks": tte_train_chunks,
        "tte_test": tte_test
    }

def combine_pkl_data(folder, file_list, given_names):
    """
    Load raw feed data stored in a .pkl file to a dataframe. Unify column names and dtypes.
    This should ALWAYS be used to load the raw bus data from .pkl files, because it unifies the column names and types from different networks.
    folder: the folder to search in
    file_list: the file names to read and combine
    given_names: list of the names of the features in the raw data
    Returns: a dataframe of all data concatenated together, a column 'file' is added, also a list of all files with no data.
    """
    data_list = []
    no_data_list = []
    for file in file_list:
        try:
            data = load_pkl(folder + "/" + file)
            data['file'] = file
            # Get unified column names, data types
            data = data[given_names]            
            data.columns = FEATURE_LOOKUP.keys()
            # Fix locationtimes that were read as floats during data download
            try:
                data = data.astype(FEATURE_LOOKUP)
            except ValueError:
                data['locationtime'] = data['locationtime'].astype(float)
                data = data.astype(FEATURE_LOOKUP)
            data_list.append(data)
        except FileNotFoundError:
            no_data_list.append(file)
    data = pd.concat(data_list, axis=0)
    # Fix IDs that were read as 0s by numpy during data download
    try:
        data['trip_id'] = [str(int(x)) for x in data['trip_id'].values]
    except:
        x = 1
        print("This should not print unless processing AtB data")
    # Critical to ensure data frame is sorted by date, then trip_id, then locationtime
    data = data.sort_values(['file','trip_id','locationtime'], ascending=True)
    return data, no_data_list

def calculate_trace_df(data, timezone, remove_stopped_pts=True):
    """
    Calculate difference in metrics between two consecutive trip points.
    This is the only place where points are filtered rather than entire shingles.
    data: pandas df with all bus trips
    timezone: string for timezone the data were collected in
    remove_stopeed_pts: whether to include consecutive points with no bus movement
    Returns: combination of original point values, and new _diff values
    """
    # Some points with collection issues
    data = data[data['lat']!=0]
    data = data[data['lon']!=0]
    # Calculate feature values between consecutive points, assign to the latter point
    data['speed_m_s'], data['dist_calc_m'], data['time_calc_s'], data['bearing'] = calculate_trip_speeds(data)
    # Remove first point of every trip (not shingle), since its features are based on a different trip
    data = data.groupby(['file','trip_id'], as_index=False).apply(lambda group: group.iloc[1:,:])
    # Remove any points which seem to be erroneous or repeated
    data = data[data['dist_calc_m']>=0.0]
    data = data[data['dist_calc_m']<5000.0]
    data = data[data['time_calc_s']>0.0]
    data = data[data['time_calc_s']<120.0]
    data = data[data['speed_m_s']>=0.0]
    data = data[data['speed_m_s']<35.0]
    if remove_stopped_pts:
        data = data[data['dist_calc_m']>0.0]
        data = data[data['speed_m_s']>0.0]
    # Now error points are removed, recalculate time and speed features
    # From here out, must filter shingles in order to not change time/dist calcs
    # Note that any point filtering necessitates recalculating travel times for individual points
    data['speed_m_s'], data['dist_calc_m'], data['time_calc_s'], data['bearing'] = calculate_trip_speeds(data)
    data = data.groupby(['shingle_id'], as_index=False).apply(lambda group: group.iloc[1:,:])
    # Remove (shingles this time) based on final calculation of speeds, distances, times
    invalid_shingles = []
    invalid_shingles.append(data[data['dist_calc_m']<0.0].shingle_id)
    invalid_shingles.append(data[data['dist_calc_m']>5000.0].shingle_id)
    invalid_shingles.append(data[data['time_calc_s']<=0.0].shingle_id)
    invalid_shingles.append(data[data['time_calc_s']>120.0].shingle_id)
    invalid_shingles.append(data[data['speed_m_s']<0.0].shingle_id)
    invalid_shingles.append(data[data['speed_m_s']>35.0].shingle_id)
    if remove_stopped_pts:
        invalid_shingles.append(data[data['dist_calc_m']<=0.0].shingle_id)
        invalid_shingles.append(data[data['speed_m_s']<=0.0].shingle_id)
    invalid_shingles = pd.concat(invalid_shingles).values
    data = data[~data['shingle_id'].isin(invalid_shingles)]
    data['dist_calc_km'] = data['dist_calc_m'] / 1000.0
    data = data.dropna()
    # Time values for deeptte
    data['datetime'] = pd.to_datetime(data['locationtime'], unit='s', utc=True).map(lambda x: x.tz_convert(timezone))
    data['dateID'] = (data['datetime'].dt.day)
    data['weekID'] = (data['datetime'].dt.dayofweek)
    # (be careful with these last two as they change across the trajectory)
    data['timeID'] = (data['datetime'].dt.hour * 60) + (data['datetime'].dt.minute)
    data['timeID_s'] = (data['datetime'].dt.hour * 60 * 60) + (data['datetime'].dt.minute * 60) + (data['datetime'].dt.second)
    return data

def calculate_cumulative_values(data):
    """
    Calculate values that accumulate across each trajectory.
    """
    unique_traj = data.groupby('shingle_id')
    # Get cumulative values from trip start
    data['time_cumulative_s'] = unique_traj['time_calc_s'].cumsum()
    data['dist_cumulative_km'] = unique_traj['dist_calc_km'].cumsum()
    data['time_cumulative_s'] = data.time_cumulative_s - unique_traj.time_cumulative_s.transform('min')
    data['dist_cumulative_km'] = data.dist_cumulative_km - unique_traj.dist_cumulative_km.transform('min')
    # Remove shingles that don't traverse more than a kilometer, or have less than n points
    data = data.groupby(['shingle_id']).filter(lambda x: np.max(x.dist_cumulative_km) >= 1.0)
    data = data.groupby(['shingle_id']).filter(lambda x: len(x) >= 5)
    return data

def parallel_clean_trace_df_w_timetables(args):
    data_chunk, gtfs_data = args
    return clean_trace_df_w_timetables(data_chunk, gtfs_data)

def parallelize_clean_trace_df_w_timetables(data, gtfs_data, n_processes=4):
    with Pool(n_processes) as p:
        data_chunks = np.array_split(data, n_processes)
        args = [(chunk, gtfs_data) for chunk in data_chunks]
        results = p.map(parallel_clean_trace_df_w_timetables, args)
    return pd.concat(results)

def clean_trace_df_w_timetables(data, gtfs_data):
    """
    Validate a set of tracked bus locations against GTFS.
    data: pandas dataframe with unified bus data
    gtfs_data: merged GTFS files
    Returns: dataframe with only trips that are in GTFS, and are reasonably close to scheduled stop ids.
    """
    # Remove any trips that are not in the GTFS
    data.drop(data[~data['trip_id'].isin(gtfs_data.trip_id)].index, inplace=True)
    # Filter trips with less than n observations
    shingle_counts = data['shingle_id'].value_counts()
    valid_trips = shingle_counts.index[shingle_counts >= 5]
    data = data[data['shingle_id'].isin(valid_trips)]
    # Save start time of first points in trajectories
    first_points = data[['shingle_id','timeID_s']].drop_duplicates('shingle_id')
    first_points.columns = ['shingle_id','trip_start_timeID_s']
    closest_stops = get_scheduled_arrival(
        data['trip_id'].values,
        data['lon'].values,
        data['lat'].values,
        gtfs_data
    )
    data = data.assign(stop_dist_km=closest_stops[0])
    data = data.assign(stop_arrival_s=closest_stops[1])
    data = data.assign(stop_lon=closest_stops[2])
    data = data.assign(stop_lat=closest_stops[3])
    # Filter out shingles with stops that are too far away
    valid_trips = data.groupby('shingle_id').filter(lambda x: x['stop_dist_km'].max() <= 1.0)['shingle_id'].unique()
    data = data[data['shingle_id'].isin(valid_trips)]
    # Get the timeID_s (for the first point of each trajectory)
    data = pd.merge(data, first_points, on='shingle_id')
    # Calculate the scheduled travel time from the first to each point in the shingle
    data = data.assign(scheduled_time_s=data['stop_arrival_s'] - data['trip_start_timeID_s'])
    # Filter out shingles where the data started after midnight, but the trip started before
    # If the data started before the scheduled time difference is still accurate
    valid_trips = data.groupby('shingle_id').filter(lambda x: x['scheduled_time_s'].max() <= 10000)['shingle_id'].unique()
    data = data[data['shingle_id'].isin(valid_trips)]
    return data

def get_scheduled_arrival(trip_ids, lons, lats, gtfs_data):
    """
    Find the nearest stop to a set of trip-coordinates, and return the scheduled arrival time.
    trip_ids: list of trip_ids
    lons/lats: lists of places where the bus will be arriving (end point of traj)
    gtfs_data: merged GTFS files
    Returns: (distance to closest stop in km, scheduled arrival time at that stop).
    """
    data = np.column_stack([lons, lats, trip_ids])
    gtfs_data_ary = gtfs_data[['stop_lon','stop_lat','trip_id','arrival_s']].values

    # Create dictionary mapping trip_ids to lists of points in gtfs
    id_to_points = {}
    for point in gtfs_data_ary:
        id_to_points.setdefault(point[2],[]).append(point)

    # For each point find the closest stop that shares the trip_id
    results = np.zeros((len(data), 5), dtype=object)
    for i, point in enumerate(data):
        corresponding_points = np.vstack(id_to_points.get(point[2], []))
        point = np.expand_dims(point, 0)
        # Find closest point and add to results
        closest_point_dist, closest_point_idx = shape_utils.get_closest_point(corresponding_points[:,0:2], point[:,0:2])
        closest_point = corresponding_points[closest_point_idx]
        closest_point = np.append(closest_point, closest_point_dist * 111)
        results[i,:] = closest_point
    return results[:,4], results[:,3], results[:,0], results[:,1]

def remap_ids(df_list, id_col):
    """
    Remap each ID in all dfs to start from 0, maintaining order.
    df_list: list of pandas dataframes with unified bus data
    id_col: the column to re-index the unique values of
    Returns: list of pandas dataframes with new column for id_recode.
    """
    all_ids = pd.concat([x[id_col] for x in df_list]).values.flatten()
    # Recode ids to start from 0
    mapping = {v:k for k,v in enumerate(set(all_ids))}
    for df in df_list:
        recode = [mapping[y] for y in df[id_col].values.flatten()]
        df[f"{id_col}_recode"] = recode
    return (df_list, len(pd.unique(all_ids)), mapping)

def map_to_deeptte(trace_data, deeptte_formatted_path, n_folds, is_test=False):
    """
    Reshape pandas dataframe to the json format needed to use deeptte.
    trace_data: dataframe with bus trajectories
    Returns: path to json file where deeptte trajectories are saved.
    """    
    # Group by the desired column use correct naming schema
    # These will be trip-totals
    trace_data['dist'] = trace_data['dist_cumulative_km']
    trace_data['time'] = trace_data['time_cumulative_s']
    # Cumulative values
    trace_data['dist_gap'] = trace_data['dist_cumulative_km']
    trace_data['time_gap'] = trace_data['time_cumulative_s']
    # Lists
    trace_data['lats'] = trace_data['lat']
    trace_data['lngs'] = trace_data['lon']
    trace_data['vehicleID'] = trace_data['vehicle_id_recode']
    trace_data['tripID'] = trace_data['trip_id_recode']

    groups = trace_data.groupby('shingle_id')

    # Save grid with past n timesteps, all occurring before shingle start
    grid, grid_features = shape_utils.get_grid_features(trace_data, n_prior=5)

    # Get necessary features as scalar or lists
    result = groups.agg({
        # Cumulative point time/dist
        'time_gap': lambda x: x.tolist(),
        'dist_gap': lambda x: x.tolist(),
        # Trip time/dist
        'time': 'max',
        'dist': 'max',
        # IDs
        'vehicleID': 'min',
        'tripID': 'min',
        'weekID': 'min',
        'timeID': 'min',
        'dateID': 'min',
        'trip_id': 'min',
        'file': 'min',
        # Individual point time/dist
        'lats': lambda x: x.tolist(),
        'lngs': lambda x: x.tolist(),
        'speed_m_s': lambda x: x.tolist(),
        'time_calc_s': lambda x: x.tolist(),
        'dist_calc_km': lambda x: x.tolist(),
        'bearing': lambda x: x.tolist(),
        # Trip start time
        'trip_start_timeID_s': 'min',
        # Trip ongoing time
        'timeID_s': lambda x: x.tolist(),
        # Nearest stop
        'stop_lat': lambda x: x.tolist(),
        'stop_lon': lambda x: x.tolist(),
        'stop_dist_km': lambda x: x.tolist(),
        'scheduled_time_s': lambda x: x.tolist(),
    })

    # Convert the DataFrame to a dictionary with the original format
    if is_test:
        print(f"Saving formatted data to '{deeptte_formatted_path}', across 1 testing file...")
        result_json_string = result.to_json(orient='records', lines=True)
        with open(deeptte_formatted_path+"test", mode='w+') as out_file:
            out_file.write(result_json_string)
    else:
        print(f"Saving formatted data to '{deeptte_formatted_path}', across {n_folds} training files...")
        dfs = np.array_split(result, n_folds)
        for i, df in enumerate(dfs):
            df_json_string = df.to_json(orient='records', lines=True)
            with open(deeptte_formatted_path+"train_0"+str(i), mode='w+') as out_file:
                out_file.write(df_json_string)
    return trace_data

def get_summary_config(trace_data, n_unique_veh, n_unique_trip, gtfs_folder, n_folds):
    """
    Get a dict of means and sds which are used to normalize data by DeepTTE.
    trace_data: pandas dataframe with unified columns and calculated distances
    Returns: dict of mean and std values, as well as train/test filenames.
    """
    # config.json
    grouped = trace_data.groupby('shingle_id')
    summary_dict = {
        # DeepTTE variables:
        # Total trip values
        "time_mean": np.mean(grouped.max()[['time_cumulative_s']].values.flatten()),
        "time_std": np.std(grouped.max()[['time_cumulative_s']].values.flatten()),
        "dist_mean": np.mean(grouped.max()[['dist_cumulative_km']].values.flatten()),
        'dist_std': np.std(grouped.max()[['dist_cumulative_km']].values.flatten()),
        # Individual point values (no cumulative)
        'time_gap_mean': np.mean(trace_data['time_calc_s']),
        'time_gap_std': np.std(trace_data['time_calc_s']),
        'dist_gap_mean': np.mean(trace_data['dist_calc_km']),
        'dist_gap_std': np.std(trace_data['dist_calc_km']),
        'lngs_mean': np.mean(trace_data['lon']),
        'lngs_std': np.std(trace_data['lon']),
        'lats_mean': np.mean(trace_data['lat']),
        "lats_std": np.std(trace_data['lat']),
        # Other variables:
        "speed_m_s_mean": np.mean(trace_data['speed_m_s']),
        "speed_m_s_std": np.std(trace_data['speed_m_s']),
        "bearing_mean": np.mean(trace_data['bearing']),
        "bearing_std": np.std(trace_data['bearing']),
        # Normalization for both cumulative and individual time/dist values
        "dist_cumulative_km_mean": np.mean(trace_data['dist_cumulative_km']),
        "dist_cumulative_km_std": np.std(trace_data['dist_cumulative_km']),
        "time_cumulative_s_mean": np.mean(trace_data['time_cumulative_s']),
        "time_cumulative_s_std": np.std(trace_data['time_cumulative_s']),
        "dist_calc_km_mean": np.mean(trace_data['dist_calc_km']),
        "dist_calc_km_std": np.std(trace_data['dist_calc_km']),
        "time_calc_s_mean": np.mean(trace_data['time_calc_s']),
        "time_calc_s_std": np.std(trace_data['time_calc_s']),
        # Nearest stop
        "stop_dist_km_mean": np.mean(trace_data['stop_dist_km']),
        "stop_dist_km_std": np.std(trace_data['stop_dist_km']),
        "scheduled_time_s_mean": np.mean(grouped.max()[['scheduled_time_s']].values.flatten()),
        "scheduled_time_s_std": np.std(grouped.max()[['scheduled_time_s']].values.flatten()),
        "stop_lng_mean": np.mean(trace_data['stop_lon']),
        "stop_lng_std": np.std(trace_data['stop_lon']),
        "stop_lat_mean": np.mean(trace_data['stop_lat']),
        "stop_lat_std": np.std(trace_data['stop_lat']),
        # Not variables
        "n_unique_veh": n_unique_veh,
        "n_unique_trip": n_unique_trip,
        "gtfs_folder": gtfs_folder,
        "n_folds": n_folds,
        "train_set": ["train_0"+str(x) for x in range(0,n_folds)],
        "eval_set": ["test"]
    }
    return summary_dict

def extract_operator(old_folder, new_folder, source_col, op_name):
    """
    Make a copy of raw bus data with only a single operator.
    old_folder: location of data which contains the desired operator + others
    new_folder: where to save the new filtered files
    source_col: column name to filter on
    op_name: column value to keep in new files
    """
    files = os.listdir(old_folder)
    for file in files:
        if file != ".DS_Store":
            with open(f"{old_folder}{file}", 'rb') as f:
                data = pickle.load(f)
                data = data[data[source_col]==op_name]
            with open(f"{new_folder}{file}", 'wb') as f:
                pickle.dump(data, f)

def extract_operator_gtfs(old_folder, new_folder, source_col, op_name):
    """
    First make a copy of the GTFS directory, then this function will overwrite the key files.
    """
    gtfs_folders = os.listdir(old_folder)
    for file in gtfs_folders:
        if file != ".DS_Store" and len(file)==10:
            z = pd.read_csv(f"{old_folder}{file}/trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
            st = pd.read_csv(f"{old_folder}{file}/stop_times.txt", low_memory=False, dtype=GTFS_LOOKUP)
            z = z[z[source_col].str[:3]==op_name]
            st = st[st[source_col].str[:3]==op_name]
            z.to_csv(f"{new_folder}{file}/trips.txt")
            st.to_csv(f"{new_folder}{file}/stop_times.txt")

def merge_gtfs_files(gtfs_folder):
    """
    Join a set of GTFS files into a single dataframe. Each row is a trip + arrival time.
    gtfs_folder: location to search for GTFS files
    Returns: pandas dataframe with merged GTFS data.
    """
    z = pd.read_csv(gtfs_folder+"trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
    st = pd.read_csv(gtfs_folder+"stop_times.txt", low_memory=False, dtype=GTFS_LOOKUP)
    sl = pd.read_csv(gtfs_folder+"stops.txt", low_memory=False, dtype=GTFS_LOOKUP)
    z = pd.merge(z,st,on="trip_id")
    z = pd.merge(z,sl,on="stop_id")

    gtfs_data = z.sort_values(['trip_id','stop_sequence'])
    gtfs_data['arrival_s'] = [int(x[0])*60*60 + int(x[1])*60 + int(x[2]) for x in gtfs_data['arrival_time'].str.split(":")]
    return gtfs_data

def get_date_from_filename(filename):
    """
    Get the date from a .pkl raw bus data filename.
    filename: string in format "YYYY_MM_DD.pkl"
    Returns: Datetime object.
    """
    file_parts = filename.split("_")
    date_obj = datetime(int(file_parts[0]), int(file_parts[1]), int(file_parts[2].split(".")[0]))
    return date_obj

def calc_data_metrics(data, timezone):
    """
    Summarize some metrics for a set of bus data on a given day.
    data: pandas dataframe with unified bus data
    timezone: string similar to "America/Los_Angeles"
    Returns: Dictionary with keys corresponding to metrics. Some are grouped by hour of day.
    """
    # Speed
    data['speed_m_s'], data['dist_calc_m'], data['time_calc_s'] = calculate_trip_speeds(data)
    data = data[data['dist_calc_m']>0.0]
    data = data[data['dist_calc_m']<5000.0]
    data = data[data['time_calc_s']>0.0]
    data = data[data['time_calc_s']<120.0]
    data = data[data['speed_m_s']>0.0]
    data = data[data['speed_m_s']<35.0]
    data['dist_calc_km'] = data['dist_calc_m'] / 1000.0
    data = data.dropna()
    # Simple metrics
    points = len(data)
    trajs = len(data.drop_duplicates(['trip_id']))
    unique_trips = len(pd.unique(data['trip_id']))
    unique_vehs = len(pd.unique(data['vehicle_id']))
    # Obs and Speed by time of day
    data['datetime'] = pd.to_datetime(data['locationtime'], unit='s', utc=True).map(lambda x: x.tz_convert(timezone))
    data['timeID'] = data['datetime'].dt.hour
    data['timeID'] = pd.Categorical(data['timeID'], categories=np.arange(24))
    hourly_agg = data[['timeID','speed_m_s']].groupby('timeID')
    mean_speeds = hourly_agg.mean(numeric_only=True)['speed_m_s'].values.flatten()
    sd_speeds = hourly_agg.std(numeric_only=True)['speed_m_s'].values.flatten()
    n_obs = hourly_agg.count()['speed_m_s'].values.flatten()
    # Group metrics to return in dict
    summary = {
        "n_points": points,
        "n_trajs": trajs,
        "nunq_trips": unique_trips,
        "nunq_vehs": unique_vehs,
        "hourly_points": n_obs,
        "hourly_mean_speeds": mean_speeds,
        "hourly_sd_speeds": sd_speeds
    }
    return summary

def full_dataset_summary(folder, given_names, timezone):
    """
    Calculate summaries for every raw bus data file in a folder.
    folder: where to look for .pkl files
    given_names: feature names (in order) used by the raw feed
    timezone: string similar to "America/Los_Angeles"
    Returns: list of dates that data was found for, and list of dicts with summary for each date.
    """
    file_list = os.listdir(folder)
    dates = []
    data_summaries = []
    for file in file_list:
        if file != ".DS_Store":
            data, _ = combine_pkl_data(folder, [file], given_names)
            dates.append(get_date_from_filename(file))
            data_summaries.append(calc_data_metrics(data, timezone))
    return dates, data_summaries

def resample_deeptte_gps(deeptte_data, n_samples):
    """
    Resamples tracked gps points evenly to specified count.
    deeptte_data: json loaded from a training/testing file formatted for DeepTTE
    n_samples: the number of gps points to resample each record to
    Returns: dataframe with lat, lon, dist, and time resampled. Each timestep is averaged or linearly interpolated.
    """
    all_res = []
    for i in range(0, len(deeptte_data)):
        time_gaps = deeptte_data[i]['time_gap']
        lats = [float(x) for x in deeptte_data[i]['lats']]
        lngs = [float(x) for x in deeptte_data[i]['lngs']]
        dists = [float(x) for x in deeptte_data[i]['dist_gap']]
        z = pd.DataFrame({"time_gaps":time_gaps, "lat":lats, "lng":lngs, "dist":dists})
        z.index = pd.to_datetime(z['time_gaps'], unit='s')
        first = z.index.min()
        last = z.index.max()
        secs = int((last-first).total_seconds())
        secs_per_sample = secs // n_samples
        periodsize = '{:f}S'.format(secs_per_sample)
        result = z.resample(periodsize).mean()
        result = result.interpolate(method='linear')
        result = result.iloc[0:n_samples,:]
        result['deeptte_index'] = i
        all_res.append(result)
    return pd.concat(all_res, axis=0)

def format_deeptte_to_features(deeptte_data, resampled_deeptte_data):
    """
    Reformat the DeepTTE json format into a numpy array that can be used for modeling with sklearn.
    deeptte_data: json loaded from a training/testing file formatted for DeepTTE
    resampled_deeptte_data: resampled dataframe from 'resample_deeptte_gps' in which all tracks have the same length
    Returns: 2d numpy array (samples x features) and 1d numpy array (tt true values). Each point in a trace has a set of resampled features, which are concatenated such that each value at each point is its own feature.
    """
    # Gather the trip attribute features
    timeIDs = np.array([x['timeID'] for x in deeptte_data]).reshape(len(deeptte_data),1)
    weekIDs = np.array([x['weekID'] for x in deeptte_data]).reshape(len(deeptte_data),1)
    dateIDs = np.array([x['dateID'] for x in deeptte_data]).reshape(len(deeptte_data),1)
    driverIDs = np.array([x['driverID'] for x in deeptte_data]).reshape(len(deeptte_data),1)
    dists = np.array([x['dist'] for x in deeptte_data]).reshape(len(deeptte_data),1)
    df = np.concatenate((timeIDs, weekIDs, dateIDs, driverIDs, dists), axis=1) # X
    times = np.array([x['time'] for x in deeptte_data]).reshape(len(deeptte_data),1).ravel() # y
    # Add resampled features (each point is own feature)
    # resampled_features = resampled_deeptte_data.groupby("deeptte_index").apply(lambda x: np.concatenate([x['dist'].values, x['lat'].values, x['lng'].values]))
    resampled_features = resampled_deeptte_data.groupby("deeptte_index").apply(lambda x: np.concatenate([x['lat'].values, x['lng'].values]))
    resampled_features = np.array([x for x in resampled_features])
    df = np.hstack((df, resampled_features))
    return df, times

def extract_results(city, model_results):
    # Extract metric results
    fold_results = [x['All Losses'] for x in model_results]
    cities = []
    models = []
    mapes = []
    rmses = []
    maes = []
    fold_nums = []
    for fold_num in range(0,len(fold_results)):
        for value in range(0,len(fold_results[0])):
            cities.append(city)
            fold_nums.append(fold_num)
            models.append(fold_results[fold_num][value][0])
            mapes.append(fold_results[fold_num][value][1])
            rmses.append(fold_results[fold_num][value][2])
            maes.append(fold_results[fold_num][value][3])
    result_df = pd.DataFrame({
        "Model": models,
        "City": cities,
        "Fold": fold_nums,
        "MAPE": mapes,
        "RMSE": rmses,
        "MAE": maes
    })
    # Extract NN loss curves
    loss_df = []
    # Iterate folds
    for fold_results in model_results:
        # Iterate models
        for model in fold_results['Loss Curves']:
            for mname, loss_curves in model.items():
                # Iterate loss curves
                for lname, loss in loss_curves.items():
                    df = pd.DataFrame({
                        "City": city,
                        "Fold": fold_results['Fold'],
                        "Model": mname,
                        "Loss Set": lname,
                        "Epoch": np.arange(len(loss)),
                        "Loss": loss
                    })
                    loss_df.append(df)
    loss_df = pd.concat(loss_df)
    return result_df, loss_df

def extract_deeptte_results(city, run_folder, network_folder):
    # Extract all fold and epoch losses from deeptte run
    all_run_data = []
    for res_file in os.listdir(f"{run_folder}{network_folder}deeptte_results/result"):
        res_preds = pd.read_csv(
            f"{run_folder}{network_folder}deeptte_results/result/{res_file}",
            delimiter=" ",
            header=None,
            names=["Label", "Pred"],
            dtype={"Label": float, "Pred": float}
        )
        res_labels = res_file.split("_")
        if len(res_labels)==5:
            model_name, test_file_name, test_file_num, fold_num, epoch_num = res_labels
            test_file_name = test_file_name + "_" + test_file_num
        elif len(res_labels)==4:
            model_name, test_file_name, fold_num, epoch_num = res_labels
        epoch_num = epoch_num.split(".")[0]
        res_data = [
            "DeepTTE",
            city,
            test_file_name,
            fold_num,
            epoch_num,
            metrics.mean_absolute_percentage_error(res_preds['Label'], res_preds['Pred']),
            np.sqrt(metrics.mean_squared_error(res_preds['Label'], res_preds['Pred'])),
            metrics.mean_absolute_error(res_preds['Label'], res_preds['Pred'])
        ]
        all_run_data.append(res_data)
    all_run_data = pd.DataFrame(
        all_run_data,
        columns=[
            "Model",
            "City",
            "Loss Set",
            "Fold",
            "Epoch",
            "MAPE",
            "RMSE",
            "MAE"
        ]
    )
    all_run_data['Fold'] = all_run_data['Fold'].astype(int)
    all_run_data['Epoch'] = all_run_data['Epoch'].astype(int)
    return all_run_data.sort_values(['Fold','Epoch'])

def shingle(trace_df, min_len, max_len):
    """
    Split a df into even chunks randomly between min and max length.
    Each split comes from a group representing a trajectory in the dataframe.
    trace_df: dataframe of raw bus data
    min_len: minimum number of chunks to split a trajectory into
    max_lan: maximum number of chunks to split a trajectory into
    Returns: A copy of trace_df with a new index, traces with <=2 points removed.
    """
    shingle_groups = trace_df.groupby(['file','trip_id']).count()['lat'].values
    idx = 0
    new_idx = []
    for num_pts in shingle_groups:
        dummy = np.array([0 for i in range(0,num_pts)])
        dummy = np.array_split(dummy, np.random.randint(min_len, max_len))
        dummy = [len(x) for x in dummy]
        for x in dummy:
            [new_idx.append(idx) for y in range(0,x)]
            idx += 1
    z = trace_df.copy()
    z['shingle_id'] = new_idx
    return z

def extract_all_dataloader(dataloader, sequential_flag=False):
    """
    Get all contents of a dataloader from batches.
    """
    all_context = []
    all_X = []
    all_y = []
    seq_lens = []
    for i, data in enumerate(dataloader):
        data, y = data
        context, X = data[:2]
        all_context.append(context)
        all_X.append(X)
        all_y.append(y)
        if sequential_flag:
            seq_lens.append(data[2])
    if sequential_flag:
        # Pad batches to match batch w/longest sequence
        max_len = max(torch.cat(seq_lens))
        all_X = [torch.nn.functional.pad(tensor, (0, 0, 0, max_len - tensor.shape[1])) for tensor in all_X]
        all_y = [torch.nn.functional.pad(tensor, (0, max_len - tensor.shape[1])) for tensor in all_y]
        return torch.cat(all_context, dim=0), torch.cat(all_X, dim=0), torch.cat(all_y, dim=0), torch.cat(seq_lens, dim=0)
    else:
        return torch.cat(all_context, dim=0), torch.cat(all_X, dim=0), torch.cat(all_y, dim=0)

def get_seq_info(seq_dataloader):
    """
    Get lengths and mask for sequence lengths in data.
    """
    context, X, y, seq_lens = extract_all_dataloader(seq_dataloader, sequential_flag=True)
    seq_lens = list(seq_lens.numpy())
    max_length = max(seq_lens)
    mask = [[1] * length + [0] * (max_length - length) for length in seq_lens]
    return seq_lens, np.array(mask, dtype='bool')

def pad_tensors(tensor_list, pad_dim):
    """
    Pad list of tensors with unequal lengths on pad_dim and combine.
    """
    tensor_lens = [tensor.shape[pad_dim] for tensor in tensor_list]
    max_len = max(tensor_lens)
    total_dim = len(tensor_list[0].shape)
    paddings = []
    for tensor in tensor_list:
        padding = list(0 for i in range(total_dim))
        padding[pad_dim] = max_len - tensor.shape[pad_dim]
        paddings.append(tuple(padding))
    padded_tensor_list = [torch.nn.functional.pad(tensor, paddings[i]) for i, tensor in enumerate(tensor_list)]
    padded_tensor_list = torch.cat(padded_tensor_list, dim=0)
    return padded_tensor_list

def convert_speeds_to_tts(speeds, dataloader, mask, config):
    """
    Convert a sequence of predicted speeds to travel times.
    """
    context, X, y, seq_lens = extract_all_dataloader(dataloader, sequential_flag=True)
    dists = X[:,:,2].numpy()
    dists = de_normalize(dists, config['dist_calc_km_mean'], config['dist_calc_km_std'])
    # Replace speeds near 0.0 with small number
    speeds[speeds<=0.0001] = 0.0001
    travel_times = dists*1000.0 / speeds
    res = aggregate_tts(travel_times, mask)
    return res

def aggregate_tts(tts, mask, drop_first=True):
    """
    Convert a sequence of predicted travel times to total travel time.
    """
    if drop_first:
        # The first point has a predicted tt, but don't sum it to match total time
        mask[:,0] = False
    masked_tts = (tts*mask)
    total_tts = np.sum(masked_tts, axis=1)
    return total_tts

def aggregate_cumulative_tts(tts, mask):
    """
    Convert a sequence of cumulative predicted travel times to total travel time.
    """
    # Take the final unmasked point
    max_idxs = np.apply_along_axis(lambda x: np.max(np.where(x)), 1, mask)
    max_idxs = max_idxs.reshape(len(max_idxs),1)
    res = np.take_along_axis(tts, max_idxs, axis=1)
    return res

def create_tensor_mask(seq_lens):
    """
    Create a mask based on a tensor of sequence lengths.
    """
    max_len = max(seq_lens)
    mask = torch.zeros(len(seq_lens), max_len, dtype=torch.bool)
    for i, seq_len in enumerate(seq_lens):
        mask[i, :seq_len] = 1
    return mask