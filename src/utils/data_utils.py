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
from statsmodels.stats.weightstats import DescrStatsW
import torch

from models import grids
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
    ary[ary==0] = 1e-6
    if mean==0:
        mean = 1e-6
    if std==0:
        std = 1e-6
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

def calculate_gps_dist(end_x, end_y, start_x, start_y):
    """
    Calculate the euclidean distance between a series of points.
    Returns: array of distances in meters.
    """
    x_diff = (end_x - start_x)
    y_diff = (end_y - start_y)
    dists = np.sqrt(x_diff**2 + y_diff**2)
    # Measured in degrees from the positive x axis, + is 1/4 quadrants, - is 2/3 quadrants
    bearings = np.arctan2(y_diff, x_diff)*180/np.pi
    return dists, bearings

def calculate_trip_speeds(data):
    """
    Calculate speeds between consecutive trip locations.
    Returns: array of speeds, dist_diff, time_diff between consecutive points.
    Nan for first point of a trip.
    """
    x = data[['shingle_id','x','y','locationtime']]
    y = data[['shingle_id','x','y','locationtime']].shift()
    y.columns = [colname+"_shift" for colname in y.columns]
    z = pd.concat([x,y], axis=1)
    z['dist_diff'], z['bearing'] = calculate_gps_dist(z['x'].values, z['y'].values, z['x_shift'].values, z['y_shift'].values)
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

def load_fold_data(data_folder, filename, fold_num, total_folds):
    ngrid = load_pkl(f"{data_folder}/../{filename}_ngrid.pkl")
    train_data = []
    contents = open(f"{data_folder}{filename}", "r").read()
    train_data.append([json.loads(str(item)) for item in contents.strip().split('\n')])
    train_data = list(itertools.chain.from_iterable(train_data))
    # Select all data that is not part of this testing fold
    n_per_fold = len(train_data) // total_folds
    mask = np.ones(len(train_data), bool)
    mask[fold_num*n_per_fold:(fold_num+1)*n_per_fold] = 0
    return ([item for item, keep in zip(train_data, mask) if keep], [item for item, keep in zip(train_data, mask) if not keep], ngrid)

def load_all_data(data_folder, filename):
    ngrid = load_pkl(f"{data_folder}/../{filename}_ngrid.pkl")
    valid_data = []
    contents = open(f"{data_folder}{filename}", "r").read()
    valid_data.append([json.loads(str(item)) for item in contents.strip().split('\n')])
    valid_data = list(itertools.chain.from_iterable(valid_data))
    return valid_data, ngrid

def load_all_inputs(run_folder, network_folder, file_num):
    with open(f"{run_folder}{network_folder}/deeptte_formatted/train_config.json") as f:
        config = json.load(f)
    train_traces = load_pkl(f"{run_folder}{network_folder}train{file_num}_traces.pkl")
    train_data, train_ngrid = load_all_data(f"{run_folder}{network_folder}deeptte_formatted/", f"train{file_num}")
    return {
        "config": config,
        "train_traces": train_traces,
        "train_data": train_data,
        "train_ngrid": train_ngrid,
    }

def combine_config_files(cfg_folder, n_save_files, train_or_test):
    temp = []
    for i in range(n_save_files):
        # Load config
        with open(f"{cfg_folder}{train_or_test}{i}_config.json") as f:
            config = json.load(f)
        # Save temp
        temp.append(config)
        # Delete existing config
        os.remove(f"{cfg_folder}{train_or_test}{i}_config.json")
    # Set up train config
    summary_config = {}
    for k in temp[0].keys():
        if k[-4:]=="mean":
            values = [x[k] for x in temp]
            weights = [x["n_points"] for x in temp]
            wtd_mean = float(DescrStatsW(values, weights=weights, ddof=len(weights)).mean)
            summary_config.update({k:wtd_mean})
        elif k[-3:]=="std":
            values = [x[k]**2 for x in temp]
            weights = [x["n_points"] for x in temp]
            wtd_std = float(np.sqrt(DescrStatsW(values, weights=weights, ddof=len(weights)).mean))
            summary_config.update({k:wtd_std})
    summary_config["n_points"] = int(np.sum([x['n_points'] for x in temp]))
    summary_config["n_samples"] = int(np.sum([x['n_samples'] for x in temp]))
    summary_config["gtfs_folder"] = temp[0]["gtfs_folder"]
    summary_config["n_save_files"] = temp[0]["n_save_files"]
    summary_config["epsg"] = temp[0]["epsg"]
    summary_config["train_set"] = temp[0]["train_set"]
    summary_config["test_set"] = temp[0]["test_set"]
    # Save final configs
    with open(f"{cfg_folder}{train_or_test}_config.json", mode="a") as out_file:
        json.dump(summary_config, out_file)

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
            # Get unified column names
            data = data[given_names]
            data.columns = FEATURE_LOOKUP.keys()
            # Nwy locationtimes are downloaded as floats and have a decimal point; must go from object through float again to get int
            data['locationtime'] = data['locationtime'].astype(float)
            # Get unified data types
            data = data.astype(FEATURE_LOOKUP)
            data_list.append(data)
        except FileNotFoundError:
            no_data_list.append(file)
    data = pd.concat(data_list, axis=0)
    # Critical to ensure data frame is sorted by date, then trip_id, then locationtime
    data = data.sort_values(['file','trip_id','locationtime'], ascending=True)
    return data, no_data_list

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

def calculate_trace_df(data, timezone, epsg, grid_bounds, coord_ref_center, data_dropout=.10, remove_stopped_pts=True):
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
    # Drop out random points from all shingles
    data = data.reset_index()
    drop_indices = np.random.choice(data.index, int(data_dropout*len(data)), replace=False)
    data = data[~data.index.isin(drop_indices)].reset_index()
    # Project to local coordinate system
    default_crs = pyproj.CRS.from_epsg(4326)
    proj_crs = pyproj.CRS.from_epsg(epsg)
    transformer = pyproj.Transformer.from_crs(default_crs, proj_crs, always_xy=True)
    data['x'], data['y'] = transformer.transform(data['lon'], data['lat'])
    # Add coordinates that are translated s.t. CBD is 0,0
    data['x_cent'] = data['x'] - coord_ref_center[0]
    data['y_cent'] = data['y'] - coord_ref_center[1]
    # Drop points outside of the network/grid bounding box
    data = data[data['x']>grid_bounds[0]]
    data = data[data['y']>grid_bounds[1]]
    data = data[data['x']<grid_bounds[2]]
    data = data[data['y']<grid_bounds[3]]
    # Calculate feature values between consecutive points, assign to the latter point
    data['speed_m_s'], data['dist_calc_m'], data['time_calc_s'], data['bearing'] = calculate_trip_speeds(data)
    # Remove first point of every trip (not shingle), since its features are based on a different trip
    data = data.groupby(['file','trip_id'], as_index=False).apply(lambda group: group.iloc[1:,:])
    # Remove any points which seem to be erroneous or repeated
    data = data[data['dist_calc_m']>=0.0]
    data = data[data['time_calc_s']>0.0]
    data = data[data['speed_m_s']>=0.0]
    data = data[data['speed_m_s']<=35.0]
    if remove_stopped_pts:
        data = data[data['dist_calc_m']>0.0]
        data = data[data['speed_m_s']>0.0]
    # Now error points are removed, recalculate time and speed features
    # From here out, must filter shingles in order to not change time/dist calcs
    # Note that any point filtering necessitates recalculating travel times for individual points
    data['speed_m_s'], data['dist_calc_m'], data['time_calc_s'], data['bearing'] = calculate_trip_speeds(data)
    shingles = data.groupby(['shingle_id'], as_index=False)
    data = shingles.apply(lambda group: group.iloc[1:,:])
    shingle_dists = shingles[['dist_calc_m']].sum()
    shingle_times = shingles[['time_calc_s']].sum()
    # Remove (shingles this time) based on final calculation of speeds, distances, times
    invalid_shingles = []
    # Total distance
    invalid_shingles.append(shingle_dists[shingle_dists['dist_calc_m']<=0].shingle_id)
    invalid_shingles.append(shingle_dists[shingle_dists['dist_calc_m']>=20000].shingle_id)
    # Total time
    invalid_shingles.append(shingle_times[shingle_times['time_calc_s']<=0].shingle_id)
    invalid_shingles.append(shingle_times[shingle_times['time_calc_s']>=3*60*60].shingle_id)
    # Invidiual point distance, time, speed
    invalid_shingles.append(data[data['dist_calc_m']<0.0].shingle_id)
    invalid_shingles.append(data[data['time_calc_s']<=0.0].shingle_id)
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
    # Get number of passed stops
    data['passed_stops_n'] = unique_traj['stop_sequence'].diff()
    data['passed_stops_n'] = data['passed_stops_n'].fillna(0)
    # Get cumulative values from trip start
    data['time_cumulative_s'] = unique_traj['time_calc_s'].cumsum()
    data['dist_cumulative_km'] = unique_traj['dist_calc_km'].cumsum()
    data['time_cumulative_s'] = data.time_cumulative_s - unique_traj.time_cumulative_s.transform('min')
    data['dist_cumulative_km'] = data.dist_cumulative_km - unique_traj.dist_cumulative_km.transform('min')
    # Remove shingles that don't traverse more than a kilometer, or have less than n points
    data = data.groupby(['shingle_id']).filter(lambda x: np.max(x.dist_cumulative_km) >= 1.0)
    # Trips that cross over midnight can end up with outlier travel times; there are very few so remove trips over 3hrs
    data = data.groupby(['shingle_id']).filter(lambda x: np.max(x.time_cumulative_s) <= 3000)
    data = data.groupby(['shingle_id']).filter(lambda x: len(x) >= 5)
    return data

def apply_gtfs_timetables(data, gtfs_data, gtfs_folder_date):
    data['gtfs_folder_date'] = gtfs_folder_date
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
        data['x'].values,
        data['y'].values,
        gtfs_data
    )
    data = data.assign(stop_x=closest_stops[:,0])
    data = data.assign(stop_y=closest_stops[:,1])
    data = data.assign(stop_arrival_s=closest_stops[:,3])
    data = data.assign(stop_sequence=closest_stops[:,4])
    data = data.assign(stop_dist_km=closest_stops[:,5]/1000)
    data = data.assign(stop_x_cent=closest_stops[:,5])
    data = data.assign(stop_y_cent=closest_stops[:,6])
    data = data.assign(route_id=closest_stops[:,7])
    data = data.assign(service_id=closest_stops[:,8])
    data = data.assign(direction_id=closest_stops[:,9])

    # Get the timeID_s (for the first point of each trajectory)
    data = pd.merge(data, first_points, on='shingle_id')
    # Calculate the scheduled travel time from the first to each point in the shingle
    data = data.assign(scheduled_time_s=data['stop_arrival_s'] - data['trip_start_timeID_s'])
    # Filter out shingles where the data started after midnight, but the trip started before
    # If the data started before the scheduled time difference is still accurate
    valid_trips = data.groupby('shingle_id').filter(lambda x: x['scheduled_time_s'].max() <= 10000)['shingle_id'].unique()
    data = data[data['shingle_id'].isin(valid_trips)]
    return data

def get_scheduled_arrival(trip_ids, x, y, gtfs_data):
    """
    Find the nearest stop to a set of trip-coordinates, and return the scheduled arrival time.
    trip_ids: list of trip_ids
    lons/lats: lists of places where the bus will be arriving (end point of traj)
    gtfs_data: merged GTFS files
    Returns: (distance to closest stop in km, scheduled arrival time at that stop).
    """
    data = np.column_stack([x, y, trip_ids])
    gtfs_data_ary = gtfs_data[['stop_x','stop_y','trip_id','arrival_s','stop_sequence','stop_x_cent','stop_y_cent','route_id','service_id','direction_id']].values
    # Create dictionary mapping trip_ids to lists of points in gtfs
    id_to_points = {}
    for point in gtfs_data_ary:
        id_to_points.setdefault(point[2],[]).append(point)
    # For each point find the closest stop that shares the trip_id
    results = np.zeros((len(data), 11), dtype=object)
    for i, point in enumerate(data):
        corresponding_points = np.vstack(id_to_points.get(point[2], []))
        point = np.expand_dims(point, 0)
        # Find closest point and add to results
        closest_point_dist, closest_point_idx = shape_utils.get_closest_point(corresponding_points[:,0:2], point[:,0:2])
        closest_point = corresponding_points[closest_point_idx]
        closest_point = np.append(closest_point, closest_point_dist)
        results[i,:] = closest_point
    return results

def get_best_gtfs_lookup(traces, gtfs_folder):
    # Get the most recent GTFS files available corresponding to each unique file in the traces
    gtfs_available = [f for f in os.listdir(gtfs_folder) if not f.startswith('.')]
    gtfs_available = [datetime.strptime(x, "%Y_%m_%d") for x in gtfs_available]
    dates_needed_string = list(pd.unique(traces['file']))
    dates_needed = [datetime.strptime(x[:10], "%Y_%m_%d") for x in dates_needed_string]
    best_gtfs_dates = []
    for fdate in dates_needed:
        matching_gtfs = [x for x in gtfs_available if x < fdate]
        best_gtfs = max(matching_gtfs)
        best_gtfs_dates.append(best_gtfs)
    best_gtfs_dates_string = [x.strftime("%Y_%m_%d") for x in best_gtfs_dates]
    file_to_gtfs_map = {k:v for k,v in zip(dates_needed_string, best_gtfs_dates_string)}
    return file_to_gtfs_map

def clean_trace_df_w_timetables(traces, gtfs_folder, epsg, coord_ref_center):
    """
    Validate a set of tracked bus locations against GTFS.
    data: pandas dataframe with unified bus data
    gtfs_data: merged GTFS files
    Returns: dataframe with only trips that are in GTFS, and are reasonably close to scheduled stop ids.
    """
    # Process each chunk of traces using corresponding GTFS files, load 1 set of GTFS at a time
    # The dates should be in order so that each GTFS file set is loaded only once
    # Also seems best to only run the merge once, with as many dates as possible
    file_to_gtfs_map = get_best_gtfs_lookup(traces, gtfs_folder)
    result = []
    unique_gtfs = pd.unique(list(file_to_gtfs_map.values()))
    for current_gtfs_name in unique_gtfs:
        keys = [k for k,v in file_to_gtfs_map.items() if v==current_gtfs_name]
        current_gtfs_data = merge_gtfs_files(f"{gtfs_folder}{current_gtfs_name}/", epsg, coord_ref_center)
        result.append(apply_gtfs_timetables(traces[traces['file'].isin(keys)].copy(), current_gtfs_data, current_gtfs_name))
    result = pd.concat(result)
    result = result.sort_values(["file","shingle_id","locationtime"], ascending=True)
    return result

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

def map_to_deeptte(trace_data, deeptte_formatted_path, grid_bounds, grid_s_res, grid_t_res, grid_n_res):
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

    # Calculate and add grid features
    ngrid, tbin_idxs, xbin_idxs, ybin_idxs = grids.traces_to_ngrid(trace_data, grid_bounds, grid_s_res=grid_s_res, grid_t_res=grid_t_res, grid_n_res=grid_n_res)
    grids.fill_ngrid_forward(ngrid)
    trace_data['tbin_idx'] = tbin_idxs
    trace_data['xbin_idx'] = xbin_idxs
    trace_data['ybin_idx'] = ybin_idxs

    # Gather by shingle
    groups = trace_data.groupby('shingle_id')

    # Get necessary features as scalar or lists
    result = groups.agg({
        # Cumulative point time/dist
        'time_gap': lambda x: x.tolist(),
        'dist_gap': lambda x: x.tolist(),
        # Trip time/dist
        'time': 'max',
        'dist': 'max',
        # IDs
        'weekID': 'min',
        'timeID': 'min',
        'dateID': 'min',
        'trip_id': 'min',
        'file': 'min',
        'route_id': 'min',
        'service_id': 'min',
        'direction_id': 'min',
        # Individual point time/dist
        'lats': lambda x: x.tolist(),
        'lngs': lambda x: x.tolist(),
        'x': lambda x: x.tolist(),
        'y': lambda y: y.tolist(),
        'x_cent': lambda x: x.tolist(),
        'y_cent': lambda y: y.tolist(),
        'speed_m_s': lambda x: x.tolist(),
        'time_calc_s': lambda x: x.tolist(),
        'dist_calc_km': lambda x: x.tolist(),
        'bearing': lambda x: x.tolist(),
        # Trip start time
        'trip_start_timeID_s': 'min',
        # Trip ongoing time
        'timeID_s': lambda x: x.tolist(),
        # Nearest stop
        'stop_x': lambda x: x.tolist(),
        'stop_y': lambda x: x.tolist(),
        'stop_x_cent': lambda x: x.tolist(),
        'stop_y_cent': lambda x: x.tolist(),
        'stop_dist_km': lambda x: x.tolist(),
        'scheduled_time_s': lambda x: x.tolist(),
        'passed_stops_n': lambda x: x.tolist(),
        # Grid
        'tbin_idx': lambda x: x.tolist(),
        'xbin_idx': lambda x: x.tolist(),
        'ybin_idx': lambda x: x.tolist()
    })

    # Convert the DataFrame to a dictionary with the original format
    result_json_string = result.to_json(orient='records', lines=True)
    with open(deeptte_formatted_path, mode='w+') as out_file:
        out_file.write(result_json_string)
    return trace_data, ngrid

def get_summary_config(trace_data, gtfs_folder, n_save_files, epsg):
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
        "x_mean": np.mean(trace_data['x']),
        "x_std": np.std(trace_data['x']),
        "y_mean": np.mean(trace_data['y']),
        "y_std": np.std(trace_data['y']),
        "x_cent_mean": np.mean(trace_data['x_cent']),
        "x_cent_std": np.std(trace_data['x_cent']),
        "y_cent_mean": np.mean(trace_data['y_cent']),
        "y_cent_std": np.std(trace_data['y_cent']),
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
        "stop_x_mean": np.mean(trace_data['stop_x']),
        "stop_x_std": np.std(trace_data['stop_x']),
        "stop_y_mean": np.mean(trace_data['stop_y']),
        "stop_y_std": np.std(trace_data['stop_y']),
        "stop_x_cent_mean": np.mean(trace_data['stop_x_cent']),
        "stop_x_cent_std": np.std(trace_data['stop_x_cent']),
        "stop_y_cent_mean": np.mean(trace_data['stop_y_cent']),
        "stop_y_cent_std": np.std(trace_data['stop_y_cent']),
        "stop_dist_km_mean": np.mean(trace_data['stop_dist_km']),
        "stop_dist_km_std": np.std(trace_data['stop_dist_km']),
        "scheduled_time_s_mean": np.mean(grouped.max()[['scheduled_time_s']].values.flatten()),
        "scheduled_time_s_std": np.std(grouped.max()[['scheduled_time_s']].values.flatten()),
        "passed_stops_n_mean": np.mean(trace_data['passed_stops_n']),
        "passed_stops_n_std": np.std(trace_data['passed_stops_n']),
        # Not variables
        "n_points": len(trace_data),
        "n_samples": len(grouped),
        "gtfs_folder": gtfs_folder,
        "n_save_files": n_save_files,
        "epsg": epsg,
        "train_set": ["train"+str(x) for x in range(0,n_save_files)],
        "test_set": ["test"+str(x) for x in range(0,n_save_files)]
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

def extract_operator_gtfs(old_folder, new_folder, source_col_trips, source_col_stop_times, op_name):
    """
    First make a copy of the GTFS directory, then this function will overwrite the key files.
    Example SCP with ipv6: scp -6 ./test.file osis@\[2001:db8:0:1\]:/home/osis/test.file
    """
    gtfs_folders = os.listdir(old_folder)
    for file in gtfs_folders:
        if file != ".DS_Store" and len(file)==10:
            z = pd.read_csv(f"{old_folder}{file}/trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
            st = pd.read_csv(f"{old_folder}{file}/stop_times.txt", low_memory=False, dtype=GTFS_LOOKUP)
            z = z[z[source_col_trips].str[:3]==op_name]
            st = st[st[source_col_stop_times].str[:3]==op_name]
            z.to_csv(f"{new_folder}{file}/trips.txt")
            st.to_csv(f"{new_folder}{file}/stop_times.txt")

def merge_gtfs_files(gtfs_folder, epsg, coord_ref_center):
    """
    Join a set of GTFS files into a single dataframe. Each row is a trip + arrival time.
    Returns: pandas dataframe with merged GTFS data.
    """
    z = pd.read_csv(gtfs_folder+"trips.txt", low_memory=False, dtype=GTFS_LOOKUP)
    st = pd.read_csv(gtfs_folder+"stop_times.txt", low_memory=False, dtype=GTFS_LOOKUP)
    sl = pd.read_csv(gtfs_folder+"stops.txt", low_memory=False, dtype=GTFS_LOOKUP)
    z = pd.merge(z,st,on="trip_id")
    z = pd.merge(z,sl,on="stop_id")
    # Calculate stop arrival from midnight
    gtfs_data = z.sort_values(['trip_id','stop_sequence'])
    gtfs_data['arrival_s'] = [int(x[0])*60*60 + int(x[1])*60 + int(x[2]) for x in gtfs_data['arrival_time'].str.split(":")]
    # Resequence stops from 0 with increment of 1
    gtfs_data['stop_sequence'] = gtfs_data.groupby('trip_id').cumcount()
    # Project stop locations to local coordinate system
    default_crs = pyproj.CRS.from_epsg(4326)
    proj_crs = pyproj.CRS.from_epsg(epsg)
    transformer = pyproj.Transformer.from_crs(default_crs, proj_crs, always_xy=True)
    gtfs_data['stop_x'], gtfs_data['stop_y'] = transformer.transform(gtfs_data['stop_lon'], gtfs_data['stop_lat'])
    gtfs_data['stop_x_cent'] = gtfs_data['stop_x'] - coord_ref_center[0]
    gtfs_data['stop_y_cent'] = gtfs_data['stop_y'] - coord_ref_center[1]
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
    fold_results = [x['All_Losses'] for x in model_results]
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
        for model in fold_results['Loss_Curves']:
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
    # Extract train times
    names_df = np.array([x['Model_Names'] for x in model_results]).flatten()
    train_time_df = np.array([x['Train_Times'] for x in model_results]).flatten()
    folds_df = np.array([np.repeat(i,len(model_results[i]['Model_Names'])) for i in range(len(model_results))]).flatten()
    city_df = np.array(np.repeat(city,len(folds_df))).flatten()
    train_time_df = pd.DataFrame({
        "City": city_df,
        "Fold": folds_df,
        "Model":  names_df,
        "Time": train_time_df
    })
    return result_df, loss_df, train_time_df

def extract_gen_results(city, gen_results):
    # Extract generalization results
    res = []
    experiments = ["Train_Losses","Test_Losses","Holdout_Losses","Tune_Train_Losses","Tune_Test_Losses","Extract_Train_Losses","Extract_Test_Losses"]
    for ex in experiments:
        fold_results = [x[ex] for x in gen_results]
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
        gen_df = pd.DataFrame({
            "Model": models,
            "City": cities,
            "Loss": ex,
            "Fold": fold_nums,
            "MAPE": mapes,
            "RMSE": rmses,
            "MAE": maes
        })
        res.append(gen_df)
    return pd.concat(res, axis=0)

def extract_deeptte_results(city, run_folder, network_folder, generalization_flag=False):
    # Extract all fold and epoch losses from deeptte run
    all_run_data = []
    if generalization_flag:
        dest_dir = f"{run_folder}{network_folder}deeptte_results/generalization"
    else:
        dest_dir = f"{run_folder}{network_folder}deeptte_results/result"
    for res_file in os.listdir(dest_dir):
        res_preds = pd.read_csv(
            f"{dest_dir}/{res_file}",
            delimiter=" ",
            header=None,
            names=["Label", "Pred"],
            dtype={"Label": float, "Pred": float}
        )
        res_labels = res_file.split("_")
        if generalization_flag:
            # model_name, test_file_name, fold_num, epoch_num = res_labels
            _, _, test_file_name, _, _ = res_labels
            fold_num = 0
            epoch_num = 0
        elif len(res_labels)==5:
            model_name, test_file_name, test_file_num, fold_num, epoch_num = res_labels
            test_file_name = test_file_name + "_" + test_file_num
            epoch_num = epoch_num.split(".")[0]
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
            seq_lens.append(data[-1])
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