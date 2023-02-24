"""
Functions for processing tracked bus and timetable data.
"""

from datetime import date, datetime, timedelta
import itertools
import json
from math import degrees, radians, atan2, cos, sin, asin, sqrt
import os
from random import sample
from sklearn import metrics

import numpy as np
import pandas as pd
import pickle
from zipfile import ZipFile

from database import shape_utils


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

def calculate_bearing(pos1, pos2):
    """
    Calculate the bearing between two GPS coordinates.
    lons1/lats1: arrays of coordinates for the end points
    lons2/lats2: arrays of coordinates for the start points
    Returns: array of distances in meters.
    """
    # end_lon, end_lat = pos1
    # start_lon, start_lat = pos2
    bearings = []
    for i in range(0, pos1.shape[0]):
        end_lon, end_lat = pos1[i]
        start_lon, start_lat = pos2[i]
        # Convert to radians
        start_lat, start_lon, end_lat, end_lon = map(radians, [start_lat, start_lon, end_lat, end_lon])
        # Calculate differences
        d_lon = end_lon - start_lon
        # Calculate bearing
        y = sin(d_lon) * cos(end_lat)
        x = cos(start_lat) * sin(end_lat) - sin(start_lat) * cos(end_lat) * cos(d_lon)
        bearing = atan2(y, x)
        # Convert to degrees
        bearing = degrees(bearing)
        # Normalize to 0-360
        if bearing < 0:
            bearing += 360
        bearings.append(np.round(bearing,1))
    return bearings

def spherical_dist(pos1, pos2, r=6371000):
    """
    Calculate spherical distance between two coordinates.
    pos1/pos2: 2d array with lon and lat for start/end points
    r: radius of earth in desired units
    Returns: array of distances.
    """
    # r is in meters (6371000)
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

def calculate_gps_dist(end_lat, end_lon, start_lat, start_lon):
    """
    Calculate the Haversine distance between a series of points.
    lons1/lats1: arrays of coordinates for the end points
    lons2/lats2: arrays of coordinates for the start points
    Returns: array of distances in meters.
    """
    end_points = np.array((end_lon, end_lat)).T
    start_points = np.array((start_lon, start_lat)).T
    return spherical_dist(end_points, start_points), calculate_bearing(end_points, start_points)

def calculate_trip_speeds(data):
    """
    Calculate speeds between consecutive trip locations.
    data: pandas dataframe of bus data with unified columns
    Returns: array of speeds, dist_diff, time_diff between consecutive points.
    Nan for first point of a trip.
    """
    x = data[['trip_id','lat','lon','locationtime']]
    y = data[['trip_id','lat','lon','locationtime']].shift()
    y.columns = [colname+"_shift" for colname in y.columns]
    z = pd.concat([x,y], axis=1)
    z['dist_diff'], z['bearing'] = calculate_gps_dist(z['lon'], z['lat'], z['lon_shift'], z['lat_shift'])
    z['time_diff'] = z['locationtime'] - z['locationtime_shift']
    z['speed_m_s'] = z['dist_diff'] / z['time_diff']
    return z['speed_m_s'].values.flatten(), z['dist_diff'].values.flatten(), z['time_diff'].values.flatten(), z['bearing'].values.flatten()

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
    data = data.sort_values(['file','trip_id','locationtime'], ascending=True)
    return data, no_data_list

def calculate_trace_df(data, timezone):
    """
    Calculate difference in metrics between two consecutive trip points.
    data: pandas df with all bus trips
    timezone: string for timezone the data were collected in
    Returns: combination of original point values, and new _diff values
    """
    # Gets speeds between consecutive points, drop first points, filter
    data['speed_m_s'], data['dist_calc_m'], data['time_calc_s'], data['bearing'] = calculate_trip_speeds(data)
    data = data[data['dist_calc_m']>0.0]
    data = data[data['dist_calc_m']<5000.0]
    data = data[data['time_calc_s']>0.0]
    data = data[data['time_calc_s']<120.0]
    data = data[data['speed_m_s']>0.0]
    data = data[data['speed_m_s']<35.0]
    data['dist_calc_km'] = data['dist_calc_m'] / 1000.0
    data = data.dropna()
    # Get unique trajectories in pd groupby, only keep trajectories with at least 10 points
    data = data.groupby(['file','trip_id']).filter(lambda x: len(x) > 10)
    unique_traj = data.groupby(['file','trip_id'])
    # Get cumulative values from trip start
    data['time_cumulative_s'] = data.locationtime - unique_traj.locationtime.transform('min')
    data['dist_cumulative_km'] = unique_traj['dist_calc_km'].cumsum()
    data['dist_cumulative_km'] = data.dist_cumulative_km - unique_traj.dist_cumulative_km.transform('min')
    # Time values for deeptte
    data['datetime'] = pd.to_datetime(data['locationtime'], unit='s', utc=True).map(lambda x: x.tz_convert(timezone))
    data['dateID'] = (data['datetime'].dt.day)
    data['weekID'] = (data['datetime'].dt.dayofweek)
    # (be careful with these last two as they change across the trajectory)
    data['timeID'] = (data['datetime'].dt.hour * 60) + (data['datetime'].dt.minute)
    data['timeID_s'] = (data['datetime'].dt.hour * 60 * 60) + (data['datetime'].dt.minute * 60) + (data['datetime'].dt.second)
    return data

def clean_trace_df_w_timetables(data, gtfs_data):
    """
    Validate a set of tracked bus locations against GTFS.
    data: pandas dataframe with unified bus data
    gtfs_data: merged GTFS files
    Returns: dataframe with only trips that are in GTFS, and are reasonably close to scheduled stop ids.
    """
    # Remove any trips that are not in the GTFS
    data = data[data['trip_id'].astype(str).isin(gtfs_data.trip_id)].copy()
    # Save start time of first points in trajectories
    first_points = data.groupby('shingle_id').first().reset_index()[['shingle_id','timeID_s']]
    first_points.columns = ['shingle_id','trip_start_timeID_s']
    closest_stops = get_scheduled_arrival(
        data['trip_id'].values,
        data['lon'].values,
        data['lat'].values,
        gtfs_data
    )
    data['stop_dist_km'], data['stop_arrival_s'], data['stop_lon'], data['stop_lat'] = closest_stops
    # Get the timeID_s (for the first point of each trajectory)
    data = pd.merge(data, first_points, on='shingle_id')
    # Calculate the scheduled travel time from the first to each point in the shingle
    data['scheduled_time_s'] = data['stop_arrival_s'] - data['trip_start_timeID_s']
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
    results = []
    for point in data:
        corresponding_points = np.vstack(id_to_points.get(point[2], []))
        point = np.expand_dims(point, 0)
        # Find closest point and add to results
        closest_point_dist, closest_point_idx = shape_utils.get_closest_point(corresponding_points[:,0:2], point[:,0:2])
        closest_point = corresponding_points[closest_point_idx]
        closest_point = np.append(closest_point, closest_point_dist * 111)
        results.append(closest_point)
    results = np.vstack(results)
    return results[:,4], results[:,3], results[:,0], results[:,1]

def remap_vehicle_ids(df_list):
    """
    Remap each vehicle ID in all dfs to start from 0, maintaining order.
    df_list: list of pandas dataframes with unified bus data
    Returns: list of pandas dataframes with new column for vehicle_id_recode.
    """
    all_vehicle_ids = pd.concat([x['vehicle_id'] for x in df_list]).values.flatten()
    # Recode vehicle ids to start from 0
    mapping = {v:k for k,v in enumerate(set(all_vehicle_ids))}
    for df in df_list:
        recode = [mapping[y] for y in df['vehicle_id'].values.flatten()]
        df['vehicle_id_recode'] = recode
    return (df_list, len(pd.unique(all_vehicle_ids)))

def map_to_deeptte(trace_data):
    """
    Reshape pandas dataframe to the json format needed to use deeptte.
    trace_data: dataframe with bus trajectories
    Returns: path to json file where deeptte trajectories are saved.
    """    
    # Group by the desired column
    groups = trace_data.groupby('shingle_id')
    # Create an empty dictionary to store the JSON data
    result = {}
    for name, group in groups:
        result[name] = {
            # DeepTTE Features
            'time_gap': group['time_cumulative_s'].tolist(),
            'dist_gap': group['dist_cumulative_km'].tolist(),
            'dist': max(group['dist_cumulative_km']),
            'lats': group['lat'].tolist(),
            'lngs': group['lon'].tolist(),
            'driverID': min(group['vehicle_id_recode']),
            'weekID': min(group['weekID']),
            'timeID': min(group['timeID']),
            'dateID': min(group['dateID']),
            # Other Features
            'trip_id': min(group['trip_id']),
            'file': min(group['file']),
            'speed_m_s': group['speed_m_s'].tolist(),
            'time_calc_s': group['time_calc_s'].tolist(),
            'dist_calc_km': group['dist_calc_km'].tolist(),
            'trip_start_timeID_s': min(group['trip_start_timeID_s']),
            'timeID_s': group['timeID_s'].tolist(),
            'stop_lat': group['stop_lat'].tolist(),
            'stop_lon': group['stop_lon'].tolist(),
            'stop_dist_km': group['stop_dist_km'].tolist(),
            'scheduled_time_s': group['scheduled_time_s'].tolist(),
            # Labels
            'time': max(group['time_cumulative_s'].tolist())
        }
    return result

def get_summary_config(trace_data, n_unique_veh, gtfs_folder, n_folds):
    """
    Get a dict of means and sds which are used to normalize data by DeepTTE.
    trace_data: pandas dataframe with unified columns and calculated distances
    Returns: dict of mean and std values, as well as train/test filenames.
    """
    # config.json
    summary_dict = {
        # DeepTTE
        'time_gap_mean': np.mean(trace_data['time_calc_s']),
        'time_gap_std': np.std(trace_data['time_calc_s']),
        'dist_gap_mean': np.mean(trace_data['dist_calc_km']),
        'dist_gap_std': np.std(trace_data['dist_calc_km']),
        "dist_mean": np.mean(trace_data.groupby(['shingle_id']).max()[['dist_cumulative_km']].values.flatten()),
        'dist_std': np.std(trace_data.groupby(['shingle_id']).max()[['dist_cumulative_km']].values.flatten()),
        'lngs_mean': np.mean(trace_data['lon']),
        'lngs_std': np.std(trace_data['lon']),
        'lats_mean': np.mean(trace_data['lat']),
        "lats_std": np.std(trace_data['lat']),
        "time_mean": np.mean(trace_data.groupby(['shingle_id']).max()[['time_cumulative_s']].values.flatten()),
        "time_std": np.std(trace_data.groupby(['shingle_id']).max()[['time_cumulative_s']].values.flatten()),
        # Others
        "speed_m_s_mean": np.mean(trace_data['speed_m_s']),
        "speed_m_s_std": np.std(trace_data['speed_m_s']),
        "stop_dist_km_mean": np.mean(trace_data['stop_dist_km']),
        "stop_dist_km_std": np.std(trace_data['stop_dist_km']),
        "scheduled_time_s_mean": np.mean(trace_data.groupby(['shingle_id']).max()[['scheduled_time_s']].values.flatten()),
        "scheduled_time_s_std": np.std(trace_data.groupby(['shingle_id']).max()[['scheduled_time_s']].values.flatten()),
        "stop_lng_mean": np.mean(trace_data['stop_lon']),
        "stop_lng_std": np.std(trace_data['stop_lon']),
        "stop_lat_mean": np.mean(trace_data['stop_lat']),
        "stop_lat_std": np.std(trace_data['stop_lat']),
        # Not variables
        "n_unique_veh": n_unique_veh,
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
            with open(old_folder+'/'+file, 'rb') as f:
                data = pickle.load(f)
                data = data[data[source_col]==op_name]
            with open(f"{new_folder}/{file}", 'wb') as f:
                pickle.dump(data, f)

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
    # Extract fold losses for all models
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
    # Extract FF loss curves
    loss_df = []
    fold_train_losses = [x['FF Train Losses'] for x in model_results]
    fold_test_losses = [x['FF Valid Losses'] for x in model_results]
    for fold_num in range(0,len(fold_train_losses)):
        df_train = pd.DataFrame({
            "City": city,
            "Fold": fold_num,
            "Loss Set": "Train",
            "Epoch": np.arange(len(fold_train_losses[0])),
            "Loss": fold_train_losses[fold_num]
        })
        df_test = pd.DataFrame({
            "City": city,
            "Fold": fold_num,
            "Loss Set": "Test",
            "Epoch": np.arange(len(fold_train_losses[0])),
            "Loss": fold_test_losses[fold_num]
        })
        loss_df.append(df_train)
        loss_df.append(df_test)
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
    Each split comes from a group representing a trajector in the dataframe.
    trace_df: unified dataframe
    min_len: minimum number of chunks to split a trajectory into.
    max_lan: maximum number of chunks to split a trajectory into.
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
    throwouts = z.groupby(['shingle_id']).count()['lat'].reset_index()
    throwouts = throwouts[throwouts['lat']<3]['shingle_id'].values
    z = z[~z['shingle_id'].isin(throwouts)]
    return z

# def extract_validation():
#     # Extract zip files of all validation tracks in folder
#     folder = "../data/kcm_validation_sensor/"
#     for file in os.listdir(folder):
#         if file != ".DS_Store" and file[-4] == ".":
#             with ZipFile(folder+file, 'r') as zip:
#                 zip.extractall(folder+file[:-4])

#     # Combine all validation data into a dict, filenames are the keys
#     folder = "../data/kcm_validation_sensor/"
#     validation_data_lookup = {}
#     for file in os.listdir(folder):
#         if file != ".DS_Store" and file[-4:] != ".zip":
#             df = [pd.read_csv(folder+file+"/"+validation_file) for validation_file in os.listdir(folder+file)]
#             validation_data_lookup[file] = df

#     # Show available keys
#     validation_data_lookup.keys()