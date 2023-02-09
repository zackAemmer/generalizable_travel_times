"""
Functions for processing and working with tracked bus data.
"""

from datetime import date, timedelta
from math import radians, cos, sin, asin, sqrt
import os
from random import sample

import geopandas
import numpy as np
import pandas as pd
import pickle
import shapely.geometry


def recode_nums(ary):
    old_codes = np.sort(ary).astype(int)
    new_codes = np.arange(0,len(old_codes))
    return dict(zip(old_codes, new_codes))

def calculate_gps_dist(lons1, lats1, lons2, lats2):
    end_points = np.array((lons1, lats1)).T
    start_points = np.array((lons2, lats2)).T
    return spherical_dist(end_points, start_points)

def spherical_dist(pos1, pos2, r=6371000):
    pos1 = pos1 * np.pi / 180
    pos2 = pos2 * np.pi / 180
    cos_lat1 = np.cos(pos1[..., 0])
    cos_lat2 = np.cos(pos2[..., 0])
    cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
    cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
    return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

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

def combine_specific_folder_data(folder, file_list, given_names, feature_lookup):
    """
    Combine all the daily .pkl files containing feed data into a single dataframe.
    folder: the folder to search in
    file_list: the file names to read and combine
    given_names: list of the names of the features in the raw data
    feature_lookup: dict mapping feature names : data types
    Returns: a dataframe of all data concatenated together, a column 'file' is added, also a list of all dates with no data.
    """
    data_list = []
    no_data_list = []
    for file in file_list:
        try:
            with open(folder+'/'+file, 'rb') as f:
                data = pickle.load(f)
                data['file'] = file
                # Get unified column names, data types
                data = data[given_names]
                data.columns = feature_lookup.keys()
                data = data.astype(feature_lookup)
                data_list.append(data)
        except FileNotFoundError:
            no_data_list.append(file)
    data = pd.concat(data_list, axis=0)
    data = data.sort_values(['file','trip_id','locationtime'], ascending=True)
    return data, no_data_list

def combine_all_folder_data(folder, given_names, feature_lookup, n_sample=None):
    """
    Combine all the daily .pkl files containing feed data into a single dataframe.
    folder: the folder full of .pkl files to combine
    given_names: list of the names of the features in the raw data
    feature_lookup: dict mapping feature names : data types
    n_sample: the number of files to sample from the folder (all if false)
    Returns: a dataframe of all data concatenated together, a column 'file' is added
    """
    data_list = []
    if n_sample is not None:
        files = sample(os.listdir(folder), n_sample)
    else:
        files = os.listdir(folder)
    for file in files:
        if file != ".DS_Store":
            with open(folder+'/'+file, 'rb') as f:
                data = pickle.load(f)
                # Get unified column names, data types
                data = data[given_names]
                data.columns = feature_lookup.keys()
                data = data.astype(feature_lookup)
                data['file'] = file
                data_list.append(data)
    data = pd.concat(data_list, axis=0)
    data = data.sort_values(['file','trip_id','locationtime'],ascending=True).dropna()
    return data

def calculate_trace_df(data, timezone):
    """
    Calculate difference in metrics between two consecutive trip points.
    data: pandas df with all bus trips
    timezone: string for timezone the data were collected in
    Returns: combination of original point values, and new _diff values
    """
    # Calculate differences between consecutive locations; drops first point in every trajectory
    shifted = data[['file','trip_id','locationtime','lat','lon']].groupby(['file','trip_id']).shift()
    shifted.columns = [colname+'_prev' for colname in shifted.columns]
    traces = pd.concat([data, shifted], axis=1).dropna()
    # Shifting makes everything a float; revert back where necessary
    traces['locationtime_prev'] = traces['locationtime_prev'].astype(int)
    # Calculate GPS distance
    end_points = np.array((traces.lon, traces.lat)).T
    start_points = np.array((traces.lon_prev, traces.lat_prev)).T
    traces['dist_calc_m'] = spherical_dist(end_points, start_points)
    traces['dist_calc_km'] = traces['dist_calc_m'] / 1000.0
    # Calculate and remove speeds that are unreasonable
    traces['time_calc_s'] = traces['locationtime'] - traces['locationtime_prev']
    traces['speed_m_s'] = traces['dist_calc_m'] / traces['time_calc_s']
    traces = traces.loc[traces['speed_m_s']>0]
    traces = traces.loc[traces['speed_m_s']<35]
    # Only keep trajectories with at least 10 points
    traces = traces.groupby(['file','trip_id']).filter(lambda x: len(x) > 10)
    # Get cumulative values from trip start
    traces['time_cumulative_s'] = traces.locationtime - traces.groupby(['file','trip_id']).locationtime.transform('min')
    traces['dist_cumulative_km'] = traces.groupby(['file','trip_id'])['dist_calc_km'].cumsum()
    traces['dist_cumulative_km'] = traces.dist_cumulative_km - traces.groupby(['file','trip_id']).dist_cumulative_km.transform('min')
    # Time values for deeptte
    traces['datetime'] = pd.to_datetime(traces['locationtime'], unit='s', utc=True).map(lambda x: x.tz_convert(timezone))
    traces['dateID'] = (traces['datetime'].dt.day)
    traces['weekID'] = (traces['datetime'].dt.dayofweek)
    traces['timeID'] = (traces['datetime'].dt.hour * 60) + (traces['datetime'].dt.minute)
    return traces

def clean_trace_df_w_timetables(trace_df, gtfs_folder):
    # Read in GTFS files
    gtfs_data = merge_gtfs_files(gtfs_folder)
    # Remove any trips that are not in the GTFS
    trace_df = trace_df[trace_df['trip_id'].isin(gtfs_data.trip_id)]
    # Match the first data point of each trajectory to first stop in its trip
    first_points_and_stops = pd.merge(trace_df.drop_duplicates(['file','trip_id']), gtfs_data.drop_duplicates(['trip_id']), on='trip_id')
    end_points = np.array((first_points_and_stops.lon, first_points_and_stops.lat)).T
    start_points = np.array((first_points_and_stops.stop_lon, first_points_and_stops.stop_lat)).T
    first_points_and_stops['stop_distances_m'] = spherical_dist(end_points, start_points)
    # Remove the 95th percentile of distances and up
    first_points_and_stops = first_points_and_stops[first_points_and_stops['stop_distances_m'] < np.quantile(first_points_and_stops['stop_distances_m'], .95)]
    first_points_and_stops = first_points_and_stops[['file','trip_id']].drop_duplicates()
    trace_df = pd.merge(trace_df, first_points_and_stops, on=['file','trip_id'])    
    return trace_df

def map_to_deeptte(trace_data):
    """
    Reshape pandas dataframe to the json format needed to use deeptte.
    trace_data: dataframe with bus trajectories
    Returns: path to json file where deeptte trajectories are saved
    """
    # group by the desired column
    groups = trace_data.groupby(['file', 'trip_id'])
    # create an empty dictionary to store the JSON data
    result = {}
    for name, group in groups:
        result[name] = {
            'time_gap': group['time_cumulative_s'].tolist(),
            'dist': max(group['dist_cumulative_km']),
            'lats': group['lat'].tolist(),
            'driverID': max(group['vehicle_id_recode']),
            'weekID': max(group['weekID']),
            'states': [1.0 for x in group['vehicle_id_recode']],
            'timeID': max(group['timeID']),
            'dateID': max(group['dateID']),
            'time': max(group['time_cumulative_s'].tolist()),
            'lngs': group['lon'].tolist(),
            'dist_gap': group['dist_cumulative_km'].tolist()
        }
    return result

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

def extract_operator():    
    folder = '../data/nwy_all/'
    new_folder = '../data/atb_all/'
    files = os.listdir(folder)
    for file in files:
        if file != ".DS_Store":
            with open(folder+'/'+file, 'rb') as f:
                data = pickle.load(f)
                data = data[data['datasource']=='ATB']
            with open(f"{new_folder}/{file}", 'wb') as f:
                pickle.dump(data, f)

def merge_gtfs_files(gtfs_folder):
    z = pd.read_csv(gtfs_folder+"trips.txt", low_memory=False)
    st = pd.read_csv(gtfs_folder+"stop_times.txt", low_memory=False)
    sl = pd.read_csv(gtfs_folder+"stops.txt", low_memory=False)
    z = pd.merge(z,st,on="trip_id")
    z = pd.merge(z,sl,on="stop_id")
    gtfs_data = z.sort_values(['trip_id','stop_sequence'])
    return gtfs_data

def gtfs_to_graph(self):
    # Read in GTFS data, get travel times
    gtfs_data = merge_gtfs_files("../data/kcm_gtfs/2022_09_19/")[['trip_id','stop_id','arrival_time']]
    gtfs_data['arrival_s'] = [int(x[0])*60*60 + int(x[1])*60 + int(x[2]) for x in gtfs_data['arrival_time'].str.split(":")]
    gtfs_data_shifted = gtfs_data.groupby(['trip_id']).shift()
    gtfs_data_shifted.columns = [x+"_shift" for x in gtfs_data_shifted.columns]
    gtfs_data = pd.concat([gtfs_data, gtfs_data_shifted], axis=1).dropna()
    gtfs_data['travel_time_s'] = gtfs_data['arrival_s'] - gtfs_data['arrival_s_shift']
    gtfs_data = gtfs_data.sort_values(['trip_id','stop_id','arrival_time'])
    # Recode nodes to start from 0
    node_ids = np.unique(pd.concat([gtfs_data['stop_id'], gtfs_data['stop_id_shift']], axis=0))
    node_recode_dict = recode_nums(node_ids)
    # Get edge connections, and recode to match new node IDs
    edges = gtfs_data[['stop_id_shift','stop_id','travel_time_s']].astype(int)
    edges = edges.groupby(['stop_id_shift','stop_id']).mean().reset_index()
    edges.columns = ['start_node','end_node','weight']
    edges_data = edges.values[:,:2].T
    edges_data = np.vectorize(node_recode_dict.get)(edges_data)
    # Create graph
    edge_index = edges_data
    edge_attr = edges.values[:,2].reshape(edges.shape[0],1)
    x = np.random.random(node_ids.shape[0]).reshape(node_ids.shape[0],1)
    return {"x":x, "edge_index":edge_index, "edge_attr":edge_attr, "node_recode_dict":node_recode_dict}