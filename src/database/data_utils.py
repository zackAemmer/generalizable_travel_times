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

def get_unique_line_geometries(shape_data):
    """
    Combine points into line segments, limit to only unique line segments, calculate their geometry.
    shape_data: dataframe containing columns for point 1, and a shifted point 2.
    Returns: geodataframe with unique segments, all having a line geometry
    """
    # Keep record of all unique segments
    shape_segment_list = shape_data.drop_duplicates(['segment_id'])
    # Get line geometries for each segment
    # Each segment ID should have two rows; point 1 and point 2
    shape_segment_list_copy = shape_segment_list.copy()
    shape_segment_list = geopandas.GeoDataFrame(shape_segment_list, geometry=geopandas.points_from_xy(shape_segment_list.shape_pt_lon, shape_segment_list.shape_pt_lat))
    shape_segment_list_copy = geopandas.GeoDataFrame(shape_segment_list_copy, geometry=geopandas.points_from_xy(shape_segment_list_copy.shape_pt_lon_shift, shape_segment_list_copy.shape_pt_lat_shift))
    segment_shapes = pd.concat([shape_segment_list, shape_segment_list_copy], axis=0).sort_values('segment_id')
    # Join each point-set across the two segment-rows into a line
    segment_shapes = segment_shapes.groupby(['segment_id'])['geometry'].apply(lambda x: shapely.geometry.LineString(x.tolist()))
    segment_shapes = geopandas.GeoDataFrame(segment_shapes, geometry='geometry', crs="EPSG:4326")
    segment_shapes.reset_index(inplace=True)
    return segment_shapes

def create_shape_segment_lookup(shape_data):
    """
    For each unique line segment in the route shapes, get a lookup for the shape_ids that use that segment.
    shape_data: dataframe with 'segment_id' column with unique segment ids, 'shape_id' column with unique shape ids
    Returns: a lookup table that maps each unique 'segment_id' to a list of corresponding shape_ids
    """
    route_segment_lookup = {}
    for i, shape in shape_data.iterrows():
        if shape.loc['segment_id'] not in route_segment_lookup.keys():
            route_segment_lookup[shape.loc['segment_id']] = [shape.loc['shape_id']]
        else:
            route_segment_lookup[shape.loc['segment_id']].append(shape.loc['shape_id'])
    return route_segment_lookup

def get_consecutive_values(shape_data):
    """
    Break dataframe of consecutive points from GTFS shapes.txt into line segments.
    shape_data: dataframe loaded from a GTFS shapes.txt file, points should be consecutively ordered
    Returns: dataframe with each row having the current, and previous point lat/lon information
    """
    route_shape_data = shape_data.copy()
    # Keep lat/lon as strings to use as id
    route_shape_data['shape_pt_lat_str'] = route_shape_data['shape_pt_lat'].astype(str)
    route_shape_data['shape_pt_lon_str'] = route_shape_data['shape_pt_lon'].astype(str)
    route_shape_data['point_id'] = route_shape_data['shape_pt_lat_str'] + '_' + route_shape_data['shape_pt_lon_str']
    # Get a segment id for each consecutive set of points
    route_shape_data_shift = route_shape_data.shift(fill_value='blank').iloc[1:,:]
    route_shape_data_shift.columns = [col+"_shift" for col in route_shape_data_shift.columns]
    route_shape_data = pd.concat([route_shape_data[['shape_id','shape_pt_lat','shape_pt_lon','point_id']], route_shape_data_shift[['shape_id_shift','shape_pt_lat_shift','shape_pt_lon_shift','point_id_shift']]], axis=1).dropna()
    route_shape_data = route_shape_data[route_shape_data['shape_id']==route_shape_data['shape_id_shift']]
    # Unique id for each lat/lon -> lat/lon segment
    route_shape_data['segment_id'] = route_shape_data['point_id'] + '_' + route_shape_data['point_id_shift']
    return route_shape_data

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

def map_to_network(traces, segments, n_sample):
    # Sample from the traces, match to the network
    traces_n = traces.sample(n_sample)
    # There are duplicates when distance to segments is tied; just take first
    traces_n = geopandas.sjoin_nearest(traces_n, segments, distance_col="join_dist").drop_duplicates(['tripid','locationtime'])
    return traces_n

def merge_gtfs_files(gtfs_folder):
    z = pd.read_csv(gtfs_folder+"trips.txt", low_memory=False)
    st = pd.read_csv(gtfs_folder+"stop_times.txt", low_memory=False)
    sl = pd.read_csv(gtfs_folder+"stops.txt", low_memory=False)
    z = pd.merge(z,st,on="trip_id")
    z = pd.merge(z,sl,on="stop_id")
    gtfs_data = z.sort_values(['trip_id','stop_sequence'])
    return gtfs_data