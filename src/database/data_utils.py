#!/usr/bin/env python3
"""Functions for processing and working with tracked bus data.
"""


from datetime import datetime, timezone, timedelta
import json
from math import radians, cos, sin, asin, sqrt
import os
from random import sample
import requests
from zipfile import ZipFile

import geopandas
import numpy as np
import pandas as pd
import pickle
import shapely.geometry

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = float(lon1)
    lon2 = float(lon2)
    lat1 = float(lat1)
    lat2 = float(lat2)
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r * 1000

def get_validation_dates(validation_path):
    dates = []
    files = os.listdir(validation_path)
    for file in files:
        labels = file.split("-")
        dates.append(labels[2] + "-" + labels[3] + "-" + labels[4].split("_")[0])
    return dates

def extract_validation_trips(validation_path):
    files = os.listdir(validation_path)
    for file in files:
        labels = file.split("-")
        vehicle_id = labels[0]
        route_num = labels[1]
        year = labels[2]
        month = labels[3]
        day = labels[4].split("_")[0]
        hour = labels[4].split("_")[1]
        minute = labels[5]
        day_data = pd.read_csv(f"./data/kcm_validation_tracks/{labels}")
        with open('10M.pkl', 'rb') as f:
            df = pickle.load(f)
    # Get the tracks for the route id and time of the validation data +/- amount
    return vehicle_id

def combine_all_folder_data(folder, n_sample=None):
    """
    Combine all the daily .pkl files containing feed data into a single dataframe.
    folder: the folder full of .pkl files to combine
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
                data['file'] = file
                data_list.append(data)
    data = pd.concat(data_list, axis=0)
    return data

def calculate_trace_df(data, file_col, tripid_col, locid_col, lat_col, lon_col, diff_cols, use_coord_dist=False):
    """
    Calculate difference in metrics between two consecutive trip points.
    data: pandas df with all bus trips
    file_col: column name that separates each pkl file
    tripid_col: column name that has unique trip ids
    locid_col: column name that has the time of the observation
    lat_col: column name with latitude
    lon_col: column name with longitude
    diff_cols: list of column names to calculate metrics across
    use_coord_dist: whether or not to calculate lat/lon distances vs odometer
    Returns: combination of original point values, and new _diff values
    """
    # Deal with cases where strings are passed
    for col in diff_cols:
        data[col] = data[col].astype(float)
    # Get in order by locationtime
    df = data.sort_values([file_col, tripid_col, locid_col], ascending=True)
    # Calculate differences between consecutive locations
    diff = df.groupby([file_col, tripid_col])[diff_cols].diff()
    diff.columns = [colname+'_diff' for colname in diff.columns]
    traces = pd.concat([df, diff], axis=1)
    if use_coord_dist:
        shift = df.groupby([file_col, tripid_col]).shift()
        hav = pd.concat([df[[lat_col,lon_col]], shift[[lat_col,lon_col]]], axis=1)
        hav.columns = ["lat1","lon1","lat2","lon2"]
        traces['dist_calc'] = hav.apply(lambda x: haversine(x.lon1, x.lat1, x.lon2, x.lat2), axis=1)
    traces.dropna(inplace=True)

    # Calculate and remove speeds that are unreasonable
    traces['speed_m_s'] = traces['dist_calc'] / traces['locationtime_diff']
    traces = traces.loc[traces['speed_m_s']>0]
    traces = traces.loc[traces['speed_m_s']<35]
    traces['dist_calc_km'] = traces['dist_calc'] / 1000.0
    # Get time from trip start
    traces['time_cumulative'] = traces.groupby(['file','tripid'])['locationtime'].transform(lambda x: x - x.iloc[0])
    traces['dist_cumulative'] = traces.groupby(['file','tripid'])['dist_calc_km'].cumsum()
    traces['dist_cumulative'] = traces.groupby(['file','tripid'])['dist_cumulative'].transform(lambda x: x - x.iloc[0])
    # Only keep trajectories with at least 10 points
    traces = traces.groupby(['file','tripid']).filter(lambda x: len(x) > 10)
    # Time values for deeptte
    traces['datetime'] = (pd.to_datetime(traces['locationtime'], unit='s')) 
    traces['dateID'] = (traces['datetime'].dt.day)
    traces['weekID'] = (traces['datetime'].dt.dayofweek)
    traces['timeID'] = (traces['datetime'].dt.hour * 60) + (traces['datetime'].dt.minute)

    # Recode vehicle id to start from 0
    mapping = {v:k for k,v in enumerate(set(traces['vehicleid'].values.flatten()))}
    recode = [mapping[y] for y in traces['vehicleid'].values.flatten()]
    traces['vehicleid_recode'] = recode
    return traces

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
    groups = trace_data.groupby(['file','tripid'])
    # create an empty dictionary to store the JSON data
    result = {}
    for name, group in groups:
        result[name] = {
            'time_gap': group['time_cumulative'].tolist(),
            'dist': max(group['dist_cumulative']),
            'lats': group['lat'].tolist(),
            'driverID': max(group['vehicleid_recode']),
            'weekID': max(group['weekID']),
            'states': [1.0 for x in group['vehicleid']],
            'timeID': max(group['timeID']),
            'dateID': max(group['dateID']),
            'time': max(group['time_cumulative'].tolist()),
            'lngs': group['lon'].tolist(),
            'dist_gap': group['dist_cumulative'].tolist()
        }
    return result