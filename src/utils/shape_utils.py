"""
Functions for processing and working with tracked bus data.
"""

from datetime import date, timedelta
from math import radians, cos, sin, asin, sqrt
import os
from random import sample

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shapely
import shapely.geometry
from scipy.spatial import KDTree
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def upscale(rast, scalar_dims):
    scalars = [np.ones((x,x), dtype=float) for x in scalar_dims]
    res = [np.kron(rast[x], scalars[x]) for x in range(len(scalar_dims))]
    return res

def get_points_within_dist(points, query_points, distance):
    """
    Get unique indices in points for all that are within distance of a query point.
    """
    tree = KDTree(points)
    idxs = tree.query_ball_point(query_points, distance)
    flat_list = [item for sublist in idxs for item in sublist]
    return flat_list

def get_closest_point(points, query_points):
    tree = KDTree(points)
    dists, idxs = tree.query(query_points)
    return dists, idxs

def decompose_vector(scalars, bearings, data_to_attach=None):
    """
    Break speed vector into its x and y components.
    scalars: array of speeds
    bearings: azimuth from north, negative for westward movement, positive for east
    data_to_attach: additional columns to keep with the lat/lon/decomposed values
    Returns: array of x +/-, y +/- scalar components.
    """
    # Use bearing to break value into 
    x = np.round(np.sin(bearings * np.pi/180) * scalars, 1)
    y = np.round(np.cos(bearings * np.pi/180) * scalars, 1)
    x_all = np.column_stack([x, data_to_attach])
    y_all = np.column_stack([y, data_to_attach])
    # Include 0.0 observations in both channels
    x_pos = x_all[x>=0.0]
    x_neg = x_all[x<=0.0]
    y_pos = y_all[y>=0.0]
    y_neg = y_all[y<=0.0]
    
    x_neg[:,0] = np.abs(x_neg[:,0])
    y_neg[:,0] = np.abs(y_neg[:,0])

    return (x_pos, x_neg, y_pos, y_neg)

def create_grid_features(features, bearings, lons, lats, times, timestep, resolution):
    # Get regularly spaced bins at given resolution/timestep across bbox for all collected points
    # Need to flip bins for latitude because it should decrease downward through array
    latbins = np.linspace(np.min(lats), np.max(lats), resolution)
    latbins = np.append(latbins, latbins[-1]+.0000001)
    lonbins = np.linspace(np.min(lons), np.max(lons), resolution)
    lonbins = np.append(lonbins, lonbins[-1]+.0000001)
    tbins = np.arange(0,1440*60,timestep)
    tbins = np.append(tbins, tbins[-1]+.0000001)
    # Split features into quadrant channels
    channel_obs = decompose_vector(features, bearings, np.column_stack([lats, lons, times]))
    # For each channel, aggregate by location and timestep bins
    all_channel_rasts = np.zeros((len(latbins)-1, len(lonbins)-1, len(tbins)-1, len(channel_obs)), dtype='float16')
    for i, channel in enumerate(channel_obs):
        # Get the average feature value in each bin
        count_hist, count_edges = np.histogramdd(channel[:,1:4], bins=[latbins, lonbins, tbins])
        sum_hist, edges = np.histogramdd(channel[:,1:4], weights=channel[:,0], bins=[latbins, lonbins, tbins])
        rast = sum_hist / np.maximum(1, count_hist)
        # Fill zeros with very small value
        rast[rast==0] = .0000001
        # Invert latitude values (the bins must be increasing in hist dd, but in reality latitude decreases)
        # This would not need to be flipped for data below the equator
        # For longitudes, AtB may need to be flipped
        rast = np.flip(rast, axis=0)
        # Save binned values for each channel
        all_channel_rasts[:,:,:,i] = rast
    return all_channel_rasts

def get_adjacent_points(df, shingle_id, t_buffer, dist):
    # Critical to not reset index on df or intermediate results
    shingle_data = df[df['shingle_id']==shingle_id]
    adjacent_data = df[df['shingle_id']!=shingle_id]
    # Filter on time
    t_buffer = 120
    t_min = np.min(shingle_data.locationtime) - t_buffer
    t_max = np.max(shingle_data.locationtime)
    adjacent_data = adjacent_data[adjacent_data['locationtime'].between(t_min, t_max)]
    # Filter on distance
    points = np.array([adjacent_data['lon'], adjacent_data['lat']]).T.tolist()
    query_points = np.array([shingle_data['lon'], shingle_data['lat']]).T.tolist()
    pt_indices = get_points_within_dist(points, query_points, dist)
    adjacent_data = adjacent_data.iloc[pt_indices].sort_values(['shingle_id','locationtime'])
    return (shingle_data, adjacent_data)

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
    shape_segment_list = geopandas.GeoDataFrame(shape_segment_list, geometry=geopandas.points_from_xy(np.array(shape_segment_list.shape_pt_lon), np.array(shape_segment_list.shape_pt_lat)))
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

def map_to_network(traces, segments, n_sample):
    # Sample from the traces, match to the network
    traces_n = traces.sample(n_sample)
    # There are duplicates when distance to segments is tied; just take first
    traces_n = geopandas.sjoin_nearest(traces_n, segments, distance_col="join_dist").drop_duplicates(['tripid','locationtime'])
    return traces_n

def interpolate_trajectories(df, group_col):
    traj_bounds = df.groupby(group_col)['timeID_s'].agg(['min', 'max'])
    ary_len = np.sum(traj_bounds['max']+1 - traj_bounds['min'])
    interp_lon = np.empty((ary_len,), dtype=float)
    interp_lat = np.empty((ary_len,), dtype=float)
    interp_t = np.empty((ary_len,), dtype=int)
    interp_id = np.empty((ary_len,), dtype=object)
    i = 0
    for traj_id, (traj_min, traj_max) in traj_bounds.iterrows():
        time_steps = np.arange(traj_min, traj_max+1)
        num_steps = len(time_steps)
        traj_data = df[df[group_col] == traj_id][['lon', 'lat', 'timeID_s']].values
        lonint = np.interp(time_steps, traj_data[:,2], traj_data[:,0])
        latint = np.interp(time_steps, traj_data[:,2], traj_data[:,1])
        interp_lon[i:i+num_steps] = lonint
        interp_lat[i:i+num_steps] = latint
        interp_t[i:i+num_steps] = time_steps
        interp_id[i:i+num_steps] = np.full((num_steps,), traj_id)
        i += num_steps
    # Put in dataframe and format
    interp = pd.DataFrame({
        'lon':interp_lon,
        'lat': interp_lat,
        'timeID_s': interp_t,
        group_col: interp_id
    })
    return interp

def fill_trajectories(df, min_timeID, max_timeID, group_id):
    traj_bounds = df.groupby(group_id)['timeID_s'].agg(['min', 'max'])
    time_steps = np.arange(min_timeID, max_timeID + 1)
    # Create an array to hold the filled trajectories
    num_trajectories = len(traj_bounds)
    num_steps = len(time_steps)
    fill_arr = np.empty((num_trajectories * num_steps, 4), dtype=object)
    # Loop over each trajectory and fill in the positions at each time step
    i = 0
    for traj_id, (traj_min, traj_max) in traj_bounds.iterrows():
        traj_data = df[df[group_id] == traj_id][['lat', 'lon', 'timeID_s']].values
        for j, t in enumerate(time_steps):
            if t < traj_min:
                # Use the first observation for this trajectory before the trajectory start time
                fill_arr[i+j] = [traj_data[0,1], traj_data[0,0], t, traj_id]
            elif t > traj_max:
                # Use the last observation for this trajectory after the trajectory end time
                fill_arr[i+j] = [traj_data[-1,1], traj_data[-1,0], t, traj_id]
            else:
                # Use the most recent observation for this trajectory at the current time step
                mask = traj_data[:,2].astype(int) <= t
                if np.any(mask):
                    fill_arr[i+j] = [traj_data[mask][-1,1], traj_data[mask][-1,0], t, traj_id]
                else:
                    # There are no observations for this trajectory at or before the current time step
                    fill_arr[i+j] = [np.nan, np.nan, t, traj_id]
        i += num_steps
    # Put in dataframe and format
    fill = pd.DataFrame(fill_arr)
    fill.columns = ['lon','lat','timeID_s',group_id]
    fill['lon'] = fill['lon'].astype(float)
    fill['lat'] = fill['lat'].astype(float)
    fill['timeID_s'] = fill['timeID_s'].astype(int)
    return fill

def plot_gtfsrt_trip(ax, trace_df):
    """
    Plot a single real-time bus trajectory on a map.
    ax: where to plot
    trace_df: data from trip to plot
    Returns: None.
    """
    to_plot = trace_df.copy()
    to_plot = geopandas.GeoDataFrame(to_plot, geometry=geopandas.points_from_xy(to_plot.lon, to_plot.lat), crs="EPSG:4326")
    to_plot_stop = trace_df.iloc[-1:,:]
    to_plot_stop = geopandas.GeoDataFrame(to_plot_stop, geometry=geopandas.points_from_xy(to_plot_stop.stop_lon, to_plot_stop.stop_lat), crs="EPSG:4326")
    # Plot all points
    to_plot.plot(ax=ax, marker='.', color='purple', markersize=20)
    # Plot first/last points
    to_plot.iloc[0:1,:].plot(ax=ax, marker='*', color='green', markersize=40)
    to_plot.iloc[-1:,:].plot(ax=ax, marker='*', color='red', markersize=40)
    # to_plot.apply(lambda x: ax.annotate(text=x['bearing'], xy=x.geometry.centroid.coords[0], ha='center', size=2), axis=1)
    # Also plot closest stop to final point
    to_plot_stop.plot(ax=ax, marker='x', color='blue', markersize=20)

    return None

def plot_gtfs_trip(ax, trip_id, gtfs_data):
    """
    Plot scheduled stops for a single bus trip on a map.
    ax: where to plot
    trip_id: which trip in GTFS to plot
    gtfs_data: merged GTFS data
    Returns: None.
    """
    to_plot = gtfs_data.copy()
    to_plot = to_plot[to_plot['trip_id']==trip_id]
    to_plot = geopandas.GeoDataFrame(to_plot, geometry=geopandas.points_from_xy(to_plot.stop_lon, to_plot.stop_lat), crs="EPSG:4326")
    to_plot.plot(ax=ax, marker='x', color='lightgreen', markersize=10)
    return None