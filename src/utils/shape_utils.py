"""
Functions for processing and working with tracked bus data.
"""
import warnings
from random import sample

import geopandas
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shapely
import shapely.geometry
from scipy.spatial import KDTree
from shapely.errors import ShapelyDeprecationWarning

from utils import data_utils

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def apply_bbox(lats, lons, bbox):
    min_lat = bbox[0]
    min_lon = bbox[1]
    max_lat = bbox[2]
    max_lon = bbox[3]
    a = [lats>=min_lat]
    b = [lats>=min_lat]
    c = [lons>=min_lon]
    d = [lons>=min_lon]
    return [a and b and c and d]

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

def get_adjacent_metric(shingle_group, adj_traces, d_buffer, t_buffer, b_buffer=None, orthogonal=False):
    """
    Calculate adjacent metric for each shingle from all other shingles in adj_traces.
    """
    # Set up spatial index for the traces
    tree = KDTree(adj_traces[:,:2])
    # Get time filter for the traces
    t_end = np.min(shingle_group[['locationtime']].values)
    t_start = t_end - t_buffer
    # Get the indices of adj_traces which fit dist buffer
    d_idxs = tree.query_ball_point(shingle_group[['x','y']].values, d_buffer)
    d_idxs = set([item for sublist in d_idxs for item in sublist])
    # Get the indices of adj_traces which fit time buffer
    t_idxs = (adj_traces[:,2] <= t_end) & (adj_traces[:,2] >= t_start)
    t_idxs = set(np.where(t_idxs)[0])
    # Get the indices of adj_traces which fit heading buffer
    if b_buffer is not None:
        if orthogonal == True:
            b_left = np.mean(shingle_group[['bearing']].values) + 90
            b_left_end = b_left + b_buffer
            b_left_start = b_left - b_buffer
            b_right = np.mean(shingle_group[['bearing']].values) - 90
            b_right_end = b_right + b_buffer
            b_right_start = b_right - b_buffer
            b_idxs = ((adj_traces[:,3] <= b_left_end) & (adj_traces[:,3] >= b_left_start)) | ((adj_traces[:,3] <= b_right_end) & (adj_traces[:,3] >= b_right_start))
        else:
            b_end = np.mean(shingle_group[['bearing']].values) + b_buffer
            b_start = np.mean(shingle_group[['bearing']].values) - b_buffer
            b_idxs = (adj_traces[:,3] <= b_end) & (adj_traces[:,3] >= b_start)
        b_idxs = set(np.where(b_idxs)[0])
        idxs = d_idxs & t_idxs & b_idxs
    else:
        idxs = d_idxs & t_idxs
    # Get the average speed of the trace and the relevant adj_traces
    target = np.mean(shingle_group[['speed_m_s']].values)
    if len(idxs) != 0:
        pred = np.mean(np.take(adj_traces[:,4], list(idxs), axis=0))
    else:
        pred = np.nan
    return (target, pred)

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

def plot_gtfsrt_trip(ax, trace_df, epsg, gtfs_folder):
    """
    Plot a single real-time bus trajectory on a map.
    ax: where to plot
    trace_df: data from trip to plot
    Returns: None.
    """
    # Plot trip stops from GTFS
    trace_date = trace_df['file'].iloc[0]
    trip_id = trace_df['trip_id'].iloc[0]
    file_to_gtfs_map = data_utils.get_best_gtfs_lookup(trace_df, gtfs_folder)
    gtfs_data = data_utils.merge_gtfs_files(f"{gtfs_folder}{file_to_gtfs_map[trace_date]}/", epsg, [0,0])
    to_plot_gtfs = gtfs_data[gtfs_data['trip_id']==trip_id]
    to_plot_gtfs = geopandas.GeoDataFrame(to_plot_gtfs, geometry=geopandas.points_from_xy(to_plot_gtfs.stop_x, to_plot_gtfs.stop_y), crs=f"EPSG:{epsg}")
    to_plot_gtfs.plot(ax=ax, marker='x', color='lightblue', markersize=10)
    # Plot observations
    to_plot = trace_df.copy()
    to_plot = geopandas.GeoDataFrame(to_plot, geometry=geopandas.points_from_xy(to_plot.x, to_plot.y), crs=f"EPSG:{epsg}")
    to_plot_stop = trace_df.iloc[-1:,:]
    to_plot_stop = geopandas.GeoDataFrame(to_plot_stop, geometry=geopandas.points_from_xy(to_plot_stop.stop_x, to_plot_stop.stop_y), crs=f"EPSG:{epsg}")
    to_plot.plot(ax=ax, marker='.', color='purple', markersize=20)
    # Plot first/last observations
    to_plot.iloc[:1,:].plot(ax=ax, marker='*', color='green', markersize=40)
    to_plot.iloc[-1:,:].plot(ax=ax, marker='*', color='red', markersize=40)
    # Plot closest stop to final observation
    to_plot_stop.plot(ax=ax, marker='x', color='blue', markersize=20)
    # Add custom legend
    ax.legend(["Scheduled Trip Stops","Shingle Observations","Shingle Start","Shingle End", "Closest Stop"], loc="upper right")
    return None