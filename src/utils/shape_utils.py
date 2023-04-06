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

def extract_grid_features(grid, tbins, xbins, ybins, n_prior=1, buffer=20):
    """
    Given sequence of bins from a trip, reconstruct grid features.
    Normalize the grid based on the starting point.
    """
    tbin_start_idx = tbins[0]
    grid_features = []
    for i in range(len(tbins)):
        # If the point occurs before start, or buffer would put off edge of grid, use 0s
        if tbin_start_idx - n_prior < 0:
            feature = np.ones((n_prior, grid.shape[1], 2*buffer+1, 2*buffer+1))*-1
        elif xbins[i]-buffer-1 < 0 or ybins[i]-buffer-1 < 0:
            feature = np.ones((n_prior, grid.shape[1], 2*buffer+1, 2*buffer+1))*-1
        elif xbins[i]+buffer > grid.shape[3] or ybins[i]+buffer > grid.shape[2]:
            feature = np.ones((n_prior, grid.shape[1], 2*buffer+1, 2*buffer+1))*-1
        else:
            # Filter grid based on shingle start time (pts<start), and adjacent squares to buffer (pts +/- buffer, including middle point)
            feature = grid[tbin_start_idx-n_prior:tbin_start_idx,:,ybins[i]-buffer-1:ybins[i]+buffer,xbins[i]-buffer-1:xbins[i]+buffer]
        grid_features.append(feature)
    grid_features = np.concatenate(grid_features)
    # Normalize the grid to the information present when the trip starts=
    if len(grid_features[grid_features!=-1])==0:
        # The grid may be completely empty
        grid_avg = 0.0
        grid_std = 1.0
    else:
        grid_avg = np.mean(grid_features[grid_features!=-1])
        grid_std = np.std(grid_features[grid_features!=-1])
    # All unknown cells are given the average, all are then normalized
    grid_features[grid_features==-1] = grid_avg
    grid_features = data_utils.normalize(grid_features, grid_avg, grid_std)
    return grid_features

def get_grid_features(traces, resolution=64, timestep=30):
    # Create grid
    grid, tbins, xbins, ybins = decompose_and_rasterize(traces['speed_m_s'].values, traces['bearing'].values, traces['x'].values, traces['y'].values, traces['locationtime'].values, timestep, resolution)
    # Get tbins for each trace. No overlap between current trip and grid values.
    # Grid assigned values: binedge[i-1] <= x < binedge[i]
    # Trace values: binedge[i-1] < x <= binedge[i]
    # Want all values up through the previous bin index (since that is guaranteed < x)
    # [i-n_prior:i] will give give n_prior total values, including up to the bin before i
    tbin_idxs = np.digitize(traces['locationtime'].values, tbins, right=True) - 1
    tbin_idxs = np.maximum(0,tbin_idxs)
    # Opposite is true for lat/lon: want the exact bin that the value falls in
    # [i-buffer-1:i+buffer] will give 2*buffer+1 total values, with bin i in the middle
    xbin_idxs = np.digitize(traces['x'].values, xbins, right=False)
    xbin_idxs = np.maximum(0,xbin_idxs)
    ybin_idxs = np.digitize(traces['y'].values, ybins, right=False)
    ybin_idxs = np.maximum(0,ybin_idxs)
    return grid, tbin_idxs, xbin_idxs, ybin_idxs

def decompose_and_rasterize(features, bearings, x, y, times, timestep, resolution):
    # Get regularly spaced bins at given resolution/timestep across bbox for all collected points
    # Need to flip bins for latitude because it should decrease downward through array
    # Add a bin to the upper end; all obs are assigned such that bin_edge[i-1] <= x < bin_edge[i]
    ybins = np.linspace(np.min(y), np.max(y), resolution)
    ybins = np.append(ybins, ybins[-1]+.0000001)
    xbins = np.linspace(np.min(x), np.max(x), resolution)
    xbins = np.append(xbins, xbins[-1]+.0000001)
    tbins = np.arange(np.min(times),np.max(times),timestep)
    tbins = np.append(tbins, tbins[-1]+1)
    # Split features into quadrant channels
    channel_obs = decompose_vector(features, bearings, np.column_stack([x, y, times]))
    # For each channel, aggregate by location and timestep bins
    # T x C x H x W
    all_channel_rasts = np.ones((len(tbins)-1, len(channel_obs), len(ybins)-1, len(xbins)-1), dtype='float64') * -1
    for i, channel in enumerate(channel_obs):
        # Get the average feature value in each bin
        count_hist, count_edges = np.histogramdd(np.column_stack([channel[:,3], channel[:,2], channel[:,1]]), bins=[tbins, ybins, xbins])
        sum_hist, edges = np.histogramdd(np.column_stack([channel[:,3], channel[:,2], channel[:,1]]), weights=channel[:,0], bins=[tbins, ybins, xbins])
        rast = sum_hist / np.maximum(1, count_hist)
        # Mask cells with no information as -1
        mask = count_hist==0
        rast[mask] = -1
        # Save binned values for each channel
        all_channel_rasts[:,i,:,:] = rast
    return all_channel_rasts, tbins, xbins, ybins

def decompose_vector(scalars, bearings, data_to_attach=None):
    """
    Break speed vector into its x and y components.
    scalars: array of speeds
    data_to_attach: additional columns to keep with the decomposed values
    Returns: array of x +/-, y +/- scalar components.
    """
    # Decompose each scalar into its x/y components
    x = np.round(np.cos(bearings * np.pi/180) * scalars, 1)
    y = np.round(np.sin(bearings * np.pi/180) * scalars, 1)
    # Attach additional variables
    x_all = np.column_stack([x, data_to_attach])
    y_all = np.column_stack([y, data_to_attach])
    # Include 0.0 observations in both channels
    x_pos = x_all[x>=0.0]
    x_neg = x_all[x<=0.0]
    y_pos = y_all[y>=0.0]
    y_neg = y_all[y<=0.0]
    # Get absolute value of negative-direction observations
    x_neg[:,0] = np.abs(x_neg[:,0])
    y_neg[:,0] = np.abs(y_neg[:,0])
    return (x_pos, x_neg, y_pos, y_neg)

def save_grid_anim(data, file_name, vmin, vmax):
    # Plot all channels
    fig, axes = plt.subplots(2,2)
    axes = axes.reshape(-1)
    fig.tight_layout()
    # Define the update function that will be called for each frame of the animation
    def update(frame):
        fig.suptitle(f"Frame {frame}")
        for i, ax in enumerate(axes):
            ax.clear()
            ax.imshow(data[frame,i,:,:], cmap='plasma', vmin=vmin, vmax=vmax, origin="lower")
    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0])
    # Save the animation object
    ani.save(f"../plots/{file_name}", fps=10, dpi=300)

def get_adjacent_metric(shingle_group, adj_traces, d_buffer, t_buffer):
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
    # Get indices in both filters
    idxs = d_idxs & t_idxs
    # Get the average speed of the trace and the relevant adj_traces
    target = np.mean(shingle_group[['speed_m_s']].values)
    if len(idxs) != 0:
        pred = np.mean(np.take(adj_traces[:,3], list(idxs), axis=0))
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

def plot_gtfsrt_trip(ax, trace_df, epsg):
    """
    Plot a single real-time bus trajectory on a map.
    ax: where to plot
    trace_df: data from trip to plot
    Returns: None.
    """
    to_plot = trace_df.copy()
    to_plot = geopandas.GeoDataFrame(to_plot, geometry=geopandas.points_from_xy(to_plot.x, to_plot.y), crs=f"EPSG:{epsg}")
    to_plot_stop = trace_df.iloc[-1:,:]
    to_plot_stop = geopandas.GeoDataFrame(to_plot_stop, geometry=geopandas.points_from_xy(to_plot_stop.stop_x, to_plot_stop.stop_y), crs=f"EPSG:{epsg}")
    # Plot all points
    to_plot.plot(ax=ax, marker='.', color='purple', markersize=20)
    # Plot first/last points
    to_plot.iloc[0:1,:].plot(ax=ax, marker='*', color='green', markersize=40)
    to_plot.iloc[-1:,:].plot(ax=ax, marker='*', color='red', markersize=40)
    # Also plot closest stop to final point
    to_plot_stop.plot(ax=ax, marker='x', color='blue', markersize=20)

    return None

def plot_gtfs_trip(ax, trip_id, gtfs_data, epsg):
    """
    Plot scheduled stops for a single bus trip on a map.
    ax: where to plot
    trip_id: which trip in GTFS to plot
    gtfs_data: merged GTFS data
    Returns: None.
    """
    to_plot = gtfs_data.copy()
    to_plot = to_plot[to_plot['trip_id']==trip_id]
    to_plot = geopandas.GeoDataFrame(to_plot, geometry=geopandas.points_from_xy(to_plot.stop_x, to_plot.stop_y), crs=f"EPSG:{epsg}")
    to_plot.plot(ax=ax, marker='x', color='lightgreen', markersize=10)
    return None

def plot_closest_stop_anim(shingle_data):
    # Plot closest stop updates across shingle
    plot_data = shingle_data
    next_stops = plot_data[['stop_lon','stop_lat','timeID_s']]
    next_stops.columns = ["lon","lat","timeID_s"]
    next_stops['Type'] = "Nearest Scheduled Stop"
    next_points = shingle_data[['lon','lat','timeID_s']]
    next_points['Type'] = "Current Position"
    next_points = interpolate_trajectories(next_points, 'Type')
    next_stops = fill_trajectories(next_stops, np.min(next_points['timeID_s']), np.max(next_points['timeID_s']), 'Type')
    plot_data = pd.concat([next_points, next_stops], axis=0)

    fig = px.scatter(
        plot_data,
        title=f"Nearest Stop to Target",
        x="lon",
        y="lat",
        range_x=[np.min(plot_data['lon'])-.01, np.max(plot_data['lon'])+.01],
        range_y=[np.min(plot_data['lat'])-.01, np.max(plot_data['lat'])+.01],
        animation_frame="timeID_s",
        animation_group="Type",
        color="Type"
    )
    fig.update_traces(marker={'size': 15})
    fig.update_layout(
    template='plotly_dark',
    margin=dict(r=60, t=25, b=40, l=60)
    )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
    # fig.write_html("../plots/nearest_stop.html")
    fig.show()
    return None

def plot_adjacent_trips(test_traces):
    shingle_id = 5885
    dist = .002
    t_buffer = 60
    shingle_data, adjacent_data = get_adjacent_points(test_traces, shingle_id, t_buffer, dist)

    # Join and interpolate each trajectory
    plot_shingle_data = interpolate_trajectories(shingle_data, 'shingle_id')
    plot_shingle_data['Type'] = 'Trajectory'
    plot_adjacent_data = interpolate_trajectories(adjacent_data, 'shingle_id')
    plot_adjacent_data['Type'] = 'Adjacent Trip'
    # For some reason Plotly needs data to be sorted by the animation frame
    plot_data = pd.concat([plot_shingle_data, plot_adjacent_data], axis=0).sort_values(['timeID_s','shingle_id'])

    # Plot adjacent shingles
    fig = px.scatter(
        plot_data,
        title=f"Active Shingles Within {dist*111*1000}m and {t_buffer}s of Target",
        x="lon",
        y="lat",
        range_x=[np.min(plot_data['lon'])-.01, np.max(plot_data['lon'])+.01],
        range_y=[np.min(plot_data['lat'])-.01, np.max(plot_data['lat'])+.01],
        animation_frame="timeID_s",
        animation_group="shingle_id",
        # color="Type", # For some reason this breaks the animation order
        text="Type"
    )
    fig.update_layout(
    template='plotly_dark',
    margin=dict(r=60, t=25, b=40, l=60)
    )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
    # fig.write_html("../plots/adjacent_trips.html")
    fig.show()

def plot_traces_on_map(mapbox_token, plot_data):
    # Show overview of trace and adjacent on a map
    px.set_mapbox_access_token(mapbox_token)
    fig = px.scatter_mapbox(
        plot_data,
        lon="lon",
        lat="lat",
        color="Type"
    )
    fig.update_layout(
    margin=dict(r=60, t=25, b=40, l=60)
    )
    # fig.write_html("../plots/adjacent_trip_traces.html")
    fig.show()

def fit_poly(x, y):
    # Fit polynomial to the data
    z = np.polyfit(x=x, y=y, deg=2)
    # Get values for plotting the line of fit
    x_val = np.arange(np.min(x),np.max(x),.1)
    p = np.poly1d(z)
    y_val = p(x_val)
    # R2: variance explained by model / total variance
    SSR = np.sum((y - p(x))**2)
    SST = np.sum((y - np.mean(y))**2)
    R2 = 1 - (SSR/SST)
    return R2, x_val, y_val

def plot_poly(x, y, R2, x_val, y_val):
    plot_data = pd.concat([
        pd.DataFrame({
            "x": x,
            "y": y,
            "Type": "data"
        }),
        pd.DataFrame({
            "x": x_val,
            "y": y_val,
            "Type": "pred"
        })
    ])
    fig = px.scatter(
        plot_data,
        y="y",
        color="Type",
        x="x",
        title=f"Mean Adjacent Speed D=2 Polynomial R-Squared: {np.round(R2,3)}",
        labels={
            "x": "Mean Speed of Adjacent Trips (m/s)",
            "y": "Mean Speed of Target Trip (m/s)",
        }
    )
    # fig.write_image("../plots/speed_reg.png")
    fig.show()