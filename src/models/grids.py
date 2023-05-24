import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from utils import data_utils


class Grid:
    def __init__(self, content, counts, mask, c_resolution, x_resolution, y_resolution, t_resolution):
        self.content = sparse.csr_matrix(content.reshape(content.shape[0],-1))
        self.counts = sparse.csr_matrix(counts.reshape(counts.shape[0],-1))
        self.mask = sparse.csr_matrix(mask.reshape(mask.shape[0],-1))
        self.c_resolution = c_resolution
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.t_resolution = t_resolution
        self.fill_content = None
        self.fill_counts = None
    def get_content(self):
        c = self.content.toarray()
        c = np.reshape(c, (c.shape[0], self.c_resolution, self.y_resolution, self.x_resolution))
        return c
    def get_fill_content(self):
        c = self.fill_content.toarray()
        o = self.fill_counts.toarray()
        c = np.reshape(c, (c.shape[0], self.c_resolution, self.y_resolution, self.x_resolution))
        o = np.reshape(o, (o.shape[0], self.c_resolution, self.y_resolution, self.x_resolution))
        return np.concatenate([c,o], axis=1)
    def get_masked_content(self):
        # Same regardless if applied to fill or not
        c = self.content.toarray()
        m = self.mask.toarray()
        return c[m==1]
    def get_masked_counts(self):
        # Fill counts are just 0's
        c = self.counts.toarray()
        m = self.mask.toarray()
        return c[m==1]
    def get_density(self):
        m = self.content.toarray()
        return np.sum(m!=-1) / m.size
    def get_fill_density(self):
        m = self.fill_content.toarray()
        return np.sum(m!=-1) / m.size
    def set_fill_content(self, fill_content, fill_counts):
        self.fill_content = sparse.csr_matrix(fill_content.reshape(fill_content.shape[0],-1))
        self.fill_counts = sparse.csr_matrix(fill_counts.reshape(fill_counts.shape[0],-1))

class NGrid:
    def __init__(self, content, c_resolution, x_resolution, y_resolution, t_resolution, n_resolution):
        self.content = content
        self.c_resolution = c_resolution
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.t_resolution = t_resolution
        self.n_resolution = n_resolution
        self.fill_content = None
    def get_content(self, tbin_idx):
        c = np.zeros((self.c_resolution, self.n_resolution, self.y_resolution, self.x_resolution))
        c[:,:,:,:] = np.nan
        counter = np.zeros((self.y_resolution, self.x_resolution), dtype='int')
        t_idx_content = self.content[self.content['tbin_idx'] == tbin_idx]
        x_indices = t_idx_content['xbin_idx'].values
        y_indices = t_idx_content['ybin_idx'].values
        obs_values = t_idx_content[['x','y','bearing','speed_m_s']].values
        for x, y, obs in zip(x_indices, y_indices, obs_values):
            counter_state = counter[y,x]
            c[:,counter_state,y,x] = obs
            counter[y,x] += 1
        return c
    def get_all_content(self):
        c = np.array([self.get_content(t) for t in range(self.t_resolution-1)])
        return c
    def get_range_content(self, tbin_idx_start, tbin_idx_end):
        c = np.array([self.get_content(t) for t in range(0, (tbin_idx_end-tbin_idx_start)-1, 1)])
        return c
    def get_masked_content(self):
        feature_content = []
        c = self.get_all_content()
        for feature_idx in range(c.shape[1]):
            c_sub = c[:,feature_idx,:,:,:]
            f = c_sub[~np.isnan(c_sub)]
            feature_content.append(f)
        return feature_content
    def get_density(self):
        c = self.get_all_content()
        return np.sum(~np.isnan(c)) / c.size
    # def get_fill_content(self):
    #     c = self.fill_content.toarray()
    #     o = self.fill_counts.toarray()
    #     c = np.reshape(c, (c.shape[0], self.c_resolution, self.y_resolution, self.x_resolution))
    #     o = np.reshape(o, (o.shape[0], self.c_resolution, self.y_resolution, self.x_resolution))
    #     return np.concatenate([c,o], axis=1)
    # def get_masked_content(self):
    #     # Same regardless if applied to fill or not
    #     c = self.content.toarray()
    #     m = self.mask.toarray()
    #     return c[m==1]
    # def get_masked_counts(self):
    #     # Fill counts are just 0's
    #     c = self.counts.toarray()
    #     m = self.mask.toarray()
    #     return c[m==1]
    # def get_density(self):
    #     m = self.content.toarray()
    #     return np.sum(m!=-1) / m.size
    # def get_fill_density(self):
    #     m = self.fill_content.toarray()
    #     return np.sum(m!=-1) / m.size
    # def set_fill_content(self, fill_content, fill_counts):
    #     self.fill_content = sparse.csr_matrix(fill_content.reshape(fill_content.shape[0],-1))
    #     self.fill_counts = sparse.csr_matrix(fill_counts.reshape(fill_counts.shape[0],-1))

def traces_to_ngrid(traces, grid_x_bounds, grid_y_bounds, grid_s_res, grid_t_res, grid_n_res):
    times = traces['locationtime'].values
    # Create grid boundaries and cells
    y_resolution = (grid_y_bounds[1] - grid_y_bounds[0]) // grid_s_res
    x_resolution = (grid_x_bounds[1] - grid_x_bounds[0]) // grid_s_res
    ybins = np.linspace(grid_y_bounds[0], grid_y_bounds[1], y_resolution)
    ybins = np.append(ybins, ybins[-1]+.0000001)
    xbins = np.linspace(grid_x_bounds[0], grid_x_bounds[1], x_resolution)
    xbins = np.append(xbins, xbins[-1]+.0000001)
    tbins = np.arange(np.min(times),np.max(times),grid_t_res)
    tbins = np.append(tbins, tbins[-1]+1)
    # Get indexes of all datapoints in the grid
    tbin_idxs = np.digitize(traces['locationtime'].values, tbins, right=True) - 1
    tbin_idxs = np.maximum(0,tbin_idxs)
    xbin_idxs = np.digitize(traces['x'].values, xbins, right=False)
    xbin_idxs = np.maximum(0,xbin_idxs)
    ybin_idxs = np.digitize(traces['y'].values, ybins, right=False)
    ybin_idxs = np.maximum(0,ybin_idxs)
    # For every grid cell, collect the most recent 3 observations, mask with -1 if unavailable
    traces['tbin_idx'] = tbin_idxs
    traces['xbin_idx'] = xbin_idxs
    traces['ybin_idx'] = ybin_idxs
    obs = traces.groupby(['tbin_idx','xbin_idx','ybin_idx']).head(grid_n_res)[['tbin_idx','xbin_idx','ybin_idx','x','y','bearing','speed_m_s']]
    ngrid = NGrid(content=obs, c_resolution=4, x_resolution=x_resolution, y_resolution=y_resolution, t_resolution=len(tbins), n_resolution=grid_n_res)
    return ngrid, tbin_idxs, xbin_idxs, ybin_idxs

def traces_to_grid(traces, grid_x_bounds, grid_y_bounds, grid_s_res, grid_t_res):
    # Create grid
    grid, tbins, xbins, ybins = decompose_and_rasterize(traces['speed_m_s'].values, traces['bearing'].values, traces['x'].values, traces['y'].values, traces['locationtime'].values, grid_x_bounds, grid_y_bounds, grid_s_res, grid_t_res)
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

def decompose_and_rasterize(features, bearings, x, y, times, grid_x_bounds, grid_y_bounds, grid_s_res, grid_t_res):
    # Get regularly spaced bins at given resolution/timestep across bbox for all collected points
    # Need to flip bins for latitude because it should decrease downward through array
    # Add a bin to the upper end; all obs are assigned such that bin_edge[i-1] <= x < bin_edge[i]
    y_resolution = (grid_y_bounds[1] - grid_y_bounds[0]) // grid_s_res
    x_resolution = (grid_x_bounds[1] - grid_x_bounds[0]) // grid_s_res
    ybins = np.linspace(grid_y_bounds[0], grid_y_bounds[1], y_resolution)
    ybins = np.append(ybins, ybins[-1]+.0000001)
    xbins = np.linspace(grid_x_bounds[0], grid_x_bounds[1], x_resolution)
    xbins = np.append(xbins, xbins[-1]+.0000001)
    tbins = np.arange(np.min(times),np.max(times),grid_t_res)
    tbins = np.append(tbins, tbins[-1]+1)
    # Split features into quadrant channels
    channel_obs = decompose_vector(features, bearings, np.column_stack([x, y, times]))
    # For each channel, aggregate by location and timestep bins
    # T x C x H x W
    grid_content = np.ones((len(tbins)-1, len(channel_obs), len(ybins)-1, len(xbins)-1), dtype='float64') * -1
    count_content = np.ones((len(tbins)-1, len(channel_obs), len(ybins)-1, len(xbins)-1), dtype='int64')
    mask_content = np.ones((len(tbins)-1, len(channel_obs), len(ybins)-1, len(xbins)-1), dtype='int64')
    for i, channel in enumerate(channel_obs):
        # Get the average feature value in each bin
        count_hist, count_edges = np.histogramdd(np.column_stack([channel[:,3], channel[:,2], channel[:,1]]), bins=[tbins, ybins, xbins])
        sum_hist, edges = np.histogramdd(np.column_stack([channel[:,3], channel[:,2], channel[:,1]]), weights=channel[:,0], bins=[tbins, ybins, xbins])
        rast = sum_hist / np.maximum(1, count_hist)
        # Mask cells with no information as -1
        mask = count_hist==0
        rast[mask] = -1
        # Save binned values for each channel
        grid_content[:,i,:,:] = rast
        count_content[:,i,:,:] = count_hist
        mask_content[:,i,:,:] = mask
    # Invert mask when saving so that True/1 points to specified values instead of unspecified
    grid = Grid(grid_content, count_content, (1-mask_content), len(channel_obs), x_resolution, y_resolution, grid_t_res)
    return grid, tbins, xbins, ybins

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

def fill_grid_forward(grid_normal):
    """
    Fill forward (in time) each channel in the grid for timesteps w/o an observation.
    Returns: copy of grid with filled forward values, and 4 new channels holding fill counts.
    """
    grid_content = grid_normal.get_content()
    filled_channels = np.zeros(grid_content.shape)
    filled_counts = np.zeros(grid_content.shape)
    for channel in range(grid_content.shape[1]):
        ffilled, channel_counts = fill_channel_forward(grid_content[:,channel,:,:])
        filled_channels[:,channel,:,:] = ffilled
        filled_counts[:,channel,:,:] = channel_counts
    # First n channels are original speeds, second n are the obs histories in grid resolution
    grid_normal.set_fill_content(filled_channels, filled_counts)

def fill_channel_forward(grid_channel):
    tsteps, rows, cols = grid_channel.shape
    mask = grid_channel==-1
    ffilled = np.copy(grid_channel)
    channel_counts = np.zeros(grid_channel.shape)
    # For each cell, fill the value at this timestep w/previous value if it is masked
    for i in range(rows):
        for j in range(cols):
            counter = 0
            for t in range(1,tsteps):
                if mask[t][i][j]:
                    # Keep record of how many steps have been filled since last known value
                    counter += 1
                    ffilled[t][i][j] = ffilled[t-1][i][j]
                    channel_counts[t][i][j] = counter
                else:
                    # When a known value is found, reset counter to 0
                    counter = 0
                    channel_counts[t][i][j] = counter
    return ffilled, channel_counts

def extract_grid_features(g, tbins, xbins, ybins, config, buffer=1):
    """
    Given sequence of bins from a trip, reconstruct grid features.
    """
    # All points in the sequence will have the information at the time of the starting point
    # However the starting information is shifted in space to center features on each point
    tbin_start_idx = tbins[0]
    # Points in data correspond to final tbin
    if tbin_start_idx==g.shape[0]:
        tbin_start_idx = tbin_start_idx-1
    grid_features = []
    for i in range(len(tbins)):
        # Handle case where buffer goes off edge of grid (-1's)
        if xbins[i]-buffer-1 < 0 or ybins[i]-buffer-1 < 0:
            feature = np.ones((g.shape[1], 2*buffer+1, 2*buffer+1))*-1
        elif xbins[i]+buffer > g.shape[3] or ybins[i]+buffer > g.shape[2]:
            feature = np.ones((g.shape[1], 2*buffer+1, 2*buffer+1))*-1
        else:
            # Filter grid based on shingle start time, and adjacent squares to buffer (pts +/- buffer, including middle point)
            feature = g[tbin_start_idx,:,ybins[i]-buffer-1:ybins[i]+buffer,xbins[i]-buffer-1:xbins[i]+buffer].copy()
        # Fill undefined cells with global average
        feature[:4,:,:][feature[:4,:,:]==-1] = config['speed_m_s_mean']
        # Normalize all cells
        feature[:4,:,:] = data_utils.normalize(feature[:4,:,:], config['speed_m_s_mean'], config['speed_m_s_std'])
        grid_features.append(feature)
    return grid_features

def extract_ngrid_features(grid, tbins, xbins, ybins, config, buffer=1):
    """
    Given sequence of bins from a trip, reconstruct grid features.
    """
    # All points in the sequence will have the information at the time of the starting point
    # However the starting information is shifted in space to center features on each point
    tbin_start_idx = tbins[0]
    # Points in data correspond to final tbin
    if tbin_start_idx==grid.t_resolution:
        tbin_start_idx = tbin_start_idx-1
    g = grid.get_content(tbin_start_idx)
    grid_features = []
    for i, tbin in enumerate(tbins):
        # Handle case where buffer goes off edge of grid
        if xbins[i]-buffer-1 < 0 or ybins[i]-buffer-1 < 0:
            feature = np.ones((g.shape[0], g.shape[1], 2*buffer+1, 2*buffer+1))*-1
            feature[:,:,:,:] = np.nan
        elif xbins[i]+buffer > g.shape[3] or ybins[i]+buffer > g.shape[2]:
            feature = np.ones((g.shape[0], g.shape[1], 2*buffer+1, 2*buffer+1))*-1
            feature[:,:,:,:] = np.nan
        else:
            # Filter grid based on adjacent squares to buffer (pts +/- buffer, including middle point)
            feature = g[:,:,ybins[i]-buffer-1:ybins[i]+buffer,xbins[i]-buffer-1:xbins[i]+buffer].copy()
        # Fill undefined cells with global averages (per variable)
        feature[0,:,:,:][np.isnan(feature[0,:,:,:])] = config['x_mean']
        feature[1,:,:,:][np.isnan(feature[1,:,:,:])] = config['y_mean']
        feature[2,:,:,:][np.isnan(feature[2,:,:,:])] = config['bearing_mean']
        feature[3,:,:,:][np.isnan(feature[3,:,:,:])] = config['speed_m_s_mean']
        # Normalize all cells
        feature[0,:,:,:] = data_utils.normalize(feature[0,:,:,:], config['x_mean'], config['x_std'])
        feature[1,:,:,:] = data_utils.normalize(feature[1,:,:,:], config['y_mean'], config['y_std'])
        feature[2,:,:,:] = data_utils.normalize(feature[2,:,:,:], config['bearing_mean'], config['bearing_std'])
        feature[3,:,:,:] = data_utils.normalize(feature[3,:,:,:], config['speed_m_s_mean'], config['speed_m_s_std'])
        grid_features.append(feature)
    return grid_features

def save_grid_anim(data, file_name, vmin, vmax):
    # Plot all channels (first 4 of axis=0, second 4 are times)
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