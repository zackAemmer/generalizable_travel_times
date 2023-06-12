import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from utils import data_utils


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
    def get_fill_content(self):
        c = self.fill_content.toarray()
        c = np.reshape(c, (c.shape[0], self.c_resolution+1, self.n_resolution, self.y_resolution, self.x_resolution))
        return c
    def get_all_content(self):
        c = np.array([self.get_content(t) for t in range(self.t_resolution-1)])
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
    def set_fill_content(self, fill_content):
        self.fill_content = sparse.csr_matrix(fill_content.reshape(fill_content.shape[0],-1))

def traces_to_ngrid(traces, grid_bounds, grid_s_res, grid_t_res, grid_n_res):
    times = traces['locationtime'].values
    # Create grid boundaries and cells
    y_resolution = (grid_bounds[3] - grid_bounds[1]) // grid_s_res
    x_resolution = (grid_bounds[2] - grid_bounds[0]) // grid_s_res
    ybins = np.linspace(grid_bounds[1], grid_bounds[3], y_resolution)
    ybins = np.append(ybins, ybins[-1]+.0000001)
    xbins = np.linspace(grid_bounds[0], grid_bounds[2], x_resolution)
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
    # For every grid cell, collect the most recent observations
    traces['tbin_idx'] = tbin_idxs
    traces['xbin_idx'] = xbin_idxs
    traces['ybin_idx'] = ybin_idxs
    obs = traces.groupby(['tbin_idx','xbin_idx','ybin_idx']).head(grid_n_res)[['tbin_idx','xbin_idx','ybin_idx','x','y','bearing','speed_m_s']]
    ngrid = NGrid(content=obs, c_resolution=4, x_resolution=x_resolution, y_resolution=y_resolution, t_resolution=len(tbins), n_resolution=grid_n_res)
    return ngrid, tbin_idxs, xbin_idxs, ybin_idxs

def fill_ngrid_forward(ngrid):
    """
    Fill forward (in time) the grid for timesteps w/o complete observations.
    """
    arr = ngrid.get_all_content()
    timesteps, features, n_obs, rows, cols = arr.shape
    filled_arr = np.copy(arr)
    # Initialize age for all obs to 0
    # Any sample that has nan for any feature should have nan for its age
    mask = np.isnan(arr.sum(axis=1))
    age = np.where(mask, np.nan, 0)
    age = np.expand_dims(age, 1)
    filled_arr = np.concatenate([filled_arr, age], axis=1)
    for t in range(1, timesteps):
        # Get all obs for current timestep, and previous
        current_obs = filled_arr[t].copy()
        prev_obs = filled_arr[t-1].copy()
        # Increment age for all previous obs that will potentially be filled forwards to this timestep
        prev_obs[4,:,:,:] += 1
        # Concatenate current obs to prev
        result = np.concatenate([current_obs, prev_obs], axis=1)
        # Sort by age lowest to highest, nan's last
        sidx = result[4,:,:,:]
        sidx = np.argsort(sidx, axis=0)
        # Take first 3
        sidx = sidx[:3,:,:] 
        sidx = np.expand_dims(sidx, axis=0)
        sidx = np.repeat(sidx, result.shape[0], axis=0)
        result = np.take_along_axis(result, sidx, axis=1)
        filled_arr[t] = result
    ngrid.set_fill_content(filled_arr)

def extract_ngrid_features(g, tbins, xbins, ybins, config, buffer=1):
    """
    Given sequence of bins from a trip, reconstruct grid features.
    """
    # All points in the sequence will have the information at the time of the starting point
    # However the starting information is shifted in space to center features on each point
    tbin_start_idx = tbins[0]
    grid_features = []
    for i in range(len(tbins)):
        # Handle case where buffer goes off edge of grid
        if xbins[i]-buffer-1 < 0 or ybins[i]-buffer-1 < 0:
            feature = np.zeros((g.shape[1], g.shape[2], 2*buffer+1, 2*buffer+1))
            feature[:,:,:,:] = np.nan
        elif xbins[i]+buffer > g.shape[4] or ybins[i]+buffer > g.shape[3]:
            feature = np.zeros((g.shape[1], g.shape[2], 2*buffer+1, 2*buffer+1))
            feature[:,:,:,:] = np.nan
        else:
            # Filter grid based on adjacent squares to buffer (pts +/- buffer, including middle point)
            feature = g[tbin_start_idx,:,:,ybins[i]-buffer-1:ybins[i]+buffer,xbins[i]-buffer-1:xbins[i]+buffer].copy()
        # Fill undefined cells with global averages (per variable)
        feature[0,:,:,:][np.isnan(feature[0,:,:,:])] = config['bearing_mean']
        feature[1,:,:,:][np.isnan(feature[1,:,:,:])] = config['speed_m_s_mean']
        feature[2,:,:,:][np.isnan(feature[2,:,:,:])] = 100
        # Normalize all cells
        feature[0,:,:,:] = data_utils.normalize(feature[0,:,:,:], config['bearing_mean'], config['bearing_std'])
        feature[1,:,:,:] = data_utils.normalize(feature[1,:,:,:], config['speed_m_s_mean'], config['speed_m_s_std'])
        grid_features.append(feature)
    return grid_features

def save_grid_anim(data, file_name):
    # Plot first 4 channels of second axis
    fig, axes = plt.subplots(2,2)
    axes = axes.reshape(-1)
    fig.tight_layout()
    # Define the update function that will be called for each frame of the animation
    def update(frame):
        fig.suptitle(f"Frame {frame}")
        for i, ax in enumerate(axes):
            ax.clear()
            im = ax.imshow(data[frame,i,:,:], cmap='plasma', vmin=np.min(data[:,i,:,:]), vmax=np.max(data[:,i,:,:]), origin="lower")
    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0])
    # Save the animation object
    ani.save(f"../plots/{file_name}", fps=10, dpi=300)