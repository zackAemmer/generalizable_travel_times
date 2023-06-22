import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from utils import data_utils


class NGridBetter:
    def __init__(self, grid_bounds, grid_s_size):
        self.grid_bounds = grid_bounds
        self.grid_s_size = grid_s_size
        self.points = None
        # Create grid boundaries and cells
        x_resolution = (grid_bounds[2] - grid_bounds[0]) // grid_s_size
        y_resolution = (grid_bounds[3] - grid_bounds[1]) // grid_s_size
        xbins = np.linspace(grid_bounds[0], grid_bounds[2], x_resolution)
        xbins = np.append(xbins, xbins[-1]+.0000001)
        ybins = np.linspace(grid_bounds[1], grid_bounds[3], y_resolution)
        ybins = np.append(ybins, ybins[-1]+.0000001)
        self.xbins=xbins
        self.ybins=ybins
        self.cell_lookup = {}
    def digitize_points(self, x_vals, y_vals):
        xbin_idxs = np.digitize(x_vals, self.xbins, right=False)
        xbin_idxs = np.maximum(0,xbin_idxs)
        ybin_idxs = np.digitize(y_vals, self.ybins, right=False)
        ybin_idxs = np.maximum(0,ybin_idxs)
        return xbin_idxs, ybin_idxs
    def add_grid_content(self, data, trace_format=False):
        if trace_format:
            data = data[['locationtime','x','y','speed_m_s','bearing']].values
        if self.points is None:
            self.points = data
        else:
            self.points = np.concatenate((self.points, data), axis=0)
    def build_cell_lookup(self):
        # Sort on time from low to high; no longer continuous shingles
        self.points = self.points[np.argsort(self.points[:,0]),:]
        point_xbins, point_ybins = self.digitize_points(self.points[:,1], self.points[:,2])
        # Build lookup table for grid cells to sorted point lists
        for x, xbin in enumerate(self.xbins):
            for y, ybin in enumerate(self.ybins):
                cell_points = self.points[(point_xbins==x) & (point_ybins==y)]
                self.cell_lookup[x,y] = cell_points
    def get_grid_features(self, x_idxs, y_idxs, locationtimes, n_points=3, buffer=2):
        seq_len = len(x_idxs)
        grid_size = (2 * buffer) + 1
        # For every point, want grid buffer in x and y
        x_range = np.arange(grid_size)
        y_range = np.arange(grid_size)
        x_buffer_range = x_idxs[:, np.newaxis] - buffer + x_range
        y_buffer_range = y_idxs[:, np.newaxis] - buffer + y_range
        # For every 1d set of X,Y grid ranges, want a 2d buffer
        buffer_range = [np.meshgrid(arr1,arr2) for arr1,arr2 in zip(x_buffer_range,y_buffer_range)]
        # Each element in each list is total enumeration of x,y cell indices for 1 point
        x_queries = np.concatenate([x[0].flatten() for x in buffer_range])
        y_queries = np.concatenate([y[1].flatten() for y in buffer_range])
        # Limit to bounds of the grid
        x_queries = np.clip(x_queries,0,len(self.xbins)-1)
        y_queries = np.clip(y_queries,0,len(self.ybins)-1)
        t_queries = np.array(locationtimes).repeat(grid_size*grid_size)
        n_recent_points = self.get_recent_points(x_queries, y_queries, t_queries, n_points)
        n_recent_points = n_recent_points.reshape((seq_len,grid_size,grid_size,n_points,6))
        # TxXxYxNxC -> TxCxYxNxX -> TxCxNxYxX
        n_recent_points = np.swapaxes(n_recent_points, 1, 4)
        n_recent_points = np.swapaxes(n_recent_points, 2, 3)
        return n_recent_points
    def get_full_grid(self, grid_t_size, n_points=3):
        tbins = np.arange(np.min(self.points[:,0]),np.max(self.points[:,0]), grid_t_size)
        tbins = np.append(tbins, tbins[-1]+1)
        all_res = np.ones((len(tbins), n_points, 3, len(self.xbins), len(self.ybins)))
        all_res[:] = np.nan
        for t, tbin in enumerate(tbins):
            for x, xbin in enumerate(self.xbins):
                for y, ybin in enumerate(self.ybins):
                    all_res[t,:,:,x,y] = self.get_recent_points(x,y,tbin,n_points)
        # tsteps X channels X samples X height X width
        all_res = np.swapaxes(all_res, 1, 2)
        all_res = np.swapaxes(all_res, 3, 4)
        return all_res
    # def get_recent_points(self, x_idx, y_idx, locationtime, n_points):
    #     # Get lookup values for every pt/cell
    #     cell_points = list(map(lambda pt: self.cell_lookup[(pt[0], pt[1])], zip(x_idx, y_idx)))
    #     # Get only values that occurred after the pt locationtime
    #     cell_points = [pts[pts[:,0]<t][-n_points:] for pts,t in zip(cell_points, locationtime)]
    #     # Reverse the values so that the most recent are first
    #     cell_points = [np.expand_dims(pts[::-1],0) for pts in cell_points]
    #     # Fill with nan so that all are same shape
    #     cell_points = [np.pad(pts, [(0,0), (0,n_points-pts.shape[1]), (0,6-pts.shape[2])], mode='constant', constant_values=np.nan) for pts in cell_points]
    #     # Join and add obs_age feature
    #     cell_points = np.concatenate(cell_points, axis=0)
    #     cell_points[:,:,-1] = np.repeat(np.expand_dims(np.array(locationtime),1),n_points,1) - cell_points[:,:,0]
    #     return cell_points
    def get_recent_points(self, x_idx, y_idx, locationtime, n_points):
        num_cells = len(x_idx)
        cell_points = np.empty((num_cells, n_points, 6))
        cell_points.fill(np.nan)
        for i, (x, y, time) in enumerate(zip(x_idx, y_idx, locationtime)):
            # Get lookup values for every pt/cell
            cell = self.cell_lookup[(x,y)]
            if cell.size==0:
                continue
            # Get only values that occurred after the pt locationtime
            points = cell[cell[:,0] < time][-n_points:][::-1]
            cell_points[i,:points.shape[0],:5] = points
        # Add obs_age feature
        cell_points[:,:,-1] = np.repeat(np.expand_dims(np.array(locationtime),1),n_points,1) - cell_points[:,:,0]
        return cell_points

# class NGrid:
#     def __init__(self, content, c_resolution, x_resolution, y_resolution, t_resolution, n_resolution):
#         self.content = content
#         self.c_resolution = c_resolution
#         self.x_resolution = x_resolution
#         self.y_resolution = y_resolution
#         self.t_resolution = t_resolution
#         self.n_resolution = n_resolution
#         self.fill_content = None
#     def get_content(self, tbin_idx):
#         # Get grid for a given timestep
#         # Using loose observations, fill in grid cells; keeping nan when < n samples available
#         c = np.zeros((self.c_resolution, self.n_resolution, self.y_resolution, self.x_resolution))
#         c[:,:,:,:] = np.nan
#         # Keep track of the number of samples in a cell
#         counter = np.zeros((self.y_resolution, self.x_resolution), dtype='int')
#         t_idx_content = self.content[self.content['tbin_idx'] == tbin_idx]
#         x_indices = t_idx_content['xbin_idx'].values
#         y_indices = t_idx_content['ybin_idx'].values
#         # Get features from each observation to include in grid
#         obs_values = t_idx_content[['bearing','speed_m_s']].values
#         for x, y, obs in zip(x_indices, y_indices, obs_values):
#             counter_state = counter[y,x]
#             c[:,counter_state,y,x] = obs
#             counter[y,x] += 1
#         return c
#     def get_all_content(self):
#         c = np.array([self.get_content(t) for t in range(self.t_resolution-1)])
#         return c
#     def set_fill_content(self, fill_content):
#         self.fill_content = sparse.csr_matrix(fill_content.reshape(fill_content.shape[0],-1))
#     def get_fill_content(self):
#         c = self.fill_content.toarray()
#         c = np.reshape(c, (c.shape[0], self.c_resolution+1, self.n_resolution, self.y_resolution, self.x_resolution))
#         return c
#     def get_density(self):
#         c = self.get_all_content()
#         return np.sum(~np.isnan(c)) / c.size
#     def get_masked_content(self):
#         feature_content = []
#         c = self.get_all_content()
#         for feature_idx in range(c.shape[1]):
#             c_sub = c[:,feature_idx,:,:,:]
#             f = c_sub[~np.isnan(c_sub)]
#             feature_content.append(f)
#         return feature_content

# def traces_to_ngrid(traces, grid_bounds, grid_s_res, grid_t_res, grid_n_res):
#     times = traces['locationtime'].values
#     # Create grid boundaries and cells
#     y_resolution = (grid_bounds[3] - grid_bounds[1]) // grid_s_res
#     x_resolution = (grid_bounds[2] - grid_bounds[0]) // grid_s_res
#     ybins = np.linspace(grid_bounds[1], grid_bounds[3], y_resolution)
#     ybins = np.append(ybins, ybins[-1]+.0000001)
#     xbins = np.linspace(grid_bounds[0], grid_bounds[2], x_resolution)
#     xbins = np.append(xbins, xbins[-1]+.0000001)
#     tbins = np.arange(np.min(times),np.max(times),grid_t_res)
#     tbins = np.append(tbins, tbins[-1]+1)
#     # Get indexes of all datapoints in the grid
#     tbin_idxs = np.digitize(traces['locationtime'].values, tbins, right=True) - 1
#     tbin_idxs = np.maximum(0,tbin_idxs)
#     xbin_idxs = np.digitize(traces['x'].values, xbins, right=False)
#     xbin_idxs = np.maximum(0,xbin_idxs)
#     ybin_idxs = np.digitize(traces['y'].values, ybins, right=False)
#     ybin_idxs = np.maximum(0,ybin_idxs)
#     # For every grid cell, collect the most recent observations
#     traces['tbin_idx'] = tbin_idxs
#     traces['xbin_idx'] = xbin_idxs
#     traces['ybin_idx'] = ybin_idxs
#     obs = traces.groupby(['tbin_idx','xbin_idx','ybin_idx']).head(grid_n_res)
#     ngrid = NGrid(content=obs, c_resolution=2, x_resolution=x_resolution, y_resolution=y_resolution, t_resolution=len(tbins), n_resolution=grid_n_res)
#     return ngrid, tbin_idxs, xbin_idxs, ybin_idxs

# def fill_ngrid_forward(ngrid):
#     """
#     Fill forward (in time) the grid for timesteps w/o complete observations.
#     """
#     arr = ngrid.get_all_content()
#     timesteps, features, n_obs, rows, cols = arr.shape
#     filled_arr = np.copy(arr)
#     # Initialize age for all obs to 0
#     # Any sample that has nan for any feature should have nan for its age
#     mask = np.isnan(arr.sum(axis=1))
#     age = np.where(mask, np.nan, 0)
#     # Add age as new channel/feature to be filled in the filled arr
#     age = np.expand_dims(age, 1)
#     filled_arr = np.concatenate([filled_arr, age], axis=1)
#     for t in range(1, timesteps):
#         # Gather all obs for current timestep, and previous
#         current_obs = filled_arr[t].copy()
#         prev_obs = filled_arr[t-1].copy()
#         # Increment age for all previous obs that will potentially be filled forwards to this timestep
#         prev_obs[-1,:,:,:] += 1
#         # Concatenate current obs to prev
#         result = np.concatenate([current_obs, prev_obs], axis=1)
#         # Sort by age feature from lowest to highest, nan's last
#         sidx = result[-1,:,:,:]
#         sidx = np.argsort(sidx, axis=0)
#         # Take first n observations
#         sidx = sidx[:ngrid.n_resolution,:,:]
#         sidx = np.expand_dims(sidx, axis=0)
#         sidx = np.repeat(sidx, result.shape[0], axis=0)
#         result = np.take_along_axis(result, sidx, axis=1)
#         filled_arr[t] = result
#     ngrid.set_fill_content(filled_arr)

# def extract_ngrid_features(g, tbins, xbins, ybins, config, buffer=1):
#     """
#     Given sequence of bins from a trip, reconstruct grid features.
#     """
#     # All points in the sequence will have the information at the time of the starting point
#     # However the starting information is shifted in space to center features on each point
#     tbin_start_idx = tbins[0]
#     grid_features = []
#     for i in range(len(tbins)):
#         # Handle case where buffer goes off edge of grid
#         if xbins[i]-buffer-1 < 0 or ybins[i]-buffer-1 < 0:
#             feature = np.zeros((g.shape[1], g.shape[2], 2*buffer+1, 2*buffer+1))
#             feature[:,:,:,:] = np.nan
#         elif xbins[i]+buffer > g.shape[4] or ybins[i]+buffer > g.shape[3]:
#             feature = np.zeros((g.shape[1], g.shape[2], 2*buffer+1, 2*buffer+1))
#             feature[:,:,:,:] = np.nan
#         else:
#             # Filter grid based on adjacent squares to buffer (pts +/- buffer, including middle point)
#             feature = g[tbin_start_idx,:,:,ybins[i]-buffer-1:ybins[i]+buffer,xbins[i]-buffer-1:xbins[i]+buffer].copy()
#         # Fill undefined cells with global averages (per variable)
#         feature[0,:,:,:][np.isnan(feature[0,:,:,:])] = config['bearing_mean']
#         feature[1,:,:,:][np.isnan(feature[1,:,:,:])] = config['speed_m_s_mean']
#         feature[2,:,:,:][np.isnan(feature[2,:,:,:])] = 100
#         # Normalize all cells
#         feature[0,:,:,:] = data_utils.normalize(feature[0,:,:,:], config['bearing_mean'], config['bearing_std'])
#         feature[1,:,:,:] = data_utils.normalize(feature[1,:,:,:], config['speed_m_s_mean'], config['speed_m_s_std'])
#         grid_features.append(feature)
#     return grid_features

def save_grid_anim(data, file_name):
    # Plot first 4 channels of second axis
    fig, axes = plt.subplots(2,2)
    axes = axes.reshape(-1)
    fig.tight_layout()
    # Define the update function that will be called for each frame of the animation
    def update(frame):
        fig.suptitle(f"Frame {frame}")
        for i in range(data.shape[1]):
            d = data[:,i,:,:]
            vmin=np.min(d[~np.isnan(d)])
            vmax=np.max(d[~np.isnan(d)])
            ax = axes[i]
            ax.clear()
            im = ax.imshow(data[frame,i,:,:], cmap='plasma', vmin=vmin, vmax=vmax, origin="lower")
    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=data.shape[0])
    # Save the animation object
    ani.save(f"../plots/{file_name}", fps=10, dpi=300)