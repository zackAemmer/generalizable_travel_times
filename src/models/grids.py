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
        # xbins = np.append(xbins, xbins[-1]+.0000001)
        ybins = np.linspace(grid_bounds[1], grid_bounds[3], y_resolution)
        # ybins = np.append(ybins, ybins[-1]+.0000001)
        self.xbins=xbins
        self.ybins=ybins
        self.cell_lookup = {}
    def digitize_points(self, x_vals, y_vals):
        xbin_idxs = np.digitize(x_vals, self.xbins, right=False)
        ybin_idxs = np.digitize(y_vals, self.ybins, right=False)
        xbin_idxs = np.clip(xbin_idxs,0,len(self.xbins))
        ybin_idxs = np.clip(ybin_idxs,0,len(self.ybins))
        return xbin_idxs, ybin_idxs
    def add_grid_content(self, data, trace_format=False):
        if trace_format:
            data = data[['locationtime','x','y','speed_m_s','bearing']].values
        if self.points is None:
            self.points = data
        else:
            self.points = np.concatenate((self.points, data), axis=0)
    def build_cell_lookup(self):
        # Sort on time from high to low; no longer continuous shingles
        self.points = self.points[np.argsort(self.points[:,0]),:]
        point_xbins, point_ybins = self.digitize_points(self.points[:,1], self.points[:,2])
        # Build lookup table for grid cells to sorted point lists
        for x,xbin in enumerate(self.xbins):
            for y,ybin in enumerate(self.ybins):
                self.cell_lookup[x,y] = self.points[(point_xbins==x) & (point_ybins==y)]
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
        x_queries = np.clip(x_queries,0,len(self.xbins))
        y_queries = np.clip(y_queries,0,len(self.ybins))
        # Limit to bounds of the grid
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
    def get_recent_points(self, x_idx, y_idx, locationtime, n_points):
        num_cells = len(x_idx)
        cell_points = np.empty((num_cells, n_points, 6))
        cell_points.fill(np.nan)
        for i, (x, y, t) in enumerate(zip(x_idx, y_idx, locationtime)):
            # Get lookup values for every pt/cell
            cell = self.cell_lookup[(x,y)]
            if cell.size==0:
                continue
            else:
                # Get only values that occurred after the pt locationtime
                idx = np.searchsorted(cell[:,0],t)
                points = cell[:idx,:][-n_points:][::-1]
            cell_points[i,:points.shape[0],:5] = points
        # Add obs_age feature
        cell_points[:,:,-1] = np.repeat(np.expand_dims(np.array(locationtime),1),n_points,1) - cell_points[:,:,0]
        return cell_points

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