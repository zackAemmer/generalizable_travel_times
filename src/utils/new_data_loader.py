import json
import os
from random import sample
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from models import grids
from utils import data_utils, shape_utils

FEATURE_COLS = [
    "shingle_id",
    "weekID",
    "timeID",
    "timeID_s",
    "locationtime",
    "lon",
    "lat",
    "x",
    "y",
    "x_cent",
    "y_cent",
    "dist_calc_km",
    "time_calc_s",
    "dist_cumulative_km",
    "time_cumulative_s",
    "speed_m_s",
    "bearing",
    "stop_x_cent",
    "stop_y_cent",
    "scheduled_time_s",
    "stop_dist_km",
    "passed_stops_n"
]
SKIP_FEATURE_COLS = [
    "shingle_id",
    "weekID",
    "timeID",
    "timeID_s",
    "locationtime",
    "lon",
    "lat",
    "x",
    "y",
    "x_cent",
    "y_cent",
    "dist_calc_km",
    "time_calc_s",
    "dist_cumulative_km",
    "time_cumulative_s",
    "speed_m_s",
    "bearing",
]

class LoadAllDataset(Dataset):
    def __init__(self, file_path, config, grid=None, holdout_routes=None, keep_only_holdout=False, add_grid_features=False, skip_gtfs=False):
        self.file_path = file_path
        self.config = config
        self.grid = grid
        self.holdout_routes = holdout_routes
        self.keep_only_holdout = keep_only_holdout
        self.add_grid_features = add_grid_features
        self.skip_gtfs = skip_gtfs
        # Defined in run preparation; should be made constant somewhere
        # Necessary to convert from np array tabular format saved in h5 files
        if not self.skip_gtfs:
            self.col_names = FEATURE_COLS
        else:
            self.col_names = SKIP_FEATURE_COLS
        # Cache indexes, means and stds for normalization
        self.col_indices = [i for i, var_name in enumerate(self.col_names) if f"{var_name}_mean" in self.config]
        self.means = np.array([self.config[f"{self.col_names[i]}_mean"] for i in self.col_indices])
        self.stds = np.array([self.config[f"{self.col_names[i]}_std"] for i in self.col_indices])
        # Load data
        self.h5_lookup = {}
        self.base_path = "/".join(self.file_path.split("/")[:-1])+"/"
        self.train_or_test = self.file_path.split("/")[-1]
        for filename in os.listdir(self.base_path):
            if filename.startswith(self.train_or_test) and filename.endswith(".h5"):
                self.h5_lookup[filename] = h5py.File(f"{self.base_path}{filename}", 'r')['tabular_data']
        self.data = self.get_all_samples(keep_cols=self.col_names)
        # Point to a dataset
        with open(f"{self.file_path}_shingle_config.json") as f:
            # This contains all of the shingle information in the data file
            self.shingle_lookup = json.load(f)
            # This is a list of keys that will be filtered and each point to a sample
            self.shingle_keys = list(self.shingle_lookup.keys())
        # Filter out (or keep exclusively) any routes that are used for generalization tests
        if self.holdout_routes is not None:
            holdout_idxs = [self.shingle_lookup[x]['route_id'] in self.holdout_routes for x in self.shingle_lookup]
            if self.keep_only_holdout==True:
                self.shingle_keys = [i for (i,v) in zip(self.shingle_keys, holdout_idxs) if v]
            else:
                self.shingle_keys = [i for (i,v) in zip(self.shingle_keys, holdout_idxs) if not v]
    def __getitem__(self, index):
        # Get information on shingle file location and lines; read specific shingle lines from specific shingle file
        samp_dict = self.shingle_lookup[self.shingle_keys[index]]
        df = self.data[self.data['shingle_id']==samp_dict['shingle_id']].values
        # df = self.h5_lookup[f"{self.train_or_test}_data_{samp_dict['network']}_{samp_dict['file_num']}.h5"][samp_dict['start_idx']:samp_dict['end_idx']]
        # df[:,self.col_indices] = (df[:,self.col_indices] - self.means) / self.stds
        # return df
        # Convert tabular to dict of keys; inefficient but works with all other code for now
        res = {}
        for i in range(df.shape[1]):
            res[self.col_names[i]] = df[:,i]
        # Some keys should not be repeated across timesteps (use first or last value of sequence)
        res['shingle_id'] = res['shingle_id'][0]
        res['time'] = res['time_cumulative_s'][-1]
        res['dist'] = res['dist_cumulative_km'][-1]
        res['weekID'] = res['weekID'][0]
        res['timeID'] = res['timeID'][0]
        # Add grid if applicable
        if self.grid is not None and self.add_grid_features:
            xbin_idxs, ybin_idxs = self.grid.digitize_points(res['x'], res['y'])
            grid_features = self.grid.get_grid_features(xbin_idxs, ybin_idxs, [res['locationtime'][0] for x in res['locationtime']])
            grid_features = apply_grid_normalization(grid_features, self.config)
            res['grid_features'] = grid_features
        res = apply_normalization(res, self.config)
        return res
    def __len__(self):
        return len(self.shingle_keys)
    def get_all_samples(self, keep_cols, indexes=None):
        # Read all h5 files in run base directory; get all point obs
        res = []
        for k in list(self.h5_lookup.keys()):
            df = self.h5_lookup[k][:]
            df = pd.DataFrame(df, columns=self.col_names)
            df['shingle_id'] = df['shingle_id'].astype(int)
            df = df[keep_cols]
            res.append(df)
        res = pd.concat(res)
        if indexes is not None:
            # Indexes are in order, but shingle_id's are not; get shingle id for each keep index and filter
            keep_shingles = [self.shingle_lookup[self.shingle_keys[i]]['shingle_id'] for i in indexes]
            res = res[res['shingle_id'].isin(keep_shingles)]
        return res

class LoadSliceDataset(Dataset):
    def __init__(self, file_path, config, holdout_routes=None, keep_only_holdout=False, add_grid_features=False, skip_gtfs=False):
        self.file_path = file_path
        self.config = config
        self.holdout_routes = holdout_routes
        self.keep_only_holdout = keep_only_holdout
        self.add_grid_features = add_grid_features
        self.skip_gtfs = skip_gtfs
        if not self.skip_gtfs:
            self.col_names = FEATURE_COLS
        else:
            self.col_names = SKIP_FEATURE_COLS
        # Cache indexes, means and stds for normalization
        self.col_indices = [i for i, var_name in enumerate(self.col_names) if f"{var_name}_mean" in self.config]
        self.means = np.array([self.config[f"{self.col_names[i]}_mean"] for i in self.col_indices])
        self.stds = np.array([self.config[f"{self.col_names[i]}_std"] for i in self.col_indices])
        # Load data
        self.h5_lookup = {}
        self.base_path = "/".join(self.file_path.split("/")[:-1])+"/"
        self.train_or_test = self.file_path.split("/")[-1]
        for filename in os.listdir(self.base_path):
            if filename.startswith(self.train_or_test) and filename.endswith(".h5"):
                self.h5_lookup[filename] = h5py.File(f"{self.base_path}{filename}", 'r')['tabular_data']
        # Point to a dataset
        with open(f"{self.file_path}_shingle_config.json") as f:
            # This contains all of the shingle information in the data file
            self.shingle_lookup = json.load(f)
            # This is a list of keys that will be filtered and each point to a sample
            self.shingle_keys = list(self.shingle_lookup.keys())
        # Filter out (or keep exclusively) any routes that are used for generalization tests
        if self.holdout_routes is not None:
            holdout_idxs = [self.shingle_lookup[x]['route_id'] in self.holdout_routes for x in self.shingle_lookup]
            if self.keep_only_holdout==True:
                self.shingle_keys = [i for (i,v) in zip(self.shingle_keys, holdout_idxs) if v]
            else:
                self.shingle_keys = [i for (i,v) in zip(self.shingle_keys, holdout_idxs) if not v]
    def __getitem__(self, index):
        # Get information on shingle file location and lines; read specific shingle lines from specific shingle file
        samp_dict = self.shingle_lookup[self.shingle_keys[index]]
        # df = self.data[self.data['shingle_id']==samp_dict['shingle_id']].values
        df = self.h5_lookup[f"{self.train_or_test}_data_{samp_dict['network']}_{samp_dict['file_num']}.h5"][samp_dict['start_idx']:samp_dict['end_idx']]
        df[:,self.col_indices] = (df[:,self.col_indices] - self.means) / self.stds
        # Convert tabular to dict of keys; inefficient but works with all other code for now
        res = {}
        for i in range(df.shape[1]):
            res[self.col_names[i]] = df[:,i]
        # Some keys should not be repeated across timesteps (use first or last value of sequence)
        res['shingle_id'] = res['shingle_id'][0]
        res['time'] = res['time_cumulative_s'][-1]
        res['dist'] = res['dist_cumulative_km'][-1]
        res['weekID'] = res['weekID'][0]
        res['timeID'] = res['timeID'][0]
        # Add grid if applicable
        if self.add_grid_features:
            xbin_idxs, ybin_idxs = self.grid.digitize_points(res['x'], res['y'])
            grid_features = self.grid.get_grid_features(xbin_idxs, ybin_idxs, [res['locationtime'][0] for x in res['locationtime']])
            grid_features = apply_grid_normalization(grid_features, self.config)
            res['grid_features'] = grid_features
        # res = apply_normalization(res, self.config)
        return res
    def __len__(self):
        return len(self.shingle_keys)
    def get_all_samples(self, keep_cols, indexes=None):
        # Read all h5 files in run base directory; get all point obs
        res = []
        for k in list(self.h5_lookup.keys()):
            df = self.h5_lookup[k][:]
            df = pd.DataFrame(df, columns=self.col_names)
            df['shingle_id'] = df['shingle_id'].astype(int)
            df = df[keep_cols]
            res.append(df)
        res = pd.concat(res)
        if indexes is not None:
            # Indexes are in order, but shingle_id's are not; get shingle id for each keep index and filter
            keep_shingles = [self.shingle_lookup[self.shingle_keys[i]]['shingle_id'] for i in indexes]
            res = res[res['shingle_id'].isin(keep_shingles)]
        return res

def apply_normalization(sample, config):
    for var_name in sample.keys():
        if f"{var_name}_mean" in config.keys():
            sample[var_name] = data_utils.normalize(np.array(sample[var_name]), config[f"{var_name}_mean"], config[f"{var_name}_std"])
    return sample

def apply_grid_normalization(grid_features, config):
    # Get the average age of observations in this shingle; or use estimate of global mean if all observations are nan
    obs_ages = grid_features[:,-1,:,:,:]
    obs_ages = obs_ages[~np.isnan(obs_ages)]
    if len(obs_ages)==0:
        # Estimate of global distribution
        obs_mean = 82000
        obs_std = 51000
    else:
        obs_mean = np.mean(obs_ages)
        obs_std = np.std(obs_ages)
    # Fill all nan values with the mean, then normalize
    grid_features[:,3,:,:,:] = np.nan_to_num(grid_features[:,3,:,:,:], config['speed_m_s_mean'])
    grid_features[:,4,:,:,:] = np.nan_to_num(grid_features[:,4,:,:,:], config['bearing_mean'])
    grid_features[:,-1,:,:,:] = np.nan_to_num(grid_features[:,-1,:,:,:], obs_mean)
    grid_features[:,3,:,:,:] = data_utils.normalize(grid_features[:,3,:,:,:], config[f"speed_m_s_mean"], config[f"speed_m_s_std"])
    grid_features[:,4,:,:,:] = data_utils.normalize(grid_features[:,4,:,:,:], config[f"bearing_mean"], config[f"bearing_std"])
    grid_features[:,-1,:,:,:] = data_utils.normalize(grid_features[:,-1,:,:,:], obs_mean, obs_std)
    # Only return the features we are interested in
    grid_features = grid_features[:,3:,:,:,:]
    return grid_features

# def basic_collate(batch):
#     y_col = "time_cumulative_s"
#     em_cols = ["timeID","weekID"]
#     first_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","speed_m_s","bearing"]
#     last_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","dist_cumulative_km","bearing"]
#     batch = [torch.tensor(b) for b in batch]
#     first = torch.cat([b[0,:].unsqueeze(0) for b in batch], axis=0)
#     last = torch.cat([b[-1,:].unsqueeze(0) for b in batch], axis=0)
#     X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])]
#     X_ct_first = first[:,([FEATURE_COLS.index(z) for z in first_cols])]
#     X_ct_last = last[:,([FEATURE_COLS.index(z) for z in last_cols])]
#     X_ct = torch.cat([X_ct_first, X_ct_last], axis=1)
#     y = last[:,(FEATURE_COLS.index(y_col))]
#     return [X_em.int(), X_ct.float()], y.float()
def basic_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['x_cent']) for x in batch]).int()
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    # Continuous features
    X_ct = np.zeros((len(batch), 12))
    X_ct[:,0] = [torch.tensor(x['x_cent'][0]) for x in batch]
    X_ct[:,1] = [torch.tensor(x['y_cent'][0]) for x in batch]
    X_ct[:,2] = [torch.tensor(x['x_cent'][-1]) for x in batch]
    X_ct[:,3] = [torch.tensor(x['y_cent'][-1]) for x in batch]
    X_ct[:,4] = [torch.tensor(x['scheduled_time_s'][0]) for x in batch]
    X_ct[:,5] = [torch.tensor(x['scheduled_time_s'][-1]) for x in batch]
    X_ct[:,6] = [torch.tensor(x['stop_dist_km'][0]) for x in batch]
    X_ct[:,7] = [torch.tensor(x['stop_dist_km'][-1]) for x in batch]
    X_ct[:,8] = [torch.sum(torch.tensor(x['passed_stops_n'])) for x in batch]
    X_ct[:,9] = [torch.tensor(x['speed_m_s'][0]) for x in batch]
    X_ct[:,10] = [torch.mean(torch.tensor(x['bearing'])) for x in batch]
    X_ct[:,11] = [torch.tensor(x['dist']) for x in batch]
    # Target featurex
    y = torch.from_numpy(np.array([x['time'] for x in batch]))
    # Sort all dataloaders so that they are consistent in the results
    X_em = torch.from_numpy(X_em)
    X_ct = torch.from_numpy(X_ct)
    X_em = X_em.int()
    X_ct = X_ct.float()
    y = y.float()
    return [X_em, X_ct], y

def basic_collate_nosch(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['x_cent']) for x in batch]).int()
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    # Continuous features
    X_ct = np.zeros((len(batch), 7))
    X_ct[:,0] = [torch.tensor(x['x_cent'][0]) for x in batch]
    X_ct[:,1] = [torch.tensor(x['y_cent'][0]) for x in batch]
    X_ct[:,2] = [torch.tensor(x['x_cent'][-1]) for x in batch]
    X_ct[:,3] = [torch.tensor(x['y_cent'][-1]) for x in batch]
    X_ct[:,4] = [torch.tensor(x['speed_m_s'][0]) for x in batch]
    X_ct[:,5] = [torch.mean(torch.tensor(x['bearing'])) for x in batch]
    X_ct[:,6] = [torch.tensor(x['dist']) for x in batch]
    # Target featurex
    y = torch.from_numpy(np.array([x['time'] for x in batch]))
    # Sort all dataloaders so that they are consistent in the results
    X_em = torch.from_numpy(X_em)
    X_ct = torch.from_numpy(X_ct)
    X_em = X_em.int()
    X_ct = X_ct.float()
    y = y.float()
    return [X_em, X_ct], y

def basic_grid_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['x_cent']) for x in batch]).int()
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    # Continuous features
    X_ct = np.zeros((len(batch), 12))
    X_ct[:,0] = [torch.tensor(x['x_cent'][0]) for x in batch]
    X_ct[:,1] = [torch.tensor(x['y_cent'][0]) for x in batch]
    X_ct[:,2] = [torch.tensor(x['x_cent'][-1]) for x in batch]
    X_ct[:,3] = [torch.tensor(x['y_cent'][-1]) for x in batch]
    X_ct[:,4] = [torch.tensor(x['scheduled_time_s'][0]) for x in batch]
    X_ct[:,5] = [torch.tensor(x['scheduled_time_s'][-1]) for x in batch]
    X_ct[:,6] = [torch.tensor(x['stop_dist_km'][0]) for x in batch]
    X_ct[:,7] = [torch.tensor(x['stop_dist_km'][-1]) for x in batch]
    X_ct[:,8] = [torch.sum(torch.tensor(x['passed_stops_n'])) for x in batch]
    X_ct[:,9] = [torch.tensor(x['speed_m_s'][0]) for x in batch]
    X_ct[:,10] = [torch.mean(torch.tensor(x['bearing'])) for x in batch]
    X_ct[:,11] = [torch.tensor(x['dist']) for x in batch]
    # Grid features
    X_gr = np.array([np.mean(np.concatenate([np.expand_dims(x, 0) for x in batch[i]['grid_features']]), axis=0) for i in range(len(batch))])
    # Target feature
    y = torch.from_numpy(np.array([x['time'] for x in batch]))
    # Sort all dataloaders so that they are consistent in the results
    X_em = torch.from_numpy(X_em)
    X_ct = torch.from_numpy(X_ct)
    X_gr = torch.from_numpy(X_gr)
    X_em = X_em.int()
    X_ct = X_ct.float()
    X_gr = X_gr.float()
    y = y.float()
    return [X_em, X_ct, X_gr], y

def basic_grid_collate_nosch(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['x_cent']) for x in batch]).int()
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    # Continuous features
    X_ct = np.zeros((len(batch), 7))
    X_ct[:,0] = [torch.tensor(x['x_cent'][0]) for x in batch]
    X_ct[:,1] = [torch.tensor(x['y_cent'][0]) for x in batch]
    X_ct[:,2] = [torch.tensor(x['x_cent'][-1]) for x in batch]
    X_ct[:,3] = [torch.tensor(x['y_cent'][-1]) for x in batch]
    X_ct[:,4] = [torch.tensor(x['speed_m_s'][0]) for x in batch]
    X_ct[:,5] = [torch.mean(torch.tensor(x['bearing'])) for x in batch]
    X_ct[:,6] = [torch.tensor(x['dist']) for x in batch]
    # Grid features
    X_gr = np.array([np.mean(np.concatenate([np.expand_dims(x, 0) for x in batch[i]['grid_features']]), axis=0) for i in range(len(batch))])
    # Target feature
    y = torch.from_numpy(np.array([x['time'] for x in batch]))
    # Sort all dataloaders so that they are consistent in the results
    X_em = torch.from_numpy(X_em)
    X_ct = torch.from_numpy(X_ct)
    X_gr = torch.from_numpy(X_gr)
    X_em = X_em.int()
    X_ct = X_ct.float()
    X_gr = X_gr.float()
    y = y.float()
    return [X_em, X_ct, X_gr], y

def sequential_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['x_cent']) for x in batch]).int()
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    X_em = torch.from_numpy(X_em)
    # Continuous features
    X_ct = torch.zeros((len(batch), max_len, 10))
    X_ct[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['x_cent']) for x in batch], batch_first=True)
    X_ct[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['y_cent']) for x in batch], batch_first=True)
    X_ct[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
    X_ct[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['scheduled_time_s']) for x in batch], batch_first=True)
    X_ct[:,:,4] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_dist_km']) for x in batch], batch_first=True)
    X_ct[:,:,5] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_x_cent']) for x in batch], batch_first=True)
    X_ct[:,:,6] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_y_cent']) for x in batch], batch_first=True)
    X_ct[:,:,7] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['bearing']) for x in batch], batch_first=True)
    X_ct[:,:,8] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['passed_stops_n']) for x in batch], batch_first=True)
    X_ct[:,:,9] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['passed_stops_n']) for x in batch], batch_first=True)
    # Target feature
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['time_calc_s']) for x in batch], batch_first=True)
    X_em = X_em.int()
    X_ct = X_ct.float()
    y = y.float()
    return [X_em, X_ct, seq_lens], y

def sequential_collate_nosch(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['x_cent']) for x in batch]).int()
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    X_em = torch.from_numpy(X_em)
    # Continuous features
    X_ct = torch.zeros((len(batch), max_len, 4))
    X_ct[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['x_cent']) for x in batch], batch_first=True)
    X_ct[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['y_cent']) for x in batch], batch_first=True)
    X_ct[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
    X_ct[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['bearing']) for x in batch], batch_first=True)
    # Target feature
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['time_calc_s']) for x in batch], batch_first=True)
    X_em = X_em.int()
    X_ct = X_ct.float()
    y = y.float()
    return [X_em, X_ct, seq_lens], y

def sequential_grid_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['x_cent']) for x in batch]).int()
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    X_em = torch.from_numpy(X_em)
    # Continuous features
    X_ct = torch.zeros((len(batch), max_len, 10))
    X_ct[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['x_cent']) for x in batch], batch_first=True)
    X_ct[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['y_cent']) for x in batch], batch_first=True)
    X_ct[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
    X_ct[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['scheduled_time_s']) for x in batch], batch_first=True)
    X_ct[:,:,4] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_dist_km']) for x in batch], batch_first=True)
    X_ct[:,:,5] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_x_cent']) for x in batch], batch_first=True)
    X_ct[:,:,6] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_y_cent']) for x in batch], batch_first=True)
    X_ct[:,:,7] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['bearing']) for x in batch], batch_first=True)
    X_ct[:,:,8] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['passed_stops_n']) for x in batch], batch_first=True)
    X_ct[:,:,9] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['passed_stops_n']) for x in batch], batch_first=True)
    # Grid features (NxCxLxHxW)
    z = [torch.tensor(np.array(x['grid_features'])) for x in batch]
    X_gr = torch.nn.utils.rnn.pad_sequence(z, batch_first=True)
    # Target feature
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['time_calc_s']) for x in batch], batch_first=True)
    X_em = X_em.int()
    X_ct = X_ct.float()
    X_gr = X_gr.float()
    y = y.float()
    return [X_em, X_ct, X_gr, seq_lens], y

def sequential_grid_collate_nosch(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['x_cent']) for x in batch]).int()
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    X_em = torch.from_numpy(X_em)
    # Continuous features
    X_ct = torch.zeros((len(batch), max_len, 4))
    X_ct[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['x_cent']) for x in batch], batch_first=True)
    X_ct[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['y_cent']) for x in batch], batch_first=True)
    X_ct[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
    X_ct[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['bearing']) for x in batch], batch_first=True)
    # Grid features (NxCxLxHxW)
    z = [torch.tensor(np.array(x['grid_features'])) for x in batch]
    X_gr = torch.nn.utils.rnn.pad_sequence(z, batch_first=True)
    # Target feature
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['time_calc_s']) for x in batch], batch_first=True)
    X_em = X_em.int()
    X_ct = X_ct.float()
    X_gr = X_gr.float()
    y = y.float()
    return [X_em, X_ct, X_gr, seq_lens], y

def deeptte_collate(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['weekID', 'timeID']
    traj_attrs = ['y_cent','x_cent','time_calc_s','dist_calc_km','bearing','scheduled_time_s','stop_dist_km','stop_x_cent','stop_y_cent','passed_stops_n']
    attr, traj = {}, {}
    lens = np.asarray([len(item['x_cent']) for item in data])
    for key in stat_attrs:
        attr[key] = torch.FloatTensor([item[key] for item in data])
    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])
    for key in traj_attrs:
        seqs = np.asarray([item[key] for item in data], dtype=object)
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype = np.float32)
        padded[mask] = np.concatenate(seqs)
        padded = torch.from_numpy(padded).float()
        traj[key] = padded
    lens = lens.tolist()
    traj['lens'] = lens
    return [attr, traj]

def deeptte_collate_nosch(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['weekID', 'timeID']
    traj_attrs = ['y_cent', 'x_cent', 'time_calc_s', 'dist_calc_km']
    attr, traj = {}, {}
    lens = np.asarray([len(item['x_cent']) for item in data])
    for key in stat_attrs:
        attr[key] = torch.FloatTensor([item[key] for item in data])
    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])
    for key in traj_attrs:
        seqs = np.asarray([item[key] for item in data], dtype=object)
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype = np.float32)
        padded[mask] = np.concatenate(seqs)
        padded = torch.from_numpy(padded).float()
        traj[key] = padded
    lens = lens.tolist()
    traj['lens'] = lens
    return [attr, traj]