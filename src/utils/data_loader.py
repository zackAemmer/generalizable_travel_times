import json
from random import sample

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from models import grids
from utils import data_utils, shape_utils


class GenericDataset(Dataset):
    def __init__(self, file_path, config, grid=None, subset=None, holdout_routes=None, keep_only_holdout=False, add_grid_features=False, skip_gtfs=False):
        self.file_path = file_path
        self.config = config
        self.grid = grid
        self.subset = subset
        self.holdout_routes = holdout_routes
        self.keep_only_holdout = keep_only_holdout
        self.add_grid_features = add_grid_features
        self.skip_gtfs = skip_gtfs
        # Point to a parquet datafile
        self.pq_dataset = ds.dataset(self.file_path, format="parquet")
        # Read all into memory and reformat
        self.content = data_utils.map_to_records(self.pq_dataset.to_table().to_pandas(), self.skip_gtfs)
        # Filter out (or keep exclusively) any routes that are used for generalization tests
        if self.holdout_routes is not None:
            is_holdout = [x['route_id'] in holdout_routes for x in self.content]
            if self.keep_only_holdout==True:
                self.content = [sample for (sample, is_holdout) in zip(self.content,is_holdout) if is_holdout]
            else:
                self.content = [sample for (sample, is_holdout) in zip(self.content,is_holdout) if not is_holdout]
        if self.subset is not None:
            if self.subset < 1:
                self.content = sample(self.content, int(len(self.content)*self.subset))
            else:
                self.content = sample(self.content, self.subset)
        # Save length of all shingles
        self.lengths = list(map(lambda x: len(x['lon']), self.content))
        # if self.holdout_routes is not None:
        #     if self.keep_only_holdout==True:
        #         data = data[data["route_id"].isin(self.holdout_routes)]
        #     else:
        #         data = data[~data["route_id"].isin(self.holdout_routes)]
        # if self.subset is not None:
        #     if self.subset < 1:
        #         keep_amnt = int(len(pd.unique(data["shingle_id"])) * self.subset)
        #         keep_ids = np.random.choice(pd.unique(data["shingle_id"]), keep_amnt)
        #         data = data[data["shingle_id"].isin(keep_ids)]
        #     else:
        #         keep_ids = np.random.choice(pd.unique(data["shingle_id"]), self.subset)
        #         data = data[data["shingle_id"].isin(keep_ids)]

    def __getitem__(self, index):
        # If precomputing:
        sample = self.content[index].copy()
        # # If trying to read partially:
        # sample = self.pq_dataset.to_table(filter=ds.field("shingle_id")==index).to_pandas()
        # sample = data_utils.map_to_records(sample, self.skip_gtfs)[0]
        # Add grid if applicable
        if self.grid is not None and self.add_grid_features:
            xbin_idxs, ybin_idxs = self.grid.digitize_points(sample['x'], sample['y'])
            grid_features = self.grid.get_grid_features(xbin_idxs, ybin_idxs, [sample['locationtime'][0] for x in sample['lon']])
            grid_features = apply_grid_normalization(grid_features, self.config)
            sample['grid_features'] = grid_features
        sample = apply_normalization(sample, self.config)
        return sample
    def __len__(self):
        # If precomputing:
        return len(self.content)
        # # If trying to read partially:
        # data = self.pq_dataset.to_table(filter=ds.field("shingle_id")==index).to_pandas()

def apply_normalization(sample, config):
    for var_name in sample.keys():
        # DeepTTE is inconsistent with sample and config naming schema.
        # These variables are in the sample (as cumulatives), but not the config.
        # The same variable names are in the config, but refer to non-cumulative values.
        # They have been added here to the config as time/dist cumulative.
        if var_name == "time_gap":
            sample[var_name] = data_utils.normalize(np.array(sample[var_name]), config[f"time_cumulative_s_mean"], config[f"time_cumulative_s_std"])
        elif var_name == "dist_gap":
            sample[var_name] = data_utils.normalize(np.array(sample[var_name]), config["dist_cumulative_km_mean"], config["dist_cumulative_km_std"])
        elif f"{var_name}_mean" in config.keys():
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

def basic_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = torch.tensor([len(x['lat']) for x in batch]).int()
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
    seq_lens = torch.tensor([len(x['lat']) for x in batch]).int()
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
    seq_lens = torch.tensor([len(x['lat']) for x in batch]).int()
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
    seq_lens = torch.tensor([len(x['lat']) for x in batch]).int()
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
    seq_lens = torch.tensor([len(x['lat']) for x in batch]).int()
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
    seq_lens = torch.tensor([len(x['lat']) for x in batch]).int()
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
    seq_lens = torch.tensor([len(x['lat']) for x in batch]).int()
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
    seq_lens = torch.tensor([len(x['lat']) for x in batch]).int()
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
    lens = np.asarray([len(item['lon']) for item in data])
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
    lens = np.asarray([len(item['lon']) for item in data])
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