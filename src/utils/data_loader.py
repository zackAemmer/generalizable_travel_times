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

class LoadSliceDataset(Dataset):
    def __init__(self, file_path, config, grid=None, holdout_routes=None, keep_only_holdout=False, add_grid_features=False, skip_gtfs=False):
        self.file_path = file_path
        self.config = config
        self.grid = grid
        self.holdout_routes = holdout_routes
        self.keep_only_holdout = keep_only_holdout
        self.add_grid_features = add_grid_features
        self.skip_gtfs = skip_gtfs
        # Necessary to convert from np array tabular format saved in h5 files
        if not self.skip_gtfs:
            self.col_names = FEATURE_COLS
        else:
            self.col_names = SKIP_FEATURE_COLS
        # Keep open files for the dataset
        self.h5_lookup = {}
        self.base_path = "/".join(self.file_path.split("/")[:-1])+"/"
        self.train_or_test = self.file_path.split("/")[-1]
        for filename in os.listdir(self.base_path):
            if filename.startswith(self.train_or_test) and filename.endswith(".h5"):
                self.h5_lookup[filename] = h5py.File(f"{self.base_path}{filename}", 'r')['tabular_data']
        # Read shingle lookup corresponding to dataset
        # This is a list of keys that will be filtered and each point to a sample
        with open(f"{self.file_path}_shingle_config.json") as f:
            self.shingle_lookup = json.load(f)
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
        samp = self.h5_lookup[f"{self.train_or_test}_data_{samp_dict['network']}_{samp_dict['file_num']}.h5"][samp_dict['start_idx']:samp_dict['end_idx']]
        if not self.add_grid_features:
            return {"samp": samp}
        else:
            xbin_idxs, ybin_idxs = self.grid.digitize_points(samp[:,self.col_names.index("x")], samp[:,self.col_names.index("y")])
            grid_features = self.grid.get_grid_features(xbin_idxs, ybin_idxs, samp[:,self.col_names.index("locationtime")])
            return {"samp": samp, "grid": grid_features}
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

def avg_collate(batch):
    cols = ["speed_m_s","dist_calc_km","timeID_s","time_cumulative_s"]
    col_idxs = [FEATURE_COLS.index(cname) for cname in cols]
    avg_speeds = [np.mean(b['samp'][:,col_idxs[0]]) for b in batch]
    tot_dists = [np.sum(b['samp'][:,col_idxs[1]])*1000 for b in batch]
    start_times = [b['samp'][0,col_idxs[2]]//3600 for b in batch]
    tot_times = [b['samp'][-1,col_idxs[3]] for b in batch]
    return (avg_speeds, tot_dists, start_times, tot_times)
def schedule_collate(batch):
    cols = ["scheduled_time_s","time_cumulative_s"]
    col_idxs = [FEATURE_COLS.index(cname) for cname in cols]
    sch_times = [b['samp'][-1,col_idxs[0]] for b in batch]
    tot_times = [b['samp'][-1,col_idxs[1]] for b in batch]
    return (sch_times, tot_times)
def persistent_collate(batch):
    seq_lens = [b['samp'].shape[0] for b in batch]
    cols = ["time_cumulative_s"]
    col_idxs = [FEATURE_COLS.index(cname) for cname in cols]
    tot_times = [b['samp'][-1,col_idxs[0]] for b in batch]
    return (seq_lens, tot_times)

def basic_collate(batch):
    y_col = "time_cumulative_s"
    em_cols = ["timeID","weekID"]
    first_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","speed_m_s","bearing"]
    last_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","dist_cumulative_km","bearing"]
    batch = [torch.tensor(b['samp']) for b in batch]
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch], axis=0)
    last = torch.cat([b[-1,:].unsqueeze(0) for b in batch], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])]
    X_ct_first = first[:,([FEATURE_COLS.index(z) for z in first_cols])]
    X_ct_last = last[:,([FEATURE_COLS.index(z) for z in last_cols])]
    X_ct = torch.cat([X_ct_first, X_ct_last], axis=1)
    y = last[:,(FEATURE_COLS.index(y_col))]
    return [X_em.int(), X_ct.float()], y.float()
def basic_collate_nosch(batch):
    y_col = "time_cumulative_s"
    em_cols = ["timeID","weekID"]
    first_cols = ["x_cent","y_cent","speed_m_s","bearing"]
    last_cols = ["x_cent","y_cent","dist_cumulative_km","bearing"]
    batch = [torch.tensor(b['samp']) for b in batch]
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch], axis=0)
    last = torch.cat([b[-1,:].unsqueeze(0) for b in batch], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])]
    X_ct_first = first[:,([FEATURE_COLS.index(z) for z in first_cols])]
    X_ct_last = last[:,([FEATURE_COLS.index(z) for z in last_cols])]
    X_ct = torch.cat([X_ct_first, X_ct_last], axis=1)
    y = last[:,(FEATURE_COLS.index(y_col))]
    return [X_em.int(), X_ct.float()], y.float()
def basic_grid_collate(batch):
    y_col = "time_cumulative_s"
    em_cols = ["timeID","weekID"]
    first_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","speed_m_s","bearing"]
    last_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","passed_stops_n","dist_cumulative_km","bearing"]
    batch_ct = [torch.tensor(b['samp']) for b in batch]
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch_ct], axis=0)
    last = torch.cat([b[-1,:].unsqueeze(0) for b in batch_ct], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])]
    X_ct_first = first[:,([FEATURE_COLS.index(z) for z in first_cols])]
    X_ct_last = last[:,([FEATURE_COLS.index(z) for z in last_cols])]
    X_ct = torch.cat([X_ct_first, X_ct_last], axis=1)
    y = last[:,(FEATURE_COLS.index(y_col))]
    # Get speed/bearing/obs age from grid results; average out all values across timesteps; 1 value per cell/obs/variable
    X_gr = torch.cat([torch.nanmean(torch.tensor(z['grid'])[:,3:,:,:,:], dim=0).unsqueeze(0) for z in batch])
    # Replace any nans with the average for that feature
    means = torch.nanmean(torch.swapaxes(X_gr, 0, 1).flatten(1), dim=1)
    for i,m in enumerate(means):
        X_gr[:,i,:,:,:] = torch.nan_to_num(X_gr[:,i,:,:,:], m)
    return [X_em.int(), X_ct.float(), X_gr.float()], y.float()
def basic_grid_collate_nosch(batch):
    y_col = "time_cumulative_s"
    em_cols = ["timeID","weekID"]
    first_cols = ["x_cent","y_cent","speed_m_s","bearing"]
    last_cols = ["x_cent","y_cent","dist_cumulative_km","bearing"]
    batch_ct = [torch.tensor(b['samp']) for b in batch]
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch_ct], axis=0)
    last = torch.cat([b[-1,:].unsqueeze(0) for b in batch_ct], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])]
    X_ct_first = first[:,([FEATURE_COLS.index(z) for z in first_cols])]
    X_ct_last = last[:,([FEATURE_COLS.index(z) for z in last_cols])]
    X_ct = torch.cat([X_ct_first, X_ct_last], axis=1)
    y = last[:,(FEATURE_COLS.index(y_col))]
    # Get speed/bearing/obs age from grid results; average out all values across timesteps; 1 value per cell/obs/variable
    X_gr = torch.cat([torch.nanmean(torch.tensor(z['grid'])[:,3:,:,:,:], dim=0).unsqueeze(0) for z in batch])
    # Replace any nans with the average for that feature
    means = torch.nanmean(torch.swapaxes(X_gr, 0, 1).flatten(1), dim=1)
    for i,m in enumerate(means):
        X_gr[:,i,:,:,:] = torch.nan_to_num(X_gr[:,i,:,:,:], m)
    return [X_em.int(), X_ct.float(), X_gr.float()], y.float()

def sequential_collate(batch):
    y_col = "time_calc_s"
    em_cols = ["timeID","weekID"]
    ct_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","stop_x_cent","stop_y_cent","passed_stops_n","bearing","dist_calc_km","dist_calc_km"]
    batch = [torch.tensor(b['samp']) for b in batch]
    X_sl = torch.tensor([len(b) for b in batch])
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    X_ct = batch[:,:,([FEATURE_COLS.index(z) for z in ct_cols])]
    y = batch[:,:,(FEATURE_COLS.index(y_col))]
    return [X_em.int(), X_ct.float(), X_sl.int()], y.float()
def sequential_collate_nosch(batch):
    y_col = "time_calc_s"
    em_cols = ["timeID","weekID"]
    ct_cols = ["x_cent","y_cent","bearing","dist_calc_km"]
    batch = [torch.tensor(b['samp']) for b in batch]
    X_sl = torch.tensor([len(b) for b in batch])
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    X_ct = batch[:,:,([FEATURE_COLS.index(z) for z in ct_cols])]
    y = batch[:,:,(FEATURE_COLS.index(y_col))]
    return [X_em.int(), X_ct.float(), X_sl.int()], y.float()
def sequential_grid_collate(batch):
    y_col = "time_calc_s"
    em_cols = ["timeID","weekID"]
    ct_cols = ["x_cent","y_cent","scheduled_time_s","stop_dist_km","stop_x_cent","stop_y_cent","passed_stops_n","bearing","dist_calc_km","dist_calc_km"]
    batch_ct = [torch.tensor(b['samp']) for b in batch]
    batch_gr = [torch.tensor(b['grid']) for b in batch]
    X_sl = torch.tensor([len(b) for b in batch_ct])
    first = torch.cat([b[0,:].unsqueeze(0) for b in batch_ct], axis=0)
    X_em = first[:,([FEATURE_COLS.index(z) for z in em_cols])]
    batch_ct = torch.nn.utils.rnn.pad_sequence(batch_ct, batch_first=True)
    X_ct = batch_ct[:,:,([FEATURE_COLS.index(z) for z in ct_cols])]
    y = batch_ct[:,:,(FEATURE_COLS.index(y_col))]
    # Get speed/bearing/obs age from grid results; average out all values across timesteps; 1 value per cell/obs/variable
    X_gr = [b[:,3:,:,:,:] for b in batch_gr]
    X_gr = torch.nn.utils.rnn.pad_sequence(X_gr, batch_first=True)
    # Replace any nans with the average for that feature
    means = torch.nanmean(torch.swapaxes(X_gr, 0, 2).flatten(1), dim=1)
    for i,m in enumerate(means):
        X_gr[:,:,i,:,:,:] = torch.nan_to_num(X_gr[:,:,i,:,:,:], m)
    return [X_em.int(), X_ct.float(), X_gr.float(), X_sl.int()], y.float()

def deeptte_collate(data):
    stat_attrs = ['dist_cumulative_km', 'time_cumulative_s']
    stat_names = ['dist','time']
    info_attrs = ['weekID', 'timeID']
    traj_attrs = ['y_cent','x_cent','time_calc_s','dist_calc_km','bearing','scheduled_time_s','stop_dist_km','stop_x_cent','stop_y_cent','passed_stops_n']
    attr, traj = {}, {}
    batch_ct = [torch.tensor(b['samp']) for b in data]
    lens = np.array([len(b) for b in batch_ct])
    for n,key in zip(stat_names, stat_attrs):
        attr[n] = torch.FloatTensor([d['samp'][-1,FEATURE_COLS.index(key)] for d in data])
    for key in info_attrs:
        attr[key] = torch.LongTensor([int(d['samp'][0,FEATURE_COLS.index(key)]) for d in data])
    for key in traj_attrs:
        seqs = [d['samp'][:,FEATURE_COLS.index(key)] for d in data]
        seqs = np.asarray(seqs, dtype=object)
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate(seqs)
        padded = torch.from_numpy(padded).float()
        traj[key] = padded
    lens = lens.tolist()
    traj['lens'] = lens
    return [attr, traj]
def deeptte_collate_nosch(data):
    stat_attrs = ['dist_cumulative_km', 'time_cumulative_s']
    stat_names = ['dist','time']
    info_attrs = ['weekID', 'timeID']
    traj_attrs = ['y_cent', 'x_cent', 'time_calc_s', 'dist_calc_km']
    attr, traj = {}, {}
    batch_ct = [torch.tensor(b['samp']) for b in data]
    lens = np.array([len(b) for b in batch_ct])
    for n,key in zip(stat_names, stat_attrs):
        attr[n] = torch.FloatTensor([d['samp'][-1,FEATURE_COLS.index(key)] for d in data])
    for key in info_attrs:
        attr[key] = torch.LongTensor([int(d['samp'][0,FEATURE_COLS.index(key)]) for d in data])
    for key in traj_attrs:
        seqs = [d['samp'][:,FEATURE_COLS.index(key)] for d in data]
        seqs = np.asarray(seqs, dtype=object)
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate(seqs)
        padded = torch.from_numpy(padded).float()
        traj[key] = padded
    lens = lens.tolist()
    traj['lens'] = lens
    return [attr, traj]