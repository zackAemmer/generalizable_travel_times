import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models import grids
from utils import data_utils, shape_utils


class GenericDataset(Dataset):
    def __init__(self, content, config, grid=None, is_ngrid=None, buffer=1):
        self.content = content
        self.config = config
        self.grid = grid
        self.is_ngrid = is_ngrid
        self.buffer = buffer
    def __getitem__(self, index):
        sample = self.content[index]
        if self.grid is not None:
            # Handles normalization, and selection of the specific buffered t/x/y bins
            if self.is_ngrid:
                grid_features = grids.extract_ngrid_features(self.grid, sample['tbin_idx'], sample['xbin_idx'], sample['ybin_idx'], self.config, self.buffer)
            else:
                grid_features = grids.extract_grid_features(self.grid, sample['tbin_idx'], sample['xbin_idx'], sample['ybin_idx'], self.config, self.buffer)
            sample['grid_features'] = grid_features
        sample = apply_normalization(sample.copy(), self.config)
        return sample
    def __len__(self):
        return len(self.content)

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

def make_generic_dataloader(data, config, batch_size, collate_fn, num_workers, grid=None, is_ngrid=None, buffer=None):
    dataset = GenericDataset(data, config, grid, is_ngrid, buffer)
    if num_workers > 0:
        pin_memory=True
    else:
        pin_memory=False
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    return dataloader

def basic_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = [len(x['lats']) for x in batch]
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    # Continuous features
    X_ct = np.zeros((len(batch), 12))
    X_ct[:,0] = [torch.tensor(x['x'][0]) for x in batch]
    X_ct[:,1] = [torch.tensor(x['y'][0]) for x in batch]
    X_ct[:,2] = [torch.tensor(x['x'][-1]) for x in batch]
    X_ct[:,3] = [torch.tensor(x['y'][-1]) for x in batch]
    X_ct[:,4] = [torch.tensor(x['scheduled_time_s'][0]) for x in batch]
    X_ct[:,5] = [torch.tensor(x['stop_dist_km'][0]) for x in batch]
    X_ct[:,6] = [torch.tensor(x['stop_x'][-1]) for x in batch]
    X_ct[:,7] = [torch.tensor(x['stop_y'][-1]) for x in batch]
    X_ct[:,8] = [torch.tensor(x['speed_m_s'][0]) for x in batch]
    X_ct[:,9] = [torch.tensor(x['bearing'][0]) for x in batch]
    X_ct[:,10] = [torch.tensor(x['dist']) for x in batch]
    X_ct[:,11] = [torch.sum(torch.tensor(x['passed_stops_n'])) for x in batch]
    # Target featurex
    y = torch.from_numpy(np.array([x['time'] for x in batch]))
    # Sort all dataloaders so that they are consistent in the results
    X_em = torch.from_numpy(X_em)
    X_ct = torch.from_numpy(X_ct)
    sorted_slens, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
    sorted_slens = sorted_slens.int()
    X_em = X_em[sorted_indices,:].int()
    X_ct = X_ct[sorted_indices,:].float()
    y = y[sorted_indices].float()
    return [X_em, X_ct], y

def basic_grid_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = [len(x['lats']) for x in batch]
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    # Continuous features
    X_ct = np.zeros((len(batch), 12))
    X_ct[:,0] = [torch.tensor(x['x'][0]) for x in batch]
    X_ct[:,1] = [torch.tensor(x['y'][0]) for x in batch]
    X_ct[:,2] = [torch.tensor(x['x'][-1]) for x in batch]
    X_ct[:,3] = [torch.tensor(x['y'][-1]) for x in batch]
    X_ct[:,4] = [torch.tensor(x['scheduled_time_s'][0]) for x in batch]
    X_ct[:,5] = [torch.tensor(x['stop_dist_km'][0]) for x in batch]
    X_ct[:,6] = [torch.tensor(x['stop_x'][-1]) for x in batch]
    X_ct[:,7] = [torch.tensor(x['stop_y'][-1]) for x in batch]
    X_ct[:,8] = [torch.tensor(x['speed_m_s'][0]) for x in batch]
    X_ct[:,9] = [torch.tensor(x['bearing'][0]) for x in batch]
    X_ct[:,10] = [torch.tensor(x['dist']) for x in batch]
    X_ct[:,11] = [torch.sum(torch.tensor(x['passed_stops_n'])) for x in batch]
    # Grid features
    X_gr = np.zeros((len(batch), 8, 3, 3))
    X_gr = np.array([np.mean(np.concatenate([np.expand_dims(x, 0) for x in batch[i]['grid_features']]), axis=0) for i in range(len(batch))])
    # Target feature
    y = torch.from_numpy(np.array([x['time'] for x in batch]))
    # Sort all dataloaders so that they are consistent in the results
    X_em = torch.from_numpy(X_em)
    X_ct = torch.from_numpy(X_ct)
    X_gr = torch.from_numpy(X_gr)
    sorted_slens, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
    sorted_slens = sorted_slens.int()
    X_em = X_em[sorted_indices,:].int()
    X_ct = X_ct[sorted_indices,:].float()
    X_gr = X_gr[sorted_indices,:].float()
    y = y[sorted_indices].float()
    return [X_em, X_ct, X_gr], y

def sequential_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = [len(x['lats']) for x in batch]
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    X_em = torch.from_numpy(X_em)
    # Continuous features
    X_ct = torch.zeros((len(batch), max_len, 9))
    X_ct[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['x']) for x in batch], batch_first=True)
    X_ct[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['y']) for x in batch], batch_first=True)
    X_ct[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
    X_ct[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['scheduled_time_s']) for x in batch], batch_first=True)
    X_ct[:,:,4] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_dist_km']) for x in batch], batch_first=True)
    X_ct[:,:,5] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_x']) for x in batch], batch_first=True)
    X_ct[:,:,6] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_y']) for x in batch], batch_first=True)
    X_ct[:,:,7] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['bearing']) for x in batch], batch_first=True)
    X_ct[:,:,8] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['passed_stops_n']) for x in batch], batch_first=True)
    # Target feature
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['time_calc_s']) for x in batch], batch_first=True)
    # Sort all by sequence length descending, for potential packing of each batch
    sorted_slens, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
    sorted_slens = sorted_slens.int()
    X_em = X_em[sorted_indices,:].int()
    X_ct = X_ct[sorted_indices,:,:].float()
    y = y[sorted_indices,:].float()
    return [X_em, X_ct, sorted_slens], y

def sequential_grid_collate(batch):
    # Last dimension is number of sequence variables below
    seq_lens = [len(x['lats']) for x in batch]
    max_len = max(seq_lens)
    # Embedded context features
    X_em = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
    X_em = torch.from_numpy(X_em)
    # Continuous features
    X_ct = torch.zeros((len(batch), max_len, 9))
    X_ct[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['x']) for x in batch], batch_first=True)
    X_ct[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['y']) for x in batch], batch_first=True)
    X_ct[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
    X_ct[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['scheduled_time_s']) for x in batch], batch_first=True)
    X_ct[:,:,4] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_dist_km']) for x in batch], batch_first=True)
    X_ct[:,:,5] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_x']) for x in batch], batch_first=True)
    X_ct[:,:,6] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_y']) for x in batch], batch_first=True)
    X_ct[:,:,7] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['bearing']) for x in batch], batch_first=True)
    X_ct[:,:,8] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['passed_stops_n']) for x in batch], batch_first=True)
    # Grid features (NxCxLxHxW)
    z = [torch.tensor(np.array(x['grid_features'])) for x in batch]
    X_gr = torch.nn.utils.rnn.pad_sequence(z, batch_first=True)
    # Target feature
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['time_calc_s']) for x in batch], batch_first=True)
    # Sort all by sequence length descending, for potential packing of each batch
    sorted_slens, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
    sorted_slens = sorted_slens.int()
    X_em = X_em[sorted_indices,:].int()
    X_ct = X_ct[sorted_indices,:,:].float()
    X_gr = X_gr[sorted_indices,:,:].float()
    y = y[sorted_indices,:].float()
    return [X_em, X_ct, X_gr, sorted_slens], y

# def sequential_mto_collate(batch):
#     # Context variables to embed
#     context = np.array([np.array([x['timeID'], x['weekID']]) for x in batch], dtype='int32')
#     # Last dimension is number of sequence variables below
#     seq_lens = [len(x['lats']) for x in batch]
#     max_len = max(seq_lens)
#     X = torch.zeros((len(batch), max_len, 9))
#     # Sequence variables
#     X[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['x']) for x in batch], batch_first=True)
#     X[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['y']) for x in batch], batch_first=True)
#     X[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
#     X[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['scheduled_time_s']) for x in batch], batch_first=True)
#     X[:,:,4] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_dist_km']) for x in batch], batch_first=True)
#     X[:,:,5] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_x']) for x in batch], batch_first=True)
#     X[:,:,6] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_y']) for x in batch], batch_first=True)
#     X[:,:,7] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['bearing']) for x in batch], batch_first=True)
#     X_ct[:,:,8] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['passed_stops_n']) for x in batch], batch_first=True)
#     context = torch.from_numpy(context)
#     y = torch.from_numpy(np.array([x['time'] for x in batch]))
#     # Sort all by sequence length descending, for potential packing of each batch
#     sorted_slens, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
#     sorted_slens = sorted_slens.int()
#     context = context[sorted_indices,:].int()
#     X = X[sorted_indices,:,:].float()
#     y = y[sorted_indices].float()
#     return [context, X, sorted_slens], y