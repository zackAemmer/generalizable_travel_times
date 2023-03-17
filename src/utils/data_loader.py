import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import data_utils


class GenericDataset(Dataset):
    def __init__(self, data, config):
        self.content = data
        self.config = config

    def __getitem__(self, index):
        sample = self.content[index]
        sample = apply_normalization(sample.copy(), self.config)
        return sample

    def __len__(self):
        return len(self.content)

def basic_collate(batch):
    # Context variables to embed
    context = np.array([np.array([x['timeID'], x['weekID'], x['vehicleID'], x['tripID']]) for x in batch], dtype='int32')
    # Last dimension is number of sequence variables below
    seq_lens = [len(x['lats']) for x in batch]
    max_len = max(seq_lens)
    X = np.zeros((len(batch), 8))
    # Sequence variables
    for i in range(len(batch)):
        X[i,0] = batch[i]['lats'][0]
        X[i,1] = batch[i]['lngs'][0]
        X[i,2] = batch[i]['lats'][-1]
        X[i,3] = batch[i]['lngs'][-1]
        X[i,4] = batch[i]['scheduled_time_s'][-1]
        X[i,5] = batch[i]['stop_dist_km'][-1]
        X[i,6] = batch[i]['speed_m_s'][0]
        X[i,7] = batch[i]['dist']
    X = torch.from_numpy(X)
    context = torch.from_numpy(context)
    # Prediction variable (travel time from first to last observation)
    y = torch.from_numpy(np.array([x['time'] for x in batch]))
    # Sort all dataloaders so that they are consistent in the results
    sorted_slens, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
    sorted_slens = sorted_slens.int()
    context = context[sorted_indices,:].int()
    X = X[sorted_indices,:].float()
    y = y[sorted_indices].float()
    return [context, X], y

def sequential_collate(batch):
    # Context variables to embed
    context = np.array([np.array([x['timeID'], x['weekID'], x['vehicleID'], x['tripID']]) for x in batch], dtype='int32')
    # Last dimension is number of sequence variables below
    seq_lens = [len(x['lats']) for x in batch]
    max_len = max(seq_lens)
    X = torch.zeros((len(batch), max_len, 7))
    # Sequence variables
    X[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['lats']) for x in batch], batch_first=True)
    X[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['lngs']) for x in batch], batch_first=True)
    X[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
    X[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['scheduled_time_s']) for x in batch], batch_first=True)
    X[:,:,4] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_dist_km']) for x in batch], batch_first=True)
    # Used for persistent model, do not use in RNN
    X[:,:,5] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['speed_m_s']) for x in batch], batch_first=True)
    context = torch.from_numpy(context)
    # Prediction variable (speed of each step)
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['speed_m_s']) for x in batch], batch_first=True)
    # Sort all by sequence length descending, for potential packing of each batch
    sorted_slens, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
    sorted_slens = sorted_slens.int()
    context = context[sorted_indices,:].int()
    X = X[sorted_indices,:,:].float()
    y = y[sorted_indices,:].float()
    return [context, X, sorted_slens], y

def sequential_tt_collate(batch):
    # Context variables to embed
    context = np.array([np.array([x['timeID'], x['weekID'], x['vehicleID'], x['tripID']]) for x in batch], dtype='int32')
    # Last dimension is number of sequence variables below
    seq_lens = [len(x['lats']) for x in batch]
    max_len = max(seq_lens)
    X = torch.zeros((len(batch), max_len, 9))
    # Sequence variables
    X[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['lats']) for x in batch], batch_first=True)
    X[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['lngs']) for x in batch], batch_first=True)
    X[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['dist_calc_km']) for x in batch], batch_first=True)
    X[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['scheduled_time_s']) for x in batch], batch_first=True)
    X[:,:,4] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_dist_km']) for x in batch], batch_first=True)
    X[:,:,5] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_lat']) for x in batch], batch_first=True)
    X[:,:,6] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['stop_lon']) for x in batch], batch_first=True)
    X[:,:,7] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['bearing']) for x in batch], batch_first=True)
    # Used for persistent model, do not use in RNN
    X[:,:,8] = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['speed_m_s']) for x in batch], batch_first=True)
    context = torch.from_numpy(context)
    # Prediction variable (speed of each step)
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(x['time_calc_s']) for x in batch], batch_first=True)
    # Sort all by sequence length descending, for potential packing of each batch
    sorted_slens, sorted_indices = torch.sort(torch.tensor(seq_lens), descending=True)
    sorted_slens = sorted_slens.int()
    context = context[sorted_indices,:].int()
    X = X[sorted_indices,:,:].float()
    y = y[sorted_indices,:].float()
    return [context, X, sorted_slens], y

def make_generic_dataloader(data, config, batch_size, task_type, num_workers):
    dataset = GenericDataset(data, config)
    if num_workers > 0:
        pin_memory=True
    else:
        pin_memory=False
    if task_type == "basic":
        dataloader = DataLoader(dataset, collate_fn=basic_collate, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    elif task_type == "sequential":
        dataloader = DataLoader(dataset, collate_fn=sequential_collate, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    elif task_type == "sequential_tt":
        dataloader = DataLoader(dataset, collate_fn=sequential_tt_collate, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    return dataloader

def apply_normalization(sample, config):
    for var_name in sample.keys():
        # DeepTTE is inconsistent with sample and config naming schema.
        # These variables are in the sample (as cumulatives), but not the config.
        # The same variable names are in the config, but refer to non-cumulative values.
        if var_name == "time_gap":
            sample[var_name] = data_utils.normalize(np.array(sample[var_name]), config[f"time_cumulative_s_mean"], config[f"time_cumulative_s_std"])
        elif var_name == "dist_gap":
            sample[var_name] = data_utils.normalize(np.array(sample[var_name]), config["dist_cumulative_km_mean"], config["dist_cumulative_km_std"])
        elif f"{var_name}_mean" in config.keys():
            sample[var_name] = data_utils.normalize(np.array(sample[var_name]), config[f"{var_name}_mean"], config[f"{var_name}_std"])
    return sample