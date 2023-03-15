import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import data_utils


class BasicDataset(Dataset):
    def __init__(self, dataset, transform_list=None):
        self.tensors = dataset
        self.transforms = transform_list

    def __getitem__(self, index):
        # Return the X and y tensors at the specified index. Transform X if applicable.
        x = self.tensors[0][index]
        if self.transforms:
            x = self.transforms(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.tensors[1])

def make_dataset(data, config):
    # Context variables to embed
    context = np.array([np.array([x['timeID'], x['weekID'], x['vehicleID'], x['tripID']]) for x in data], dtype='int32')
    X = np.zeros((len(data), 8), dtype='float32')
    # Sequence variables
    for i in range(len(data)):
        X[i,0] = data_utils.normalize(np.array(data[i]['lats'][0], dtype='float32'), config['lats_mean'], config['lats_std'])
        X[i,1] = data_utils.normalize(np.array(data[i]['lngs'][0], dtype='float32'), config['lngs_mean'], config['lngs_std'])
        X[i,2] = data_utils.normalize(np.array(data[i]['lats'][-1], dtype='float32'), config['lats_mean'], config['lats_std'])
        X[i,3] = data_utils.normalize(np.array(data[i]['lngs'][-1], dtype='float32'), config['lngs_mean'], config['lngs_std'])
        X[i,4] = data_utils.normalize(np.array(data[i]['scheduled_time_s'][-1], dtype='float32'), config['scheduled_time_s_mean'], config['scheduled_time_s_std'])
        X[i,5] = data_utils.normalize(np.array(data[i]['stop_dist_km'][-1], dtype='float32'), config['stop_dist_km_mean'], config['stop_dist_km_std'])
        X[i,6] = data_utils.normalize(np.array(data[i]['speed_m_s'][0], dtype='float32'), config['speed_m_s_mean'], config['speed_m_s_std'])
        X[i,7] = data_utils.normalize(np.array(data[i]['dist'], dtype='float32'), config['dist_mean'], config['dist_std'])
    X = torch.from_numpy(X)
    context = torch.from_numpy(context)
    # Prediction variable (travel time from first to last observation)
    y = torch.from_numpy(data_utils.normalize(np.array([x['time'] for x in data], dtype='float32'), config['time_mean'], config['time_std']))
    X = [(i,j) for i,j in zip(X,context)]
    dataset = BasicDataset((X,y))
    return dataset

def make_seq_dataset(data, config):
    # Context variables to embed
    context = np.array([np.array([x['timeID'], x['weekID'], x['vehicleID'], x['tripID']]) for x in data], dtype='int32')
    # Last dimension is number of sequence variables below
    seq_lens = [len(x['lats']) for x in data]
    max_len = np.max(seq_lens)
    mask = np.array([np.concatenate([np.repeat(True,slen), np.repeat(False,max_len-slen)]) for slen in seq_lens], dtype='bool')
    X = np.zeros((len(data), max_len, 7), dtype='float32')
    # Sequence variables
    X[:,:,0] = torch.nn.utils.rnn.pad_sequence([torch.tensor(data_utils.normalize(np.array(x['lats'], dtype='float32'), config['lats_mean'], config['lats_std'])) for x in data], batch_first=True)
    X[:,:,1] = torch.nn.utils.rnn.pad_sequence([torch.tensor(data_utils.normalize(np.array(x['lngs'], dtype='float32'), config['lngs_mean'], config['lngs_std'])) for x in data], batch_first=True)
    X[:,:,2] = torch.nn.utils.rnn.pad_sequence([torch.tensor(data_utils.normalize(np.array(x['dist_calc_km'], dtype='float32'), config['dist_calc_km_mean'], config['dist_calc_km_std'])) for x in data], batch_first=True)
    X[:,:,3] = torch.nn.utils.rnn.pad_sequence([torch.tensor(data_utils.normalize(np.array(x['scheduled_time_s'], dtype='float32'), config['scheduled_time_s_mean'], config['scheduled_time_s_std'])) for x in data], batch_first=True)
    X[:,:,4] = torch.nn.utils.rnn.pad_sequence([torch.tensor(data_utils.normalize(np.array(x['stop_dist_km'], dtype='float32'), config['stop_dist_km_mean'], config['stop_dist_km_std'])) for x in data], batch_first=True)
    # Note that last two have info that should not be included with RNN model
    X[:,:,5] = torch.nn.utils.rnn.pad_sequence([torch.tensor(data_utils.normalize(np.array(x['time_gap'], dtype='float32'), config['time_gap_mean'], config['time_gap_std'])) for x in data], batch_first=True)
    X[:,:,6] = torch.nn.utils.rnn.pad_sequence([torch.tensor(data_utils.normalize(np.array(x['speed_m_s'], dtype='float32'), config['speed_m_s_mean'], config['speed_m_s_std'])) for x in data], batch_first=True)
    X = torch.from_numpy(X)
    context = torch.from_numpy(context)
    # Prediction variable (speed of final step)
    y = torch.nn.utils.rnn.pad_sequence([torch.tensor(data_utils.normalize(np.array(x['speed_m_s'], dtype='float32'), config['speed_m_s_mean'], config['speed_m_s_std'])) for x in data], batch_first=True)
    X = [(i,j) for i,j in zip(X,context)]
    dataset = BasicDataset((X,y))
    return dataset, mask

def make_dataloader(data, config, batch_size, task_type):
    if task_type == "basic":
        dataset = make_dataset(data, config)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
        return dataloader
    elif task_type == "sequential":
        dataset, mask = make_seq_dataset(data, config)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
        return dataloader, mask