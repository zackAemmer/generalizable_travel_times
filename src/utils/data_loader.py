import copy

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
    # Coordinates
    start_lng = torch.from_numpy(data_utils.normalize(np.array([x['lngs'][0] for x in data]).astype('float32'), config['lngs_mean'], config['lngs_std'])).unsqueeze(1)
    start_lat = torch.from_numpy(data_utils.normalize(np.array([x['lats'][0] for x in data]).astype('float32'), config['lats_mean'], config['lats_std'])).unsqueeze(1)
    end_lng = torch.from_numpy(data_utils.normalize(np.array([x['lngs'][-1] for x in data]).astype('float32'), config['lngs_mean'], config['lngs_std'])).unsqueeze(1)
    end_lat = torch.from_numpy(data_utils.normalize(np.array([x['lats'][-1] for x in data]).astype('float32'), config['lats_mean'], config['lats_std'])).unsqueeze(1)
    # Schedule
    stop_dist_km = torch.from_numpy(data_utils.normalize(np.array([x['stop_dist_km'][-1] for x in data]).astype('float32'), config['stop_dist_km_mean'], config['stop_dist_km_std'])).unsqueeze(1)
    scheduled_time_s = torch.from_numpy(data_utils.normalize(np.array([x['scheduled_time_s'][-1] for x in data]).astype('float32'), config['scheduled_time_s_mean'], config['scheduled_time_s_std'])).unsqueeze(1)
    # Previous
    speed_m_s = torch.from_numpy(data_utils.normalize(np.array([x['speed_m_s'][0] for x in data]).astype('float32'), config['speed_m_s_mean'], config['speed_m_s_std'])).unsqueeze(1)
    # Network
    timeID = torch.from_numpy(np.array([x['timeID'] for x in data]).astype('float32')).unsqueeze(1)
    weekID = torch.from_numpy(np.array([x['weekID'] for x in data]).astype('float32')).unsqueeze(1)
    driverID = torch.from_numpy(np.array([x['vehicleID'] for x in data]).astype('float32')).unsqueeze(1)
    dist = torch.from_numpy(data_utils.normalize(np.array([x['dist_gap'][-1] for x in data]).astype('float32'), config['dist_mean'], config['dist_std'])).unsqueeze(1)

    # Features
    X = torch.cat((
        scheduled_time_s,
        start_lng,
        start_lat,
        end_lng,
        end_lat,
        stop_dist_km,
        speed_m_s,
        dist,
        timeID,
        weekID,
        driverID
    ), dim=1)

    # Labels
    y = torch.from_numpy(data_utils.normalize(np.array([x['time_gap'][-1] for x in data]).astype('float32'), config['time_mean'], config['time_std'])).unsqueeze(1)

    # Pack everything into dataset/dataloader for training loop
    dataset = BasicDataset((X,y))
    return dataset

def make_seq_dataset(data, config, seq_len=2):
    # Context variables to embed
    context = np.array([np.array([x['timeID'], x['weekID'], x['vehicleID']]) for x in data], dtype='int32')
    # Last dimension is num sequence variables below
    X = np.zeros((len(data), seq_len+1, 7), dtype='float32')
    # Sequence variables
    for i in range(len(data)):
        X[i,:,0] = data_utils.normalize(np.array(data[i]['lats'][:seq_len+1], dtype='float32'), config['lats_mean'], config['lats_std'])
        X[i,:,1] = data_utils.normalize(np.array(data[i]['lngs'][:seq_len+1], dtype='float32'), config['lngs_mean'], config['lngs_std'])
        X[i,:,2] = data_utils.normalize(np.array(data[i]['dist_gap'][:seq_len+1], dtype='float32'), config['dist_gap_mean'], config['dist_gap_std'])
        X[i,:,3] = data_utils.normalize(np.array(data[i]['scheduled_time_s'][:seq_len+1], dtype='float32'), config['scheduled_time_s_mean'], config['scheduled_time_s_std'])
        X[i,:,4] = data_utils.normalize(np.array(data[i]['stop_dist_km'][:seq_len+1], dtype='float32'), config['stop_dist_km_mean'], config['stop_dist_km_std'])
        X[i,:,5] = data_utils.normalize(np.array(data[i]['speed_m_s'][:seq_len+1], dtype='float32'), config['speed_m_s_mean'], config['speed_m_s_std'])
        X[i,:,6] = data_utils.normalize(np.array(data[i]['time_gap'][:seq_len+1], dtype='float32'), config['time_gap_mean'], config['time_gap_std'])
    # Mask out the speed/time observations for all unknown points
    X[:,seq_len:,5] = 0.0
    X[:,seq_len:,6] = 0.0
    X = torch.from_numpy(X)
    context = torch.from_numpy(context)
    # Prediction variable (speed of final step)
    y = torch.from_numpy(data_utils.normalize(np.array([x['speed_m_s'][seq_len] for x in data], dtype='float32'), config['speed_m_s_mean'], config['speed_m_s_std']))
    X = [(i,j) for i,j in zip(X,context)]
    dataset = BasicDataset((X,y))
    return dataset