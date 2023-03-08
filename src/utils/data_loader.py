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

def make_seq_dataset(data, seq_len=2):
    # For example, if you have 10 GPS points with 2 features (latitude and longitude) for each sample, and you use a batch size of 32, then your input shape would be [10, 32, 2].
    context = np.array([np.array([x['timeID'], x['weekID'], x['vehicleID']]) for x in data])
    # Take only the first n steps of each sequence, keep all attr data
    # Keep info from the predicted step, since it's lat/lon etc. can be used to predict
    X = np.zeros((len(data), seq_len+1, 5))
    for i in range(len(data)):
        X[i,:,0] = data[i]['lats'][:seq_len+1]
        X[i,:,1] = data[i]['lngs'][:seq_len+1]
        X[i,:,2] = data[i]['speed_m_s'][:seq_len+1]
        X[i,:,3] = data[i]['scheduled_time_s'][:seq_len+1]
        X[i,:,4] = data[i]['dist_calc_km'][:seq_len+1]
    X = torch.from_numpy(X.astype('float32'))
    # Predict average speed between the last point in sequence, and the next point
    y = torch.from_numpy(np.array([x['speed_m_s'][seq_len] for x in data]).astype('float32'))
    X = [(i,j) for i,j in zip(X,context)]
    dataset = BasicDataset((X,y))
    return dataset