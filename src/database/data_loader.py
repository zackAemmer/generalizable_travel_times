import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from database import data_utils


class DeepTTEDataset(Dataset):
    def __init__(self, dataset, transform_list=None, device="cpu"):
        X, y = dataset
        X_tensor, y_tensor = X.to(device), y.to(device)
        self.tensors = (X_tensor, y_tensor)
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
        return self.tensors[0].size(0)

def make_dataset(data, config, device):
    # Get all features that can be made available to the model
    # Coordinates
    start_lng = torch.from_numpy(data_utils.normalize(np.array([x['lngs'][0] for x in data]).astype('float32'), config['lngs_mean'], config['lngs_std'])).unsqueeze(1)
    start_lat = torch.from_numpy(data_utils.normalize(np.array([x['lats'][0] for x in data]).astype('float32'), config['lats_mean'], config['lats_std'])).unsqueeze(1)
    end_lng = torch.from_numpy(data_utils.normalize(np.array([x['lngs'][-1] for x in data]).astype('float32'), config['lngs_mean'], config['lngs_std'])).unsqueeze(1)
    end_lat = torch.from_numpy(data_utils.normalize(np.array([x['lats'][-1] for x in data]).astype('float32'), config['lats_mean'], config['lats_std'])).unsqueeze(1)
    # Schedule
    stop_dist_km = torch.from_numpy(data_utils.normalize(np.array([x['stop_dist_km'] for x in data]).astype('float32'), config['stop_dist_km_mean'], config['stop_dist_km_std'])).unsqueeze(1)
    scheduled_time_s = torch.from_numpy(data_utils.normalize(np.array([x['scheduled_time_s'] for x in data]).astype('float32'), config['scheduled_time_s_mean'], config['scheduled_time_s_std'])).unsqueeze(1)
    # Previous
    speed_m_s = torch.from_numpy(data_utils.normalize(np.array([x['speed_m_s'][0] for x in data]).astype('float32'), config['speed_m_s_mean'], config['speed_m_s_std'])).unsqueeze(1)
    # Network
    timeID = torch.from_numpy(np.array([x['timeID'] for x in data]).astype('float32')).unsqueeze(1)
    weekID = torch.from_numpy(np.array([x['weekID'] for x in data]).astype('float32')).unsqueeze(1)
    # Misc
    driverID = torch.from_numpy(np.array([x['driverID'] for x in data]).astype('float32')).unsqueeze(1)
    dist = torch.from_numpy(data_utils.normalize(np.array([x['dist_gap'][-1] for x in data]).astype('float32'), config['dist_mean'], config['dist_std'])).unsqueeze(1)
    noise = torch.from_numpy(np.random.uniform(0.0, 1.0, len(data)).astype('float32')).unsqueeze(1)

    # Testing different feature combinations
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
    dataset = DeepTTEDataset((X,y), device=device)
    return dataset