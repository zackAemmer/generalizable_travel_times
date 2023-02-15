import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from database.data_utils import normalize


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

def make_dataloader(data, config, batch_size, device):
    # Get all features that can be made available to the model
    start_lng = torch.from_numpy(normalize(np.array([x['lngs'][0] for x in data]).astype('float32'), config['lngs_mean'], config['lngs_std'])).unsqueeze(1)
    start_lat = torch.from_numpy(normalize(np.array([x['lats'][0] for x in data]).astype('float32'), config['lats_mean'], config['lats_std'])).unsqueeze(1)
    end_lng = torch.from_numpy(normalize(np.array([x['lngs'][-1] for x in data]).astype('float32'), config['lngs_mean'], config['lngs_std'])).unsqueeze(1)
    end_lat = torch.from_numpy(normalize(np.array([x['lats'][-1] for x in data]).astype('float32'), config['lats_mean'], config['lats_std'])).unsqueeze(1)
    stop_dist_km = torch.from_numpy(normalize(np.array([x['stop_dist_km'] for x in data]).astype('float32'), config['stop_dist_km_mean'], config['stop_dist_km_std'])).unsqueeze(1)
    scheduled_time_s = torch.from_numpy(normalize(np.array([x['scheduled_time_s'] for x in data]).astype('float32'), config['scheduled_time_s_mean'], config['scheduled_time_s_std'])).unsqueeze(1)
    speed_m_s = torch.from_numpy(normalize(np.array([x['speed_m_s'][0] for x in data]).astype('float32'), config['speed_m_s_mean'], config['speed_m_s_std'])).unsqueeze(1)
    timeID = torch.from_numpy(np.array([x['timeID'] for x in data]).astype('float32')).unsqueeze(1)
    weekID = torch.from_numpy(np.array([x['weekID'] for x in data]).astype('float32')).unsqueeze(1)

    # Testing different feature combinations
    X = torch.cat((start_lng, start_lat, end_lng, end_lat, stop_dist_km, scheduled_time_s, speed_m_s, timeID, weekID), dim=1)
    # X = torch.cat((start_lng, start_lat, end_lng, end_lat, timeID, weekID), dim=1)

    # Labels
    y = torch.from_numpy(normalize(np.array([x['time_gap'][-1] for x in data]).astype('float32'), config['time_gap_mean'], config['time_gap_std'])).unsqueeze(1)

    # Pack everything into dataset/dataloader for training loop
    dataset = DeepTTEDataset((X,y), device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    return dataloader