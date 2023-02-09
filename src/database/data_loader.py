import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DeepTTEDataset(Dataset):
    def __init__(self, dataset, transform_list=None):
        X_tensor, y_tensor = dataset
        tensors = (X_tensor, y_tensor)
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transforms:
            x = self.transforms(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    
def make_dataloader(data, batch_size):
    start_lng = np.array([x['lngs'][0] for x in data]).astype('float32')
    start_lat= np.array([x['lats'][0] for x in data]).astype('float32')
    end_lng = np.array([x['lngs'][-1] for x in data]).astype('float32')
    end_lat= np.array([x['lats'][-1] for x in data]).astype('float32')
    timeID = np.array([x['timeID'] for x in data]).astype('float32')
    weekID = np.array([x['weekID'] for x in data]).astype('float32')
    y = torch.from_numpy(np.array([x['time_gap'][-1] for x in data]).astype('float32'))
    X = torch.from_numpy(np.column_stack([start_lng, start_lat, end_lng, end_lat, timeID, weekID]))
    dataset = DeepTTEDataset((X,y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader