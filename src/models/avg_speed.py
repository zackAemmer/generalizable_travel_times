import numpy as np
import pandas as pd

from utils import data_utils, data_loader


class AvgHourlySpeedModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.speed_lookup = {}
        self.requires_grid = False
        self.collate_fn = data_loader.basic_collate
        self.train_time = 0.0
        self.is_nn = False
        return None
    def train(self, dataloader, config):
        data = np.array(dataloader.dataset.content)[dataloader.sampler.indices]
        speeds = np.array([sample['speed_m_s'][0] for sample in data])
        hours = np.array([sample['timeID']//60 for sample in data])
        # Calculate average speed grouped by time of day
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None
    def evaluate(self, dataloader, config):
        data = np.array(dataloader.dataset.content)[dataloader.sampler.indices]
        hours = np.array([sample['timeID']//60 for sample in data])
        dists = np.array([sample['dist'] for sample in data])
        speeds = np.array([self.get_speed_if_available(x) for x in hours])
        preds = dists*1000.0 / speeds
        labels = np.array([sample['time'] for sample in data])
        return labels, preds
    def get_speed_if_available(self, hour):
        # If no data was available for the requested hour, return the mean of all available hours
        if hour in self.speed_lookup['speed'].keys():
            speed = self.speed_lookup['speed'][hour]
            # If there is an hour with 0.0 speeds due to small sample size it will cause errors
            if speed == 0.0:
                return np.mean(list(self.speed_lookup['speed'].values()))
            else:
                return speed
        else:
            return np.mean(list(self.speed_lookup['speed'].values()))
    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None