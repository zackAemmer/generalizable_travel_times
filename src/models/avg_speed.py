import numpy as np
import pandas as pd

from utils import data_utils


class AvgHourlySpeedModel:
    def __init__(self, config):
        self.config = config
        self.speed_lookup = {}
        return None

    def fit(self, dataloader):
        data, labels = dataloader.dataset.tensors
        speeds = np.array([x[0][6].numpy() for x in data])
        speeds = data_utils.de_normalize(speeds, self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        hours = np.array([x[1][0].numpy() for x in data]) // 60
        # Calculate average speed grouped by time of day
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None

    def predict(self, dataloader):
        data, labels = dataloader.dataset.tensors
        hours = np.array([x[1][0].numpy() for x in data]) // 60
        dists = np.array([x[0][7].numpy() for x in data])
        dists = data_utils.de_normalize(dists, self.config['dist_mean'], self.config['dist_std'])
        speeds = np.array([self.get_speed_if_available(x) for x in hours], dtype='float32')
        preds = dists*1000.0 / speeds
        labels = data_utils.de_normalize(labels.numpy(), self.config['time_mean'], self.config['time_std'])
        return labels, preds

    def get_speed_if_available(self, hour):
        # If no data was available for the requested hour, return the mean of all available hours
        if hour in self.speed_lookup['speed'].keys():
            return self.speed_lookup['speed'][hour]
        else:
            return np.mean(list(self.speed_lookup['speed'].values()))

    def save_to(self, path):
        data_utils.write_pkl(self, path)
        return None