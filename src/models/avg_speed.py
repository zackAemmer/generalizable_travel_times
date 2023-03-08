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
        speeds = data[:,6]
        speeds = data_utils.de_normalize(speeds.numpy(), self.config['speed_m_s_mean'], self.config['speed_m_s_std'])
        hours = data[:,8]
        hours = hours.numpy() // 60
        hours = hours.astype('int32')
        # Calculate average speed grouped by time of day
        self.speed_lookup = pd.DataFrame({"hour":hours, "speed":speeds}).groupby("hour").mean().to_dict()
        return None

    def predict(self, dataloader):
        data, labels = dataloader.dataset.tensors
        hours = data[:,8]
        hours = hours.numpy() // 60
        hours = hours.astype('int32')
        dists = data[:,7]
        dists = data_utils.de_normalize(dists.numpy(), self.config['dist_mean'], self.config['dist_std'])
        speeds = np.array([self.get_speed_if_available(x) for x in hours], dtype='float32')
        preds = dists*1000.0 / speeds
        labels = data_utils.de_normalize(labels.numpy().flatten(), self.config['time_mean'], self.config['time_std'])
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